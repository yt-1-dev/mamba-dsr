#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes = 1024,code_dim = 64,decay=0.99,epsilon=1e-5, reset_thresh = 2):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.epsilon = epsilon
        self.reset_thresh = reset_thresh

        embedding = torch.randn(num_codes,code_dim)

        # not learnable parameters
        self.register_buffer("embedding",embedding)

        self.register_buffer("ema_cluster_size",torch.zeros(num_codes))
        self.register_buffer("ema_embedding_sum",embedding.clone())

        self.register_buffer('initialized', torch.zeros(1))
        # mirrors the buffer so the common path never calls .item() (a GPU sync)
        self._init_done = False
        self.verbose = False

    def forward(self,x):

        B,T,D = x.shape
        flat = x.reshape(-1,D) #B*T,D

        #Inialize the notebook - Pseudo random intialization
        if not self._init_done and self.initialized.item():
            self._init_done = True          # restored from a checkpoint

        if not self._init_done:

            n = flat.shape[0]

            if n>=self.num_codes:
                # Generates a random permutation of integers from 0 to n-1.
                indices = torch.randperm(n,device=flat.device)[:self.num_codes]
                self.embedding.data.copy_(flat[indices])
            else:
                ## Repeat the input
                self.embedding.data.copy_(flat.repeat((self.num_codes//n)+1, 1)[:self.num_codes])

            self.ema_embedding_sum.data.copy_(self.embedding.data.clone())
            self.ema_cluster_size.data.fill_(1.0)
            self.initialized.fill_(True)
            self._init_done = True



        distances = flat.pow(2).sum(dim=1, keepdim=True) + self.embedding.pow(2).sum(dim=1, keepdim=True).t() - 2* flat @ self.embedding.t()

        codes_flat = distances.argmin(dim=1)


        encodings = F.one_hot(codes_flat,self.num_codes).float()

        # entropy_loss = _compute_entropy_loss(encodings)

        # entropy_loss is -H, so perplexity = exp(H) = exp(-entropy_loss)
        # perplexity = torch.exp(-entropy_loss)   # effective codes in use, max = num_codes

        # print(entropy_loss, perplexity)

        quantized_flat = F.embedding(codes_flat,self.embedding)
        if self.training:
            with torch.no_grad():

                batch_cluster_size = encodings.sum(0)  # (num_codes,)
                self.ema_cluster_size.mul_(self.decay).add_(
                    batch_cluster_size, alpha=1 - self.decay
                )

                # Update embedding sums
                batch_embedding_sum = encodings.t() @ flat  # (num_codes, D)
                self.ema_embedding_sum.mul_(self.decay).add_(
                    batch_embedding_sum, alpha=1 - self.decay
                )

                # Laplace smoothing to avoid division by zero
                n = self.ema_cluster_size.sum()
                smoothed_cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_codes * self.epsilon)
                    * n
                )

                # Update codebook
                self.embedding.data.copy_(
                    self.ema_embedding_sum / smoothed_cluster_size.unsqueeze(1)
                )

                # Reset dead codes
                self._reset_dead_codes(flat, encodings)

         # Straight-through estimator: forward uses quantized, backward copies
        quantized_flat = flat + (quantized_flat - flat).detach()

        quantized = quantized_flat.reshape(B, T, D)
        codes = codes_flat.reshape(B, T)

        return quantized, codes, encodings



    def _reset_dead_codes(self,flatten_ip,encodings):
        with torch.no_grad():
            usage = encodings.sum(0)
            dead_mask = usage< self.reset_thresh
            num_dead = dead_mask.sum().item()
            # print(num_dead)
            if(num_dead == 0):
                return
            elif self.verbose:
                # was a missing-f string printing literal braces, on every forward
                print(f"{num_dead} codes will be reset")
            n = flatten_ip.shape[0]
            indices = torch.randint(0,n,(num_dead,),device=flatten_ip.device)
            replacements = flatten_ip[indices]

            self.embedding.data[dead_mask] = replacements
            self.ema_embedding_sum.data[dead_mask] = replacements
            self.ema_cluster_size.data[dead_mask] = 1.0




def _compute_entropy_loss(encodings: torch.Tensor) -> torch.Tensor:

    avg_usage = encodings.mean(dim=0) + 1e-10

    return (avg_usage * avg_usage.log()).sum()         # = -entropy


if __name__ == "__main__":
    vq = VectorQuantizerEMA(code_dim=5)
    vq.train()
    for i in range(3):
        vq(torch.rand([500,10,5]))


# In[ ]:


class ProductQuantizer(nn.Module):
    def __init__(self,input_dim=512,num_groups=4,num_codes = 1024, code_dim=64,
                 commitment_weight=1.0, entropy_weight=0.1, decay=0.9, reset_thresh=2):
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.entropy_weight = entropy_weight

        projected_dim = num_groups * code_dim

        self.pre_proj = nn.Linear(input_dim, projected_dim)


        self.post_proj = nn.Linear(projected_dim, projected_dim)

        # Independent quantizer per group
        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(
                num_codes=num_codes,
                code_dim=code_dim,
                decay=decay,
                reset_thresh=reset_thresh,
            )
            for _ in range(num_groups)
        ])

        # Layer norm before quantization (stabilizes training)
        self.pre_norm = nn.LayerNorm(projected_dim)

    def forward(self,x):
        B,T,D = x.shape
        x_proj = self.pre_proj(x)
        x_proj = self.pre_norm(x_proj)


        x_groups = x_proj.chunk(self.num_groups,dim=-1)

        quantized_groups = []
        all_codes = []
        all_encodings = []
        total_commitment = torch.tensor(0.0,device=x.device)

        for g,(x_g,quantizer) in enumerate(zip(x_groups,self.quantizers)):

            q_g,codes_g,encodings_g = quantizer(x_g)
            all_codes.append(codes_g)
            all_encodings.append(encodings_g)
            quantized_groups.append(q_g)
            commitment = F.mse_loss(x_g, q_g.detach())
            total_commitment += commitment

        entropy_loss, perplexity, active_ratio = self.compute_metrics(all_encodings)

        commitment_loss = total_commitment / self.num_groups

        total_loss = self.commitment_weight*commitment_loss + self.entropy_weight*entropy_loss

        quantized = torch.cat(quantized_groups, dim=-1)  # (B, T, G*code_dim)
        quantized = self.post_proj(quantized)

        # Stack codes: (B, T, G)
        codes = torch.stack(all_codes, dim=-1)

        return {
            "total_loss":total_loss,
            "codes":codes,
            "quantized":quantized,
            "perplexity":perplexity,
            "active_ratio":active_ratio,
            "entropy_loss":entropy_loss
        }

    def compute_metrics(self,all_encodings):
        # everything stays on-device: the old version called .item() twice per
        # group, forcing ~8 GPU syncs per forward.
        device = all_encodings[0].device
        entropy_loss  = torch.zeros((), device=device)
        total_entropy = torch.zeros((), device=device)
        total_active  = torch.zeros((), device=device)

        for encodings in all_encodings:
            avg_usage = encodings.mean(0) + 1e-10 #avoid log 0
            entropy = (avg_usage * torch.log(avg_usage)).sum()   # this is -H
            entropy_loss = entropy_loss - entropy
            total_entropy = total_entropy + entropy
            total_active  = total_active + (encodings.sum(dim=0) > 0).sum()

        avg_entropy = total_entropy / self.num_groups            # mean of -H

        # BEHAVIOUR CHANGE: was exp(avg_entropy) = exp(-H), which inverts the
        # metric -- collapse showed as HIGH perplexity. Correct is exp(H).
        perplexity = torch.exp(-avg_entropy)

        active_ratio = total_active.float() / (self.num_groups * self.num_codes)
        entropy_loss = entropy_loss/self.num_groups

        return entropy_loss, perplexity, active_ratio




if __name__ == "__main__":
    pq = ProductQuantizer(input_dim=256)
    pq.train()
    op = pq(torch.rand([1,22,256]))
    print(op['quantized'].shape)


# In[ ]:




