#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import torch.nn.functional as F


# In[ ]:


class SpeakerAdversary(nn.Module):
    
    def __init__(self,input_dim, hidden_dim, num_speakers=100, grl_scale=1.0):
        
        super().__init__()
        self.grl_scale = grl_scale
        self.current_scale = 0.0
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim,num_speakers)
        )
        
    def set_current_scale(self,scale):
        self.current_scale = scale
    
    def forward(self,x,speaker_ids):
        
        pooled = x.mean(dim=1)
        
        pooled  = GradientReversalFn.apply(pooled,self.current_scale)
        
        logits = self.classifier(pooled)  # (B, num_speakers)
        loss = F.cross_entropy(logits, speaker_ids)
 
        return loss
 
        


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reversal
        return -ctx.scale * grad_output, None


# In[ ]:


if __name__ == "__main__":
    sa = SpeakerAdversary(256,128,100)
    # speaker_ids are class indices for cross_entropy, not a float matrix
    print(sa(torch.rand([31,12,256]), torch.randint(0, 100, (31,))))

