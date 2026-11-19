#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import torch
import torch.nn as nn


# In[ ]:


USE_LSTM = False


# In[ ]:


if USE_LSTM == False:
    from mamba_ssm import Mamba3   
else:
    print("USING LSTM NOT Mamba")


# In[ ]:


class BirectionalMambaLayer(nn.Module):
    def __init__(self,d_model=256,d_state=16,d_conv=4,expand=2,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.RMSNorm(d_model)

        if(USE_LSTM):

            self.forward_layer = nn.LSTM(
                    input_size=d_model,
                    hidden_size=d_model,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,  # we handle bidirectional at encoder level
                )
            self.backward_layer = nn.LSTM(
                        input_size=d_model,
                        hidden_size=d_model,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=False,  # we handle bidirectional at encoder level
                    )
        else:

            self.forward_layer = Mamba3(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand)

            self.backward_layer = Mamba3(
                                d_model=d_model,
                                d_state=d_state,
                                d_conv=d_conv,
                                expand=expand)

        self.fusion = nn.Linear(2*d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        residual = x

        x= self.norm(x)
        if(USE_LSTM):
            forward_out, _ = self.forward_layer(x)
            backward_out, _ = self.backward_layer(x.flip(dims=[1]))

        else:
            forward_out = self.forward_layer(x)
            backward_out = self.backward_layer(x.flip(dims=[1]))

        backward_out = backward_out.flip(dims=[1])

        merged = torch.cat([forward_out, backward_out],dim=-1)

        merged = self.fusion(merged)

        merged = self.dropout(merged)

        return merged + residual





if __name__ == "__main__":
    Encoder = BirectionalMambaLayer()
    op = Encoder(torch.rand([3,2,256]))


# In[ ]:


class ConvDownSample(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels=d_model,out_channels=d_model,kernel_size=3,stride=2,padding=1)
        self.norm = nn.RMSNorm(d_model)
    def forward(self,x):
        x=x.transpose(1,2)
        x= self.conv(x)
        x = x.transpose(1,2)
        x = self.norm(x)
        return x
   
# m = ConvDownSample()

# m(torch.randn(([1,4,256]))).shape


# In[ ]:





# In[ ]:


class BidirectionalMambaEncoder(nn.Module):
    def __init__(self,n_mels=80,d_model=256,d_state=16,
                 d_conv=4,expand=2,dropout=0.1,downsample_after=3,n_layers=6 ):

        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.downsample_after = downsample_after

        self.input_proj = nn.Linear(n_mels,d_model)

        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            self.layers.append(BirectionalMambaLayer(d_model=d_model,
                                                     d_state=d_state,
                                                     d_conv=d_conv,expand=expand,dropout=dropout))

        self.downsample = ConvDownSample(d_model=d_model)
        self.final_norm = nn.RMSNorm(d_model)


    def forward(self,mel,padding_mask):
        x = mel.transpose(1,2)
        x= self.input_proj(x)
        x = self.input_dropout(x)
        if(padding_mask is not None):
            x = x.masked_fill(padding_mask.unsqueeze(-1),0.0) # (4,400,1)

        for i,layer in enumerate(self.layers):
            x = layer(x)
            if i == self.downsample_after - 1:
                x = self.downsample(x)
                if padding_mask is not None:
                    padding_mask = padding_mask[:,::2]
                    x = x.masked_fill(padding_mask.unsqueeze(-1),0.0)

        x=self.final_norm(x)
        return x



if __name__ == "__main__":
    bdme = BidirectionalMambaEncoder()
    op = bdme(torch.randint(0, 2, size=(4, 80, 2)).to(dtype=torch.float),
              torch.randint(0, 2, size=(4, 2), dtype=torch.bool))

# print(op, op.shape)


# In[ ]:


class MambaDecoderLayer(nn.Module):
    def __init__(self,d_model=256,d_state=16,d_conv=4,expand=2,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.RMSNorm(d_model)

        if(USE_LSTM):

            self.forward_layer = nn.LSTM(
                    input_size=d_model,
                    hidden_size=d_model,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,
                )

        else:

            self.forward_layer = Mamba3(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand)




        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        residual = x

        x= self.norm(x)
        if(USE_LSTM):
            x,_ = self.forward_layer(x)

        else:
            x = self.forward_layer(x)

        x = self.dropout(x)

        return x + residual





if __name__ == "__main__":
    Decoder = MambaDecoderLayer()
    # was torch.rand([3,256]): Mamba3 does `batch, seqlen, dim = u.shape`, so a
    # 2-D tensor raised "not enough values to unpack (expected 3, got 2)".
    op = Decoder(torch.rand([3, 2, 256]))

# print(op.shape)


# In[ ]:


class ConvUpSample(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()

        self.conv = nn.ConvTranspose1d(in_channels=d_model,out_channels=d_model,kernel_size=4,stride=2,padding=1)
        self.norm = nn.RMSNorm(d_model)
    def forward(self,x):
        x=x.transpose(1,2)
        x= self.conv(x)
        x = x.transpose(1,2)
        x = self.norm(x)
        return x

if __name__ == "__main__":
    m = ConvUpSample()
    print(m(torch.randn([1,4,256])).shape)


# In[ ]:


class CausalMambaDecoder(nn.Module):
    def __init__(self,n_mels=80,d_model=256,d_state=16,
                 d_conv=4,expand=2,dropout=0.1,n_layers=6 ):

        super().__init__()

        self.d_model = d_model
        self.d_state = d_state




        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            self.layers.append(MambaDecoderLayer(d_model=d_model,
                                                     d_state=d_state,
                                                     d_conv=d_conv,expand=expand,dropout=dropout))

        self.upsample = ConvUpSample(d_model=d_model)
        self.final_norm = nn.RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_mels)

    def forward(self,x):


        x = self.input_dropout(x)
        x = self.upsample(x)

        for i,layer in enumerate(self.layers):
            x = layer(x)


        x=self.final_norm(x)
        mel = self.out_proj(x)

        return mel



if __name__ == "__main__":
    cmd = CausalMambaDecoder()
    op = cmd(torch.rand([1,22,256]))

# print(op.shape)

