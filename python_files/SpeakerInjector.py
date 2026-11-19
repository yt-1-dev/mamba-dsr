#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn

class SpeakerInjector(nn.Module):
    def __init__(self, vq_dim=256, spk_dim=256, d_model=256):
        super().__init__()
        self.proj = nn.Linear(vq_dim+spk_dim, d_model)
    def forward(self,x, speaker_emb):
        B,T,_ = x.shape
        spk_expanded = speaker_emb.unsqueeze(1).expand(-1,T,-1)

        combined = torch.cat([x,spk_expanded],dim=-1)
        return self.proj(combined)





if __name__ == "__main__":
    si = SpeakerInjector()
    print(si(torch.rand([2,12,256]), torch.rand([2,256])).shape)


