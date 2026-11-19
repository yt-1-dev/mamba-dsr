#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import argparse
from pathlib import Path
from collections import defaultdict
 
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
 
from dataloader import create_libritts_loader, create_uaspeech_loader


# In[ ]:


def train(mode, data_dir, checkpoint, device,num_speakers=100):
    
    pipeline =SpeechRe

