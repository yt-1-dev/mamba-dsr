#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from pathlib import Path
from collections import defaultdict
from typing import Optional
 
import torch
from torch.utils.data import Dataset, DataLoader
LIBRE_TTS_PATH = '../data/processed/libreTTS/'
UA_PATH = '../data/processed/ua/'

# raw wavs, for ECAPA speaker embeddings (the *_PATH dirs above hold .pt mels only)
LIBRE_TTS_RAW_PATH = '../data/LibriTTS/train-clean-100/'
UA_RAW_PATH = '../data/Noise Reduced UASPEECH Dysarthria Dataset/noisereduced-uaspeech/'


# In[2]:


class LibreTTSDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.samples = []
        self.speakers = set()
        self.max_frames = 860
        self.min_frames = 86
        self.raw_audio_path = LIBRE_TTS_RAW_PATH
         
        
        print(Path(LIBRE_TTS_PATH).rglob("*.pt"))

        for pt_file in sorted(Path(LIBRE_TTS_PATH).rglob("*.pt")):

            self.samples.append({
                "path":str(pt_file),
                "speaker_id": pt_file.parent.name
            })

            self.speakers.add(pt_file.parent.name)

    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
 
        data = torch.load(sample["path"], weights_only=False)
        mel = data["mel"]  # (80, T)
 
        n_frames = mel.shape[-1]
 
        # Skip if too short
        if n_frames < self.min_frames:
            return self.__getitem__((idx + 1) % len(self))
 
        # Random crop if too long
        if n_frames > self.max_frames:
            start = random.randint(0, n_frames - self.max_frames)
            mel = mel[:, start:start + self.max_frames]
            n_frames = self.max_frames
 
        return {
            "input_mel": mel,           # (80, T)
            "target_mel": mel.clone(),  # same — self-reconstruction
            "speaker_id": sample["speaker_id"],
            "input_length": n_frames,
            "target_length": n_frames,
        }



# In[3]:


def collate_fn(batch):
    input_mels = [b["input_mel"] for b in batch]
    target_mels = [b["target_mel"] for b in batch]
    target_lengths = [b["target_length"] for b in batch]
    input_lengths = [b["input_length"] for b in batch]
    
    max_input_len = max(input_lengths)
    max_target_len = max(target_lengths)
    
    
    ##Input 
    padded_inputs = [] 
    input_masks = []
    for mel,len in zip(input_mels,input_lengths):
        n_mels,T = mel.shape
        
        if T < max_input_len:
            mel = torch.concat([mel,torch.zeros(n_mels,max_input_len-T)],dim=-1)

        padded_inputs.append(mel)
        mask = torch.zeros(max_input_len,dtype=torch.bool)
        mask[len:] = True
        input_masks.append(mask)    

    #Target
    padded_targets = [] 
    target_masks = []
    for mel,len in zip(target_mels,target_lengths):
        n_mels,T = mel.shape
        
        if T < max_target_len:
            mel = torch.concat([mel,torch.zeros(n_mels,max_target_len-T)],dim=-1)

        padded_targets.append(mel)
        mask = torch.zeros(max_target_len,dtype=torch.bool)
        mask[len:] = True
        target_masks.append(mask)
        
    return {
         "input_mel": torch.stack(padded_inputs),         
        "target_mel": torch.stack(padded_targets),       
        "input_lengths": torch.tensor(input_lengths),    
        "target_lengths": torch.tensor(target_lengths),  
        "padding_mask": torch.stack(input_masks),        
        "target_mask": torch.stack(target_masks),        
        "speaker_ids": [b["speaker_id"] for b in batch],
    }
        

if __name__ == "__main__":
    dl = DataLoader(
        LibreTTSDataset(),
        batch_size=2,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=collate_fn
    )
    

    data = next(iter(dl))
    assert data['input_mel'].shape[0] == 2
    assert data['padding_mask'].shape[1] == data['input_mel'].shape[-1]
    print("LibreTTS ok:", data['input_mel'].shape)


# ## UASpeech

# In[4]:


SEVERITY = {
    "M01": "low",     "M04": "low",     "M05": "mid",
    "M07": "mid",     "M08": "mid",     "M09": "mid",
    "M10": "high",    "M11": "low",     "M12": "high",
    "M14": "mid",
    "F02": "low",     "F03": "mid",     "F04": "mid",
    "F05": "high",
}
 
DYSARTHRIC_SPEAKERS = set(SEVERITY.keys())
 
CONTROL_SPEAKERS = {
    "CM01", "CM04", "CM05", "CM06", "CM08", "CM09",
    "CM10", "CM12", "CM13",
    "CF02", "CF03", "CF04", "CF05",
}

REFERENCE_CONTROL = "CF02"   # targets are DTW-aligned to this speaker in preprocess.ipynb


# In[8]:


class UASpeechDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.max_frames = 860
        
        self.word_to_dysarthric = defaultdict(list)
        self.word_to_control = defaultdict(list)
        self.speakers = set()
        
        self.raw_audio_path = UA_RAW_PATH
        # ponytail: speaker/word live in the filename (SPK_B1_CW75_M8.pt), so the index
        # builds without torch.load-ing all ~136k shards.
        for pt_file in sorted(Path(UA_PATH).rglob("*.pt")):
            
            parts = pt_file.stem.split("_")
            if len(parts) != 4:
                continue
            speaker_id, block, word_id, _mic = parts
            word = f"{block}_{word_id}"
            
            if(speaker_id in DYSARTHRIC_SPEAKERS):
                self.speakers.add(speaker_id)
                self.word_to_dysarthric[word].append((speaker_id,str(pt_file)))
                
            elif(speaker_id in CONTROL_SPEAKERS):
                self.word_to_control[word].append((speaker_id,str(pt_file)))
                
                
    
        self.pairs = [];
        
        for word in self.word_to_dysarthric:
            
            control_options = self.word_to_control.get(word, [])
            if not control_options:      # no control recording of this word -> no target
                continue
            
            options = dict(control_options)
            if REFERENCE_CONTROL not in options:
                continue
            
            for dys_speaker, dys_path in self.word_to_dysarthric[word]:
                self.pairs.append({
                    "dysarthric_path": dys_path,
                    "dysarthric_speaker": dys_speaker,
                    "word":word,
                    "severity":SEVERITY.get(dys_speaker),
                    "control_options": control_options,
                })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        data = torch.load(pair["dysarthric_path"], weights_only=False)
        input_mel = data["mel"]
        # KeyError here = run the DTW cell at the end of preprocess.ipynb
        target_mel = data["dtw_target"]   # REFERENCE_CONTROL's take, aligned frame-by-frame to input_mel
        
 
        
     
      
        
        
        
        if input_mel.shape[-1] > self.max_frames:
            input_mel = input_mel[:, :self.max_frames]
        if target_mel.shape[-1] > self.max_frames:
            target_mel = target_mel[:, :self.max_frames]

        
            
            
            
            
        
        
        return {
            "input_mel": input_mel,
            "target_mel": target_mel,
            "speaker_id": pair["dysarthric_speaker"],
            "input_length": input_mel.shape[-1],
            "target_length": target_mel.shape[-1],
            "word": pair["word"],
            "severity": pair["severity"],
            "control_speaker": REFERENCE_CONTROL,
        }
            
          

if __name__ == "__main__":
    dl = DataLoader(
        UASpeechDataset(),
        batch_size=2,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=collate_fn
    )

    b = next(iter(dl))
    # input and target are different utterances, so they pad to different lengths
    assert b['target_mask'].shape[1] == b['target_mel'].shape[-1]
    assert b['padding_mask'].shape[1] == b['input_mel'].shape[-1]
    assert set(b['speaker_ids']) <= DYSARTHRIC_SPEAKERS
    print("UASpeech ok:", len(dl.dataset), "pairs,", len(dl.dataset.speakers), "speakers")


# In[ ]:




