#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict
 
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

CONFIG = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mels": 80,
    "f_min": 0.0,
    "f_max": 8000.0,
    "min_duration_sec": 1.0,
    "max_duration_sec": 10.0,
    "silence_threshold_db": -40.0,
    "target_lufs": -23.0,
}
 


# In[3]:


mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=CONFIG["sample_rate"],
    n_fft=CONFIG["n_fft"],
    hop_length=CONFIG["hop_length"],
    win_length=CONFIG["win_length"],
    n_mels=CONFIG["n_mels"],
    f_min=CONFIG["f_min"],
    f_max=CONFIG["f_max"],
    power=2.0,
    norm="slaney",
    mel_scale="slaney",
)


# In[4]:


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform)
 
 
def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert multi-channel to mono."""
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        return waveform.mean(dim=0, keepdim=True)
    return waveform
 
 
def normalize_loudness(waveform: torch.Tensor, target_lufs: float = -23.0) -> torch.Tensor:
    """Approximate LUFS normalization using RMS."""
    rms = waveform.pow(2).mean().sqrt()
    if rms < 1e-8:
        return waveform
    target_rms = 10 ** ((target_lufs + 0.691) / 20)
    waveform = waveform * (target_rms / rms)
    return torch.clamp(waveform, -1.0, 1.0)
 
 
def trim_silence(waveform: torch.Tensor, threshold_db: float = -40.0) -> torch.Tensor:
    """Trim leading and trailing silence."""
    threshold = 10 ** (threshold_db / 20)
    amplitude = waveform.abs().squeeze(0)
    above = (amplitude > threshold).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        return waveform
    start = above[0].item()
    end = above[-1].item() + 1
    return waveform[:, start:end]
 
 
def extract_mel(waveform: torch.Tensor) -> torch.Tensor:
    """Extract log-mel spectrogram."""
    mel = mel_transform(waveform)  # (1, n_mels, T)
    mel = mel.squeeze(0)           # (n_mels, T)
    mel = torch.clamp(mel, min=1e-5)
    mel = torch.log(mel)
    return mel
 
 
def preprocess_utterance(
    waveform: torch.Tensor,
    orig_sr: int,
) -> dict:
    """
    Full preprocessing pipeline for a single utterance.
 
    Returns:
        dict with 'mel', 'duration_sec', 'n_frames', or None if rejected
    """
    sr = CONFIG["sample_rate"]
 
    # Resample
    waveform = resample(waveform, orig_sr, sr)
 
    # Mono
    waveform = to_mono(waveform)
 
    # Normalize
    waveform = normalize_loudness(waveform, CONFIG["target_lufs"])
 
    # Trim silence
    waveform = trim_silence(waveform, CONFIG["silence_threshold_db"])
 
    # Duration check
    duration = waveform.shape[-1] / sr
    if duration < CONFIG["min_duration_sec"]:
        return None  # too short
    if duration > CONFIG["max_duration_sec"]:
        # Crop to max duration (from the start for simplicity)
        max_samples = int(CONFIG["max_duration_sec"] * sr)
        waveform = waveform[:, :max_samples]
        duration = CONFIG["max_duration_sec"]
 
    # Extract mel
    mel = extract_mel(waveform)
 
    return {
        "mel": mel,
        "duration_sec": duration,
        "n_frames": mel.shape[-1],
    }
 


# In[5]:


dataset = torchaudio.datasets.LIBRITTS(
            root='../data',
            url="train-clean-100",
            download=False,
        )


# In[6]:


test_1 = dataset[20]
print(test_1[0].shape)
processed = preprocess_utterance(test_1[0],test_1[1])
print(processed)


# In[7]:


stats = {
        "total_utterances": 0,
        "processed": 0,
        "skipped_short": 0,
        "skipped_error": 0,
        "speakers": defaultdict(int),
        "total_duration_sec": 0.0,
        "mel_frames": [],
    }

output_path = "../data/processed/libreTTS"

for idx in tqdm(range(len(dataset)), desc="LibriTTS-clean-100"):
    try:
        # torchaudio LIBRITTS returns:
        # (waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id)
        item = dataset[idx]
        waveform = item[0]
        sample_rate = item[1]
        speaker_id = str(item[4])
        utterance_id = f"{item[4]}_{item[5]}_{item[6]}"

        stats["total_utterances"] += 1

        # Preprocess
        result = preprocess_utterance(waveform, sample_rate)

        if result is None:
            stats["skipped_short"] += 1
            continue

        # Save
        speaker_dir = Path(f"{output_path}/{speaker_id}")
        speaker_dir.mkdir(exist_ok=True,parents=True)

        save_data = {
            "mel": result["mel"],
            "speaker_id": speaker_id,
            "duration_sec": result["duration_sec"],
            "n_frames": result["n_frames"],
            "utterance_id": utterance_id,
        }

        save_path = speaker_dir/f"{utterance_id}.pt"
        torch.save(save_data, save_path)

        stats["processed"] += 1
        stats["speakers"][speaker_id] += 1
        stats["total_duration_sec"] += result["duration_sec"]
        stats["mel_frames"].append(result["n_frames"])

    except Exception as e:
        stats["skipped_error"] += 1
        print(e)
        print(f"  Error on utterance {idx}: {e}")


# In[8]:


print(stats)


# ### UASPEECH
# 

# In[9]:


input_path = Path('../data/Noise Reduced UASPEECH Dysarthria Dataset/')
output_path = Path('../data/processed/ua')
wav_files = sorted(input_path.rglob("*_*_*.wav"))
print(len(wav_files))
output_path.mkdir(parents=True, exist_ok=True)


# In[10]:


ua_stats = {
        "total_utterances": 0,
        "processed": 0,
        "skipped_short": 0,
        "skipped_error": 0,
        "speakers": defaultdict(int),
        "total_duration_sec": 0.0,
        "mel_frames": [],
        "dysarthric_count": 0,
        "control_count": 0,
    }

for wav_path in tqdm(wav_files, desc="UA-Speech"):
    filename = wav_path.stem
    parts = filename.split("_")

    
    speaker_id = wav_path.parent.name


    stats["total_utterances"] += 1

    waveform, sample_rate = torchaudio.load(wav_path)

    # Preprocess
    result = preprocess_utterance(waveform, sample_rate)

    if result is None:
        stats["skipped_short"] += 1
        continue

    # Determine if dysarthric or control
    is_dysarthric = speaker_id.startswith('F') or speaker_id.startswith('M')
    is_control = speaker_id.startswith('C')

    speaker_dir = output_path / speaker_id
    speaker_dir.mkdir(exist_ok=True)

    save_data = {
        "mel": result["mel"],
        "speaker_id": speaker_id,
        "duration_sec": result["duration_sec"],
        "n_frames": result["n_frames"],
        "filename": filename,
        "is_dysarthric": is_dysarthric,
        "is_control": is_control,
        "word": "_".join(parts[1:-1]) if len(parts) > 2 else filename,
    }

    save_path = speaker_dir / f"{filename}.pt"
    torch.save(save_data, save_path)

    ua_stats["processed"] += 1
    ua_stats["speakers"][speaker_id] += 1
    ua_stats["total_duration_sec"] += result["duration_sec"]
    ua_stats["mel_frames"].append(result["n_frames"])

    if is_dysarthric:
        ua_stats["dysarthric_count"] += 1
    elif is_control:
        ua_stats["control_count"] += 1


# In[ ]:


print(ua_stats)


# In[ ]:


# DTW-align every dysarthric utterance to the reference control speaker's take of the
# same word, once, here, so UASpeechDataset only has to torch.load. Safe to re-run:
# files already aligned to the same reference are skipped.
import librosa

REFERENCE_CONTROL = "CF02"   # must match DataLoaders.REFERENCE_CONTROL

def dtw_target_indices(input_mel, target_mel, n_mfcc=20):
    """For each input frame, the index of the aligned target frame."""
    X = librosa.feature.mfcc(S=input_mel.numpy(), n_mfcc=n_mfcc)[1:]
    Y = librosa.feature.mfcc(S=target_mel.numpy(), n_mfcc=n_mfcc)[1:]
    # euclidean, not cosine: silent frames give all-zero MFCC vectors, whose cosine distance is NaN
    _, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    idx = np.zeros(input_mel.shape[-1], dtype=np.int64)
    for i, j in wp[::-1]:
        idx[i] = j
    return torch.from_numpy(idx)

# word ("B1_C10") -> the reference speaker's first mic take of it
ref = {}
for f in sorted((output_path / REFERENCE_CONTROL).glob("*.pt")):
    parts = f.stem.split("_")
    if len(parts) == 4:
        ref.setdefault(f"{parts[1]}_{parts[2]}", f)

ref_mels = {}
dtw_stats = {"aligned": 0, "already_aligned": 0, "no_reference": 0}
for f in tqdm(sorted(output_path.glob("[FM]*/*.pt")), desc="DTW"):   # dysarthric dirs: F02, M01, ...
    parts = f.stem.split("_")
    if len(parts) != 4:
        continue
    word = f"{parts[1]}_{parts[2]}"
    if word not in ref:
        dtw_stats["no_reference"] += 1
        continue

    data = torch.load(f, weights_only=False)
    if data.get("dtw_reference") == ref[word].name:
        dtw_stats["already_aligned"] += 1
        continue

    if word not in ref_mels:
        ref_mels[word] = torch.load(ref[word], weights_only=False)["mel"]
    target = ref_mels[word]
    data["dtw_target"] = target[:, dtw_target_indices(data["mel"], target)]   # (80, T_input)
    data["dtw_reference"] = ref[word].name
    torch.save(data, f)
    dtw_stats["aligned"] += 1

print(dtw_stats)

