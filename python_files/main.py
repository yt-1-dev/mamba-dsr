#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Environment setup -- do NOT pip install from inside the notebook.
#
#     pip install -r requirements.txt
#
# requirements.txt pins torch 2.9.1+cu130 with the matching torchaudio and
# torchcodec, plus prebuilt mamba-ssm / causal-conv1d wheels for cu13+torch2.9.
# The previous contents of this cell installed torch 2.11.0+cu128 instead, which
# contradicts that pin -- that mismatch is what produced the torchcodec
# "undefined symbol: torch_list_size" and "_get_pybind_ops_module_name" import
# errors. It also emitted a bare `!pip` line that made python_files/main.py
# syntactically invalid.


# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from python_files.DataLoaders import *
from python_files.MambaBlocks import *
# from python_files.preprocess import *
from python_files.VectorQuantizer import *
from python_files.SpeakerInjector import *

##Used only for training
from python_files.SpeakerAdversary import *

from speechbrain.inference.speaker import EncoderClassifier

from speechbrain.inference.vocoders import HIFIGAN

import os

import torchaudio

# --- CUDA setup -------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True          # shapes are bounded by max_frames
torch.set_float32_matmul_precision("high")


def normalize_device(device):
    """'cuda' -> 'cuda:N'.

    SpeechBrain resolves a device string by splitting on ':' and unpacking two
    values, so an index-less 'cuda' raises before it falls back to device 0.
    """
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device(f"cuda:{torch.cuda.current_device()}")
    return d


# In[2]:


## CONSTANTS

N_MELS = 80
D_STATE = 16
D_MODEL = 256
D_CONV = 4
EXPAND = 2
N_LAYERS = 6 ## NUMBER OF ENCODER/DECODER LAYERS
SPK_DIM = 256
VQ_CODES = 512
DROPOUT = 0.1
VQ_GROUPS = 4


## Configuration


MODE_CONFIG = {
    "pretrain": {
        "lr": 3e-4,
        "batch_size": 16,
        "epochs": 50,
        "warmup_steps": 5000,
        "grad_clip": 1.0,
        "log_interval": 50,
        "eval_interval": 2000,
        "save_interval": 5000,
    },
    "finetune": {
        "lr": 2e-5,
        "batch_size": 8,
        "epochs": 30,
        "warmup_steps": 1000,
        "grad_clip": 1.0,
        "log_interval": 20,
        "eval_interval": 500,
        "save_interval": 2000,
    },
    "e2e": {
        "lr": 1e-5,
        "batch_size": 8,
        "epochs": 10,
        "warmup_steps": 0,
        "grad_clip": 0.5,
        "log_interval": 20,
        "eval_interval": 500,
        "save_interval": 1000,
    },
    "vocoder_adapt": {
        "lr": 1e-5,
        "batch_size": 16,
        "epochs": 20,
        "warmup_steps": 0,
        "grad_clip": 5.0,
        "log_interval": 50,
        "eval_interval": 1000,
        "save_interval": 2000,
    },
}


# In[3]:


class SpeechReconstructionPipeline(nn.Module):
    def __init__(self, d_model=256,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 n_layers = 6,
                 spk_dim = 256,
                 n_mels = 80,
                 vq_groups=4,
                 vq_codes = 512,
                 code_dim = 64,
                 speaker_adv_weight=0.1,
                 num_speakers = 100,
                 dropout=0.1,
                 device="cuda:0"):
        super().__init__()

        # SpeechBrain caches its device at construction and .to() does NOT update
        # it, so both pretrained models have to be told now, not later.
        self.dev = normalize_device(device)
        run_opts = {"device": str(self.dev)}
        self._resamplers = {}

        self.speaker_adv_weight = speaker_adv_weight

        self.speaker_encoder = EncoderClassifier.from_hparams(
            source= "speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(
                    Path.home(), ".cache", "speechbrain", "spkrec-ecapa"
                ),
            run_opts=run_opts,
        )

        for param in self.speaker_encoder.mods.parameters():
            param.requires_grad = False


        self.vocoder = HIFIGAN.from_hparams(
            source='speechbrain/tts-hifigan-ljspeech',
            savedir=os.path.join(
                Path.home(), ".cache", "speechbrain", "tts-hifigan-ljspeech"
            ),
            run_opts=run_opts,
        )

        for param in self.vocoder.mods.parameters():
            param.requires_grad = False

        self.speaker_encoder_proj = nn.Linear(192,spk_dim)

        self.encoder =  BidirectionalMambaEncoder(
            n_mels=n_mels,
            d_model= d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            n_layers=n_layers
        )

        self.decoder = CausalMambaDecoder(
            n_mels=n_mels,
            d_model= d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            n_layers=n_layers
        )

        self.vq = ProductQuantizer(
            input_dim=d_model,
            num_groups=vq_groups,
            code_dim= code_dim,
            commitment_weight=1.0,
            entropy_weight=0.5,
            num_codes=vq_codes,
            decay=0.9,
            reset_thresh= 2
        )

        projected = vq_groups*code_dim

        self.speaker_injector = SpeakerInjector(
            vq_dim=projected,
            d_model=d_model,
            spk_dim=spk_dim,

        )

        self.speaker_adversary = SpeakerAdversary(
            input_dim=projected,
            hidden_dim= 128,
            num_speakers= num_speakers,
            grl_scale= 1.0,
        )

        self.speaker_projection = nn.Linear(192, spk_dim)


    def mel_to_waveform(self, mel):

        with torch.no_grad():

            wav = self.vocoder.decode_batch(mel)
        return wav.unsqueeze(1) if wav.dim() == 2 else wav

    
    def extract_speaker_embedding(self, waveform, sample_rate=22050):

        if sample_rate != 16000:
            key = int(sample_rate)
            if key not in self._resamplers:
                self._resamplers[key] = torchaudio.transforms.Resample(
                    key, 16000
                ).to(waveform.device)
            waveform = self._resamplers[key](waveform)

        with torch.no_grad():
            raw = self.speaker_encoder.encode_batch(waveform).squeeze(1)

        return F.normalize(self.speaker_projection(raw), p=2, dim=-1)

    def set_mode(self,mode):

        for param in self.parameters():
            param.requires_grad = False

        if(mode == 'pretrain'):
            for m in [self.encoder,self.vq,self.speaker_adversary,
                      self.speaker_injector,self.decoder,self.speaker_projection]:

                for param in m.parameters():
                            param.requires_grad = True

        elif(mode == 'finetune'):
            for i, layer in enumerate(self.encoder.layers):
                if i>= self.encoder.downsample_after:
                    for param in layer.parameters():
                        param.requires_grad = True

            for m in [self.encoder.final_norm, self.vq, self.speaker_adversary,
                      self.speaker_injector, self.decoder, self.speaker_projection]:
                for param in m.parameters():
                    param.requires_grad = True


        elif mode == "e2e":
            for m in [self.encoder, self.vq, self.speaker_adversary,
                      self.speaker_injector, self.decoder, self.speaker_projection]:
                for param in m.parameters():
                    param.requires_grad = True

        elif mode == "vocoder_adapt":
            for param in self.vocoder.mods.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Mode: {mode} — {trainable:,} / {total:,} params trainable")

    def set_adversary_scale(self, step, ramp_steps=50000):

        # grl_scale is the configured max strength; ramp up to it, not to a hardcoded 1.0
        adv = self.speaker_adversary
        adv.set_current_scale(adv.grl_scale * min(1.0, step / ramp_steps))



    def forward(self,x,padding_mask,speaker_ids,speaker_emb,target, run_vocoder):

        B,n_mels,T = x.shape


        encoder_op = self.encoder(x, padding_mask)



        vq_op = self.vq(encoder_op)

        quantized = vq_op['quantized']
        vq_loss = vq_op['total_loss']



        adv_loss = self.speaker_adversary(quantized,speaker_ids)



        si_op =  self.speaker_injector(quantized,speaker_emb)

        decoder_op = self.decoder(si_op )

        predicted = decoder_op.transpose(1,2) # Converting back to mel

        predicted_waveform = None

        if run_vocoder:
            predicted_waveform = self.mel_to_waveform(predicted)

        mel_loss = torch.tensor(0.0,device=x.device)
        if(target is not None):

            min_T = min(predicted.shape[-1],target.shape[-1])
            # print(min_T,predicted.shape[-1],target.shape[-1])
            mel_loss = F.l1_loss(predicted[:,:,:min_T],target[:,:,:min_T])




        loss = vq_loss +mel_loss + self.speaker_adv_weight * adv_loss

        return {
            "predicted":predicted,
            "target":target,
            "loss":loss,
            "mel_loss" : mel_loss,
            "vq_loss":vq_loss,
            "adv_loss":adv_loss,
            "vq_op":vq_op,
            "predicted_waveform":predicted_waveform

        }









# guarded: an `import *` of this module would otherwise download both SpeechBrain
# models and run a full forward pass before any real work starts.
if __name__ == "__main__":
    _dev = normalize_device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SpeechReconstructionPipeline(device=_dev).to(_dev)

    BATCH_SIZE = 32
    T = 45
    model(torch.rand(BATCH_SIZE, N_MELS, T, device=_dev),
          torch.randint(0, 2, size=(BATCH_SIZE, T), dtype=torch.bool, device=_dev),
          torch.randint(0, 100, size=(BATCH_SIZE,), device=_dev),
          torch.rand(BATCH_SIZE, SPK_DIM, device=_dev),
          torch.rand(BATCH_SIZE, N_MELS, T, device=_dev),
          False)

    model.set_mode('pretrain')
    del model
    if _dev.type == "cuda":
        torch.cuda.empty_cache()


# In[4]:


import os
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Subset


# In[5]:


def load_checkpoint(pipeline, path, device, current_mode, optimizer=None, scheduler=None):

    if Path(path).exists():
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Load model weights (always, regardless of mode)
        own = pipeline.state_dict()
        state = {k: v for k, v in checkpoint["model_state_dict"].items()
                 if k not in own or own[k].shape == v.shape}
        skipped = checkpoint["model_state_dict"].keys() - state.keys()
        if skipped:
            print(f"  Skipped shape-mismatched: {sorted(skipped)}")
        missing, unexpected = pipeline.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  WARNING  missing={list(missing)}  unexpected={list(unexpected)}")

        saved_mode = checkpoint.get("mode")
        speaker_embs = checkpoint.get("speaker_embeddings")

        # Same mode = resume training (restore everything)
        if saved_mode == current_mode:
            if optimizer and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            best_metric = checkpoint.get("best_metric")
            
            if isinstance(best_metric, dict):
                best_metric = best_metric.get("val/mel_loss", float("inf"))

            print(checkpoint["step"],
                checkpoint["epoch"],
                best_metric,
                speaker_embs)

            print(f"  Loaded checkpoint from {path} (step {checkpoint['step']}, mode: {saved_mode})")
            return (
                checkpoint["step"],
                checkpoint["epoch"],
                best_metric,
                speaker_embs,
            )

        # Different mode = new phase (keep weights, reset everything else)
        print(f"  Loaded weights from {path} (mode: {saved_mode} → {current_mode})")
        print(f"  Resetting step, epoch, metric, optimizer, scheduler")
        return 0, 0, float("inf"), speaker_embs

    else:
        print("No checkpoint found!!")
        return 0, 0, float("inf"), None


def save_checkpoint(pipeline, optimizer, scheduler, step, epoch, path,
                    mode, best_metric=None, speaker_embs=None):

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "step": step,
        "epoch": epoch,
        "mode": mode,
        "model_state_dict": pipeline.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
        "speaker_embeddings": speaker_embs,
    }, path)


# In[6]:


def set_checkpoint_mode(path, mode):
    """
    Update the mode stored in a checkpoint file.

    Useful when:
      - Checkpoint was saved without a mode field (older format)
      - You want to force-resume from a different phase's checkpoint
        without resetting optimizer/scheduler state

    Args:
        path: Path to checkpoint file
        mode: "pretrain", "finetune", "e2e", or "vocoder_adapt"
    """
    path = Path(path)
    if not path.exists():
        print(f"Checkpoint not found: {path}")
        return

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    old_mode = checkpoint.get("mode", "not set")
    checkpoint["mode"] = mode
    torch.save(checkpoint, path)
    print(f"  {path.name}: mode '{old_mode}' → '{mode}'")

set_checkpoint_mode('checkpoints/pretrain_best.pt','pretrain')
set_checkpoint_mode('checkpoints/finetune_best.pt','finetune')
set_checkpoint_mode('checkpoints/e2e_best.pt','e2e')



# In[7]:


def load_dataloaders(dataset, batch_size, val_fraction=0.05, seed=1234):
    """Build the dataset ONCE and split it.

    The previous version instantiated the dataset twice and returned two loaders
    over the same data, differing only in `shuffle` -- so every "validation"
    metric was a training metric, and the ~136k-file index scan ran twice.

    The split is utterance-level, not speaker-level: holding speakers out would
    leave them absent from `speaker_to_idx`, and `get_speaker_embeddings_for_batch`
    would KeyError on the first val batch.
    """
    base = (UASpeechDataset if dataset == 'ua-speech' else LibreTTSDataset)()

    n = len(base)
    print(n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = max(batch_size, int(n * val_fraction))
    val_set = Subset(base, perm[:n_val])
    train_set = Subset(base, perm[n_val:])

    # Subset hides the attributes the training code reads off .dataset
    for s in (train_set, val_set):
        s.speakers = base.speakers
        s.raw_audio_path = base.raw_audio_path

    common = dict(
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,            # required for .to(non_blocking=True) to overlap
        persistent_workers=True,
    )
    print(f"  {len(train_set)} train / {len(val_set)} val samples")

    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **common),
        DataLoader(val_set, shuffle=False, drop_last=False, **common),
    )
    
train, test = load_dataloaders('libre',16)
print(len(train))


# In[8]:


@torch.no_grad()
def precompute_speaker_embeddings(pipeline,speaker_to_idx,raw_audio_dir,device):
   
    raw_path = Path(raw_audio_dir)
    speaker_embs = {}
    for sid in sorted(speaker_to_idx.keys()):
        speaker_dir = raw_path / sid
        wav_files = []
        
        if speaker_dir.exists():
            wav_files = list(speaker_dir.rglob("*.wav"))
 
        if not wav_files:
            wav_files = list(raw_path.rglob(f"{sid}_*.wav"))
            
        if not wav_files:
            print(f"  WARNING: No audio found for speaker {sid}, skipping")
            continue
        
        
        best_file, best_duration = wav_files[0], 0.0
        for wf in wav_files[:10]:
            info = info = sf.info(str(wf))
            duration = info.duration
            if duration > best_duration:
                best_duration = duration
                best_file = wf
 
        # Load and extract embedding
        waveform, sample_rate = torchaudio.load(str(best_file))
 
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
 
        waveform = waveform.to(device)
        embedding = pipeline.extract_speaker_embedding(
            waveform, sample_rate=sample_rate
        )
 
        speaker_embs[sid] = embedding.squeeze(0)

    return speaker_embs


# In[9]:


print(LIBRE_TTS_PATH,Path.cwd())


# In[10]:


def load_speaker_embeddings(dataloader):
  
    
    return {sid:idx for idx, sid in enumerate(sorted(dataloader.dataset.speakers))}

def get_speaker_ids_for_batch(batch, speaker_to_idx, device):
    ids = [speaker_to_idx[sid] for sid in batch['speaker_ids']]
  
    return torch.tensor(ids, dtype=torch.long, device=device)

def get_speaker_embeddings_for_batch(batch, speaker_embs, device):
    embs = torch.stack([speaker_embs[sid] for sid in batch['speaker_ids']])
    return embs.to(device)

# train_loader,_ = load_dataloaders('ua-speech',16)

# speaker_to_idx = load_speaker_embeddings(train_loader)

# print(speaker_to_idx)

# batch = next(iter(train_loader))

# get_speaker_ids_for_batch(batch=batch, speaker_to_idx=speaker_to_idx, device='cpu')





# In[11]:


def train_step(pipeline,batch, optimizer, scheduler,
               speaker_embs, speaker_to_idx, device, step,config):

    input_mel = batch['input_mel'].to(device, non_blocking=True)
    target_mel = batch['target_mel'].to(device, non_blocking=True)
    padding_mask = batch['padding_mask'].to(device, non_blocking=True)


    speaker_emb = get_speaker_embeddings_for_batch(batch,speaker_embs,device)
    speaker_ids = get_speaker_ids_for_batch(batch, speaker_to_idx, device)



    pipeline.set_adversary_scale(step)

    # run_vocoder=False: predicted_waveform is not part of the loss, so running
    # HiFi-GAN every step was paying full price for a discarded tensor.
    result = pipeline(
        input_mel,padding_mask,speaker_ids,speaker_emb,target_mel, False
    )

    result["loss"].backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in pipeline.parameters() if p.requires_grad],
        max_norm=config["grad_clip"],
    )

    # Optimizer step
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # float() not .item(): drops the autograd graph so logged metrics can't pin it
    vq_op = result["vq_op"]
    return {
        "train/mel_loss": float(result["mel_loss"]),
        "train/vq_loss": float(result["vq_loss"]),
        "train/adv_loss": float(result["adv_loss"]),
        "train/total_loss": float(result["loss"]),
        "train/perplexity": float(vq_op["perplexity"]),
        "train/active_ratio": float(vq_op["active_ratio"]),
        "train/grad_norm": float(grad_norm),
        "train/lr": scheduler.get_last_lr()[0],
    }




# In[ ]:





# In[12]:


import soundfile as sf

@torch.no_grad()
def validate(pipeline, val_loader, speaker_embs, speaker_to_idx, device, max_batches=50):
    """Returns the metrics dict the training loop expects (needs 'val/mel_loss').

    eval() matters beyond dropout here: it stops the VQ's EMA updates and dead-code
    resets from firing on validation batches and mutating the codebook.
    """
    pipeline.eval()
    totals, n = defaultdict(float), 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        input_mel    = batch['input_mel'].to(device, non_blocking=True)
        target_mel   = batch['target_mel'].to(device, non_blocking=True)
        padding_mask = batch['padding_mask'].to(device, non_blocking=True)

        speaker_emb = get_speaker_embeddings_for_batch(batch, speaker_embs, device)
        speaker_ids = get_speaker_ids_for_batch(batch, speaker_to_idx, device)

        result = pipeline(input_mel, padding_mask, speaker_ids,
                          speaker_emb, target_mel, False)

        totals["val/mel_loss"]   += float(result["mel_loss"])
        totals["val/vq_loss"]    += float(result["vq_loss"])
        totals["val/adv_loss"]   += float(result["adv_loss"])
        totals["val/total_loss"] += float(result["loss"])
        totals["val/perplexity"] += float(result["vq_op"]["perplexity"])
        n += 1

    return {k: v / max(1, n) for k, v in totals.items()}


def train(mode, checkpoint_path, device):

    device = normalize_device(device)

    config = MODE_CONFIG[mode]

    train_loader, val_loader = load_dataloaders('libre-tts'if mode=='pretrain'else 'ua-speech',config['batch_size'])


    output_dir = "./checkpoints"



    start_step = 0
    start_epoch = 0
    best_metric = float("inf")
    speaker_embs = None
    speaker_to_idx = load_speaker_embeddings(train_loader)
    print(f"{len(speaker_to_idx)} Speakers")

    pipeline = SpeechReconstructionPipeline(

            n_mels = N_MELS,
            d_model = D_MODEL,
            n_layers = N_LAYERS,
            vq_groups = VQ_GROUPS,
            vq_codes = VQ_CODES,
            code_dim = 32,
            spk_dim = SPK_DIM,
            num_speakers = len(speaker_to_idx),
            speaker_adv_weight = 0.1,
            device = device,

        )

    pipeline = pipeline.to(device)
    pipeline.set_mode(mode=mode)

    

    if checkpoint_path:
            start_step, start_epoch ,saved_metric, speaker_embs = load_checkpoint(pipeline, checkpoint_path, device=device,current_mode=mode)
            if saved_metric is not None:
                best_metric = saved_metric
            # Checkpoint from another dataset (pretrain -> finetune): its speaker embs, step
            # and val loss don't apply here, so start the stage fresh with only the weights.
            if speaker_embs is not None and set(speaker_embs) != set(speaker_to_idx):
                print("  Checkpoint is from a different dataset: resetting step/metric/speaker embs")
                start_step, start_epoch, best_metric, speaker_embs = 0, 0, float("inf"), None

    if speaker_embs is None:
        speaker_embs = precompute_speaker_embeddings(pipeline, speaker_to_idx,
                                                     train_loader.dataset.raw_audio_path,
                                                     device)


    trainable_params = [p for p in pipeline.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config["lr"],
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )

    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = config['warmup_steps']

    

    def lambda_lr(step):
        if(step < warmup_steps):
            return step/max(1,warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))

    scheduler = LambdaLR(optimizer=optimizer,lr_lambda=lambda_lr)

    global_step = start_step

    epochs = config['epochs']



    pipeline.train();
    print(f"\nStarting training: {total_steps} steps ({config['epochs']} epochs)")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Warmup: {config['warmup_steps']} steps")
    print(f"  Grad clip: {config['grad_clip']}")
    print(f"  Start Epoch: {start_epoch}")
    print(f"  Start Step: {global_step}")
    print(f"  Best Metric : {best_metric}")
    


    





    batches_per_epoch = len(train_loader)

    for epoch in range(start_epoch,epochs):
    # for epoch in range(1):

        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and global_step > start_step:
                pass  # already past this point
            elif epoch == start_epoch and batch_idx < (start_step % batches_per_epoch):
                continue

            global_step+=1

            metrics = train_step(pipeline,batch, optimizer, scheduler,
                       speaker_embs, speaker_to_idx, device, global_step,config)

           

            # --- Validate ---
            if global_step % config["eval_interval"] == 0:
                val_metrics = validate(
                    pipeline, val_loader, speaker_embs, speaker_to_idx, device,
                )

                print(f"  [VAL] step {global_step} | " + " | ".join(
                    f"{k}: {v:.4f}" for k, v in val_metrics.items()
                ))
                print(metrics)
                # Save best checkpoint
                val_loss = val_metrics["val/mel_loss"]
                if val_loss < best_metric:
                    best_metric = val_loss
                    best_path = os.path.join(output_dir, f"{mode}_best.pt")
                    save_checkpoint(
                        pipeline, optimizer, scheduler,
                        global_step, epoch, best_path, mode ,best_metric,
                        speaker_embs,
                    )
                    print(f"  [BEST] New best val loss: {best_metric:.4f} → {best_path}")


                pipeline.train()
            
        epoch_time = time.time() - epoch_start
        print(f"\n  Epoch {epoch + 1}/{config['epochs']} done in {epoch_time:.1f}s")
        print()
    final_path = os.path.join(output_dir, f"{mode}_final.pt")
    save_checkpoint(
        pipeline, optimizer, scheduler,
        global_step, config["epochs"], final_path, mode ,best_metric,
        speaker_embs,
    )




# train('pretrain','./checkpoint.pt','cuda:0')



# In[ ]:


# train('finetune','./checkpoints/pretrain_best.pt','cuda:0')


# In[14]:


# train('e2e','./checkpoints/finetune_best.pt','cuda:0')


# In[15]:


# train('vocoder_adapt','./checkpoints/e2e_best.pt','cuda:0')


