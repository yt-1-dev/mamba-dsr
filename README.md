# MambaDSR: Code Predictability as the Bottleneck in Dysarthric Speech Reconstruction

MambaDSR turns dysarthric speech into clear speech in the voice of a healthy control speaker (CF02). It has five parts:
- **Encoder:** a bidirectional Mamba-3 encoder.
- **Tokenizer:** a product vector quantiser (4 groups × 512 codes × 32 dims) that turns each frame into discrete codes.
- **Code recognizer:** predicts healthy codes from dysarthric speech.
- **Decoder:** a causal Mamba-3 decoder that turns codes into a mel spectrogram.
- **Vocoder:** HiFi-GAN, which turns the mel spectrogram into audio.

The main finding: the codes carry enough content to say the word (oracle 19.8% exact words, above the original's 11.0%). But they cannot be predicted reliably from dysarthric speech (3.4% exact words). Code predictability is the bottleneck.

---

## Repository layout

| Path | What it is |
|---|---|
| `main.ipynb` | Model, training loop, all modes (`train(mode, ckpt, device)`), gates and agreement checks |
| `MambaBlocks.ipynb` | Bidirectional and causal Mamba-3 blocks |
| `VectorQuantizer.ipynb` | Product VQ with EMA codebook and dead-code reset |
| `SpeakerAdversary.ipynb` | Speaker adversary (gradient reversal) |
| `SpeakerInjector.ipynb` | Joins codes with the 192-d ECAPA speaker embedding |
| `DataLoaders.ipynb` | LibriTTS and UA-Speech loaders (pairs, DTW targets, multi-reference targets) |
| `preprocess.ipynb` | Audio to log-mel, DTW alignment to CF02 |
| `jasper.ipynb` | Jasper (NeMo) evaluation: exact words, CER, WER, oracle check |
| `reconstruction.ipynb` | Reconstruct single files or validation samples (mel plots + audio) |
| `train.ipynb`, `colab_train.ipynb` | Older training entry points |
| `retrain.ipynb` | **Full retrain on Colab, stage by stage, with all results** (see below) |
| `convert.sh` | Converts every notebook to `python_files/<name>.py` |
| `python_files/` | Generated `.py` versions of the notebooks (created by `convert.sh`) |
| `uaspeech/speaker_wordlist.csv` | UA-Speech word codes and their words (e.g. `B2_CW18`), used for labels |
| `checkpoints/`, `results/` | Created at run time (not tracked) |

---

## Data

### Datasets
| Dataset | Used for | Size |
|---|---|---|
| LibriTTS train-clean-100 (246 healthy speakers) | Pretrain, tokenizer | 29,551 train / 1,555 val utterances |
| UA-Speech, 13 control speakers (CF02 = output voice) | Tokenizer, targets | Blocks B1 + B3 |
| UA-Speech, 14 dysarthric speakers, paired with CF02 | Recognizer, rec2mel | 39,231 train / 2,050 val pairs (B1 + B3) |
| UA-Speech B2: M05, F04, M07, F02, mic M5 | Test only | 849 words |

UA-Speech audio is the noise-reduced copy. Each file is named `<speaker>_<block>_<word code>_<mic>`, for example `M05_B2_CW18_M5`. Word codes:
- `CW` common words;
- `UW` uncommon words;
- `D` digits;
- `C` computer commands;
- `L` letters.

Mics run from `M2` to `M8`.

### Data archive (`data-2.zip`)
The notebook unzips it into `/content`, next to the repo, so the code reads it as `../data/...`:

```
/content/
├── mamba-dsr/                      # this repo
└── data/
    ├── processed/
    │   ├── libreTTS/<speaker>/<utterance>.pt        # pretrain / tokenizer
    │   └── ua/<speaker>/<spk>_<block>_<word>_<mic>.pt  # UA-Speech, control + dysarthric
    └── Noise Reduced UASPEECH Dysarthria Dataset/
        └── noisereduced-uaspeech/<speaker>/<spk>_<block>_<word>_<mic>.wav  # raw audio (playback only)
```

`preprocess.ipynb` builds the `.pt` files. Each one holds:
- `mel`: the 80-bin log-mel `(80, T)`;
- `dtw_target` (dysarthric files only): CF02's take of the same word, DTW-stretched to this file's timing, `(80, T)`.

`main.ipynb` sets the paths as `UA_PATH` (processed UA-Speech), `UA_RAW_PATH` (raw wavs) and `WORDLIST`.

### Loading
`DataLoaders.ipynb` provides `load_dataloaders(dataset, batch_size)`, which returns `(train_loader, val_loader)` with a fixed seed and split:

| `dataset` | Sample | Used by |
|---|---|---|
| `"libritts"` | One utterance: mel, padding mask, speaker id | `pretrain`, `tokenizer` |
| `"ua-speech"` | One dysarthric/CF02 pair: `input_mel`, `target_mel` (DTW-aligned CF02), `speaker_id`, `word`, padding mask | `recognizer`, `rec2mel` |

On top of that:
- **Tokenizer pairs:** the tokenizer also draws pairs of healthy takes of the same word (another mic, or another control speaker or block). The consistency loss runs on these.
- **Recognizer targets:** `precompute_target_codes` and `precompute_multi_target_codes` store CF02's codes (and up to 8 healthy references, CF02–CF05) in the UA-Speech `.pt` files.
- **Batching:** batches are padded to the longest clip, and padded frames are masked out of every loss.
- **Augmentation:** SpecAugment is applied to the dysarthric input during recognizer training.

---

## Pipeline

```
dysarthric wav ─► log-mel (1, 80, T) ─► Bi-Mamba encoder (↓2) ─► code recognizer ─► 4 code indices per frame
                                                                                      │
          CF02 ECAPA (192) ─► speaker injector ◄── codebook lookup (128) ◄────────────┘
                                   │
                                   ▼
                  causal Mamba decoder (↑2) ─► mel (1, 80, T) ─► HiFi-GAN ─► wav
```

- **Encoder:**
  - Linear 80→256.
  - 3 bidirectional Mamba-3 layers.
  - Conv1d (kernel 3, stride 2) + RMSNorm.
  - 3 bidirectional Mamba-3 layers + RMSNorm.
- **Product VQ:**
  - Linear 256→128 + LayerNorm.
  - 4 groups × 512 codes × 32 dims.
  - EMA codebook, decay 0.9; dead-code reset every batch.
- **Code recognizer:** 2 bidirectional Mamba-3 layers + 4 × Linear(256→512).
- **Speaker injector:** 128 code dims + 64 zeros (the slot of the removed skip path) + 192-d ECAPA → 256.
- **Decoder:**
  - ConvTranspose ↑2.
  - 6 causal Mamba-3 layers.
  - Linear 256→80.
- **Vocoder:** SpeechBrain HiFi-GAN (LJSpeech). The mel is scaled by 0.5 before vocoding.

**Pre-processing:**
- 22.05 kHz mono, −23 LUFS loudness, silence trimmed at −40 dB, clips of 1–10 s.
- 80-bin log-mel: n_fft 1024, hop 256, 0–8 kHz.
- Each dysarthric take is paired with CF02's take of the same word. DTW on 20 MFCCs stretches CF02 onto the dysarthric timing.

---

## Training stages (modes)

| Stage | Mode | Data | Trains | Notes |
|---|---|---|---|---|
| 0 | `pretrain` | LibriTTS train-clean-100 (246 speakers) | Whole core | 10 epochs |
| 1 | `tokenizer` | LibriTTS + UA-Speech control speakers (B2 excluded) | Encoder, VQ, decoder | Plain, then a consistency loss (weight 0.1) on DTW-aligned healthy take pairs |
| 2 | `recognizer` | UA-Speech dysarthric/CF02 pairs (B1 + B3) | Encoder + code heads | Soft-target cross-entropy, multi-reference targets; codebook, injector and decoder frozen |
| 3 | `rec2mel` | Same pairs | Decoder | Adapts the decoder to the recognizer's predicted codes |
| 4 | `recognizer` with `merge_classes=256` | Same pairs | Encoder + code heads | Codes merged into 256 classes per group (usage-weighted k-means) |

The test set is UA-Speech block B2, mic M5, for speakers M05, F04, M07 and F02 (849 words). Jasper scores every system on the same words.

---

## Running `retrain.ipynb` (Google Colab)

### Requirements
- **GPU:** Colab with an L4 GPU (24 GB).
- **GitHub token:** a Colab secret named `GITHUB_TOKEN` that can read this repo. The notebook reads it with `userdata.get("GITHUB_TOKEN")`. Never write the token into the notebook.
- **Drive layout:**
  - `MyDrive/Colab Notebooks/MTP/data-2.zip`: the pre-processed LibriTTS and UA-Speech data.
  - `MyDrive/Colab Notebooks/MTP/checkpoints/pretrain_best.pt`: the starting checkpoint, read only.
  - `MyDrive/Colab Notebooks/MTP/checkpoints_retrain/`: this run's backups, created automatically.

### What the setup cells do
1. Mount Drive, clone or update this repo into `/content/mamba-dsr`, and copy `pretrain_best.pt`.
2. Restore any earlier backups from `checkpoints_retrain/`, so a run resumes after a disconnect. Then run `convert.sh`.
3. Unzip `data-2.zip` into `/content` (see [Data](#data)).
4. Install the training stack:
   - `torch 2.9.1` / `torchaudio 2.9.1` / `torchvision 0.24.1` (CUDA 12.8);
   - prebuilt wheels for `causal-conv1d 1.7.0` and `mamba_ssm 2.3.2.post1` (Mamba-3);
   - `speechbrain`, `soundfile`, and the matching `torchcodec`.
5. Install NeMo (for Jasper) and `jiwer`, with the training packages pinned so NeMo can't upgrade them.
6. Define the helpers:

| Helper | What it does |
|---|---|
| `load_src(*names)` | Runs `python_files/<name>.py` without its top-level calls |
| `stage(mode, ckpt, **cfg)` | Overrides `MODE_CONFIG[mode]`, trains, and always backs up to Drive |
| `snap(src, dst)` | Copies a checkpoint under a new name and backs up |
| `add_stats(name, df)` | Records exact words, CER, WER and empty-output rate for the summary |

   This cell also patches `torch.cuda.device` so `mamba_ssm` works on torch 2.9+.

### Stages in the notebook
| Section | Command | Output checkpoints |
|---|---|---|
| 0. Pretrain (optional) | `stage("pretrain", None, epochs=10, batch_size=16)` | `pretrain_best.pt` |
| 1. Tokenizer, plain | `stage("tokenizer", pretrain_best.pt, consistency_weight=0.0, epochs=15)` | `tokenizer_plain.pt` |
| 1. Tokenizer, consistency | `stage("tokenizer", tokenizer_init.pt, consistency_weight=0.1, epochs=5)` | `tokenizer_best.pt` |
| 1. Tokenizer stats | `codes2mel_gate`, `code_agreement_check`, codebook health | – |
| 2. Recognizer | `precompute_target_codes`, `precompute_multi_target_codes`, then `stage("recognizer", ..., epochs=3)` | `recognizer_raw.pt` |
| 3. rec2mel | `stage("rec2mel", recognizer_raw.pt, epochs=3)` | `rec2mel_raw.pt` |
| 3. Evaluation | `run_jasper` (recognizer, rec2mel), `oracle_codes_check` | – |
| 4. Merged classes | recognizer + rec2mel with `merge_classes=256` | `recognizer_m256.pt`, `rec2mel_m256.pt` |
| Stats | Summary table → `results/retrain_summary.csv` | – |

Before starting a stage from a copied checkpoint, the notebook relabels the copy with `set_checkpoint_mode`. That makes training start fresh instead of resuming.

The target codes are stored in the local data files. After a runtime restart, re-run `precompute_target_codes` and `precompute_multi_target_codes`.

---

## Results (retrain)

### Tokenizer (consistency, `tokenizer_best.pt`)

**Codes-only gate** (CF02, B2, mic M5, 255 words):

| | Exact words | Empty | CER |
|---|---|---|---|
| CF02 vocoded directly | 63.1% | 2.4% | 17.8% |
| Rebuilt from codes only | 28.2% | 8.6% | 50.0% |

**Agreement between takes** (CF02 B1 vs B3, mic M5, 155 words, 18,739 aligned frames):

| Classes | Exact | Chance | Above chance (κ) |
|---|---|---|---|
| 512 (raw) | 17.7% | 4.2% | 14.1% |
| 256 (merged) | 21.7% | 6.3% | 16.5% |
| 128 | 37.5% | 18.4% | 23.4% |
| 64 | 47.8% | 27.0% | 28.6% |

For 512 raw codes, the right code is among the 5 nearest 40.1% of the time.

**Codebook health:**

| Group | Perplexity (of 512) | Codes used under 0.1% | Near-twins (< 0.15 × median distance) |
|---|---|---|---|
| 0 | 374 | 142 | 324 |
| 1 | 363 | 146 | 274 |
| 2 | 363 | 139 | 274 |
| 3 | 362 | 150 | 273 |

### Jasper on UA-Speech B2 (M5; M05, F04, M07, F02; 849 words)

| System | Exact words | CER | WER | Empty |
|---|---|---|---|---|
| Original dysarthric speech | 11.0% | 78.8 | 140.6 | 8.8% |
| Recognizer | 2.8% | 93.5 | 124.4 | 31.6% |
| Recognizer + rec2mel | 3.4% | 90.5 | 116.0 | 31.7% |
| Recognizer (256 classes) | 2.8% | 90.1 | 120.8 | 26.9% |
| Recognizer (256 classes) + rec2mel | 3.1% | 90.3 | 117.2 | 30.9% |
| Oracle (CF02's DTW-aligned codes) | 19.8% | 59.9 | 108.5 | 10.0% |
| Oracle (256 classes) | 19.8% | 63.2 | 109.6 | 10.6% |

**Mean of per-speaker relative WER reductions (UNIT-DSR style):**
- recognizer + rec2mel: 16.7%;
- 256 classes + rec2mel: 16.1%;
- oracle: 23.9%;
- UNIT-DSR reports 28.2%.

**How to read WER here:** every test item is a single word. Insertions push WER above 100%, while an empty output costs exactly 100%. So a lower WER can come from more empty outputs rather than more correct words. Exact-word accuracy is the main metric.

### Final checkpoints
| File | Mode | Step | Best val |
|---|---|---|---|
| `tokenizer_best.pt` | tokenizer | 4,500 | 1.215 |
| `recognizer_raw.pt` | recognizer | 7,000 | 4.176 |
| `rec2mel_raw.pt` | rec2mel | 7,000 | 2.065 |
| `recognizer_m256.pt` | recognizer (256 classes) | 7,000 | 3.727 |
| `rec2mel_m256.pt` | rec2mel (256 classes) | 6,000 | 2.052 |

---

## Reconstructing audio

The last cells of `retrain.ipynb` define the reconstruction helpers.
- `reconstruct_file(pt_path, pipeline, speaker_embs)` takes one pre-processed UA-Speech file, for example `M05_B2_CW18_M5.pt`.
- `reconstruct_val(index, pipeline, speaker_embs)` takes one held-out validation sample.

Each call saves the input, reconstructed and target audio to `reconstructions/`, plots the mel spectrograms, and plays the audio inline.

A seeded loop at the end picks random B2 words for M07, F04, F02 and M05 and writes them to `recon_random/`.

Recognizer and rec2mel checkpoints always speak in CF02's voice, so male speakers come out with a female voice.
