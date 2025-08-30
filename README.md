# GSC-C: Mixing Google Speech Commands with ESC-50 for Robust KWS Evaluation

GSC-C is a lightweight toolkit to synthesize test data for Keyword Spotting (KWS) under real-world background noise. It mixes Google Speech Commands (GSC) utterances with ESC-50 environmental sounds at controlled Signal-to-Noise Ratios (SNR), producing:
- Playable mixed audio samples (for quick listening/inspection)
- MFCC features in .npy format (for model evaluation)

This repository contains:
- audio_mixer.py — batch synthesis for test sets (saves MFCC features and applies mean/variance normalization).
- visualize_mixed.py — generates a small set of mixed .wav files and their MFCCs for quick listening and sanity checks.

## Features

- Choose ESC-50 noise categories and SNRs (supports negative SNRs).
- MFCC extraction (40 dims) at 16 kHz, 40 ms window, 20 ms hop.
- Mean/variance normalization performed per generated directory (in-place).
- Simple, reproducible pipeline with minimal dependencies.

## Data Prerequisites

- Google Speech Commands (GSC, recommended V2)
  - Paper: https://arxiv.org/abs/1804.03209
  - Download (V2): https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
  - Must include testing_list.txt at the root of the GSC folder (used by audio_mixer.py to filter test samples).

- ESC-50
  - Repo: https://github.com/karolpiczak/ESC-50
  - Place all .wav files under esc50/ with original filenames:
    {FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav
  - The TARGET index is used to determine the category.

- Expected local layout:
  - speech_commands/
    - testing_list.txt
    - yes/.wav, no/.wav, ... , background_noise/*.wav
  - esc50/
    - *.wav (all ESC-50 audio files placed directly here)

## Installation
Python 3.9+ is recommended.
```
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### 1) Batch generation of mixed MFCC features
The script:
- Filters test samples according to testing_list.txt (commands, unknown, silence).
- Randomly selects ESC-50 files within specified categories and mixes them with target SNR.
- Extracts MFCC features and saves .npy dictionaries: {'feature': np.ndarray, 'label': int}.
- Computes mean/std over the generated directory and applies in-place normalization.

Run:
```
python audio_mixer.py
```

Default configuration (edit in `audio_mixer.py` → `main`):
- ESC categories (by TARGET indices): `[0, 5, 17, 19, 20, 26, 35, 36, 43, 48]`
- SNR levels: `[-20, -15]`
- Paths: `gsc_dir='speech_commands'`, `esc50_dir='esc50'`, `output_dir='processed_dataset'`

Notes:
- `process_google_sample` only keeps 1-second (16000 samples) inputs. Others are skipped.
- Silence is sampled from `_background_noise_` in 1-second segments.

### 2) Generate playable mixed audio samples (.wav)
Use this to quickly listen to the mixing quality. By default, it randomly picks one sample per command and mixes it at various SNRs and ESC-50 categories.

Run:
```
python visualize_mixed.py
```

Defaults (edit in `visualize_mixed.py`):
- SNRs: `[-10, 0, 10]`
- ESC categories (TARGET indices): `[0, 5, 17, 19, 20, 26, 35, 36, 43, 48]`
- Output dir: `testwav`


## Implementation Details

- SNR control
  - Noise is scaled to the target SNR relative to the speech signal, then added.
  - Peak normalization is applied to avoid clipping, which may slightly alter the perceived SNR—typically acceptable for robustness testing.

- MFCC
  - 16 kHz sampling rate, 40 ms window, 20 ms hop, 40 coefficients.
  - `audio_mixer.py` normalizes features per output directory using mean/std computed over that directory.

- Categories
  - `audio_mixer.py` maps ESC-50 TARGET indices to human-readable names for output paths.
  - `visualize_mixed.py` uses `esc_{index}` for folder names.

## Reference
- GSC: Warden, P. “Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.”
- ESC-50: Piczak, K. J. “ESC: Dataset for Environmental Sound Classification.”