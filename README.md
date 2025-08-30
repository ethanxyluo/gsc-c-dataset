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
```
python visualize_mixed.py
```

## Reference
- GSC: Warden, P. “Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.”
- ESC-50: Piczak, K. J. “ESC: Dataset for Environmental Sound Classification.”