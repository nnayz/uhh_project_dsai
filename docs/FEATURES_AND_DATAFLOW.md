# Features and Data Flow

This document describes the feature export pipeline used by training.

## Overview

Training expects precomputed, full-audio feature arrays stored next to each `.wav`.

```
.wav + CSV labels
  -> full-audio feature extraction
  -> per-audio .npy arrays
  -> dynamic segment sampling during training
```

The segment-level cache pipeline is not used in this repo.

## Feature export

**What gets written**
- One `.npy` per audio file, next to the `.wav`:
  - `audio_logmel.npy`
  - `audio_pcen.npy`

**Where it lives**
- Export: `preprocessing/feature_export.py`
- Sequence datasets: `preprocessing/sequence_data/*`

**Why this path**
- Matches the reference training behavior (dynamic sampling from full-audio arrays).
- Keeps preprocessing simple: no segment cache to maintain.

## CLI

```bash
# Export features
 g5 export-features --split all

# Validate files exist
 g5 check-features --split all
```
