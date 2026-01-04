# DCASE Few-Shot Bioacoustic Classification

Few-shot classification of animal vocalizations using Prototypical Networks. The model uses episodic training (N-way, K-shot) to adapt quickly to new species or call types from only a handful of labeled clips.

## Quick Start

```bash
# Install
pip install -e .

# 1. Extract features (run once)
python main.py extract-features

# 2. Train the model
python main.py train v1

# 3. Test the model
python main.py test outputs/protonet_baseline/v1_run/checkpoints/last.ckpt
```

## Training Pipeline

The pipeline follows a two-phase design:

```
Phase 1 (offline, run once):
  .wav audio → feature extraction → .npy cached features

Phase 2 (online, repeated):
  .npy features → PyTorch Lightning training → model checkpoints
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `train` | Train model with PyTorch Lightning |
| `test` | Test a trained checkpoint |
| `extract-features` | Extract and cache audio features |
| `cache-info` | Show cached feature statistics |
| `verify-cache` | Verify cache integrity |

### Training Examples

```bash
# Basic training
python main.py train v1

# Custom experiment name
python main.py train v1 --exp-name my_experiment

# Override hyperparameters
python main.py train v1 arch.training.max_epochs=100 arch.training.learning_rate=0.0005

# Change episode configuration
python main.py train v1 train_param.k_way=5 train_param.n_shot=3
```

See [docs/CLI_USAGE.md](docs/CLI_USAGE.md) for complete documentation.

## Project Structure

```
├── main.py                 # CLI entry point
├── conf/                   # Hydra configuration
│   ├── config.yaml         # Main config
│   ├── arch/v1.yaml        # V1 architecture config
│   ├── callbacks/          # Training callbacks
│   └── logger/             # MLflow logger config
├── archs/
│   ├── train.py            # Lightning trainer
│   └── v1/                 # V1 architecture
│       ├── arch.py         # ProtoNet model
│       └── lightning_module.py
├── preprocessing/
│   ├── preprocess.py       # Audio to features
│   ├── feature_cache.py    # Feature caching
│   ├── datamodule.py       # Lightning DataModule
│   └── dataset.py          # Dataset classes
├── utils/
│   ├── mlflow_logger.py    # MLflow logging
│   └── distance.py         # Distance metrics
└── docs/
    ├── CLI_USAGE.md        # CLI documentation
    └── FEATURES_AND_DATAFLOW.md
```

## Configuration

Configuration uses Hydra. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `arch.training.max_epochs` | Training epochs | 50 |
| `arch.training.learning_rate` | Learning rate | 0.001 |
| `train_param.k_way` | N-way (classes/episode) | 10 |
| `train_param.n_shot` | K-shot (samples/class) | 5 |
| `train_param.num_episodes` | Episodes/epoch | 2000 |
| `features.sr` | Sample rate | 22050 |
| `features.n_mels` | Mel bands | 128 |

Override via CLI:
```bash
python main.py train v1 arch.training.learning_rate=0.0005
```

Or edit `conf/config.yaml` directly.

## Data Format

### Directory Structure

```
/data/msc-proj/
  Training_Set/
    BV/
      BV_file1.wav
      BV_file1.csv
  Validation_Set_DSAI_2025_2026/
    ...
```

### Annotation CSV Format

CSV files with columns:
- `Audiofilename`: Audio file name
- `Starttime`: Segment start (seconds)
- `Endtime`: Segment end (seconds)
- `Q` or `CLASS_*`: Label (POS/NEG/UNK)

## MLflow Tracking

Training logs are tracked with MLflow:

```bash
# View training logs
mlflow ui --backend-store-uri outputs/protonet_baseline/v1_run/mlruns
```

Open http://localhost:5000 in your browser.

## Feature Caching

Features are cached as `.npy` files for fast training:

```bash
# Extract features (run once)
python main.py extract-features

# Check cache status
python main.py cache-info

# Force re-extraction after config change
python main.py extract-features --force
```

Cache location: `{data_dir}/features_cache/{config_hash}/`

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Lightning
- Hydra
- librosa
- MLflow (optional, for tracking)

```bash
pip install -e .
```

## License

This project is for educational purposes as part of the DCASE challenge.
