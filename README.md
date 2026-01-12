# DCASE Few-Shot Bioacoustic Classification

Few-shot classification of animal vocalizations using Prototypical Networks. The model uses episodic training (N-way, K-shot) to adapt quickly to new species or call types from only a handful of labeled clips.

## Quick Start

```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# 1. Export per-audio feature arrays (run once)
g5 export-features --split all

# 2. Train the model
g5 train v1 --exp-name my_experiment

# 3. Test the model
g5 test outputs/mlflow_experiments/my_experiment/checkpoints/last.ckpt
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `train` | Train model with PyTorch Lightning |
| `test` | Test a trained checkpoint |
| `export-features` | Export per-audio feature arrays next to `.wav` files |
| `check-features` | Validate per-audio feature arrays exist |
| `evaluate` | Evaluate a prediction CSV with baseline metrics |
| `list-data-dir` | List data directories |
| `list-all-audio-files` | List all audio files |

### Training Examples

```bash
# Basic training (exp-name is required)
g5 train v1 --exp-name my_experiment

# Train V2 with attention and augmentation
g5 train v2 --exp-name v2_experiment

# Override hyperparameters
g5 train v1 --exp-name my_experiment arch.training.max_epochs=100 arch.training.learning_rate=0.0005

# Change episode configuration
g5 train v1 --exp-name my_experiment train_param.k_way=5 train_param.n_shot=3
```

See [`docs/CLI_USAGE.md`](docs/CLI_USAGE.md) for complete documentation.

## How It Works

This project uses a two-phase pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1 (offline, run once):                                               │
│    .wav audio → STFT → Mel filterbank → Log/PCEN → .npy feature arrays     │
│                                                                             │
│  Phase 2 (online, each training iteration):                                 │
│    .npy features → DataModule → Episodic batches → ProtoNet → Loss         │
│                                                                             │
│  Inference:                                                                 │
│    Support examples → Prototypes → Distance to query → Classification      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Prototypical Networks

The core algorithm learns an embedding space where classification is done by computing distances to class prototypes (centroids):

1. **Embed** support examples using a ResNet encoder
2. **Compute prototypes** as the mean of embeddings per class
3. **Classify queries** by finding the nearest prototype

> 📖 See [`docs/PROTOTYPICAL_NETWORK.md`](docs/PROTOTYPICAL_NETWORK.md) for the full algorithm explanation.

## Project Structure

```
├── main.py                       # CLI entry point
├── conf/                         # Hydra configuration
│   ├── config.yaml               # Main config
│   ├── arch/v1.yaml              # V1 architecture config
│   ├── arch/v2.yaml              # V2 architecture config
│   ├── callbacks/                # Training callbacks
│   └── logger/                   # MLflow logger config
├── archs/
│   ├── train.py                  # Lightning trainer
│   ├── v1/                       # V1 architecture (ResNet encoder)
│   └── v2/                       # V2 architecture (ResNet + Attention)
├── preprocessing/
│   ├── preprocess.py             # Audio to features
│   ├── feature_export.py         # Feature export/validation
│   ├── datamodule.py             # Lightning DataModule
│   └── sequence_data/            # Sequence datasets + samplers
├── utils/
│   ├── mlflow_logger.py          # MLflow logging
│   ├── evaluation.py             # Baseline evaluation
│   ├── loss.py                   # Prototypical loss functions
│   └── distance.py               # Distance metrics
└── docs/                         # Documentation
    ├── DOCUMENTATION_INDEX.md    # Documentation overview
    ├── WORKFLOW.md               # End-to-end workflow guide
    ├── CLI_USAGE.md              # CLI reference
    ├── CONFIG_REFERENCE.md       # Configuration parameters
    ├── AUDIO_SIGNAL_PROCESSING.md # Signal processing details
    ├── FEATURES_AND_DATAFLOW.md  # Data loading pipeline
    ├── PREPROCESSING.md          # CSV parsing and segmentation
    ├── PROTOTYPICAL_NETWORK.md   # Algorithm explanation
    └── V2_IMPLEMENTATION_SUMMARY.md # V2 architecture details
```

## Documentation

| Document | Description |
|----------|-------------|
| [DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md) | Start here - overview of all docs |
| [WORKFLOW.md](docs/WORKFLOW.md) | End-to-end training workflow |
| [CLI_USAGE.md](docs/CLI_USAGE.md) | Command-line interface reference |
| [CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) | All configuration parameters explained |
| [AUDIO_SIGNAL_PROCESSING.md](docs/AUDIO_SIGNAL_PROCESSING.md) | How audio becomes features (with diagrams) |
| [FEATURES_AND_DATAFLOW.md](docs/FEATURES_AND_DATAFLOW.md) | DataModule and data loading pipeline |
| [PROTOTYPICAL_NETWORK.md](docs/PROTOTYPICAL_NETWORK.md) | Algorithm and math explanation |
| [V2_IMPLEMENTATION_SUMMARY.md](docs/V2_IMPLEMENTATION_SUMMARY.md) | V2 architecture (ResNet + Attention) |

## Configuration

Configuration uses Hydra. Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `arch.training.max_epochs` | Training epochs | 200 (V1) / 50 (V2) |
| `arch.training.learning_rate` | Learning rate | 0.001 |
| `train_param.k_way` | N-way (classes/episode) | 10 |
| `train_param.n_shot` | K-shot (samples/class) | 5 |
| `train_param.num_episodes` | Episodes/epoch | 2000 |
| `train_param.seg_len` | Segment length (seconds) | 0.2 |
| `features.sr` | Sample rate | 22050 |
| `features.n_mels` | Mel frequency bins | 128 |
| `features.feature_types` | Feature type(s) | logmel |

Override via CLI:
```bash
g5 train v1 --exp-name my_experiment arch.training.learning_rate=0.0005
```

> 📖 See [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md) for complete parameter documentation.

## Data Format

### Directory Structure

```
/data/msc-proj/
  Training_Set/
    BirdSpecies_A/
      audio_001.wav
      audio_001.csv
      audio_001_logmel.npy  ← Generated by g5 export-features
  Validation_Set_DSAI_2025_2026/
    ...
  Evaluation_Set_DSAI_2025_2026/
    ...
```

### Annotation CSV Format

CSV files with columns:
- `Audiofilename`: Audio file name
- `Starttime`: Segment start (seconds)
- `Endtime`: Segment end (seconds)
- `Q` or `CLASS_*`: Label (POS/NEG/UNK)

## Architectures

### V1 (Baseline)
- ResNet-style encoder with 3 BasicBlocks
- Euclidean distance metric
- 2048-dimensional embeddings

### V2 (Enhanced)
- ResNet + Channel/Temporal Attention
- SpecAugment data augmentation
- Learnable distance metric
- 1024-dimensional embeddings (memory-efficient)

```bash
# Train V1
g5 train v1 --exp-name v1_experiment

# Train V2
g5 train v2 --exp-name v2_experiment
```

> 📖 See [`docs/V2_IMPLEMENTATION_SUMMARY.md`](docs/V2_IMPLEMENTATION_SUMMARY.md) for V2 details.

## MLflow Tracking

Training logs are tracked with MLflow:

```bash
# View training logs
mlflow ui --backend-store-uri outputs/mlflow_experiments/my_experiment/mlruns
```

Open http://localhost:5000 in your browser.

## Requirements

- Python 3.12+
- uv (package manager)
- PyTorch 2.0+
- PyTorch Lightning
- Hydra
- librosa
- MLflow (optional, for tracking)

### Installation

```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

## License

This project is for educational purposes as part of the DCASE challenge.
