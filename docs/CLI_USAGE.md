# CLI Usage Guide

This document provides a complete reference for the DCASE Few-Shot Bioacoustic command-line interface.

## Setup

Before using the CLI, ensure you have installed dependencies and activated the virtual environment:

```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

## Quick Start

```bash
# 1. Export full-audio features (run once for sequence sampling)
g5 export-features

# 2. Train the model (exp-name is required)
g5 train v1 --exp-name my_experiment

# 3. Test the model
g5 test outputs/mlflow_experiments/my_experiment/checkpoints/last.ckpt
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `train` | Train model with PyTorch Lightning |
| `test` | Test a trained model checkpoint |
| `export-features` | Export per-audio feature arrays next to `.wav` files |
| `check-features` | Validate per-audio feature arrays exist |
| `list-data-dir` | List data directories |
| `list-all-audio-files` | List all audio files |

## Training

### Basic Training

```bash
g5 train v1 --exp-name my_experiment
```

This runs training with default configuration from `conf/config.yaml`. The `--exp-name` flag is required and specifies the experiment name for this run.

### Training Options

```bash
g5 train [ARCH] [OPTIONS] [OVERRIDES]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `ARCH` | Architecture to use (`v1`) | `v1` |
| `OVERRIDES` | Hydra config overrides | None |

**Options:**

| Option | Description |
|--------|-------------|
| `--exp-name`, `-e` | Experiment name for this run (required) |

### Examples

```bash
# Basic training (exp-name is required)
g5 train v1 --exp-name my_experiment

# Modify hyperparameters
g5 train v1 --exp-name my_experiment arch.training.max_epochs=100
g5 train v1 --exp-name my_experiment arch.training.learning_rate=0.0005

# Multiple overrides
g5 train v1 --exp-name my_experiment \
    arch.training.max_epochs=100 \
    arch.training.learning_rate=0.0005 \
    train_param.k_way=5

# Change episode configuration
g5 train v1 --exp-name my_experiment \
    train_param.k_way=5 \
    train_param.n_shot=3 \
    train_param.num_episodes=1000
```

### Common Overrides

| Override | Description | Default |
|----------|-------------|---------|
| `arch.training.max_epochs` | Maximum training epochs | 50 |
| `arch.training.learning_rate` | Learning rate | 0.001 |
| `arch.training.weight_decay` | Weight decay | 1e-4 |
| `train_param.k_way` | N-way (classes per episode) | 10 |
| `train_param.n_shot` | K-shot (support samples per class) | 5 |
| `train_param.num_episodes` | Episodes per epoch | 2000 |
| `exp_name` | Experiment run name (required via --exp-name) | v1_run |
| `seed` | Random seed | 1234 |

## Testing

### Test a Checkpoint

```bash
g5 test CHECKPOINT [OPTIONS] [OVERRIDES]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `CHECKPOINT` | Path to model checkpoint (.ckpt file) |
| `OVERRIDES` | Hydra config overrides |

**Options:**

| Option | Description |
|--------|-------------|
| `--arch`, `-a` | Architecture type (default: v1) |

### Examples

```bash
# Test the last checkpoint
g5 test outputs/mlflow_experiments/my_experiment/checkpoints/last.ckpt

# Test a specific checkpoint
g5 test outputs/mlflow_experiments/my_experiment/checkpoints/v1_050_0.8500.ckpt

# Test with different episode config
g5 test outputs/mlflow_experiments/my_experiment/checkpoints/last.ckpt train_param.k_way=5
```

## Feature Export (Full-Audio Arrays)

Use this path when training with the sequence-sampling datamodule.

### Export All Features

```bash
g5 export-features
```

```bash
# Export both logmel and pcen
g5 export-features --split all --type logmel@pcen
```

### Options

```bash
g5 export-features [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--split`, `-s` | Split to export (train/val/test/all) | all |
| `--force`, `-f` | Overwrite existing files | false |
| `--type`, `-t` | Feature types to export (e.g., logmel or logmel@pcen) | config default |

### Validate Exports

```bash
g5 check-features
```

```bash
# Validate specific feature types
g5 check-features --type logmel@pcen
```

### Output Location

Files are written next to each `.wav`:
```
BV_file1.wav
BV_file1_logmel.npy
```

## Data Listing

### List Data Directories

```bash
g5 list-data-dir --type TYPE
```

| Type | Description |
|------|-------------|
| `training` | Training data directories |
| `validation` | Validation data directories |
| `evaluation` | Evaluation/test data directories |
| `all` | All directories |

### Examples

```bash
g5 list-data-dir --type training
g5 list-data-dir --type all
```

### List Audio Files

```bash
g5 list-all-audio-files
```

Lists all audio files in the configured data directories.

## Configuration

### View Current Configuration

Training automatically prints the configuration. You can also check config files:

```bash
cat conf/config.yaml
cat conf/arch/v1.yaml
```

### Configuration Files

| File | Purpose |
|------|---------|
| `conf/config.yaml` | Main configuration |
| `conf/arch/v1.yaml` | V1 architecture config |
| `conf/callbacks/default.yaml` | Training callbacks |
| `conf/logger/mlflow.yaml` | MLflow logger config |
| `conf/trainer/default.yaml` | Lightning trainer config |

## Output Structure

After training, outputs are organized as:

```
outputs/
  mlflow_experiments/
    {exp_name}/
      checkpoints/
        last.ckpt
        v1_001_0.7500.ckpt
        v1_002_0.8000.ckpt
      mlruns/
        ...  # MLflow tracking data
```

## MLflow Tracking

Training logs are tracked with MLflow. To view the UI:

```bash
mlflow ui --backend-store-uri outputs/mlflow_experiments/my_experiment/mlruns
```

Then open http://localhost:5000 in your browser.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | GPU device selection |
| `HYDRA_FULL_ERROR` | Show full Hydra error traces |

### Examples

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 g5 train v1 --exp-name my_experiment

# Show full error traces
HYDRA_FULL_ERROR=1 g5 train v1 --exp-name my_experiment
```

## Troubleshooting

### Common Issues

**Missing feature files:**
```bash
# Export features first
g5 export-features --split all
```

**Out of memory:**
```bash
# Reduce batch size or episodes
g5 train v1 --exp-name my_experiment \
    annotations.batch_size=1 \
    train_param.num_episodes=500
```

**MLflow not available:**
```bash
# Install MLflow using uv
uv add mlflow

# Or train without MLflow (falls back to console logging)
g5 train v1 --exp-name my_experiment
```

## Complete Workflow Example

```bash
# 1. Check data directories
g5 list-data-dir --type all

# 2. Export features (takes time, run once)
g5 export-features --split all

# 3. Verify features were exported correctly
g5 check-features --split all

# 4. Train the model
g5 train v1 --exp-name experiment_1

# 5. View training logs in MLflow
mlflow ui --backend-store-uri outputs/mlflow_experiments/experiment_1/mlruns

# 6. Test the best model
g5 test outputs/mlflow_experiments/experiment_1/checkpoints/last.ckpt
```
