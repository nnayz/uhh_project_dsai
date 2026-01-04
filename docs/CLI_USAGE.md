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
# 1. Extract features (run once)
g5 extract-features

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
| `extract-features` | Extract and cache audio features |
| `cache-info` | Show cached feature statistics |
| `verify-cache` | Verify cache integrity |
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
| `--no-cache` | Disable feature caching (extract on-the-fly) |
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

# Train without feature cache (slower, for debugging)
g5 train v1 --exp-name my_experiment --no-cache

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
| `features.use_cache` | Use cached features | true |
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

## Feature Extraction

### Extract All Features

```bash
g5 extract-features --exp-name my_experiment
```

This extracts features for train, validation, and test splits. The `--exp-name` flag is required and organizes caches by experiment name.

### Options

```bash
g5 extract-features --exp-name EXPERIMENT [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--exp-name`, `-e` | Experiment name for this cache (required) | - |
| `--split`, `-s` | Split to extract (train/val/test/all) | all |
| `--force`, `-f` | Force re-extraction even if cache exists | false |

### Examples

```bash
# Extract all splits
g5 extract-features --exp-name my_experiment

# Extract only training features
g5 extract-features --exp-name my_experiment --split train

# Force re-extraction
g5 extract-features --exp-name my_experiment --force

# Extract specific split with force
g5 extract-features --exp-name my_experiment --split val --force
```

### Feature Cache Location

Features are cached in:
```
{path.root_dir}/features_cache/{exp_name}/{split}/
```

Example:
```
/data/msc-proj/features_cache/my_experiment/train/
  manifest.json
  BV/
    BV_file1_0.500_1.200.npy
  PB/
    PB_file1_1.000_2.500.npy
```

The cache is organized by experiment name, allowing different experiments to have separate feature caches. Each experiment maintains its own cache directory.

## Cache Management

### View Cache Information

```bash
g5 cache-info --exp-name my_experiment
```

Shows:
- Cache directory location
- Number of samples per split
- Number of classes
- Disk usage
- Class distribution

### Options

```bash
g5 cache-info --exp-name EXPERIMENT [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--exp-name`, `-e` | Experiment name for the cache (required) | - |
| `--split`, `-s` | Split to show info for | all |

### Examples

```bash
# Show all cache info
g5 cache-info --exp-name my_experiment

# Show only training cache
g5 cache-info --exp-name my_experiment --split train
```

### Verify Cache Integrity

```bash
g5 verify-cache --exp-name my_experiment
```

Checks that all cached feature files exist and are valid.

### Examples

```bash
# Verify all caches
g5 verify-cache --exp-name my_experiment

# Verify specific split
g5 verify-cache --exp-name my_experiment --split train
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

**No features cached:**
```bash
# Run feature extraction first
g5 extract-features --exp-name my_experiment
```

**Out of memory:**
```bash
# Reduce batch size or episodes
g5 train v1 --exp-name my_experiment \
    annotations.batch_size=1 \
    train_param.num_episodes=500
```

**Config hash mismatch:**
```bash
# Force re-extraction with new config
g5 extract-features --force
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

# 2. Extract features (takes time, run once)
g5 extract-features --exp-name experiment_1

# 3. Verify features were extracted correctly
g5 cache-info --exp-name experiment_1
g5 verify-cache --exp-name experiment_1

# 4. Train the model
g5 train v1 --exp-name experiment_1

# 5. View training logs in MLflow
mlflow ui --backend-store-uri outputs/mlflow_experiments/experiment_1/mlruns

# 6. Test the best model
g5 test outputs/mlflow_experiments/experiment_1/checkpoints/last.ckpt
```

