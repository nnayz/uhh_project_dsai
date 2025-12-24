# CLI Usage Guide

This document provides a complete reference for the DCASE Few-Shot Bioacoustic command-line interface.

## Quick Start

```bash
# 1. Extract features (run once)
python main.py extract-features

# 2. Train the model
python main.py train v1

# 3. Test the model
python main.py test outputs/protonet_baseline/v1_run/checkpoints/last.ckpt
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
python main.py train v1
```

This runs training with default configuration from `conf/config.yaml`.

### Training Options

```bash
python main.py train [ARCH] [OPTIONS] [OVERRIDES]
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
| `--exp-name`, `-e` | Custom experiment name for this run |

### Examples

```bash
# Basic training
python main.py train v1

# Custom experiment name
python main.py train v1 --exp-name my_experiment

# Modify hyperparameters
python main.py train v1 arch.training.max_epochs=100
python main.py train v1 arch.training.learning_rate=0.0005

# Multiple overrides
python main.py train v1 \
    arch.training.max_epochs=100 \
    arch.training.learning_rate=0.0005 \
    train_param.k_way=5

# Train without feature cache (slower, for debugging)
python main.py train v1 --no-cache

# Change episode configuration
python main.py train v1 \
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
| `exp_name` | Experiment run name | v1_run |
| `seed` | Random seed | 1234 |

## Testing

### Test a Checkpoint

```bash
python main.py test CHECKPOINT [OPTIONS] [OVERRIDES]
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
python main.py test outputs/protonet_baseline/v1_run/checkpoints/last.ckpt

# Test the best checkpoint
python main.py test outputs/protonet_baseline/v1_run/checkpoints/v1_050_0.8500.ckpt

# Test with different episode config
python main.py test checkpoints/model.ckpt train_param.k_way=5
```

## Feature Extraction

### Extract All Features

```bash
python main.py extract-features
```

This extracts features for train, validation, and test splits.

### Options

```bash
python main.py extract-features [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--split`, `-s` | Split to extract (train/val/test/all) | all |
| `--force`, `-f` | Force re-extraction even if cache exists | false |

### Examples

```bash
# Extract all splits
python main.py extract-features

# Extract only training features
python main.py extract-features --split train

# Force re-extraction
python main.py extract-features --force

# Extract specific split with force
python main.py extract-features --split val --force
```

### Feature Cache Location

Features are cached in:
```
{path.root_dir}/features_cache/{version}/{config_hash}/{split}/
```

Example:
```
/data/msc-proj/features_cache/v1/abc123def456/train/
  manifest.json
  BV/
    BV_file1_0.500_1.200.npy
  PB/
    PB_file1_1.000_2.500.npy
```

## Cache Management

### View Cache Information

```bash
python main.py cache-info
```

Shows:
- Cache directory location
- Number of samples per split
- Number of classes
- Disk usage
- Class distribution

### Options

```bash
python main.py cache-info [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--split`, `-s` | Split to show info for | all |

### Examples

```bash
# Show all cache info
python main.py cache-info

# Show only training cache
python main.py cache-info --split train
```

### Verify Cache Integrity

```bash
python main.py verify-cache
```

Checks that all cached feature files exist and are valid.

### Examples

```bash
# Verify all caches
python main.py verify-cache

# Verify specific split
python main.py verify-cache --split train
```

## Data Listing

### List Data Directories

```bash
python main.py list-data-dir --type TYPE
```

| Type | Description |
|------|-------------|
| `training` | Training data directories |
| `validation` | Validation data directories |
| `evaluation` | Evaluation/test data directories |
| `all` | All directories |

### Examples

```bash
python main.py list-data-dir --type training
python main.py list-data-dir --type all
```

### List Audio Files

```bash
python main.py list-all-audio-files
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
  {experiment_name}/
    {run_name}/
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
mlflow ui --backend-store-uri outputs/protonet_baseline/v1_run/mlruns
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
CUDA_VISIBLE_DEVICES=0 python main.py train v1

# Show full error traces
HYDRA_FULL_ERROR=1 python main.py train v1
```

## Troubleshooting

### Common Issues

**No features cached:**
```bash
# Run feature extraction first
python main.py extract-features
```

**Out of memory:**
```bash
# Reduce batch size or episodes
python main.py train v1 \
    annotations.batch_size=1 \
    train_param.num_episodes=500
```

**Config hash mismatch:**
```bash
# Force re-extraction with new config
python main.py extract-features --force
```

**MLflow not available:**
```bash
# Install MLflow
pip install mlflow

# Or train without MLflow (falls back to console logging)
python main.py train v1
```

## Complete Workflow Example

```bash
# 1. Check data directories
python main.py list-data-dir --type all

# 2. Extract features (takes time, run once)
python main.py extract-features

# 3. Verify features were extracted correctly
python main.py cache-info
python main.py verify-cache

# 4. Train the model
python main.py train v1 --exp-name experiment_1

# 5. View training logs in MLflow
mlflow ui --backend-store-uri outputs/protonet_baseline/experiment_1/mlruns

# 6. Test the best model
python main.py test outputs/protonet_baseline/experiment_1/checkpoints/last.ckpt
```

