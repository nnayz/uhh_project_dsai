# Workflow Guide

This document describes the end-to-end workflow using only the current, active code paths in this repository. It covers feature export, dataset construction, dataloaders, training, testing, and evaluation.

## Table of Contents

1. [Configuration Loading](#1-configuration-loading)
2. [Feature Export (Offline)](#2-feature-export-offline-run-once)
3. [Training Entry Point](#3-training-entry-point)
4. [DataModule Creation](#4-datamodule-creation)
5. [Datasets and Samplers](#5-datasets-and-samplers)
6. [DataLoaders](#6-dataloaders)
7. [Model Construction](#7-model-construction)
8. [Training Loop](#8-training-loop)
9. [Testing](#9-testing)
10. [Standalone Evaluation](#10-standalone-evaluation)
11. [Outputs](#11-outputs)
12. [Typical End-to-End Workflow](#12-typical-end-to-end-workflow)

---

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HIGH-LEVEL WORKFLOW                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────┐                                                           │
│   │ 1. Config       │ ← conf/config.yaml + CLI overrides                        │
│   └────────┬────────┘                                                           │
│            │                                                                    │
│            ▼                                                                    │
│   ┌─────────────────┐                                                           │
│   │ 2. Export       │ ← g5 export-features (run once)                           │
│   │    Features     │   .wav → .npy feature arrays                              │
│   └────────┬────────┘                                                           │
│            │                                                                    │
│            ▼                                                                    │
│   ┌─────────────────┐                                                           │
│   │ 3. Training     │ ← g5 train v1 --exp-name my_exp                           │
│   │    Pipeline     │                                                           │
│   │    ┌──────────────────────────────────────────────┐                         │
│   │    │ DataModule → Dataset → Sampler → DataLoader  │                         │
│   │    │      ↓                                       │                         │
│   │    │ Model → Encoder → Prototypes → Loss          │                         │
│   │    │      ↓                                       │                         │
│   │    │ Lightning Trainer → fit()                    │                         │
│   │    └──────────────────────────────────────────────┘                         │
│   └────────┬────────┘                                                           │
│            │                                                                    │
│            ▼                                                                    │
│   ┌─────────────────┐                                                           │
│   │ 4. Testing      │ ← g5 test checkpoint.ckpt                                 │
│   └────────┬────────┘                                                           │
│            │                                                                    │
│            ▼                                                                    │
│   ┌─────────────────┐                                                           │
│   │ 5. Evaluation   │ ← g5 evaluate --pred pred.csv                             │
│   └─────────────────┘                                                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1) Configuration Loading

**Entry point**: `main.py`

- `load_config()` loads Hydra configuration from `conf/config.yaml` using `initialize_config_dir()` and `compose()`.
- CLI commands pass overrides such as `arch.training.max_epochs=100` or `train_param.k_way=5`.

**Key files**:
| File | Purpose |
|------|---------|
| `conf/config.yaml` | Main configuration (entry point) |
| `conf/arch/v1.yaml` | V1 architecture settings |
| `conf/arch/v2.yaml` | V2 architecture settings |
| `conf/callbacks/default.yaml` | Training callbacks |
| `conf/logger/mlflow.yaml` | MLflow logger settings |
| `conf/trainer/default.yaml` | PyTorch Lightning trainer settings |

> 📖 **See also**: [CONFIG_REFERENCE.md](./CONFIG_REFERENCE.md) for detailed parameter explanations.

---

## 2) Feature Export (Offline, Run Once)

**Entry point**: `g5 export-features` in `main.py`

**Methods**:
- `preprocessing/feature_export.py:export_features()`
- `preprocessing/feature_export.py:validate_features()`

**Workflow**:
1. Walk train/val/test directories (`cfg.path.train_dir`, `cfg.path.eval_dir`, `cfg.path.test_dir`).
2. For each `.wav`, export feature arrays next to it.
3. Supported suffixes from `cfg.features.feature_types` (e.g., `logmel`, `pcen`).

**Feature computation**:
- `preprocessing/preprocess.py:load_audio()` loads and normalizes waveform.
- `preprocessing/preprocess.py:waveform_to_logmel()` or `waveform_to_pcen()` compute features.
- Files are saved as `*_logmel.npy` and/or `*_pcen.npy`.

```
audio.wav  →  audio_logmel.npy  (T × 128)
           →  audio_pcen.npy    (T × 128, if configured)
```

> 📖 **See also**: [AUDIO_SIGNAL_PROCESSING.md](./AUDIO_SIGNAL_PROCESSING.md) for detailed signal processing explanation.

---

## 3) Training Entry Point

**Entry point**: `g5 train v1 --exp-name ...` in `main.py`

**Methods**:
- `main.py:train()` builds a Python command that runs `archs/train.py`.
- `archs/train.py:main()` is the Hydra-driven training pipeline.

**The training command**:
- Always includes `arch=<v1|v2>`.
- Always includes `+exp_name=<experiment>`.
- Appends any CLI overrides.

**Example**:
```bash
g5 train v1 --exp-name my_experiment arch.training.max_epochs=100
```

---

## 4) DataModule Creation

**Entry point**: `archs/train.py:main()`

**Methods**:
- `preprocessing/datamodule.py:DCASEFewShotDataModule` constructor
- `preprocessing/datamodule.py:DCASEFewShotDataModule.prepare_data()`
- `preprocessing/datamodule.py:DCASEFewShotDataModule.init()`

**Workflow**:
1. `prepare_data()` validates that all required `*_logmel.npy` / `*_pcen.npy` files exist using `validate_features()`.
2. `init()` constructs datasets and samplers for train, val, and test.

> 📖 **See also**: [FEATURES_AND_DATAFLOW.md](./FEATURES_AND_DATAFLOW.md) for detailed data loading explanation.

---

## 5) Datasets and Samplers

**Entry point**: `preprocessing/datamodule.py:DCASEFewShotDataModule.init()`

### Training Datasets

| Dataset | File | When Used |
|---------|------|-----------|
| `PrototypeDynamicArrayDataSet` | `preprocessing/sequence_data/dynamic_pcen_dataset.py` | Default training |
| `PrototypeDynamicArrayDataSetWithEval` | `preprocessing/sequence_data/dynamic_pcen_dataset_first_5.py` | When `train_param.use_validation_first_5=true` |

### Validation Dataset

| Dataset | File |
|---------|------|
| `PrototypeDynamicArrayDataSetVal` | `preprocessing/sequence_data/dynamic_pcen_dataset_val.py` |

### Test Datasets

| Dataset | File | When Used |
|---------|------|-----------|
| `PrototypeTestSet` | `preprocessing/sequence_data/test_loader.py` | Default testing |
| `PrototypeAdaSeglenBetterNegTestSetV2` | `preprocessing/sequence_data/test_loader_ada_seglen_better_neg_v2.py` | When `train_param.adaptive_seg_len=true` |

### Sampler

- `preprocessing/sequence_data/identity_sampler.py:IdentityBatchSampler`
- Chooses `k_way` classes per episode and tiles indices for episodic batches.
- Can mix extra classes for training when provided.

### Feature Loading

- `preprocessing/sequence_data/pcen.py:Feature_Extractor`
- `update_mean_std()` computes mean/std from existing `.npy` files.
- `extract_feature()` loads cached features, normalizes them, and returns time-major arrays.

### Segment Selection

- `PrototypeDynamicArrayDataSet.select_positive()` and `select_negative()` choose segments.
- `select_segment()` crops or tiles segments to a fixed length (`train_param.seg_len`).

> 📖 **See also**: [PREPROCESSING.md](./PREPROCESSING.md) for CSV parsing and segment selection details.

---

## 6) DataLoaders

**Entry point**: `preprocessing/datamodule.py:DCASEFewShotDataModule.init()`

| Loader | Configuration |
|--------|---------------|
| Train | `DataLoader(dataset, batch_sampler=IdentityBatchSampler, num_workers=2)` |
| Val | `DataLoader(dataset, batch_sampler=IdentityBatchSampler, num_workers=2)` |
| Test | `DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)` |

---

## 7) Model Construction

**Entry point**: `archs/train.py:build_model()`

### V1 Architecture

| Component | File |
|-----------|------|
| Lightning Module | `archs/v1/lightning_module.py:ProtoNetLightningModule` |
| Model | `archs/v1/arch.py:ProtoNet` (ResNet-style encoder + prototypical head) |

### V2 Architecture

| Component | File |
|-----------|------|
| Lightning Module | `archs/v2/lightning_module.py:ProtoNetV2LightningModule` |
| Model | `archs/v2/arch.py:ProtoNetV2` (ResNet + Attention encoder) |
| Augmentation | `archs/v2/augmentation.py:BioacousticAugmentation` (SpecAugment) |

> 📖 **See also**: 
> - [PROTOTYPICAL_NETWORK.md](./PROTOTYPICAL_NETWORK.md) for the algorithm explanation.
> - [V2_IMPLEMENTATION_SUMMARY.md](./V2_IMPLEMENTATION_SUMMARY.md) for V2 architecture details.

---

## 8) Training Loop

**Entry point**: `archs/train.py:main()`

**Workflow**:
1. Builds callbacks and loggers with `build_callbacks()` and `build_pl_loggers()`.
2. Creates `lightning.Trainer` and calls `trainer.fit()`.

**Loss and episode logic**:
- `archs/v1/lightning_module.py:_step()` or `archs/v2/lightning_module.py:_step()`
- Uses `utils/loss.py:prototypical_loss()` or `prototypical_loss_filter_negative()`.

**V1 validation evaluation**:
- `archs/v1/lightning_module.py:_run_val_event_eval()` generates event-level CSVs under `outputs/val_eval/` and runs `utils/evaluation.py:evaluate()` with post-processing sweeps.

---

## 9) Testing

**Entry point**: `g5 test <checkpoint>` in `main.py`

**Workflow**:
1. Parses `exp_name` from the checkpoint path.
2. Calls `archs/train.py` with `train=false test=true`.
3. DataModule builds the test dataset and `trainer.test()` runs evaluation.

**Example**:
```bash
g5 test outputs/mlflow_experiments/my_experiment/checkpoints/last.ckpt
```

---

## 10) Standalone Evaluation

**Entry point**: `g5 evaluate --pred <csv> --dataset <val|test>` in `main.py`

**Methods**:
- `utils/evaluation.py:evaluate()`
- `utils/post_proc.py:post_processing()`
- `utils/post_proc_new.py:post_processing()`

**Workflow**:
1. Copies prediction CSV to `outputs/evaluation/<name>/Eval_raw.csv`.
2. Sweeps post-processing thresholds and selects the best F-measure.
3. Optionally computes PSDS scores when validation metadata is available.

---

## 11) Outputs

### Training Output

```
outputs/mlflow_experiments/<exp_name>/
├── checkpoints/
│   ├── last.ckpt
│   └── v1_001_0.7500.ckpt
└── mlruns/
    └── ... (MLflow tracking data)
```

### Validation Event Evaluation

```
outputs/val_eval/epoch_<N>/
├── 0.5/
│   ├── Eval_raw.csv
│   └── Eval_VAL_threshold_*.csv
└── ...
```

### Evaluation Outputs

```
outputs/evaluation/<pred_name>/
├── Eval_raw.csv
└── ... (post-processed CSVs)
```

---

## 12) Typical End-to-End Workflow

```bash
# 1. Export features once (takes ~10-30 minutes depending on dataset size)
g5 export-features --split all

# 2. Verify feature presence
g5 check-features --split all

# 3. Train a V1 model
g5 train v1 --exp-name my_experiment

# 4. Or train a V2 model with custom settings
g5 train v2 --exp-name v2_experiment arch.training.max_epochs=50

# 5. Test a checkpoint
g5 test outputs/mlflow_experiments/my_experiment/checkpoints/last.ckpt

# 6. Evaluate a prediction CSV
g5 evaluate --pred path/to/pred.csv --dataset val
```

---

## Related Documentation

| Document | Topic |
|----------|-------|
| [CLI_USAGE.md](./CLI_USAGE.md) | Command-line interface reference |
| [CONFIG_REFERENCE.md](./CONFIG_REFERENCE.md) | Configuration parameters |
| [AUDIO_SIGNAL_PROCESSING.md](./AUDIO_SIGNAL_PROCESSING.md) | Audio processing pipeline |
| [FEATURES_AND_DATAFLOW.md](./FEATURES_AND_DATAFLOW.md) | Data loading pipeline |
| [PREPROCESSING.md](./PREPROCESSING.md) | CSV parsing and segmentation |
| [PROTOTYPICAL_NETWORK.md](./PROTOTYPICAL_NETWORK.md) | Algorithm explanation |
| [V2_IMPLEMENTATION_SUMMARY.md](./V2_IMPLEMENTATION_SUMMARY.md) | V2 architecture details |
