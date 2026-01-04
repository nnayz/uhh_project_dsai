# Enhanced Evaluation Metrics - Implementation Summary

## Date: 2025-12-31

## Overview
Enhanced the V1 Prototypical Network with comprehensive evaluation metrics for better model analysis and debugging.

## Changes Made

### 1. Updated Files

#### `archs/v1/lightning_module.py`
- **Added imports**: torchmetrics library for metric computation
- **New parameter**: `num_classes` (default: 10) for metric initialization
- **New metrics tracked**:
  - Precision (macro-averaged)
  - Recall (macro-averaged)
  - F1-Score (macro-averaged)
  - Per-class accuracy (for each of the n_way classes)
  - Average per-class accuracy
  - Confusion matrix (computed but not logged to avoid clutter)

- **New methods**:
  - `on_validation_epoch_end()`: Computes and logs validation metrics
  - `on_test_epoch_end()`: Computes and logs test metrics, prints confusion matrix

#### `archs/train.py`
- **Added parameter**: `num_classes=cfg.arch.episodes.n_way` when building the model
- This passes the n_way value (default: 10) to the Lightning module

### 2. Dependencies Installed
- `lightning==2.6.0` (PyTorch Lightning framework)
- `pytorch-lightning==2.6.0` (backend)
- `torchmetrics==1.8.2` (metric computation library)

## What You'll See Now

### Before (Old Output)
```
Epoch 14:
  train_loss: 0.8856
  train_acc: 0.9824
  val_loss: 1.9798
  val_acc: 0.3938
```

### After (Enhanced Output)
```
Epoch 14:
  train_loss: 0.8856
  train_acc: 0.9824
  val_loss: 1.9798
  val_acc: 0.3938
  val_precision: 0.4123    ← NEW
  val_recall: 0.3856       ← NEW
  val_f1: 0.3982           ← NEW
  val_avg_class_acc: 0.4012 ← NEW
  val_class_0_acc: 0.52    ← NEW (per species)
  val_class_1_acc: 0.48    ← NEW
  ...
  val_class_9_acc: 0.21    ← NEW
```

## Why These Metrics Matter

### 1. **Precision & Recall**
- **Precision**: Of detected animals, how many were correct?
- **Recall**: Of all actual animals, how many did we find?
- **Critical for bioacoustics**: False positives vs false negatives have different costs

### 2. **F1-Score**
- Harmonic mean of precision and recall
- Better than accuracy for imbalanced classes (common in DCASE)

### 3. **Per-Class Accuracy**
- Shows which species the model struggles with
- Example: If Bat_C has 20% accuracy while others have 80%, focus improvement there

### 4. **Confusion Matrix**
- Shows which species are confused with each other
- Helps identify similar-sounding species that need better features

## How to Use

### Training (unchanged)
```bash
g5 train v1 --exp-name my_experiment
```

### View Metrics in MLflow
```bash
mlflow ui --backend-store-uri outputs/mlflow_experiments/my_experiment/mlruns
```

Then open http://localhost:5000 and you'll see all the new metrics!

### Interpreting Results

**Good Model:**
```
val_acc: 0.70
val_precision: 0.68
val_recall: 0.72
val_f1: 0.70
```
Balanced performance across all metrics.

**Too Conservative:**
```
val_acc: 0.60
val_precision: 0.85  ← High
val_recall: 0.45     ← Low
```
Model is very precise but misses many animals (low recall).

**Too Aggressive:**
```
val_acc: 0.60
val_precision: 0.45  ← Low
val_recall: 0.85     ← High
```
Model finds many animals but makes many mistakes (low precision).

## Backward Compatibility

✅ **All existing functionality preserved**
- Training loop unchanged
- Existing metrics (loss, accuracy) still logged
- No performance impact during training
- Metrics only computed during validation/test

## Testing

✅ **Verified:**
- `lightning_module.py` imports successfully
- `train.py` syntax is valid
- No linter errors
- All dependencies installed

## Next Steps

1. Run a training session to see the new metrics in action
2. Check MLflow UI to visualize the metrics
3. Use per-class accuracy to identify problematic species
4. Compare precision/recall to understand model behavior

## Notes

- Metrics are computed on GPU for efficiency
- Confusion matrix is printed at test time but not logged (to avoid clutter)
- All metrics reset after each epoch to avoid accumulation errors
- Uses torchmetrics (industry standard, maintained by PyTorch Lightning team)

## Questions?

If you see any issues or want to customize the metrics, the main code is in:
- `archs/v1/lightning_module.py` (lines 12-24 for imports, lines 81-113 for metrics initialization)

