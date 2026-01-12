# Documentation Index

This document provides an overview of all documentation files in the `docs/` folder, explaining the purpose and content of each file.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Documentation](#core-documentation)
3. [Technical Details](#technical-details)
4. [Reference Materials](#reference-materials)
5. [Test Results](#test-results)
6. [Reading Order](#recommended-reading-order)

---

## Quick Start

If you're new to this project, read these files in order:

1. **[CLI_USAGE.md](./CLI_USAGE.md)** - How to use the command-line interface
2. **[WORKFLOW.md](./WORKFLOW.md)** - End-to-end training workflow
3. **[CONFIG_REFERENCE.md](./CONFIG_REFERENCE.md)** - Configuration parameters

---

## Core Documentation

### [WORKFLOW.md](./WORKFLOW.md)
**Purpose**: Complete end-to-end workflow guide

**Contents**:
- Configuration loading process
- Feature export (offline preprocessing)
- Training entry points
- DataModule and DataLoader setup
- Model construction
- Training and testing loops
- Output locations and structure

**When to read**: When you need to understand the complete pipeline from data to trained model.

---

### [CLI_USAGE.md](./CLI_USAGE.md)
**Purpose**: Command-line interface reference

**Contents**:
- All available CLI commands (`g5 train`, `g5 test`, etc.)
- Command options and arguments
- Example usage patterns
- Troubleshooting common issues

**When to read**: When you need to run commands or understand CLI options.

---

### [CONFIG_REFERENCE.md](./CONFIG_REFERENCE.md)
**Purpose**: Complete configuration parameter reference

**Contents**:
- All configuration files and their structure
- Feature extraction parameters (sample rate, FFT, mel bins)
- Training parameters (k-way, n-shot, learning rate)
- Evaluation parameters (threshold, hop length)
- Architecture-specific settings (V1 vs V2)
- CLI override examples

**When to read**: When you need to modify training settings or understand what each parameter does.

---

## Technical Details

### [AUDIO_SIGNAL_PROCESSING.md](./AUDIO_SIGNAL_PROCESSING.md)
**Purpose**: Detailed explanation of audio processing pipeline

**Contents**:
- Raw audio loading and normalization
- Short-Time Fourier Transform (STFT)
- Mel filterbank application
- Log-mel spectrogram computation
- PCEN (Per-Channel Energy Normalization)
- Feature normalization
- Segment extraction from full-audio arrays

**Includes**: ASCII diagrams showing signal flow and transformations.

**When to read**: When you need to understand how audio becomes features, or debug feature extraction issues.

---

### [FEATURES_AND_DATAFLOW.md](./FEATURES_AND_DATAFLOW.md)
**Purpose**: Feature extraction and data loading pipeline

**Contents**:
- Offline feature export process
- DataModule architecture
- Dataset classes (training, validation, test)
- Batch sampler for episodic training
- How features flow from files to batches

**Includes**: Complete data flow diagrams.

**When to read**: When you need to understand how data is loaded during training.

---

### [PREPROCESSING.md](./PREPROCESSING.md)
**Purpose**: CSV-driven annotation parsing and segmentation

**Contents**:
- How CSV annotations are parsed
- Positive/negative segment extraction
- Time-to-frame conversion
- Segment selection strategies
- Dataset class internals

**When to read**: When you need to understand how annotations drive the training data.

---

### [PROTOTYPICAL_NETWORK.md](./PROTOTYPICAL_NETWORK.md)
**Purpose**: Explanation of the prototypical network approach

**Contents**:
- Few-shot learning concepts
- Prototypical networks theory
- Mathematical foundation (prototypes, distances, loss)
- V1 architecture implementation
- Training episodes
- Inference (testing) process

**Includes**: Visual diagrams of embedding space and prototype computation.

**When to read**: When you need to understand the machine learning approach.

---

## Reference Materials

### [V2_IMPLEMENTATION_SUMMARY.md](./V2_IMPLEMENTATION_SUMMARY.md)
**Purpose**: Summary of V2 architecture enhancements

**Contents**:
- V2 architecture overview (ResNet + Attention)
- Component descriptions (channel attention, temporal attention)
- Data augmentation (SpecAugment)
- Learnable distance metric
- V1 vs V2 comparison
- Usage examples
- Expected performance improvements

**When to read**: When working with V2 or comparing architectures.

---

### [METRICS_ENHANCEMENT_SUMMARY.md](./METRICS_ENHANCEMENT_SUMMARY.md)
**Purpose**: Summary of enhanced evaluation metrics

**Contents**:
- New metrics added (Precision, Recall, F1, per-class accuracy)
- How to interpret metrics
- Changes made to lightning module
- Dependencies added

**When to read**: When analyzing training results or understanding logged metrics.

---

## Test Results

### [test_results.txt](./test_results.txt)
**Purpose**: V1 test output records

**Contents**: Raw test output from V1 experiments.

---

### [v2_test_results.txt](./v2_test_results.txt)
**Purpose**: V2 test output records

**Contents**: Raw test output from V2 experiments.

---

## Recommended Reading Order

### For New Users

1. **[CLI_USAGE.md](./CLI_USAGE.md)** - Learn the commands
2. **[WORKFLOW.md](./WORKFLOW.md)** - Understand the pipeline
3. **[CONFIG_REFERENCE.md](./CONFIG_REFERENCE.md)** - Learn configuration options

### For Understanding the Data Pipeline

1. **[AUDIO_SIGNAL_PROCESSING.md](./AUDIO_SIGNAL_PROCESSING.md)** - Audio to features
2. **[FEATURES_AND_DATAFLOW.md](./FEATURES_AND_DATAFLOW.md)** - Features to batches
3. **[PREPROCESSING.md](./PREPROCESSING.md)** - CSV parsing details

### For Understanding the Model

1. **[PROTOTYPICAL_NETWORK.md](./PROTOTYPICAL_NETWORK.md)** - Core algorithm
2. **[V2_IMPLEMENTATION_SUMMARY.md](./V2_IMPLEMENTATION_SUMMARY.md)** - Enhanced architecture

### For Analyzing Results

1. **[METRICS_ENHANCEMENT_SUMMARY.md](./METRICS_ENHANCEMENT_SUMMARY.md)** - Understanding metrics
2. **[test_results.txt](./test_results.txt)** / **[v2_test_results.txt](./v2_test_results.txt)** - Example outputs

---

## File Structure

```
docs/
├── DOCUMENTATION_INDEX.md          # This file - overview of all docs
│
├── CLI_USAGE.md                    # Command-line interface guide
├── WORKFLOW.md                     # End-to-end workflow
├── CONFIG_REFERENCE.md             # Configuration parameters
│
├── AUDIO_SIGNAL_PROCESSING.md      # Audio processing details
├── FEATURES_AND_DATAFLOW.md        # Feature and data flow
├── PREPROCESSING.md                # CSV parsing and segmentation
├── PROTOTYPICAL_NETWORK.md         # Algorithm explanation
│
├── V2_IMPLEMENTATION_SUMMARY.md    # V2 architecture summary
├── METRICS_ENHANCEMENT_SUMMARY.md  # Metrics explanation
│
├── test_results.txt                # V1 test outputs
└── v2_test_results.txt             # V2 test outputs
```

---

## Quick Reference Table

| Document | Type | Audience | Key Topics |
|----------|------|----------|------------|
| CLI_USAGE.md | Guide | All users | Commands, examples |
| WORKFLOW.md | Guide | All users | Pipeline, training |
| CONFIG_REFERENCE.md | Reference | All users | Parameters, settings |
| AUDIO_SIGNAL_PROCESSING.md | Technical | Developers | STFT, mel, PCEN |
| FEATURES_AND_DATAFLOW.md | Technical | Developers | DataModule, datasets |
| PREPROCESSING.md | Technical | Developers | CSV parsing, segments |
| PROTOTYPICAL_NETWORK.md | Conceptual | All users | Algorithm, math |
| V2_IMPLEMENTATION_SUMMARY.md | Technical | Developers | V2 architecture |
| METRICS_ENHANCEMENT_SUMMARY.md | Reference | All users | Precision, Recall, F1 |
