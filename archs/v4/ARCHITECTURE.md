# V4 Architecture: EfficientNet-B1 Encoder

This document describes the V4 Prototypical Network with EfficientNet-B1 encoder.

## Overview

V4 replaces the ResNet encoder from V1 with EfficientNet-B1, providing:
- **Better parameter efficiency**: Compound scaling balances depth, width, and resolution
- **Built-in attention**: Squeeze-and-Excitation blocks for channel attention
- **ImageNet pretraining**: Better initialization than training from scratch
- **Memory efficient**: EfficientNet-B1 uses ~3GB GPU memory, leaving room on 8GB GPUs

## Model Architecture

### Encoder: EfficientNet-B1

The encoder uses EfficientNet-B1 pretrained on ImageNet:

```
Input: (B, T, n_mels) → Reshape to (B, 1, T, 128)
  → Replicate to 3 channels (B, 3, T, 128)
  → EfficientNet-B1 backbone (extract features)
  → Global Average Pooling
  → Linear projection (1280 → 2048)
  → Output: (B, 2048)
```

**Key features:**
- **Compound scaling**: Efficiently scales depth, width, and resolution together
- **Mobile inverted bottlenecks**: More efficient than standard ResNet blocks
- **Squeeze-and-Excitation**: Built-in channel attention mechanism
- **Global pooling**: Handles variable-length spectrograms automatically

### Prototypical Network

The prototypical network structure remains identical to V1:
- Compute prototypes as mean embeddings per class
- Euclidean distance for classification
- Same loss function and training loop

## Training Configuration

V4 uses the same training setup as V1:
- **Distance metric**: Euclidean
- **Features**: PCEN (same as best V1 configuration)
- **k-way**: 15 (same as best V1)
- **Negative segments**: CSV annotations (same as best V1)
- **Monitor**: F-measure on validation set

## Expected Benefits

1. **Better feature extraction**: EfficientNet's compound scaling provides richer embeddings
2. **Faster convergence**: ImageNet pretraining helps with initialization
3. **Memory efficient**: Fits comfortably on 8GB GPU
4. **Training time**: ~2-3 hours for 50 epochs (fits within 3-4 hour constraint)

## Comparison with V1

| Aspect | V1 (ResNet) | V4 (EfficientNet-B1) |
|--------|-------------|----------------------|
| Encoder | Custom ResNet (3-4 blocks) | EfficientNet-B1 (pretrained) |
| Parameters | ~Custom | ~8M (pretrained) |
| Memory | ~2GB | ~3GB |
| Training time | ~2-3 hours | ~2-3 hours |
| Expected F-measure | 48.925% | Target: >48.925% |

## Implementation Notes

- EfficientNet expects 3-channel input, so single-channel spectrograms are replicated
- Global average pooling handles variable time lengths automatically
- Projection head maps EfficientNet's 1280-dim output to 2048-dim embeddings
- All other components (loss, distance, evaluation) remain identical to V1

