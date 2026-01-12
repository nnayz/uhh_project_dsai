# Prototypical Networks for Few-Shot Bioacoustic Classification

This document explains the prototypical network approach used in this project for few-shot learning of bioacoustic sounds.

## Table of Contents

1. [What is Few-Shot Learning?](#what-is-few-shot-learning)
2. [Prototypical Networks Overview](#prototypical-networks-overview)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Implementation in This Project](#implementation-in-this-project)
5. [Training and Inference](#training-and-inference)
6. [Distance Metrics](#distance-metrics)

---

## What is Few-Shot Learning?

Few-shot learning is a machine learning scenario where a model must generalize to new classes with only a small number of labeled examples. This is particularly relevant for bioacoustics, where:

- **New species** may have very few recorded examples
- **Rare calls** may only occur a handful of times in recordings
- **Field conditions** make collecting large datasets difficult

### Terminology

| Term | Definition | This Project |
|------|------------|--------------|
| **N-way** | Number of classes per episode | `k_way = 10` |
| **K-shot** | Number of support examples per class | `n_shot = 5` |
| **Support Set** | Labeled examples used to build prototypes | 5 segments per class |
| **Query Set** | Samples to classify | 5 segments per class |
| **Episode** | One training iteration with N classes and K shots | 10 classes × 5 shots |

---

## Prototypical Networks Overview

Prototypical Networks (Snell et al., 2017) learn an embedding space where classification is performed by computing distances to class prototypes.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      PROTOTYPICAL NETWORK CONCEPT                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   The key idea: "Similar things should be close together in embedding space"   │
│                                                                                 │
│   Step 1: Embed all support examples                                            │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │   Support Examples           Encoder           Embeddings               │   │
│   │   ┌───────────────┐         (ResNet)           (2048-dim vectors)       │   │
│   │   │ Bird A: [s₁]  │ ───────►  fᵩ  ──────────►  [e₁]                     │   │
│   │   │         [s₂]  │ ───────►  fᵩ  ──────────►  [e₂]                     │   │
│   │   │         [s₃]  │ ───────►  fᵩ  ──────────►  [e₃]                     │   │
│   │   └───────────────┘                                                     │   │
│   │   ┌───────────────┐                                                     │   │
│   │   │ Bird B: [s₄]  │ ───────►  fᵩ  ──────────►  [e₄]                     │   │
│   │   │         [s₅]  │ ───────►  fᵩ  ──────────►  [e₅]                     │   │
│   │   │         [s₆]  │ ───────►  fᵩ  ──────────►  [e₆]                     │   │
│   │   └───────────────┘                                                     │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   Step 2: Compute prototypes (mean of class embeddings)                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │   Prototype A = mean([e₁, e₂, e₃])                                      │   │
│   │   Prototype B = mean([e₄, e₅, e₆])                                      │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│   Step 3: Classify query by nearest prototype                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                         │   │
│   │   Embedding Space:                                                      │   │
│   │                                                                         │   │
│   │         ●  ●                                                            │   │
│   │        ●  ★A                          ●  ●                              │   │
│   │         ●                            ●  ★B                              │   │
│   │                                       ●                                 │   │
│   │                   ◆ query                                               │   │
│   │                                                                         │   │
│   │   ● = support embeddings                                                │   │
│   │   ★ = prototypes (class centers)                                        │   │
│   │   ◆ = query to classify                                                 │   │
│   │                                                                         │   │
│   │   Distance to A: d(◆, ★A) = 2.3                                         │   │
│   │   Distance to B: d(◆, ★B) = 4.1                                         │   │
│   │                                                                         │   │
│   │   → Query is closer to A, so classify as Bird A                         │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundation

### Prototype Computation

For a class $k$ with support set $S_k = \{(x_1, y_1), ..., (x_n, y_n)\}$, the prototype is:

$$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i)$$

Where:
- $c_k$ is the prototype (centroid) for class $k$
- $f_\phi$ is the embedding function (neural network encoder)
- $|S_k|$ is the number of support examples in class $k$

### Classification

A query point $x$ is classified by computing distances to all prototypes:

$$p(y = k | x) = \frac{\exp(-d(f_\phi(x), c_k))}{\sum_{k'} \exp(-d(f_\phi(x), c_{k'}))}$$

Where $d(\cdot, \cdot)$ is a distance function (Euclidean or cosine).

### Loss Function

The training loss is cross-entropy over query predictions:

$$\mathcal{L} = -\log p(y = k | x)$$

Where $k$ is the true class of query $x$.

### Why Mean for Prototypes?

The choice of mean (not median, not mode) is theoretically motivated:

> When using Bregman divergences (including squared Euclidean distance), the mean is the optimal representative that minimizes total distance to all points in the cluster.

This means the prototype is the point that best represents all support examples.

---

## Implementation in This Project

### V1 Architecture

**Location**: `archs/v1/arch.py`

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           V1 ProtoNet Architecture                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Input: Spectrogram segment (17 frames × 128 mel bins)                         │
│          │                                                                      │
│          ▼                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        ResNet Encoder                                   │   │
│   │                                                                         │   │
│   │   ┌───────────────┐                                                     │   │
│   │   │ BasicBlock 1  │  64 channels, 3×3 conv, BN, LeakyReLU, MaxPool(2)  │   │
│   │   └───────┬───────┘                                                     │   │
│   │           ▼                                                             │   │
│   │   ┌───────────────┐                                                     │   │
│   │   │ BasicBlock 2  │  128 channels                                       │   │
│   │   └───────┬───────┘                                                     │   │
│   │           ▼                                                             │   │
│   │   ┌───────────────┐                                                     │   │
│   │   │ BasicBlock 3  │  64 channels + DropBlock                            │   │
│   │   └───────┬───────┘                                                     │   │
│   │           ▼                                                             │   │
│   │   ┌───────────────┐                                                     │   │
│   │   │ AdaptivePool  │  Global Average Pool → (batch, 64, 4, 8)            │   │
│   │   └───────┬───────┘                                                     │   │
│   │           ▼                                                             │   │
│   │   ┌───────────────┐                                                     │   │
│   │   │   Flatten     │  → (batch, 2048) embedding vector                   │   │
│   │   └───────────────┘                                                     │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                      │
│          ▼                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     Prototypical Classification                         │   │
│   │                                                                         │   │
│   │   1. Compute prototypes from support embeddings                         │   │
│   │   2. Compute distances from queries to prototypes                       │   │
│   │   3. Apply softmax over negative distances                              │   │
│   │   4. Compute cross-entropy loss                                         │   │
│   │                                                                         │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Code Components

**Encoder** (`archs/v1/arch.py:ResNetEncoder`):
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x shape: (batch, seq_len, mel_bins)
    x = x.view(-1, 1, seq_len, mel_bins)  # Add channel dim
    x = self.layer1(x)  # 64 channels
    x = self.layer2(x)  # 128 channels
    x = self.layer3(x)  # 64 channels
    x = self.pool_avg(x)  # Adaptive pooling
    return x.view(x.size(0), -1)  # Flatten to (batch, 2048)
```

**Prototype Computation** (`archs/v1/arch.py:ProtoNet._compute_prototypes`):
```python
def _compute_prototypes(self, embeddings, labels):
    class_ids = torch.unique(labels)
    protos = []
    for c in class_ids:
        mask = labels == c
        protos.append(embeddings[mask].mean(dim=0))
    return torch.stack(protos, dim=0), class_ids
```

**Distance Computation** (`archs/v1/arch.py:ProtoNet.euclidean_dist`):
```python
@staticmethod
def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between all rows of x and y."""
    n, m, d = x.size(0), y.size(0), x.size(1)
    x_exp = x.unsqueeze(1).expand(n, m, d)
    y_exp = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x_exp - y_exp, 2).sum(dim=2)
```

---

## Training and Inference

### Training Episode

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING EPISODE                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Episode Configuration: 10-way 5-shot                                          │
│                                                                                 │
│   1. Sample 10 random classes from training set                                 │
│                                                                                 │
│   2. For each class, sample 5 support + 5 query segments                        │
│      ┌───────────────────────────────────────────────────────────────────┐      │
│      │ Class 1:  [S₁, S₂, S₃, S₄, S₅] support  [Q₁, Q₂, Q₃, Q₄, Q₅] query│      │
│      │ Class 2:  [S₁, S₂, S₃, S₄, S₅] support  [Q₁, Q₂, Q₃, Q₄, Q₅] query│      │
│      │ ...                                                               │      │
│      │ Class 10: [S₁, S₂, S₃, S₄, S₅] support  [Q₁, Q₂, Q₃, Q₄, Q₅] query│      │
│      └───────────────────────────────────────────────────────────────────┘      │
│                                                                                 │
│   3. Encode all segments: embed = encoder(segment)                              │
│                                                                                 │
│   4. Compute prototypes (per class):                                            │
│      c₁ = mean(embed[S₁], embed[S₂], ..., embed[S₅])                           │
│      c₂ = mean(embed[S₁], embed[S₂], ..., embed[S₅])                           │
│      ...                                                                        │
│                                                                                 │
│   5. For each query, compute distance to all prototypes:                        │
│      d(Q, c₁), d(Q, c₂), ..., d(Q, c₁₀)                                        │
│                                                                                 │
│   6. Apply softmax: p(class | Q) = softmax(-distances)                          │
│                                                                                 │
│   7. Compute cross-entropy loss against true labels                             │
│                                                                                 │
│   8. Backpropagate and update encoder weights                                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Inference (Testing)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE (Testing)                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Given: New audio file with 5 labeled positive examples                        │
│                                                                                 │
│   1. Build Positive Prototype                                                   │
│      ┌───────────────────────────────────────────────────────────────────┐      │
│      │ 5 labeled positive segments → encoder → mean → prototype_pos     │      │
│      └───────────────────────────────────────────────────────────────────┘      │
│                                                                                 │
│   2. Build Negative Prototype (from unlabeled regions)                          │
│      ┌───────────────────────────────────────────────────────────────────┐      │
│      │ Sample N segments from unlabeled regions → encoder → mean        │      │
│      │ → prototype_neg                                                   │      │
│      └───────────────────────────────────────────────────────────────────┘      │
│                                                                                 │
│   3. Slide window across audio and classify each segment                        │
│      ┌───────────────────────────────────────────────────────────────────┐      │
│      │                                                                   │      │
│      │   Audio: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━►           │      │
│      │          ┌────┐                                                   │      │
│      │          │ W₁ │ → embed → d(W₁, pos), d(W₁, neg) → p(pos)        │      │
│      │          └────┘                                                   │      │
│      │             ┌────┐                                                │      │
│      │             │ W₂ │ → embed → d(W₂, pos), d(W₂, neg) → p(pos)     │      │
│      │             └────┘                                                │      │
│      │                ...                                                │      │
│      │                                                                   │      │
│      └───────────────────────────────────────────────────────────────────┘      │
│                                                                                 │
│   4. Threshold probabilities → onset/offset predictions                         │
│      ┌───────────────────────────────────────────────────────────────────┐      │
│      │ p(pos) > threshold → detected event                              │      │
│      │ Consecutive detections → merge into events with onset/offset     │      │
│      └───────────────────────────────────────────────────────────────────┘      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Distance Metrics

### Euclidean Distance (Default for V1)

$$d(x, y) = \sqrt{\sum_{i=1}^{D} (x_i - y_i)^2}$$

In practice, we use squared Euclidean distance (no sqrt) for efficiency:

$$d^2(x, y) = \sum_{i=1}^{D} (x_i - y_i)^2$$

**Properties**:
- Simple and intuitive
- Works well with normalized embeddings
- Sensitive to scale of dimensions

### Cosine Distance (Optional)

$$d(x, y) = 1 - \frac{x \cdot y}{\|x\| \|y\|}$$

**Properties**:
- Measures angle, not magnitude
- Invariant to embedding scale
- Good when direction matters more than length

### Learnable Distance (V2)

V2 uses an MLP to learn a task-specific distance metric:

$$d(x, y) = \text{MLP}([x; y])$$

Where $[x; y]$ is the concatenation of query and prototype embeddings.

**Properties**:
- Learns which dimensions are important
- Can capture non-linear relationships
- More flexible but requires more data

---

## References

1. **Snell, J., Swersky, K., & Zemel, R.** (2017). Prototypical Networks for Few-shot Learning. *NeurIPS 2017*. [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)

2. **DCASE Challenge**: Detection and Classification of Acoustic Scenes and Events. [https://dcase.community/](https://dcase.community/)

---

## Related Documentation

- [AUDIO_SIGNAL_PROCESSING.md](./AUDIO_SIGNAL_PROCESSING.md) - How audio becomes embeddings
- [FEATURES_AND_DATAFLOW.md](./FEATURES_AND_DATAFLOW.md) - Data loading pipeline
- [V2_IMPLEMENTATION_SUMMARY.md](./V2_IMPLEMENTATION_SUMMARY.md) - Enhanced architecture with attention
