# v1 Architecture

This document describes the v1 ProtoNet model and the current training flow.

## Model (ResNet encoder)

The encoder mirrors the baseline ResNet-style design:

- 4 residual blocks with Conv→BN→ReLU stacks and max pooling
- Optional layer_4 stage
- AdaptiveAvgPool to (time_max_pool_dim, embedding_dim / (time_max_pool_dim * 64))
- Flatten to embedding vector (no FC head, no L2 norm)

```mermaid
flowchart LR
    In["Log-mel spectrogram (B,T,n_mels)"] --> R1["ResBlock (64) + MaxPool"]
    R1 --> R2["ResBlock (128) + MaxPool"]
    R2 --> R3["ResBlock (64) + MaxPool"]
    R3 --> R4["ResBlock (64) + MaxPool (optional)"]
    R4 --> Pool["AdaptiveAvgPool"]
    Pool --> Flat["Flatten (B,D)"]
```

## Training flow (batch-based prototypical loss)

Training uses class-balanced batches. Prototypes are computed inside the loss
function by averaging support samples within each class.

```mermaid
flowchart TD
    Data["Dynamic segment sampler (sequence_data)"] --> Batch["Class-balanced batch (x,y)"]
    Batch --> Enc["ResNet encoder"]
    Enc --> Emb["Embeddings (B,D)"]
    Emb --> Proto["Prototypes (mean per class)"]
    Proto --> Dist["Distances to prototypes"]
    Dist --> Loss["Prototypical loss"]
    Loss --> Opt["AdamW update"]
```

## Notes
- Batches are produced by `DCASEFewShotDataModule` and `IdentityBatchSampler`.
- If `negative_train_contrast=true`, negative samples are concatenated into the batch
  and the loss uses the filtered-negative variant.
