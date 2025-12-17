# v1 Architecture

Below is a mermaid diagram describing how the v1 ProtoNet training pipeline is wired together.

```mermaid
flowchart TD
    Cfg["Config (utils/config.py)"] --> DL["make_fewshot_dataloaders (preprocessing/dataloaders.py)"]
    DL --> Train["train_loader (few-shot episodes)"]
    DL -->|optional| Val["val_loader"]
    Train --> Proto["ProtoNet (archs/v1/arch.py)"]
    Val --> Proto
    subgraph Episode_Flow
        Sx["support_x/support_y"] --> EncS["ConvEncoder <br/> Conv-BN-ReLU-MaxPool x2 <br/> GlobalAvgPool + FC + L2 norm"]
        Qx["query_x/query_y"] --> EncQ["ConvEncoder"]
        EncS --> ProtoGen["Compute prototypes <br/> (mean per class)"]
        ProtoGen --> Dist["Distance to prototypes <br/> (euclidean or cosine)"]
        EncQ --> Dist
        Dist --> Logits["Logits = -distance"]
        Logits --> Loss["CrossEntropy loss"]
    end
    Loss --> Opt["AdamW update (archs/v1/train.py)"]
    Proto --> CKPT["Checkpoint saved to runs/proto/checkpoints"]
```

## Notes
- Episodes come from `FewShotEpisodeDataset` wrapping `DCASEEventDataset`, sampling `N_WAY` classes with `K_SHOT` support and `N_QUERY` query per class, padded/cropped to `MAX_FRAMES`.
- `ConvEncoder` consumes log-mel spectrograms `(1, n_mels, T)` and outputs L2-normalized embeddings of dimension `EMBEDDING_DIM`.
- Prototypes are class means of support embeddings; distances drive logits, and `cross_entropy` trains the network.
- `train.py` loops over episodes, runs optional validation, and saves checkpoints to `runs/proto/checkpoints`.

## Neural network and data flow (mermaid)

```mermaid
flowchart LR
    Audio["Log-mel spectrogram (1, n_mels, T)"] --> C1["Conv2d 1->64, k=3, p=1"]
    C1 --> BN1["BatchNorm2d"]
    BN1 --> R1["ReLU"]
    R1 --> MP1["MaxPool2d 2x2"]
    MP1 --> C2["Conv2d 64->128, k=3, p=1"]
    C2 --> BN2["BatchNorm2d"]
    BN2 --> R2["ReLU"]
    R2 --> MP2["MaxPool2d 2x2"]
    MP2 --> GAP["AdaptiveAvgPool2d (1x1)"]
    GAP --> Flat["Flatten to (B,128)"]
    Flat --> FC["Linear 128->emb_dim"]
    FC --> Norm["L2 normalize embedding"]

    subgraph Episode_Processing
        Support["support_x (Ns,1,n_mels,T)"] --> EncS["Shared ConvEncoder"]
        Query["query_x (Nq,1,n_mels,T)"] --> EncQ["Shared ConvEncoder"]
        EncS --> EmbS["Support embeddings (Ns,D)"]
        EncQ --> EmbQ["Query embeddings (Nq,D)"]
        EmbS --> Proto["Mean per class -> prototypes (Nc,D)"]
        Proto --> Dist["Distance matrix (Nq,Nc)"]
        EmbQ --> Dist
        Dist --> Logits["Logits = -distance"]
        Logits --> CE["CrossEntropy with mapped query labels"]
        CE --> Update["AdamW weight update"]
    end
```

- ConvEncoder shown on top: two Conv-BN-ReLU-MaxPool blocks, global pooling, linear projection, and L2 normalization to produce embeddings.
- Both support and query batches share the same encoder weights; support embeddings are averaged per class to build prototypes.
- Distances (euclidean or cosine) between query embeddings and prototypes become logits; cross-entropy loss drives AdamW updates.
