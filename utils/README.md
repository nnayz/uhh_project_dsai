## Dataclass helpers for few-shot episodes

Utilities that handle episodic sampling for Prototypical Networks.

### What lives here
- `PrototypicalDataset`: Samples N-way, K-shot episodes on the fly. Each `__getitem__` picks `n_way` classes, takes `k_shot` support and `n_query` query examples per class, and returns tensors `(support_x, support_y, query_x, query_y)`.
- `DataClass`: Loads data from a class-per-folder layout, builds the above dataset, and wraps it in a PyTorch `DataLoader`.

### Expected data layout
Class-centric folders under a split directory (names are arbitrary but must be distinct):
```
<data_root>/<split>/<CLASS_NAME>/*.(wav|mp3|flac|npy|pt|pth|jpg|png)
```
`DataClass` skips hidden dirs and will error if a class has fewer than `k_shot + n_query` examples.

### Minimal usage
```python
from pathlib import Path
from utils.dataclass import DataClass

data = DataClass(config)  # config.DATA_DIR should point to your dataset root
data.load_data(Path("/path/to/Training"))  # or Validation/Evaluation split

ds = data.create_dataset(n_way=5, k_shot=5, n_query=15, mode="train", seed=42)
loader = data.create_dataloader(ds, batch_size=1, shuffle=True)

for support_x, support_y, query_x, query_y in loader:
    # support_x: [n_way*k_shot, ...], support_y: class ids
    # query_x: [n_way*n_query, ...], query_y: class ids
    ...
```

### Customisation tips
- Override `_load_class_examples` in `DataClass` if you need custom file loading (e.g., converting WAV to mel-spectrograms).
- Adjust `n_way`, `k_shot`, and `n_query` per experiment; set `seed` for reproducible episodic sampling.
- Keep `batch_size=1` in the dataloader when training prototypical networks, since each “batch” is already an episode.
