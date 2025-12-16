## Meta Learning on Bioacoustics

Few-shot classification of animal vocalisations using Prototypical Networks. Each training step is an episodic task (N-way, K-shot) so the model can adapt quickly to new species or call types from only a handful of labelled clips.

### Repo layout (key bits)
- `main.py`: Click CLI; `train` runs either the baseline or the new `v1` ProtoNet trainer, and data listing helpers remain available.
- `archs/v1/`: `arch.py` (ProtoNet + encoder) and `train.py` (full training loop, checkpointing, logging).
- `preprocessing/`: Annotation parsing (`ann_service.py`), flat dataset + episodic sampler (`dataset.py`), and dataloader builders (`dataloaders.py`).
- `utils/config.py`: Central config dataclass (data paths, episodes, model + optimizer settings, device).
- `utils/logger.py`: File + stdout logger (`runs/proto/logs/log.txt` by default).

### Data & annotations
- Defaults point to CSV annotation globs:
  - train: `/data/msc-proj/Training_Set/**/*.csv`
  - val: `/data/msc-proj/Validation_Set_DSAI_2025_2026/**/*.csv`
  - test: `/data/msc-proj/Evaluation_Set_DSAI_2025_2026/**/*.csv`
- Audio files are expected beside the CSVs. Supported CSV formats (see `preprocessing/ann_service.py`):
  - Single-class with `Q` column (rows marked `POS` kept).
  - Explicit class name via `Config.CLASS_NAME` + `Q`.
  - Multi-class `CLASS_*` columns (keep rows where column == `POS`).
  - Fallback: `Audiofilename/Starttime/Endtime` only (all rows treated as positives for that file’s class).
- Update `Config` paths if your data lives elsewhere.

### Running
```bash
# install (example)
pip install -e .

# explore data root
python main.py list-data-dir --type all
python main.py list-all-audio-files

# train (baseline or v1)
python main.py train v1
```
Training logs to `runs/proto/logs/log.txt`; checkpoints save to `runs/proto/checkpoints/protonet_v1_epoch*.pt`.

### Config tips
- Override defaults when constructing `Config`, e.g.:
  ```python
  cfg = Config(
      TRAIN_ANNOTATION_FILES=[Path("/path/to/train/*.csv")],
      VAL_ANNOTATION_FILES=[Path("/path/to/val/*.csv")],
      DISTANCE=Config.Distance.COSINE,
      MAX_EPOCHS=20,
  )
  ```
- Episode shape: `N_WAY`, `K_SHOT`, `N_QUERY`, `EPISODES_PER_EPOCH` govern sampler behaviour; `MAX_FRAMES` pads/crops time dimension of spectrograms.
