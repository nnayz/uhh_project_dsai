## Meta Learning on Bioacoustics

Early-stage repo for few-shot classification of animal vocalisations using Prototypical Networks. The goal is to learn an embedding that can quickly adapt to new species/call types from only a handful of labelled audio clips.

### What we're trying to do
- Build a meta-learning pipeline that treats each training batch as an “episode” (N-way, K-shot) and classifies queries by distance to class prototypes.
- Target bioacoustics corpora (hyenas, birds, meerkats, etc.) where labels are scarce and new classes appear at evaluation time.
- Provide a simple CLI to explore data directories and, eventually, launch training.

### Current pieces
- `archs/baseline_arch/`: Prototypical Network encoders (simple ConvNet and a shallow ResNet) for spectrogram inputs.
- `utils/dataclass.py`: Episodic dataset + dataloader wrapper for few-shot sampling (usage in `utils/README.md`).
- `data.py`: Helpers to list available training/validation/evaluation folders and audio files.
- `utils/config.py`: Minimal configuration (data root, sampling rate, seed).
- `main.py`: Click CLI with commands to list data folders/files and a placeholder `train-model`.

### Docs
- Dataclass helpers and episodic sampling: `utils/README.md`.

### Data layout (expected)
Under `Config.DATA_DIR` (defaults to `/data/msc-proj/`):
```
Training*/<CLASS_NAME>/*.wav
Validation*/<CLASS_NAME>/*.wav
Evaluation*/<CLASS_NAME>/*.wav
```
Class folders are mapped to friendly names in `data.py`; hidden/venv folders are skipped.

### Quick start
```bash
# list available training/validation/evaluation sets
python main.py list-data-dir --type all

# list all audio files (recurses over the data root)
python main.py list-all-audio-files

# training entrypoint (to be implemented)
python main.py train-model --arch-type baseline
```

### Next steps
- Wire up data loading of audio -> log-mel (or other) spectrograms.
- Implement the prototypical loss + episodic trainer in `train-model`.
- Add evaluation scripts for unseen classes and model checkpointing.
