## Utility helpers

Shared utilities for training and evaluation:
- Metric and loss helpers for few-shot experiments (`metrics.py`, `loss.py`, `distance.py`).
- Evaluation and post-processing utilities (`evaluation.py`, `post_proc*.py`).
- MLflow logger wrapper (`mlflow_logger.py`).

Note: legacy dataset/dataclass helpers were removed; dataset logic now lives under `preprocessing/` (see `preprocessing/datamodule.py` and `preprocessing/sequence_data/`).
