from __future__ import annotations

from pathlib import Path
from typing import List

from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, Logger


def setup_logger(cfg: DictConfig, name: str = "proto") -> List[Logger]:
    """
    Setup PyTorch Lightning loggers using Hydra config settings.

    Args:
        cfg: Hydra DictConfig with runtime.log_dir.
        name: Logger name.

    Returns:
        List[Logger]: List of configured PyTorch Lightning logger instances.
    """
    log_dir = Path(cfg.runtime.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(save_dir=log_dir, name=name, default_hp_metric=False)
    csv_logger = CSVLogger(save_dir=log_dir, name=name)

    return [tb_logger, csv_logger]
