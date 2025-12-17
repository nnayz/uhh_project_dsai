from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig


def setup_logger(cfg: DictConfig, name: str = "proto") -> logging.Logger:
    """
    Setup the logger using Hydra config settings.

    Args:
        cfg: Hydra DictConfig with runtime.log_dir.
        name: Logger name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = Path(cfg.runtime.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_dir / "log.txt")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
