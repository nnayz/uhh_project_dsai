from __future__ import annotations

import logging

from utils.config import Config


def setup_logger(config: Config, name: str = "proto") -> logging.Logger:
    """
    Setup the logger using config settings.

    Args:
        config: Configuration object (required).
        name: Logger name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(config.LOG_DIR / "log.txt")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger