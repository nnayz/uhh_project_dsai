import torch
from utils.mlflow_logger import get_logger

# Get the global MLflow logger
mf_logger = get_logger()


def resolve_device(device: str) -> str:
    """Resolve the accelerator from the device string."""

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            mf_logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        return "cuda"
    elif device == "mps":
        if not torch.backends.mps.is_available():
            mf_logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return "mps"
    else:
        return device
