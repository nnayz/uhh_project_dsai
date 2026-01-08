from pathlib import Path
from typing import Optional

from omegaconf import DictConfig


def _is_virtual_env(path: Path) -> bool:
    """
    This function is used to check if a path is inside a venv directory, [To SKip them]
    Check if a path is inside a virtual environment directory.
    Args:
        path (Path): The path to check.
    Returns:
        bool: True if the path is inside a virtual environment directory, False otherwise.
    """
    path_str = str(path)
    # Common virtual environment directory patterns
    venv_patterns = [
        "g5env",
        "4prasad_env",
        "_env",  # e.g., g5env, 4prasad_env
        "/venv/",
        "/.venv/",
        "/env/",
        "/.env/",
        "/bin/python",  # venv structure
        "/Scripts/python",  # Windows venv structure
    ]
    return any(pattern in path_str for pattern in venv_patterns)


def list_directories(folder_path: Path) -> list[Path]:
    return [d for d in folder_path.iterdir() if d.is_dir() and not _is_virtual_env(d)]


def list_files(folder_path: Path) -> list[Path]:
    return [f for f in folder_path.iterdir() if f.is_file()]


def list_all_files(folder_path: Path) -> list[Path]:
    files = []
    for f in folder_path.rglob("*"):
        try:
            if _is_virtual_env(f):
                continue  # Skip virtual environment files
            if f.is_file():
                files.append(f)
        except (PermissionError, OSError):
            # Skip files/directories we don't have permission to access
            continue
    return files


def list_all_directories(folder_path: Path) -> list[Path]:
    directories = []
    for d in folder_path.rglob("*"):
        try:
            if _is_virtual_env(d):
                continue  # Skip virtual environment directories
            if d.is_dir():
                directories.append(d)
        except (PermissionError, OSError):
            # Skip directories we don't have permission to access
            continue
    return directories
