from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    SAMPLING_RATE: int = 16000
    DATA_DIR: Path = Path("/data/msc-proj/")
