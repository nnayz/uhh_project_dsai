from dataclasses import dataclass
from pathlib import Path


@dataclass
class SegmentExample:
    """
    Represents a single labeled event segment in a file.
    
    Attributes:
        wav_path: Path to the audio file.
        start_time: Start time of the segment in seconds.
        end_time: End time of the segment in seconds.
        class_id: Integer class identifier.
    """
    wav_path: Path
    start_time: float
    end_time: float
    class_id: int
