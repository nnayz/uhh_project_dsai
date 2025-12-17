from pathlib import Path

from omegaconf import DictConfig


class ListData:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.data_dir = Path(cfg.data.data_dir)
        self.directories_mapping = {
            "HT": "Hyenas",
            "WMW": "Western Mediterranean Wetlands Birds",
            "JD": "Jackdaws",
            "BV": "BirdVox",
            "MT": "Meerkats",

            # Validation directories
            "ME": "<Hidden>",
            "PB": "<Hidden>",
            "PB24": "<Hidden>",

            # Evaluation directories
            "HB": "<Hidden>",
            "PW": "<Hidden>",
            "RD": "<Hidden>"
        }

    def _is_virtual_env(self, path: Path) -> bool:
        path_str = str(path)
        venv_patterns = [
            'g5env',
            '4prasad_env',
            '_env',  # e.g., g5env, 4prasad_env
            '/venv/',
            '/.venv/',
            '/env/',
            '/.env/',
            '/bin/python',  # venv structure
            '/Scripts/python',  # Windows venv structure
        ]
        return any(pattern in path_str for pattern in venv_patterns)

    def list_directories(self) -> None:
        for dir in self.data_dir.iterdir():
            if dir.is_dir() and not dir.name.startswith('.') and not self._is_virtual_env(dir):
                print(dir.name)

    def list_training_directories(self) -> None:
        for dir in self.data_dir.iterdir():
            if dir.is_dir() and dir.name.startswith('Training') and not self._is_virtual_env(dir):
                for subdir in dir.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.') and not self._is_virtual_env(subdir):
                        print(f"{self.directories_mapping.get(subdir.name, subdir.name)}: {subdir.name}")

    def list_validation_directories(self) -> None:
        for dir in self.data_dir.iterdir():
            if dir.is_dir() and dir.name.startswith('Validation') and not self._is_virtual_env(dir):
                for subdir in dir.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.') and not self._is_virtual_env(subdir):
                        print(f"{self.directories_mapping.get(subdir.name, subdir.name)}: {subdir.name}")

    def list_evaluation_directories(self) -> None:
        for dir in self.data_dir.iterdir():
            if dir.is_dir() and dir.name.startswith('Evaluation') and not self._is_virtual_env(dir):
                for subdir in dir.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.') and not self._is_virtual_env(subdir):
                        print(f"{self.directories_mapping.get(subdir.name, subdir.name)}: {subdir.name}")

    def list_all_audio_files(self) -> None:
        for dir in self.data_dir.iterdir():
            if dir.is_dir() and not dir.name.startswith('.') and not self._is_virtual_env(dir):
                for subdir in dir.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.') and not self._is_virtual_env(subdir):
                        for file in subdir.iterdir():
                            if file.is_file() and file.name.endswith('.wav'):
                                print(f"{subdir.name}: {file.name}")
