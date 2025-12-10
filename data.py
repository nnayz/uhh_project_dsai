from utils.config import Config
from pathlib import Path

class ListData:
    def __init__(self, config: Config):
        self.config = config
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
        for dir in self.config.DATA_DIR.iterdir():
            if dir.is_dir() and not dir.name.startswith('.') and not self._is_virtual_env(dir):
                print(dir.name)

    def list_training_directories(self) -> None:
        for dir in self.config.DATA_DIR.iterdir():
            if dir.is_dir() and dir.name.startswith('Training') and not self._is_virtual_env(dir):
                for subdir in dir.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.') and not self._is_virtual_env(subdir):
                        print(f"{self.directories_mapping[subdir.name]}: {subdir.name}")

    def list_validation_directories(self) -> None:
        for dir in self.config.DATA_DIR.iterdir():
            if dir.is_dir() and dir.name.startswith('Validation') and not self._is_virtual_env(dir):
                for subdir in dir.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.') and not self._is_virtual_env(subdir):
                        print(f"{self.directories_mapping[subdir.name]}: {subdir.name}")

    def list_evaluation_directories(self) -> None:
        for dir in self.config.DATA_DIR.iterdir():
            if dir.is_dir() and dir.name.startswith('Evaluation') and not self._is_virtual_env(dir):
                for subdir in dir.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.') and not self._is_virtual_env(subdir):
                        print(f"{self.directories_mapping[subdir.name]}: {subdir.name}")

if __name__ == "__main__":
    config = Config()
    list_data = ListData(config)
    print("-"*40)
    print("Metadata information: \n")
    print("-"*40)

    print("\n\n")
    print("Data directories: \n")
    list_data.list_directories()

    print("\n\n")
    print("Training directories: \n")
    list_data.list_training_directories()

    print("\n\n")
    print("Validation directories: \n")
    list_data.list_validation_directories()

    print("\n\n")
    print("Evaluation directories: \n")
    list_data.list_evaluation_directories()