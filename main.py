import click
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from schemas.model_choice import ModelChoice


def load_config() -> DictConfig:
    """
    Load Hydra config from conf directory.
    """
    config_dir = str(Path(__file__).parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")
    return cfg


@click.group()
@click.version_option("1.0.0", "-v", "--version", help="Show version and exit.")
def cli():
    """
    CLI for the project [Easy Access]
    """
    pass


@cli.command("list-data-dir", help="List all data directories")
@click.option(
    "--type", "-t", 
    type=click.Choice(["training", "validation", "evaluation", "all"], 
    case_sensitive=False), required=True, 
    help="Type of data to list", 
    default="training"
)
def list_data_directories(type):
    """
    List all data directories
    """
    cfg = load_config()
    from preprocessing.list_data import ListData
    list_data = ListData(cfg)
    if type == "training":
        print("Training directories: \n")
        list_data.list_training_directories()
    elif type == "validation":
        print("Validation directories: \n")
        list_data.list_validation_directories()
    elif type == "evaluation":
        print("Evaluation directories: \n")
        list_data.list_evaluation_directories()
    elif type == "all":
        print("All directories: \n")
        print("Training directories: \n")
        list_data.list_training_directories()
        print("\n\n")
        print("Validation directories: \n")
        list_data.list_validation_directories()
        print("\n\n")
        print("Evaluation directories: \n")
        list_data.list_evaluation_directories()
        print("\n\n")
    print("\n\n")


@cli.command("list-all-audio-files", help="List all audio files")
def list_all_audio_files():
    """
    List all audio files
    """
    cfg = load_config()
    from preprocessing.list_data import ListData
    list_data = ListData(cfg)
    list_data.list_all_audio_files()
    print("\n\n")


@cli.command("train", help="Train a particular architecture")
@click.argument("arch-type", type=click.Choice([model.value for model in ModelChoice], case_sensitive=False))
def train(arch_type):
    """
    Train the model with the given architecture.
    
    ARCH-TYPE: Architecture type to use (baseline or v1)

    Note: For more control over training parameters, use:
        python train.py arch=v1 arch.training.learning_rate=0.0001
    """
    print(f"Training the model with {arch_type} architecture...")
    if arch_type == ModelChoice.BASELINE.value:
        from archs.baseline.train import main as baseline_train
        baseline_train()
    elif arch_type == ModelChoice.V1.value:
        import subprocess
        import sys
        subprocess.run([sys.executable, "archs/train.py", f"arch={arch_type}"])

@cli.command("train-lightning", help="Train the model with the given architecture using Lightning")
@click.argument("arch-type", type=click.Choice([model.value for model in ModelChoice], case_sensitive=False))
@click.argument("overrides", nargs=-1)
def train_lightning(arch_type, overrides):
    """
    Train the model with the given architecture using Lightning.
    
    ARCH-TYPE: Architecture type to use (baseline or v1)
    OVERRIDES: Optional Hydra overrides (e.g., arch.training.learning_rate=0.0005)
    """
    import subprocess, sys
    if arch_type != ModelChoice.V1.value:
        raise ValueError(f"Lightning training only supports v1 architecture, got {arch_type}")
    cmd = [sys.executable, "archs/train_lightning.py", f"arch={arch_type}", *overrides]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    cli()
