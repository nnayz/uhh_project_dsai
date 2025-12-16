import click
from utils.config import Config
from utils.logger import setup_logger
from schemas.model_choice import ModelChoice

config = Config()
logger = setup_logger(config, name="main")

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
    from preprocessing.list_data import ListData
    list_data = ListData(config)
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
    from preprocessing.list_data import ListData
    list_data = ListData(config)
    list_data.list_all_audio_files()
    print("\n\n")

@cli.command("train", help="Train a particular architecture")
@click.argument("arch-type", type=click.Choice([model.value for model in ModelChoice], case_sensitive=False))
def train(arch_type):
    """
    Train the model with the given architecture.
    
    ARCH-TYPE: Architecture type to use (baseline or v1)
    """
    logger.info(f"Training the model with {arch_type} architecture...")
    if arch_type == ModelChoice.BASELINE:
        from archs.baseline.train import main as baseline_train
        baseline_train()
    elif arch_type == ModelChoice.V1:
        from archs.v1.train import main as v1_train
        v1_train()

if __name__ == "__main__":
    cli()