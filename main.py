import click
from utils.config import Config

config = Config()

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
    from data import ListData
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
    from data import ListData
    list_data = ListData(config)
    list_data.list_all_audio_files()
    print("\n\n")

@cli.command("train-model", help="Train the model")
@click.option(
    "--arch-type", "-a", 
    type=click.Choice(["baseline", "nasrul", "diksha", "gaurika"], 
    case_sensitive=False), required=True, 
    help="Architecture type to use", 
    default="baseline"
)
def train_model(arch_type):
    """
    Train the model with the given architecture
    """
    print(f"Training the model with {arch_type} architecture...")
    pass #TODO: Implement the training logic



if __name__ == "__main__":
    cli()