"""
CLI for DCASE Few-Shot Bioacoustic Project.

This module provides command-line interface for:
- Feature extraction (Phase 1)
- Training with PyTorch Lightning (Phase 2)
- Cache management
- Data listing
"""

import click
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from utils.mlflow_logger import get_logger

# Get global logger
logger = get_logger(use_mlflow=False)


def load_config(overrides: list = None) -> DictConfig:
    """Load Hydra config from conf directory."""
    config_dir = str(Path(__file__).parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


@click.group()
@click.version_option("1.0.0", "-v", "--version", help="Show version and exit.")
def cli():
    """
    CLI for DCASE Few-Shot Bioacoustic Project

    Workflow:

    1. Export features:
       g5 export-features

    2. Train model:
       g5 train v1 --exp-name my_experiment
    """
    pass


# Feature Extraction Commands (Phase 1)


@cli.command("export-features", help="Export feature files next to audio")
@click.option(
    "--exp-name",
    "-e",
    type=str,
    required=False,
    help="Experiment name override (optional)",
)
@click.option(
    "--split",
    "-s",
    type=click.Choice(["train", "val", "test", "all"]),
    default="all",
    help="Which split to export",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Overwrite existing feature files",
)
def export_features(exp_name, split, force):
    """Export per-audio feature .npy files for training."""
    overrides = [f"+exp_name={exp_name}"] if exp_name else []
    cfg = load_config(overrides)
    from preprocessing.feature_export import export_features

    splits = ["train", "val", "test"] if split == "all" else [split]
    written = export_features(cfg, splits=splits, force=force)
    logger.info(f"Exported {written} feature files for splits: {splits}")


@cli.command("check-features", help="Validate feature files exist")
@click.option(
    "--exp-name",
    "-e",
    type=str,
    required=False,
    help="Experiment name override (optional)",
)
@click.option(
    "--split",
    "-s",
    type=click.Choice(["train", "val", "test", "all"]),
    default="all",
    help="Which split to validate",
)
def check_features(exp_name, split):
    """Check for missing feature files."""
    overrides = [f"+exp_name={exp_name}"] if exp_name else []
    cfg = load_config(overrides)
    from preprocessing.feature_export import validate_features

    splits = ["train", "val", "test"] if split == "all" else [split]
    missing = validate_features(cfg, splits=splits)
    if not missing:
        logger.info(f"All feature files present for splits: {splits}")
        return
    logger.warning(f"Missing {len(missing)} feature files. Example:")
    for path in missing[:10]:
        logger.warning(f"  {path}")


# Data Listing Commands


@cli.command("list-data-dir", help="List all data directories")
@click.option(
    "--type",
    "-t",
    type=click.Choice(
        ["training", "validation", "evaluation", "all"], case_sensitive=False
    ),
    required=True,
    help="Type of data to list",
    default="training",
)
def list_data_directories(type):
    """List all data directories."""
    cfg = load_config()
    from preprocessing.list_data import ListData

    list_data = ListData(cfg)

    if type == "training":
        logger.info("Training directories:")
        list_data.list_training_directories()
    elif type == "validation":
        logger.info("Validation directories:")
        list_data.list_validation_directories()
    elif type == "evaluation":
        logger.info("Evaluation directories:")
        list_data.list_evaluation_directories()
    elif type == "all":
        logger.info("Training directories:")
        list_data.list_training_directories()
        logger.info("Validation directories:")
        list_data.list_validation_directories()
        logger.info("Evaluation directories:")
        list_data.list_evaluation_directories()


@cli.command("list-all-audio-files", help="List all audio files")
def list_all_audio_files():
    """List all audio files."""
    cfg = load_config()
    from preprocessing.list_data import ListData

    list_data = ListData(cfg)
    list_data.list_all_audio_files()


# Training Command (Phase 2) - Lightning only


@cli.command("train", help="Train model with PyTorch Lightning (Phase 2)")
@click.argument("arch", type=click.Choice(["v1", "v2"]), default="v1")
@click.option(
    "--exp-name",
    "-e",
    type=str,
    required=True,
    help="Experiment name for this run (required)",
)
@click.argument("overrides", nargs=-1)
def train(arch, exp_name, overrides):
    """
    Train the model with PyTorch Lightning.

    ARCH: Architecture to use ('v1' or 'v2')

    --exp-name: Experiment name for this run (required)

    OVERRIDES: Optional Hydra config overrides

    Examples:

        g5 train v1 --exp-name my_experiment

        g5 train v1 --exp-name my_experiment arch.training.max_epochs=100

        g5 train v1 --exp-name my_experiment
    """
    import subprocess
    import sys

    cmd = [sys.executable, "archs/train.py", f"arch={arch}"]

    cmd.append(f"+exp_name={exp_name}")

    cmd.extend(overrides)

    logger.info(f"Starting training with PyTorch Lightning")
    logger.info(f"Architecture: {arch}")
    logger.info(f"Command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)


@cli.command("test", help="Test a trained model")
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "--arch", "-a", type=click.Choice(["v1"]), default="v1", help="Architecture type"
)
@click.argument("overrides", nargs=-1)
def test(checkpoint, arch, overrides):
    """
    Test a trained model checkpoint.

    CHECKPOINT: Path to the model checkpoint file (.ckpt)

    The exp_name is automatically extracted from the checkpoint path and used for:
    - Loading the checkpoint
    - Finding the corresponding feature cache: {cache_dir}/{exp_name}/{split}/

    Expected path format: outputs/mlflow_experiments/{exp_name}/checkpoints/...

    Example:
        g5 test outputs/mlflow_experiments/my_experiment/checkpoints/last.ckpt
    """
    import subprocess
    import sys
    from pathlib import Path

    # Extract exp_name from checkpoint path
    # Expected: outputs/mlflow_experiments/{exp_name}/checkpoints/{checkpoint_file}
    # Structure: checkpoint -> checkpoints -> exp_name directory -> mlflow_experiments
    checkpoint_path = Path(checkpoint).resolve()
    try:
        # Navigate up from checkpoint file: checkpoints -> exp_name -> mlflow_experiments
        exp_name = checkpoint_path.parent.parent.name

        # Validate that we're in the correct structure
        if exp_name == "mlflow_experiments" or exp_name == "checkpoints":
            # Try going up one more level if needed
            if checkpoint_path.parent.parent.parent.name == "mlflow_experiments":
                exp_name = checkpoint_path.parent.parent.parent.parent.name
            else:
                exp_name = checkpoint_path.parent.parent.parent.name

        # Final validation
        if exp_name in ["mlflow_experiments", "checkpoints", "outputs"]:
            raise ValueError(
                f"Could not extract valid exp_name from path: {checkpoint_path}"
            )

    except (IndexError, AttributeError, ValueError) as e:
        logger.error(
            f"Could not extract exp_name from checkpoint path: {checkpoint}. "
            f"Expected format: outputs/mlflow_experiments/{{exp_name}}/checkpoints/{{checkpoint_file}}.ckpt"
        )
        raise click.ClickException(f"Invalid checkpoint path format: {e}")

    cmd = [
        sys.executable,
        "archs/train.py",
        f"arch={arch}",
        "train=false",
        "test=true",
        f"+exp_name={exp_name}",
        f"arch.training.load_weight_from={checkpoint}",
    ]
    cmd.extend(overrides)

    logger.info(f"Testing model from checkpoint: {checkpoint}")
    logger.info(
        f"Extracted exp_name: {exp_name} (used for checkpoint and cache lookup)"
    )
    logger.info(f"Command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    cli()
