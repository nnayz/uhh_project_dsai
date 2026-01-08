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

    1. Extract features (Phase 1 - offline):
       g5 extract-features --exp-name my_experiment

    2. Train model (Phase 2 - uses cached features):
       g5 train v1 --exp-name my_experiment
    """
    pass


# Feature Extraction Commands (Phase 1)


@cli.command("extract-features", help="Extract and cache features from audio (Phase 1)")
@click.option(
    "--exp-name",
    "-e",
    type=str,
    required=True,
    help="Experiment name for this cache (required)",
)
@click.option(
    "--split",
    "-s",
    type=click.Choice(["train", "val", "test", "all"]),
    default="all",
    help="Which split to extract features for",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Force re-extraction even if cache exists",
)
def extract_features(exp_name, split, force):
    """
    Extract features from audio files and cache as .npy files.

    This is Phase 1 of the baseline v1 pipeline:
        .wav audio → feature extraction → .npy files

    Cache structure: {cache_dir}/{exp_name}/{split}/
    """
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "archs/train.py",
        "train=false",
        "test=false",
        f"+exp_name={exp_name}",
    ]

    if split != "all":
        # For single split, we need to call extract_and_cache_features directly
        # But since it needs cfg, we'll pass it via Hydra
        cfg = load_config([f"+exp_name={exp_name}"])
        from preprocessing.feature_cache import extract_and_cache_features

        logger.info("Feature extraction settings:")
        logger.info(f"  Experiment: {exp_name}")
        logger.info(f"  Cache directory: {cfg.features.cache_dir}")
        logger.info(
            f"  Normalize: {cfg.features.normalize} ({cfg.features.normalize_mode})"
        )
        logger.info(f"  Force recompute: {force}")

        if split == "train":
            annotation_paths = cfg.annotations.train_files
        elif split == "val":
            annotation_paths = cfg.annotations.val_files
        else:
            annotation_paths = cfg.annotations.test_files

        if not annotation_paths:
            logger.warning(f"No annotation files configured for {split}")
            return

        cache_dir, manifest = extract_and_cache_features(
            cfg=cfg,
            split=split,
            annotation_paths=annotation_paths,
            force_recompute=force,
        )
        logger.info("Feature Extraction Complete")
        logger.info(
            f"  {split}: {manifest.num_samples} samples, {manifest.num_classes} classes"
        )
        logger.info(f"  → {cache_dir}")
    else:
        # For all splits, use extract_all_splits
        cfg = load_config([f"+exp_name={exp_name}"])
        from preprocessing.feature_cache import extract_all_splits

        logger.info("Feature extraction settings:")
        logger.info(f"  Experiment: {exp_name}")
        logger.info(f"  Cache directory: {cfg.features.cache_dir}")
        logger.info(
            f"  Normalize: {cfg.features.normalize} ({cfg.features.normalize_mode})"
        )
        logger.info(f"  Force recompute: {force}")

        results = extract_all_splits(cfg, force_recompute=force)
        logger.info("Feature Extraction Complete")
        for split_name, (cache_dir, manifest) in results.items():
            logger.info(
                f"  {split_name}: {manifest.num_samples} samples, {manifest.num_classes} classes"
            )
            logger.info(f"    → {cache_dir}")


@cli.command("cache-info", help="Show information about cached features")
@click.option(
    "--exp-name",
    "-e",
    type=str,
    required=True,
    help="Experiment name for the cache (required)",
)
@click.option(
    "--split",
    "-s",
    type=click.Choice(["train", "val", "test", "all"]),
    default="all",
    help="Which split to show info for",
)
def cache_info(exp_name, split):
    """Display information about cached features."""
    cfg = load_config([f"+exp_name={exp_name}"])
    from preprocessing.feature_cache import get_cache_dir, get_cache_stats

    splits = ["train", "val", "test"] if split == "all" else [split]

    logger.info("Feature Cache Information")
    logger.info(f"Cache root: {cfg.features.cache_dir}")
    logger.info(f"Version: {cfg.features.version}")

    for split_name in splits:
        cache_dir = get_cache_dir(cfg, split_name)
        if cache_dir.exists():
            stats = get_cache_stats(cache_dir)
            if "error" in stats:
                logger.warning(f"{split_name.upper()}: {stats['error']}")
            else:
                logger.info(f"{split_name.upper()}:")
                logger.info(f"  Directory: {cache_dir}")
                logger.info(f"  Samples: {stats['num_samples']}")
                logger.info(f"  Classes: {stats['num_classes']}")
                logger.info(f"  Size: {stats['total_size_mb']:.2f} MB")
                logger.info(f"  Config hash: {stats['config_hash']}")
                logger.info(f"  Normalization: {stats['normalization']}")
                logger.info(f"  Feature shape: {stats['feature_shape']}")
                logger.info(f"  Class distribution:")
                for class_name, count in stats["class_counts"].items():
                    logger.info(f"    {class_name}: {count}")
        else:
            logger.warning(
                f"{split_name.upper()}: Not cached (run extract-features first)"
            )


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


@cli.command("verify-cache", help="Verify integrity of cached features")
@click.option(
    "--exp-name",
    "-e",
    type=str,
    required=True,
    help="Experiment name for the cache (required)",
)
@click.option(
    "--split",
    "-s",
    type=click.Choice(["train", "val", "test", "all"]),
    default="all",
    help="Which split to verify",
)
def verify_cache(exp_name, split):
    """Verify that all cached feature files exist and are valid."""
    cfg = load_config([f"+exp_name={exp_name}"])
    from preprocessing.feature_cache import get_cache_dir, verify_cache_integrity

    splits = ["train", "val", "test"] if split == "all" else [split]

    logger.info("Cache Integrity Check")

    all_valid = True
    for split_name in splits:
        cache_dir = get_cache_dir(cfg, split_name)
        if cache_dir.exists():
            is_valid = verify_cache_integrity(cache_dir)
            status = "Valid" if is_valid else "Invalid"
            if is_valid:
                logger.info(f"{split_name}: {status}")
            else:
                logger.warning(f"{split_name}: {status}")
                all_valid = False
        else:
            logger.warning(f"{split_name}: Not cached")

    if all_valid:
        logger.info("All caches are valid!")
    else:
        logger.warning(
            "Some caches are invalid. Run extract-features --force to regenerate."
        )


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
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable feature caching (extract on-the-fly)",
)
@click.option(
    "--exp-name",
    "-e",
    type=str,
    required=True,
    help="Experiment name for this run (required)",
)
@click.argument("overrides", nargs=-1)
def train(arch, no_cache, exp_name, overrides):
    """
    Train the model with PyTorch Lightning.

    ARCH: Architecture to use ('v1' or 'v2')

    --exp-name: Experiment name for this run (required)

    OVERRIDES: Optional Hydra config overrides

    Examples:

        g5 train v1 --exp-name my_experiment

        g5 train v1 --exp-name my_experiment arch.training.max_epochs=100

        g5 train v1 --exp-name my_experiment --no-cache
    """
    import subprocess
    import sys

    cmd = [sys.executable, "archs/train.py", f"arch={arch}"]

    if no_cache:
        cmd.append("features.use_cache=false")

    cmd.append(f"exp_name={exp_name}")

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
