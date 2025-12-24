"""
CLI for DCASE Few-Shot Bioacoustic Project.

This module provides command-line interface for:
- Feature extraction (Phase 1)
- Training with PyTorch Lightning (Phase 2)
- Cache management
- Data listing
"""

import click
import logging
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
       python main.py extract-features
    
    2. Train model (Phase 2 - uses cached features):
       python main.py train v1
    """
    pass


# Feature Extraction Commands (Phase 1)

@cli.command("extract-features", help="Extract and cache features from audio (Phase 1)")
@click.option(
    "--split", "-s",
    type=click.Choice(["train", "val", "test", "all"]),
    default="all",
    help="Which split to extract features for"
)
@click.option(
    "--force", "-f",
    is_flag=True,
    default=False,
    help="Force re-extraction even if cache exists"
)
def extract_features(split, force):
    """
    Extract features from audio files and cache as .npy files.
    
    This is Phase 1 of the baseline v1 pipeline:
        .wav audio → feature extraction → .npy files
    """
    cfg = load_config()
    from preprocessing.feature_cache import extract_and_cache_features, extract_all_splits
    
    logger.info("Feature extraction settings:")
    logger.info(f"  Cache directory: {cfg.features.cache_dir}")
    logger.info(f"  Version: {cfg.features.version}")
    logger.info(f"  Normalize: {cfg.features.normalize} ({cfg.features.normalize_mode})")
    logger.info(f"  Force recompute: {force}")
    
    if split == "all":
        results = extract_all_splits(cfg, force_recompute=force)
        logger.info("Feature Extraction Complete")
        for split_name, (cache_dir, manifest) in results.items():
            logger.info(f"  {split_name}: {manifest.num_samples} samples, {manifest.num_classes} classes")
            logger.info(f"    → {cache_dir}")
    else:
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
        logger.info(f"  {split}: {manifest.num_samples} samples, {manifest.num_classes} classes")
        logger.info(f"  → {cache_dir}")


@cli.command("cache-info", help="Show information about cached features")
@click.option(
    "--split", "-s",
    type=click.Choice(["train", "val", "test", "all"]),
    default="all",
    help="Which split to show info for"
)
def cache_info(split):
    """Display information about cached features."""
    cfg = load_config()
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
                for class_name, count in stats['class_counts'].items():
                    logger.info(f"    {class_name}: {count}")
        else:
            logger.warning(f"{split_name.upper()}: Not cached (run extract-features first)")


@cli.command("verify-cache", help="Verify integrity of cached features")
@click.option(
    "--split", "-s",
    type=click.Choice(["train", "val", "test", "all"]),
    default="all",
    help="Which split to verify"
)
def verify_cache(split):
    """Verify that all cached feature files exist and are valid."""
    cfg = load_config()
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
        logger.warning("Some caches are invalid. Run extract-features --force to regenerate.")


# Data Listing Commands

@cli.command("list-data-dir", help="List all data directories")
@click.option(
    "--type", "-t", 
    type=click.Choice(["training", "validation", "evaluation", "all"], 
    case_sensitive=False), required=True, 
    help="Type of data to list", 
    default="training"
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
@click.argument("arch", type=click.Choice(["v1"]), default="v1")
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable feature caching (extract on-the-fly)"
)
@click.option(
    "--exp-name", "-e",
    type=str,
    default=None,
    help="Experiment name for this run"
)
@click.argument("overrides", nargs=-1)
def train(arch, no_cache, exp_name, overrides):
    """
    Train the model with PyTorch Lightning.
    
    ARCH: Architecture to use (currently only 'v1' supported)
    
    OVERRIDES: Optional Hydra config overrides
    
    Examples:
    
        python main.py train v1
        
        python main.py train v1 --exp-name my_experiment
        
        python main.py train v1 arch.training.max_epochs=100
        
        python main.py train v1 --no-cache
    """
    import subprocess
    import sys
    
    cmd = [sys.executable, "archs/train.py", f"arch={arch}"]
    
    if no_cache:
        cmd.append("features.use_cache=false")
    
    if exp_name:
        cmd.append(f"exp_name={exp_name}")
    
    cmd.extend(overrides)
    
    logger.info(f"Starting training with PyTorch Lightning")
    logger.info(f"Architecture: {arch}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    subprocess.run(cmd, check=True)


@cli.command("test", help="Test a trained model")
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "--arch", "-a",
    type=click.Choice(["v1"]),
    default="v1",
    help="Architecture type"
)
@click.argument("overrides", nargs=-1)
def test(checkpoint, arch, overrides):
    """
    Test a trained model checkpoint.
    
    CHECKPOINT: Path to the model checkpoint file (.ckpt)
    
    Example:
        python main.py test outputs/protonet_baseline/v1_run/checkpoints/last.ckpt
    """
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "archs/train.py",
        f"arch={arch}",
        "train=false",
        "test=true",
        f"arch.training.load_weight_from={checkpoint}",
    ]
    cmd.extend(overrides)
    
    logger.info(f"Testing model from checkpoint: {checkpoint}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    cli()
