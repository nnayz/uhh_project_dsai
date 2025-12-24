import click
import logging
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from schemas.model_choice import ModelChoice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_config(overrides: list = None) -> DictConfig:
    """
    Load Hydra config from conf directory.
    
    Args:
        overrides: Optional list of config overrides.
    """
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
       python main.py train-lightning v1
    """
    pass


# =============================================================================
# Feature Extraction Commands (Phase 1)
# =============================================================================

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
    
    Features are cached in the directory specified by cfg.features.cache_dir,
    organized by version and config hash for reproducibility.
    
    Example:
        python main.py extract-features
        python main.py extract-features --split train --force
    """
    cfg = load_config()
    from preprocessing.feature_cache import extract_and_cache_features, extract_all_splits
    
    print(f"Feature extraction settings:")
    print(f"  Cache directory: {cfg.features.cache_dir}")
    print(f"  Version: {cfg.features.version}")
    print(f"  Normalize: {cfg.features.normalize} ({cfg.features.normalize_mode})")
    print(f"  Force recompute: {force}")
    print()
    
    if split == "all":
        results = extract_all_splits(cfg, force_recompute=force)
        print("\n=== Feature Extraction Complete ===")
        for split_name, (cache_dir, manifest) in results.items():
            print(f"  {split_name}: {manifest.num_samples} samples, {manifest.num_classes} classes")
            print(f"    → {cache_dir}")
    else:
        # Get annotation paths for the specified split
        if split == "train":
            annotation_paths = cfg.annotations.train_files
        elif split == "val":
            annotation_paths = cfg.annotations.val_files
        else:
            annotation_paths = cfg.annotations.test_files
        
        if not annotation_paths:
            print(f"No annotation files configured for {split}")
            return
        
        cache_dir, manifest = extract_and_cache_features(
            cfg=cfg,
            split=split,
            annotation_paths=annotation_paths,
            force_recompute=force,
        )
        print(f"\n=== Feature Extraction Complete ===")
        print(f"  {split}: {manifest.num_samples} samples, {manifest.num_classes} classes")
        print(f"  → {cache_dir}")


@cli.command("cache-info", help="Show information about cached features")
@click.option(
    "--split", "-s",
    type=click.Choice(["train", "val", "test", "all"]),
    default="all",
    help="Which split to show info for"
)
def cache_info(split):
    """
    Display information about cached features.
    
    Shows cache location, number of samples, classes, and disk usage.
    """
    cfg = load_config()
    from preprocessing.feature_cache import get_cache_dir, get_cache_stats
    
    splits = ["train", "val", "test"] if split == "all" else [split]
    
    print("=== Feature Cache Information ===\n")
    print(f"Cache root: {cfg.features.cache_dir}")
    print(f"Version: {cfg.features.version}")
    print()
    
    for split_name in splits:
        cache_dir = get_cache_dir(cfg, split_name)
        if cache_dir.exists():
            stats = get_cache_stats(cache_dir)
            if "error" in stats:
                print(f"{split_name.upper()}: {stats['error']}")
            else:
                print(f"{split_name.upper()}:")
                print(f"  Directory: {cache_dir}")
                print(f"  Samples: {stats['num_samples']}")
                print(f"  Classes: {stats['num_classes']}")
                print(f"  Size: {stats['total_size_mb']:.2f} MB")
                print(f"  Config hash: {stats['config_hash']}")
                print(f"  Normalization: {stats['normalization']}")
                print(f"  Feature shape: {stats['feature_shape']}")
                print(f"  Class distribution:")
                for class_name, count in stats['class_counts'].items():
                    print(f"    {class_name}: {count}")
        else:
            print(f"{split_name.upper()}: Not cached (run extract-features first)")
        print()


@cli.command("verify-cache", help="Verify integrity of cached features")
@click.option(
    "--split", "-s",
    type=click.Choice(["train", "val", "test", "all"]),
    default="all",
    help="Which split to verify"
)
def verify_cache(split):
    """
    Verify that all cached feature files exist and are valid.
    """
    cfg = load_config()
    from preprocessing.feature_cache import get_cache_dir, verify_cache_integrity
    
    splits = ["train", "val", "test"] if split == "all" else [split]
    
    print("=== Cache Integrity Check ===\n")
    
    all_valid = True
    for split_name in splits:
        cache_dir = get_cache_dir(cfg, split_name)
        if cache_dir.exists():
            is_valid = verify_cache_integrity(cache_dir)
            status = "✓ Valid" if is_valid else "✗ Invalid"
            print(f"{split_name}: {status}")
            if not is_valid:
                all_valid = False
        else:
            print(f"{split_name}: Not cached")
    
    if all_valid:
        print("\nAll caches are valid!")
    else:
        print("\nSome caches are invalid. Run extract-features --force to regenerate.")


# =============================================================================
# Data Listing Commands
# =============================================================================

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


# =============================================================================
# Training Commands (Phase 2)
# =============================================================================

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


@cli.command("train-lightning", help="Train with PyTorch Lightning (Phase 2)")
@click.argument("arch-type", type=click.Choice([model.value for model in ModelChoice], case_sensitive=False))
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable feature caching (extract on-the-fly)"
)
@click.argument("overrides", nargs=-1)
def train_lightning(arch_type, no_cache, overrides):
    """
    Train the model with PyTorch Lightning using cached features.
    
    ARCH-TYPE: Architecture type to use (baseline or v1)
    OVERRIDES: Optional Hydra overrides (e.g., arch.training.learning_rate=0.0005)
    
    This is Phase 2 of the baseline v1 pipeline:
        .npy features → embedding network → few-shot classification
    
    Example:
        python main.py train-lightning v1
        python main.py train-lightning v1 arch.training.max_epochs=20
        python main.py train-lightning v1 --no-cache  # Use on-the-fly extraction
    """
    import subprocess
    import sys
    
    if arch_type != ModelChoice.V1.value:
        raise ValueError(f"Lightning training only supports v1 architecture, got {arch_type}")
    
    # Build command with overrides
    cmd = [sys.executable, "archs/train.py", f"arch={arch_type}"]
    
    if no_cache:
        cmd.append("features.use_cache=false")
    
    cmd.extend(overrides)
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    cli()
