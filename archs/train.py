"""
Generic Lightning-powered training entry point using Hydra configuration.

This trainer is architecture-agnostic and uses the DataModule for data handling.
It supports the two-phase baseline v1 workflow:

    Phase 1 (offline): python main.py extract-features
    Phase 2 (online):  python archs/train.py arch=v1

The trainer automatically uses cached features if available (cfg.features.use_cache=true).

Usage:
    python archs/train.py
    python archs/train.py arch=v1
    python archs/train.py arch=v1 arch.training.learning_rate=0.0005
    python archs/train.py features.use_cache=false  # On-the-fly extraction
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Type

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from utils.logger import setup_logger
from preprocessing.datamodule import DCASEFewShotDataModule

from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
)

logger = logging.getLogger(__name__)


def resolve_device(cfg: DictConfig) -> str:
    """
    Resolve the accelerator from the runtime config.
    
    Args:
        cfg: Hydra DictConfig with runtime.device setting.
        
    Returns:
        Accelerator string for Lightning Trainer ('cuda', 'mps', 'cpu', 'auto').
    """
    device = cfg.runtime.device
    
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        return "cuda"
    elif device == "mps":
        if not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return "mps"
    else:
        return device


def get_lightning_module(arch_name: str) -> Type[L.LightningModule]:
    """
    Dynamically import the LightningModule for the given architecture.
    
    Args:
        arch_name: Architecture name (e.g., 'v1', 'v2').
        
    Returns:
        LightningModule class.
    """
    if arch_name == "v1":
        from archs.v1.lightning_module import ProtoNetLightningModule
        return ProtoNetLightningModule
    else:
        raise ValueError(
            f"Unknown architecture: {arch_name}. "
            f"Supported: v1"
        )


def build_model(cfg: DictConfig) -> L.LightningModule:
    """
    Build the LightningModule from configuration.
    
    This is the factory function that creates the model based on arch config.
    Each architecture version can have its own model parameters.
    
    Args:
        cfg: Hydra DictConfig with arch settings.
        
    Returns:
        Configured LightningModule instance.
    """
    arch_name = cfg.arch.name
    module_class = get_lightning_module(arch_name)
    
    # Build model with architecture-specific parameters
    if arch_name == "v1":
        model = module_class(
            emb_dim=cfg.arch.model.embedding_dim,
            distance=cfg.arch.model.distance,
            lr=cfg.arch.training.learning_rate,
            weight_decay=cfg.arch.training.weight_decay,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    
    logger.info(f"Created model: {module_class.__name__}")
    return model


def build_callbacks(cfg: DictConfig) -> list:
    """
    Build training callbacks.
    
    Args:
        cfg: Hydra DictConfig.
        
    Returns:
        List of Lightning callbacks.
    """
    callbacks = []
    
    # Checkpoint callback
    ckpt_dir = Path(cfg.runtime.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=ckpt_dir,
        filename=f"{cfg.arch.name}_{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    # Early stopping (optional)
    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        verbose=True,
    )
    callbacks.append(early_stop)
    
    # Progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    return callbacks


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training entry point.
    
    This function orchestrates the entire training pipeline:
    1. Setup logging and configuration
    2. Create DataModule (handles feature caching automatically)
    3. Build model from config
    4. Configure trainer with callbacks
    5. Run training
    """
    # Suppress warnings if configured
    if cfg.ignore_warnings:
        warnings.filterwarnings("ignore")
    
    # Setup
    accelerator = resolve_device(cfg)
    loggers = setup_logger(cfg, name=f"proto_{cfg.arch.name}")
    
    logger.info("=" * 60)
    logger.info(f"Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Architecture: {cfg.arch.name}")
    logger.info(f"Accelerator: {accelerator}")
    logger.info(f"Feature caching: {cfg.features.use_cache}")
    logger.info(f"Epochs: {cfg.arch.training.max_epochs}")
    logger.info(f"Learning rate: {cfg.arch.training.learning_rate}")
    logger.info(f"Episodes per epoch: {cfg.arch.episodes.episodes_per_epoch}")
    logger.info(f"N-way: {cfg.arch.episodes.n_way}, K-shot: {cfg.arch.episodes.k_shot}")
    logger.info("=" * 60)
    
    # Set seed for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    
    # Create DataModule
    logger.info("Creating DataModule...")
    datamodule = DCASEFewShotDataModule(
        cfg=cfg,
        use_cache=cfg.features.use_cache,
        force_recompute=cfg.features.force_recompute,
    )
    
    # Prepare data (extracts features if needed)
    logger.info("Preparing data (extracting features if needed)...")
    datamodule.prepare_data()
    
    # Setup datasets
    datamodule.setup("fit")
    
    # Log cache info
    cache_info = datamodule.get_cache_info()
    logger.info(f"Cache info: {cache_info}")
    
    # Build model
    logger.info("Building model...")
    model = build_model(cfg)
    
    # Build callbacks
    callbacks = build_callbacks(cfg)
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=cfg.arch.training.max_epochs,
        accelerator=accelerator,
        devices="auto",
        precision="16-mixed" if accelerator == "cuda" else "32",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=True,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)
    
    # Log results
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    
    # Find best checkpoint
    best_ckpt = None
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_ckpt = callback.best_model_path
            break
    
    if best_ckpt:
        logger.info(f"Best model saved to: {best_ckpt}")
    
    # Run test if test data is available
    if cfg.annotations.test_files:
        logger.info("Running test evaluation...")
        datamodule.setup("test")
        if datamodule.test_dataset is not None:
            trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
