"""
Generic Lightning-powered training entry point using Hydra configuration.

This trainer is architecture-agnostic and uses the DataModule for data handling.
It expects precomputed feature arrays when using the task-specific datamodule.

Usage:
    python archs/train.py
    python archs/train.py arch=v1
    python archs/train.py arch=v1 arch.training.learning_rate=0.0005
"""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import List, Type

os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # For CUDA >= 10.2 determinism

import hydra
import lightning as L
import numpy as np
import torch
from hydra.utils import instantiate
import omegaconf
from omegaconf import DictConfig, OmegaConf

from preprocessing.datamodule import DCASEFewShotDataModule
from utils.mlflow_logger import get_logger, reset_logger
from utils.resolve_device import resolve_device


# Get global MLflow logger
mf_logger = get_logger()

# Deterministic settings for PyTorch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Must be False for determinism


def set_seed(seed: int):
    """Set random seed for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_lightning_module(arch_name: str) -> Type[L.LightningModule]:
    """Dynamically import the LightningModule for the given architecture."""
    if arch_name == "v1":
        from archs.v1.lightning_module import ProtoNetLightningModule

        return ProtoNetLightningModule
    elif arch_name == "v2":
        from archs.v2.lightning_module import ProtoNetV2LightningModule

        return ProtoNetV2LightningModule
    elif arch_name == "v3":
        from archs.v3.lightning_module import ProtoNetV3LightningModule

        return ProtoNetV3LightningModule
    elif arch_name == "v4":
        from archs.v4.lightning_module import ProtoNetV4LightningModule

        return ProtoNetV4LightningModule
    else:
        raise ValueError(f"Unknown architecture: {arch_name}. Supported: v1, v2, v3, v4")


def build_model(cfg: DictConfig) -> L.LightningModule:
    """Build the LightningModule from configuration."""
    arch_name = cfg.arch.name
    module_class = get_lightning_module(arch_name)

    if arch_name == "v1":
        model = module_class(
            emb_dim=cfg.arch.model.embedding_dim,
            distance=cfg.arch.model.distance,
            n_mels=cfg.features.n_mels,
            with_bias=cfg.arch.model.with_bias,
            drop_rate=cfg.arch.model.drop_rate,
            time_max_pool_dim=cfg.arch.model.time_max_pool_dim,
            non_linearity=cfg.arch.model.non_linearity,
            layer_4=cfg.arch.model.layer_4,
            lr=cfg.arch.training.learning_rate,
            weight_decay=cfg.arch.training.weight_decay,
            scheduler_gamma=cfg.arch.training.scheduler_gamma,
            scheduler_step_size=cfg.arch.training.scheduler_step_size,
            scheduler_type=cfg.arch.training.scheduler,
            num_classes=cfg.arch.episodes.n_way,
            n_shot=cfg.train_param.n_shot,
            negative_train_contrast=cfg.train_param.negative_train_contrast,
        )
    elif arch_name == "v2":
        model = module_class(
            emb_dim=cfg.arch.model.embedding_dim,
            distance=cfg.arch.model.distance,
            lr=cfg.arch.training.learning_rate,
            weight_decay=cfg.arch.training.weight_decay,
            scheduler_gamma=cfg.arch.training.scheduler_gamma,
            scheduler_step_size=cfg.arch.training.scheduler_step_size,
            scheduler_type=cfg.arch.training.scheduler,
            n_mels=cfg.features.n_mels,
            in_channels=cfg.arch.model.in_channels,
            channels=list(cfg.arch.model.channels),
            dropout=cfg.arch.model.dropout,
            num_classes=cfg.arch.episodes.n_way,
            n_shot=cfg.train_param.n_shot,
            negative_train_contrast=cfg.train_param.negative_train_contrast,
            # Augmentation parameters
            use_augmentation=cfg.arch.augmentation.use_augmentation,
            use_spec_augment=cfg.arch.augmentation.use_spec_augment,
            use_noise=cfg.arch.augmentation.use_noise,
            time_mask_pct=cfg.arch.augmentation.time_mask_pct,
            freq_mask_pct=cfg.arch.augmentation.freq_mask_pct,
        )
    elif arch_name == "v3":
        model = module_class(
            emb_dim=cfg.arch.model.embedding_dim,
            distance=cfg.arch.model.distance,
            n_mels=cfg.features.n_mels,
            patch_freq=cfg.arch.model.patch_freq,
            patch_time=cfg.arch.model.patch_time,
            max_time_bins=cfg.arch.model.max_time_bins,
            depth=cfg.arch.model.depth,
            num_heads=cfg.arch.model.num_heads,
            mlp_dim=cfg.arch.model.mlp_dim,
            dropout=cfg.arch.model.dropout,
            pooling=cfg.arch.model.pooling,
            lr=cfg.arch.training.learning_rate,
            weight_decay=cfg.arch.training.weight_decay,
            scheduler_gamma=cfg.arch.training.scheduler_gamma,
            scheduler_step_size=cfg.arch.training.scheduler_step_size,
            scheduler_type=cfg.arch.training.scheduler,
            warmup_epochs=cfg.arch.training.warmup_epochs,
            max_epochs=cfg.arch.training.max_epochs,
            num_classes=cfg.arch.episodes.n_way,
            n_shot=cfg.train_param.n_shot,
            negative_train_contrast=cfg.train_param.negative_train_contrast,
        )
    elif arch_name == "v4":
        model = module_class(
            emb_dim=cfg.arch.model.embedding_dim,
            distance=cfg.arch.model.distance,
            n_mels=cfg.features.n_mels,
            model_name=cfg.arch.model.model_name,
            pretrained=cfg.arch.model.pretrained,
            dropout=cfg.arch.model.dropout,
            lr=cfg.arch.training.learning_rate,
            weight_decay=cfg.arch.training.weight_decay,
            scheduler_gamma=cfg.arch.training.scheduler_gamma,
            scheduler_step_size=cfg.arch.training.scheduler_step_size,
            scheduler_type=cfg.arch.training.scheduler,
            num_classes=cfg.arch.episodes.n_way,
            n_shot=cfg.train_param.n_shot,
            negative_train_contrast=cfg.train_param.negative_train_contrast,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    mf_logger.info(f"Created model: {module_class.__name__}")

    # Log model architecture params
    if arch_name == "v1":
        mf_logger.log_params(
            {
                "model/embedding_dim": cfg.arch.model.embedding_dim,
                "model/activation": cfg.arch.model.non_linearity,
                "model/dropout": cfg.arch.model.drop_rate,
                "model/distance": cfg.arch.model.distance,
            }
        )
    elif arch_name == "v2":
        mf_logger.log_params(
            {
                "model/embedding_dim": cfg.arch.model.embedding_dim,
                "model/channels": str(list(cfg.arch.model.channels)),
                "model/dropout": cfg.arch.model.dropout,
                "model/distance": cfg.arch.model.distance,
                "model/encoder": cfg.arch.model.encoder_type,
                "augmentation/spec_augment": cfg.arch.augmentation.use_spec_augment,
                "augmentation/noise": cfg.arch.augmentation.use_noise,
                "augmentation/time_mask": cfg.arch.augmentation.time_mask_pct,
                "augmentation/freq_mask": cfg.arch.augmentation.freq_mask_pct,
            }
        )
    elif arch_name == "v3":
        mf_logger.log_params(
            {
                "model/embedding_dim": cfg.arch.model.embedding_dim,
                "model/distance": cfg.arch.model.distance,
                "model/patch_freq": cfg.arch.model.patch_freq,
                "model/patch_time": cfg.arch.model.patch_time,
                "model/max_time_bins": cfg.arch.model.max_time_bins,
                "model/depth": cfg.arch.model.depth,
                "model/num_heads": cfg.arch.model.num_heads,
                "model/mlp_dim": cfg.arch.model.mlp_dim,
                "model/dropout": cfg.arch.model.dropout,
                "model/pooling": cfg.arch.model.pooling,
            }
        )
    elif arch_name == "v4":
        mf_logger.log_params(
            {
                "model/embedding_dim": cfg.arch.model.embedding_dim,
                "model/distance": cfg.arch.model.distance,
                "model/model_name": cfg.arch.model.model_name,
                "model/pretrained": cfg.arch.model.pretrained,
                "model/dropout": cfg.arch.model.dropout,
            }
        )

    return model


def build_callbacks(cfg: DictConfig) -> List[L.Callback]:
    """Build training callbacks from configuration."""
    callbacks = []

    ckpt_dir = Path(cfg.runtime.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(cfg, "callbacks"):
        callbacks_cfg = cfg.callbacks

        if hasattr(callbacks_cfg, "model_checkpoint"):
            try:
                checkpoint = instantiate(callbacks_cfg.model_checkpoint)
                # Enforce best + last checkpoint saving; monitor is configured via Hydra.
                checkpoint.save_top_k = 1
                checkpoint.save_last = True
                
                callbacks.append(checkpoint)
                mf_logger.info(f"Instantiated ModelCheckpoint from config (monitor={checkpoint.monitor}, save_top_k={checkpoint.save_top_k}, save_last={checkpoint.save_last})")
            except Exception as e:
                mf_logger.warning(f"Failed to instantiate ModelCheckpoint: {e}")
                from lightning.pytorch.callbacks import ModelCheckpoint

                # Fallback: use checkpoint config settings
                monitor = cfg.checkpoint.monitor if hasattr(cfg, "checkpoint") else "val/acc"
                mode = cfg.checkpoint.mode if hasattr(cfg, "checkpoint") else "max"

                callbacks.append(
                    ModelCheckpoint(
                        monitor=monitor,
                        mode=mode,
                        dirpath=str(ckpt_dir),
                        filename="best",
                        save_top_k=1,
                        save_last=True,
                    )
                )

        if hasattr(callbacks_cfg, "early_stopping"):
            try:
                early_stop = instantiate(callbacks_cfg.early_stopping)
                # Monitor is configured via Hydra.
                callbacks.append(early_stop)
                mf_logger.info(f"Instantiated EarlyStopping from config (monitor={early_stop.monitor})")
            except Exception as e:
                mf_logger.warning(f"Failed to instantiate EarlyStopping: {e}")
                from lightning.pytorch.callbacks import EarlyStopping

                # Fallback: use checkpoint config settings
                monitor = cfg.checkpoint.monitor if hasattr(cfg, "checkpoint") else "val/acc"
                mode = cfg.checkpoint.mode if hasattr(cfg, "checkpoint") else "max"

                callbacks.append(
                    EarlyStopping(
                        monitor=monitor,
                        mode=mode,
                        patience=10,
                    )
                )

        if hasattr(callbacks_cfg, "model_summary"):
            try:
                summary = instantiate(callbacks_cfg.model_summary)
                callbacks.append(summary)
            except Exception as e:
                mf_logger.warning(f"Failed to instantiate ModelSummary: {e}")

        if hasattr(callbacks_cfg, "learning_rate_monitor"):
            try:
                lr_monitor = instantiate(callbacks_cfg.learning_rate_monitor)
                callbacks.append(lr_monitor)
            except Exception as e:
                mf_logger.warning(f"Failed to instantiate LRMonitor: {e}")
                from lightning.pytorch.callbacks import LearningRateMonitor

                callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    else:
        from lightning.pytorch.callbacks import (
            ModelCheckpoint,
            EarlyStopping,
            LearningRateMonitor,
            RichProgressBar,
        )

        # Use checkpoint config settings
        monitor = cfg.checkpoint.monitor if hasattr(cfg, "checkpoint") else "val/acc"
        mode = cfg.checkpoint.mode if hasattr(cfg, "checkpoint") else "max"

        callbacks.append(
            ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                dirpath=str(ckpt_dir),
                filename="best",
                save_top_k=1,  # Only save the best checkpoint
                save_last=True,  # Also save the last checkpoint
            )
        )

        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=10,
            )
        )
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        callbacks.append(RichProgressBar())

    return callbacks


def build_pl_loggers(cfg: DictConfig) -> List:
    """Build PyTorch Lightning loggers from configuration."""
    loggers = []

    log_dir = Path(cfg.runtime.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Use MLflow logger for experiment tracking
    try:
        from lightning.pytorch.loggers import MLFlowLogger as PLMLFlowLogger

        tracking_uri = str(log_dir / "mlruns")

        pl_mlflow_logger = PLMLFlowLogger(
            experiment_name=cfg.name,
            run_name=cfg.exp_name,
            tracking_uri=tracking_uri,
            save_dir=str(log_dir),
            log_model=False,  # Disable auto-logging checkpoints to avoid permission issues
        )
        loggers.append(pl_mlflow_logger)
        mf_logger.info(f"Created MLflow logger: tracking_uri={tracking_uri}")

    except Exception as e:
        mf_logger.warning(f"Failed to create MLflow logger: {e}")

    return loggers if loggers else None


def print_config(cfg: DictConfig) -> None:
    """Pretty print the configuration."""
    # Try to resolve, but fall back to unresolved if interpolation keys are missing
    # (e.g., hydra.job.num only exists during multirun/sweep mode)
    try:
        config_str = OmegaConf.to_yaml(cfg, resolve=True)
    except omegaconf.errors.InterpolationKeyError:
        config_str = OmegaConf.to_yaml(cfg, resolve=False)

    try:
        from rich import print as rprint
        from rich.panel import Panel
        from rich.syntax import Syntax

        syntax = Syntax(config_str, "yaml", theme="monokai", line_numbers=True)
        rprint(Panel(syntax, title="Configuration", border_style="blue"))
    except ImportError:
        mf_logger.info("Configuration:")
        print(config_str)


def log_config_params(cfg: DictConfig):
    """Log all configuration parameters to MLflow."""
    # Training params
    mf_logger.log_params(
        {
            "training/learning_rate": cfg.arch.training.learning_rate,
            "training/weight_decay": cfg.arch.training.weight_decay,
            "training/max_epochs": cfg.arch.training.max_epochs,
            "training/scheduler": cfg.arch.training.scheduler,
            "training/scheduler_gamma": cfg.arch.training.scheduler_gamma,
            "training/scheduler_step_size": cfg.arch.training.scheduler_step_size,
        }
    )

    # Episode params
    mf_logger.log_params(
        {
            "episodes/n_way": cfg.arch.episodes.n_way,
            "episodes/k_shot": cfg.arch.episodes.k_shot,
            "episodes/n_query": cfg.arch.episodes.n_query,
            "episodes/per_epoch": cfg.arch.episodes.episodes_per_epoch,
        }
    )

    # Feature params
    mf_logger.log_params(
        {
            "features/sr": cfg.features.sr,
            "features/n_mels": cfg.features.n_mels,
            "features/n_fft": cfg.features.n_fft,
            "features/hop_mel": cfg.features.hop_mel,
            "features/type": cfg.features.feature_types,
        }
    )

    # General params
    mf_logger.log_params(
        {
            "seed": cfg.seed,
            "architecture": cfg.arch.name,
        }
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training entry point.
    This function is the entry point for the training pipeline.
    It initializes the MLflow logger, sets the seed, and builds the data module, model, and callbacks.
    It then trains the model and logs the results to MLflow.
    Finally, it runs the test evaluation if enabled.

    Args:
        cfg: The configuration object.
    """
    global mf_logger

    if cfg.ignore_warnings:
        warnings.filterwarnings("ignore")

    # Initialize MLflow logger with tracking URI
    log_dir = Path(cfg.runtime.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = str(log_dir / "mlruns")

    reset_logger()
    mf_logger = get_logger(use_mlflow=True, tracking_uri=tracking_uri)

    # Start MLflow run
    mf_logger.start_run(
        run_name=cfg.exp_name,
        experiment_name=cfg.name,
    )

    try:
        if cfg.print_config:
            print_config(cfg)

        if cfg.disable_cudnn:
            torch.backends.cudnn.enabled = False

        accelerator = resolve_device(cfg.runtime.device)

        # Set tags
        mf_logger.set_tags(
            {
                "experiment": cfg.name,
                "run": cfg.exp_name,
                "architecture": cfg.arch.name,
                "accelerator": accelerator,
            }
        )

        # Log all config params
        log_config_params(cfg)

        mf_logger.info("Training Configuration")
        mf_logger.info(f"Experiment: {cfg.name} / {cfg.exp_name}")
        mf_logger.info(f"Architecture: {cfg.arch.name}")
        mf_logger.info(f"Accelerator: {accelerator}")
        mf_logger.info(f"Epochs: {cfg.arch.training.max_epochs}")
        mf_logger.info(f"Learning rate: {cfg.arch.training.learning_rate}")
        mf_logger.info(f"Episodes per epoch: {cfg.arch.episodes.episodes_per_epoch}")
        mf_logger.info(
            f"N-way: {cfg.arch.episodes.n_way}, K-shot: {cfg.arch.episodes.k_shot}"
        )
        mf_logger.info(
            f"Features: sr={cfg.features.sr}, n_mels={cfg.features.n_mels}, type={cfg.features.feature_types}"
        )

        if cfg.seed is not None:
            set_seed(cfg.seed)
            L.seed_everything(cfg.seed, workers=True)
            mf_logger.info(f"Random seed set to {cfg.seed} for deterministic training")

        # Allow test-only mode if checkpoint is provided
        test_only_mode = (
            not cfg.train and cfg.test and cfg.arch.training.load_weight_from
        )

        if not cfg.train and not test_only_mode:
            mf_logger.info("Training disabled (cfg.train=False), exiting")
            return

        mf_logger.info("Creating DataModule...")
        datamodule = DCASEFewShotDataModule(cfg=cfg)

        mf_logger.info("Preparing data...")
        datamodule.prepare_data()
        datamodule.setup("fit")

        mf_logger.info("Building model...")
        model = build_model(cfg)

        if cfg.arch.training.load_weight_from:
            weight_path = cfg.arch.training.load_weight_from
            mf_logger.info(f"Loading pretrained weights from: {weight_path}")
            state_dict = torch.load(weight_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            mf_logger.set_tag("pretrained", weight_path)

        callbacks = build_callbacks(cfg)
        pl_loggers = build_pl_loggers(cfg)

        precision = "32"
        mf_logger.log_param("training/precision", precision)

        trainer = L.Trainer(
            max_epochs=cfg.arch.training.max_epochs,
            accelerator=accelerator,
            devices=1 if accelerator == "cuda" else "auto",
            precision=precision,
            callbacks=callbacks,
            logger=pl_loggers,
            default_root_dir=str(log_dir),  # Sets trainer.log_dir for lightning modules
            log_every_n_steps=10,
            enable_progress_bar=True,
            num_sanity_val_steps=0,
            gradient_clip_val=cfg.arch.training.gradient_clip_val,
            deterministic="warn",  # Warn on non-deterministic ops instead of error
        )

        # Skip training if in test-only mode
        if not test_only_mode:
            mf_logger.info("Starting training...")
            trainer.fit(model, datamodule=datamodule)
            mf_logger.info("Training Complete!")
        else:
            mf_logger.info("Skipping training (test-only mode)")

        # Variables to store DCASE metrics for checkpoints
        best_dcase_metrics = None
        last_dcase_metrics = None

        # Find and log best checkpoint
        best_ckpt = None
        ckpt_callback = None
        for callback in callbacks:
            # Check if this is a ModelCheckpoint callback
            from lightning.pytorch.callbacks import ModelCheckpoint

            if isinstance(callback, ModelCheckpoint):
                best_ckpt = callback.best_model_path
                ckpt_callback = callback
                break

        ckpt_dir = Path(cfg.runtime.ckpt_dir)
        
        if best_ckpt:
            mf_logger.info(f"Best model saved to: {best_ckpt}")
            mf_logger.set_tag("best_checkpoint", best_ckpt)

            # Log the best checkpoint as artifact
            if Path(best_ckpt).exists():
                mf_logger.log_artifact(best_ckpt, "checkpoints")
            else:
                mf_logger.warning(
                    f"Best checkpoint path exists but file not found: {best_ckpt}"
                )
            
            # Write best checkpoint info file
            import json

            # Load best checkpoint to get the epoch it was saved at
            best_ckpt_data = torch.load(best_ckpt, map_location="cpu")
            best_epoch = best_ckpt_data.get("epoch", None)

            # Run DCASE evaluation on the best checkpoint to get segment-based metrics
            mf_logger.info("Running DCASE evaluation on best checkpoint...")
            trainer.validate(model, datamodule=datamodule, ckpt_path=str(best_ckpt))
            best_dcase_metrics = getattr(model, 'last_dcase_metrics', None)

            if best_dcase_metrics:
                mf_logger.info(f"Best checkpoint DCASE metrics: {best_dcase_metrics}")
            else:
                mf_logger.warning("Could not retrieve DCASE metrics for best checkpoint")

            best_info = {
                "checkpoint_path": str(best_ckpt),
                "monitor": ckpt_callback.monitor if ckpt_callback else "val/fmeasure",
                "best_epoch": best_epoch,
                "final_epoch": trainer.current_epoch,
                "seed": cfg.seed,
                "dcase_metrics": best_dcase_metrics,
            }
            best_info_path = ckpt_dir / "best_info.json"
            with open(best_info_path, "w") as f:
                json.dump(best_info, f, indent=2)
            mf_logger.info(f"Best checkpoint info saved to: {best_info_path}")
        else:
            mf_logger.warning(
                f"Best checkpoint path not set by ModelCheckpoint callback. Available checkpoints in {cfg.runtime.ckpt_dir}: {list(Path(cfg.runtime.ckpt_dir).glob('*.ckpt')) if Path(cfg.runtime.ckpt_dir).exists() else 'N/A'}"
            )
        
        # Write last checkpoint info file
        last_ckpt = ckpt_dir / "last.ckpt"
        if last_ckpt.exists():
            import json

            # Load last checkpoint to get the epoch it was saved at
            last_ckpt_data = torch.load(last_ckpt, map_location="cpu")
            last_epoch = last_ckpt_data.get("epoch", trainer.current_epoch)

            # Run DCASE evaluation on the last checkpoint to get segment-based metrics
            mf_logger.info("Running DCASE evaluation on last checkpoint...")
            trainer.validate(model, datamodule=datamodule, ckpt_path=str(last_ckpt))
            last_dcase_metrics = getattr(model, 'last_dcase_metrics', None)

            if last_dcase_metrics:
                mf_logger.info(f"Last checkpoint DCASE metrics: {last_dcase_metrics}")
            else:
                mf_logger.warning("Could not retrieve DCASE metrics for last checkpoint")

            last_info = {
                "checkpoint_path": str(last_ckpt),
                "epoch": last_epoch,
                "seed": cfg.seed,
                "dcase_metrics": last_dcase_metrics,
            }
            last_info_path = ckpt_dir / "last_info.json"
            with open(last_info_path, "w") as f:
                json.dump(last_info, f, indent=2)
            mf_logger.info(f"Last checkpoint info saved to: {last_info_path}")

        # Log DCASE metrics to MLflow
        if best_dcase_metrics:
            mf_logger.log_metrics({
                "best/dcase_precision": best_dcase_metrics["precision"],
                "best/dcase_recall": best_dcase_metrics["recall"],
                "best/dcase_fmeasure": best_dcase_metrics["fmeasure"],
            })
        if last_dcase_metrics:
            mf_logger.log_metrics({
                "last/dcase_precision": last_dcase_metrics["precision"],
                "last/dcase_recall": last_dcase_metrics["recall"],
                "last/dcase_fmeasure": last_dcase_metrics["fmeasure"],
            })

        # Run test if enabled
        if cfg.test and cfg.annotations.test_files:
            mf_logger.info("Running test evaluation...")
            datamodule.setup("test")
            if datamodule.data_test is not None:
                # Use loaded checkpoint if in test-only mode, otherwise use best from training
                if test_only_mode:
                    ckpt_path = cfg.arch.training.load_weight_from
                else:
                    # Rely on "best" string - PyTorch Lightning will resolve it using ModelCheckpoint's best_model_path
                    ckpt_path = "best"
                mf_logger.info(f"Testing with checkpoint: {ckpt_path}")
                test_results = trainer.test(
                    model, datamodule=datamodule, ckpt_path=ckpt_path
                )
                if test_results:
                    for result in test_results:
                        mf_logger.log_metrics(
                            {f"test/{k}": v for k, v in result.items()}
                        )

    finally:
        mf_logger.end_run()


if __name__ == "__main__":
    main()
