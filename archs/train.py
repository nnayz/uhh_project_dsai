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

import warnings
from pathlib import Path
from typing import List, Type

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
import omegaconf
from omegaconf import DictConfig, OmegaConf

from preprocessing.datamodule import DCASEFewShotDataModule
from utils.mlflow_logger import get_logger, reset_logger

# Get global MLflow logger
mf_logger = get_logger()


def resolve_device(cfg: DictConfig) -> str:
    """Resolve the accelerator from the runtime config."""
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
            mf_logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        return "cuda"
    elif device == "mps":
        if not torch.backends.mps.is_available():
            mf_logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return "mps"
    else:
        return device


def get_lightning_module(arch_name: str) -> Type[L.LightningModule]:
    """Dynamically import the LightningModule for the given architecture."""
    if arch_name == "v1":
        from archs.v1.lightning_module import ProtoNetLightningModule

        return ProtoNetLightningModule
    elif arch_name == "v2":
        from archs.v2.lightning_module import ProtoNetV2LightningModule

        return ProtoNetV2LightningModule
    else:
        raise ValueError(f"Unknown architecture: {arch_name}. Supported: v1, v2")


def build_model(cfg: DictConfig) -> L.LightningModule:
    """Build the LightningModule from configuration."""
    arch_name = cfg.arch.name
    module_class = get_lightning_module(arch_name)

    if arch_name == "v1":
        model = module_class(
            emb_dim=cfg.arch.model.embedding_dim,
            distance=cfg.arch.model.distance,
            n_mels=cfg.features.n_mels,
            conv_channels=list(cfg.arch.model.conv_channels),
            activation=cfg.arch.model.non_linearity,
            with_bias=cfg.arch.model.with_bias,
            drop_rate=cfg.arch.model.drop_rate,
            time_max_pool_dim=cfg.arch.model.time_max_pool_dim,
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
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    mf_logger.info(f"Created model: {module_class.__name__}")

    # Log model architecture params
    if arch_name == "v1":
        mf_logger.log_params(
            {
                "model/embedding_dim": cfg.arch.model.embedding_dim,
                "model/conv_channels": str(list(cfg.arch.model.conv_channels)),
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
                callbacks.append(checkpoint)
                mf_logger.info("Instantiated ModelCheckpoint from config")
            except Exception as e:
                mf_logger.warning(f"Failed to instantiate ModelCheckpoint: {e}")
                from lightning.pytorch.callbacks import ModelCheckpoint

                callbacks.append(
                    ModelCheckpoint(
                        monitor="val_acc",
                        mode="max",
                        dirpath=str(ckpt_dir),
                        filename=f"{cfg.arch.name}_{{epoch:03d}}_{{val_acc:.4f}}",
                        save_top_k=1,
                        save_last=True,
                    )
                )

        if hasattr(callbacks_cfg, "early_stopping"):
            try:
                early_stop = instantiate(callbacks_cfg.early_stopping)
                callbacks.append(early_stop)
                mf_logger.info("Instantiated EarlyStopping from config")
            except Exception as e:
                mf_logger.warning(f"Failed to instantiate EarlyStopping: {e}")
                from lightning.pytorch.callbacks import EarlyStopping

                callbacks.append(
                    EarlyStopping(
                        monitor="val_acc",
                        mode="max",
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

        callbacks.append(
            ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                dirpath=str(ckpt_dir),
                filename=f"{cfg.arch.name}_{{epoch:03d}}_{{val_acc:.4f}}",
                save_top_k=1,
                save_last=True,
            )
        )
        callbacks.append(
            EarlyStopping(
                monitor="val_acc",
                mode="max",
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

    # Always try to use MLflow logger for Lightning
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
    """Main training entry point."""
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

        accelerator = resolve_device(cfg)

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

        L.seed_everything(cfg.seed, workers=True)

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

        precision = "16-mixed" if accelerator == "cuda" else "32"
        mf_logger.log_param("training/precision", precision)

        trainer = L.Trainer(
            max_epochs=cfg.arch.training.max_epochs,
            accelerator=accelerator,
            devices="auto",
            precision=precision,
            callbacks=callbacks,
            logger=pl_loggers,
            log_every_n_steps=10,
            enable_progress_bar=True,
            deterministic="warn",
            gradient_clip_val=cfg.arch.training.gradient_clip_val,
        )

        # Skip training if in test-only mode
        if not test_only_mode:
            mf_logger.info("Starting training...")
            trainer.fit(model, datamodule=datamodule)
            mf_logger.info("Training Complete!")
        else:
            mf_logger.info("Skipping training (test-only mode)")

        # Find and log best checkpoint
        best_ckpt = None
        checkpoint_callback = None
        for callback in callbacks:
            # Check if this is a ModelCheckpoint callback
            from lightning.pytorch.callbacks import ModelCheckpoint

            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                best_ckpt = callback.best_model_path
                break

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
        else:
            mf_logger.warning(
                f"Best checkpoint path not set by ModelCheckpoint callback. Available checkpoints in {cfg.runtime.ckpt_dir}: {list(Path(cfg.runtime.ckpt_dir).glob('*.ckpt')) if Path(cfg.runtime.ckpt_dir).exists() else 'N/A'}"
            )

        # Log final metrics
        if hasattr(trainer, "callback_metrics"):
            final_metrics = {
                k: float(v)
                for k, v in trainer.callback_metrics.items()
                if isinstance(v, (int, float, torch.Tensor))
            }
            if final_metrics:
                mf_logger.log_metrics(final_metrics)

        # Run test if enabled
        if cfg.test and cfg.annotations.test_files:
            mf_logger.info("Running test evaluation...")
            datamodule.setup("test")
            if datamodule.test_dataset is not None:
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
