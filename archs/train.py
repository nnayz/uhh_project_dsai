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
    python archs/train.py features.use_cache=false
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Type

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
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
    else:
        raise ValueError(f"Unknown architecture: {arch_name}. Supported: v1")


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
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    
    mf_logger.info(f"Created model: {module_class.__name__}")
    
    # Log model architecture params
    mf_logger.log_params({
        "model/embedding_dim": cfg.arch.model.embedding_dim,
        "model/conv_channels": str(list(cfg.arch.model.conv_channels)),
        "model/activation": cfg.arch.model.non_linearity,
        "model/dropout": cfg.arch.model.drop_rate,
        "model/distance": cfg.arch.model.distance,
    })
    
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
                callbacks.append(ModelCheckpoint(
                    monitor="val_acc",
                    mode="max",
                    dirpath=str(ckpt_dir),
                    filename=f"{cfg.arch.name}_{{epoch:03d}}_{{val_acc:.4f}}",
                    save_top_k=1,
                    save_last=True,
                ))
        
        if hasattr(callbacks_cfg, "early_stopping"):
            try:
                early_stop = instantiate(callbacks_cfg.early_stopping)
                callbacks.append(early_stop)
                mf_logger.info("Instantiated EarlyStopping from config")
            except Exception as e:
                mf_logger.warning(f"Failed to instantiate EarlyStopping: {e}")
                from lightning.pytorch.callbacks import EarlyStopping
                callbacks.append(EarlyStopping(
                    monitor="val_acc",
                    mode="max",
                    patience=10,
                ))
        
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
        
        callbacks.append(ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            dirpath=str(ckpt_dir),
            filename=f"{cfg.arch.name}_{{epoch:03d}}_{{val_acc:.4f}}",
            save_top_k=1,
            save_last=True,
        ))
        callbacks.append(EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=10,
        ))
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
            log_model=True,
        )
        loggers.append(pl_mlflow_logger)
        mf_logger.info(f"Created MLflow logger: tracking_uri={tracking_uri}")
            
    except Exception as e:
        mf_logger.warning(f"Failed to create MLflow logger: {e}")
        
        # Fallback to TensorBoard
        try:
            from lightning.pytorch.loggers import TensorBoardLogger
            tb_logger = TensorBoardLogger(
                save_dir=str(log_dir),
                name=cfg.arch.name,
            )
            loggers.append(tb_logger)
            mf_logger.info("Created TensorBoard logger as fallback")
        except Exception as e2:
            mf_logger.warning(f"Failed to create TensorBoard logger: {e2}")
    
    return loggers if loggers else None


def print_config(cfg: DictConfig) -> None:
    """Pretty print the configuration."""
    try:
        from rich import print as rprint
        from rich.panel import Panel
        from rich.syntax import Syntax
        
        config_str = OmegaConf.to_yaml(cfg, resolve=True)
        syntax = Syntax(config_str, "yaml", theme="monokai", line_numbers=True)
        rprint(Panel(syntax, title="Configuration", border_style="blue"))
    except ImportError:
        mf_logger.info("Configuration:")
        print(OmegaConf.to_yaml(cfg, resolve=True))


def log_config_params(cfg: DictConfig):
    """Log all configuration parameters to MLflow."""
    # Training params
    mf_logger.log_params({
        "training/learning_rate": cfg.arch.training.learning_rate,
        "training/weight_decay": cfg.arch.training.weight_decay,
        "training/max_epochs": cfg.arch.training.max_epochs,
        "training/scheduler": cfg.arch.training.scheduler,
        "training/scheduler_gamma": cfg.arch.training.scheduler_gamma,
        "training/scheduler_step_size": cfg.arch.training.scheduler_step_size,
    })
    
    # Episode params
    mf_logger.log_params({
        "episodes/n_way": cfg.arch.episodes.n_way,
        "episodes/k_shot": cfg.arch.episodes.k_shot,
        "episodes/n_query": cfg.arch.episodes.n_query,
        "episodes/per_epoch": cfg.arch.episodes.episodes_per_epoch,
    })
    
    # Feature params
    mf_logger.log_params({
        "features/sr": cfg.features.sr,
        "features/n_mels": cfg.features.n_mels,
        "features/n_fft": cfg.features.n_fft,
        "features/hop_mel": cfg.features.hop_mel,
        "features/type": cfg.features.feature_types,
    })
    
    # General params
    mf_logger.log_params({
        "seed": cfg.seed,
        "architecture": cfg.arch.name,
    })

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
        mf_logger.set_tags({
            "experiment": cfg.name,
            "run": cfg.exp_name,
            "architecture": cfg.arch.name,
            "accelerator": accelerator,
        })
        
        # Log all config params
        log_config_params(cfg)
        
        mf_logger.info("Training Configuration")
        mf_logger.info(f"Experiment: {cfg.name} / {cfg.exp_name}")
        mf_logger.info(f"Architecture: {cfg.arch.name}")
        mf_logger.info(f"Accelerator: {accelerator}")
        mf_logger.info(f"Feature caching: {cfg.features.use_cache}")
        mf_logger.info(f"Epochs: {cfg.arch.training.max_epochs}")
        mf_logger.info(f"Learning rate: {cfg.arch.training.learning_rate}")
        mf_logger.info(f"Episodes per epoch: {cfg.arch.episodes.episodes_per_epoch}")
        mf_logger.info(f"N-way: {cfg.arch.episodes.n_way}, K-shot: {cfg.arch.episodes.k_shot}")
        mf_logger.info(f"Features: sr={cfg.features.sr}, n_mels={cfg.features.n_mels}, type={cfg.features.feature_types}")
        
        L.seed_everything(cfg.seed, workers=True)
        
        if not cfg.train:
            mf_logger.info("Training disabled (cfg.train=False), exiting")
            return
        
        mf_logger.info("Creating DataModule...")
        datamodule = DCASEFewShotDataModule(
            cfg=cfg,
            use_cache=cfg.features.use_cache,
            force_recompute=cfg.features.force_recompute,
        )
        
        mf_logger.info("Preparing data (extracting features if needed)...")
        datamodule.prepare_data()
        datamodule.setup("fit")
        
        cache_info = datamodule.get_cache_info()
        mf_logger.info(f"Cache info: {cache_info}")
        
        # Log cache info
        if "splits" in cache_info:
            for split, info in cache_info["splits"].items():
                mf_logger.log_params({
                    f"data/{split}_samples": info.get("num_samples", 0),
                    f"data/{split}_classes": info.get("num_classes", 0),
                })
        
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
            deterministic=True,
            gradient_clip_val=cfg.arch.training.gradient_clip_val,
        )
        
        mf_logger.info("Starting training...")
        trainer.fit(model, datamodule=datamodule)
        
        mf_logger.info("Training Complete!")
        
        # Find and log best checkpoint
        best_ckpt = None
        for callback in callbacks:
            if hasattr(callback, "best_model_path"):
                best_ckpt = callback.best_model_path
                break
        
        if best_ckpt:
            mf_logger.info(f"Best model saved to: {best_ckpt}")
            mf_logger.set_tag("best_checkpoint", best_ckpt)
            
            # Log the best checkpoint as artifact
            if Path(best_ckpt).exists():
                mf_logger.log_artifact(best_ckpt, "checkpoints")
        
        # Log final metrics
        if hasattr(trainer, "callback_metrics"):
            final_metrics = {
                k: float(v) for k, v in trainer.callback_metrics.items()
                if isinstance(v, (int, float, torch.Tensor))
            }
            if final_metrics:
                mf_logger.log_metrics(final_metrics)
        
        # Run test if enabled
        if cfg.test and cfg.annotations.test_files:
            mf_logger.info("Running test evaluation...")
            datamodule.setup("test")
            if datamodule.test_dataset is not None:
                test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")
                if test_results:
                    for result in test_results:
                        mf_logger.log_metrics({f"test/{k}": v for k, v in result.items()})
    
    finally:
        mf_logger.end_run()


if __name__ == "__main__":
    main()
