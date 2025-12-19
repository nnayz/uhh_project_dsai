"""
Lightning-powered training entry point using Hydra configuration.

Usage:
    python archs/train_lightning.py
    python archs/train_lightning.py arch-v1 training.learning_rate=0.0005
    
"""

from __future__ import annotations
from pathlib import Path

import hydra 
import lightning as L
import torch
from omegaconf import DictConfig

from utils.logger import setup_logger
from preprocessing.dataloaders import make_fewshot_dataloaders
from archs.v1.lightning_module import ProtoNetLightningModule

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# Write device agnostic code to resolve the device fromm the runtime config
def resolve_device(cfg: DictConfig) -> torch.device:
    """
    Resolve the device from the runtime config
    """
    device = cfg.runtime.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        device = torch.device("cuda")
    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is not available")
        device = torch.device("mps")
    else:
        device = torch.device(device)
    return device

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Train ProtoNet with PyTorch lightning using Hydra config
    """
    device = resolve_device(cfg)
    loggers = setup_logger(cfg, name=f"proto_lit_{cfg.arch.name}")
    print(f"Training architecture: {cfg.arch.name}")
    print(f"Device: {device}")

    # Make the few shot dataloaders
    train_loader, val_loader = make_fewshot_dataloaders(cfg=cfg)
    print("Training and Validation dataloaders created successfully")

    # Instantiate the model
    print("Instantiating the model ...")
    model = ProtoNetLightningModule(
        emb_dim=cfg.arch.model.embedding_dim,
        distance=cfg.arch.model.distance,
        lr=cfg.arch.training.learning_rate,
        weight_decay=cfg.arch.training.weight_decay
    ).to(device)
    print("Model instantiated successfully")

    # Callbacks and checkpointing
    ckpt_dir = Path(cfg.runtime.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor_metric = "val_loss" if val_loader is not None else "train_loss"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode="min",
        dirpath=ckpt_dir,
        filename="protonet_{epoch:02d}-{"+monitor_metric+"}:.4f}",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    lr_monitor = LearningRateMonitor(
        logging_interval="epoch"
    )

    trainer = L.Trainer(
        max_epochs=cfg.arch.training.max_epochs,
        accelerator=device if device != "cpu" else None,
        devices="auto",
        precision="16-mixed",
        callbacks=[checkpoint_callback, lr_monitor],
        logger=loggers,
        log_every_n_steps=10
    )

    # Fit the model
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader if val_loader else None)

    print("Training completed successfully")  
    print(f"Best model saved to {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()