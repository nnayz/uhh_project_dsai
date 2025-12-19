"""
Common training script using Hydra for configuration management.

Usage:
    # Train with default v1 architecture
    python archs/train.py

    # Train with specific architecture
    python archs/train.py arch=v1

    # Override parameters
    python archs/train.py arch=v1 training.learning_rate=0.0001 training.max_epochs=20

    # Use different distance metric
    python archs/train.py arch=v1 arch.model.distance=cosine
"""
from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from tqdm import tqdm

from preprocessing.dataloaders import make_fewshot_dataloaders
from utils.logger import setup_logger
from utils.distance import Distance


def get_model(cfg: DictConfig):
    """
    Instantiate the model based on the architecture version.
    """
    arch_name = cfg.arch.name

    # Parse distance
    distance_str = cfg.arch.model.distance.lower()
    distance = Distance.COSINE if distance_str == "cosine" else Distance.EUCLIDEAN

    if arch_name == "v1":
        from archs.v1.arch import ProtoNet
        return ProtoNet(
            emb_dim=cfg.arch.model.embedding_dim,
            distance=distance
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


def resolve_device(cfg: DictConfig) -> str:
    """Resolve the device from config."""
    device = cfg.runtime.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return device


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Training entry point with Hydra configuration.
    """
    # Print resolved config
    print(OmegaConf.to_yaml(cfg))

    arch_name = cfg.arch.name
    device = resolve_device(cfg)

    # Setup Lightning loggers (for consistency, though this script doesn't use Trainer)
    loggers = setup_logger(cfg, name=f"proto_{arch_name}")

    print(f"Training architecture: {arch_name}")
    print(f"Device: {device}")

    # Make the few shot dataloaders
    train_loader, val_loader = make_fewshot_dataloaders(cfg=cfg)

    print("Training and Validation dataloaders created successfully")

    # Instantiate the model
    print("Instantiating the model ...")
    model = get_model(cfg).to(device)

    lr = cfg.arch.training.learning_rate
    weight_decay = cfg.arch.training.weight_decay
    max_epochs = cfg.arch.training.max_epochs

    print(f"Setting Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"Number of epochs: {max_epochs}")

    for epoch in range(1, max_epochs + 1):
        model.train()

        running_loss = 0.0
        batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            support_x, support_y, query_x, query_y = batch

            # Because batch_size=1, first dim is "episode"
            support_x = support_x.squeeze(0)
            support_y = support_y.squeeze(0)
            query_x = query_x.squeeze(0)
            query_y = query_y.squeeze(0)

            optimizer.zero_grad()
            loss, _ = model(support_x, support_y, query_x, query_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        avg_loss = running_loss / max(1, batches)
        print(f"Epoch {epoch}: train loss = {avg_loss:.4f}")

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.inference_mode():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                    support_x, support_y, query_x, query_y = batch
                    support_x = support_x.squeeze(0)
                    support_y = support_y.squeeze(0)
                    query_x = query_x.squeeze(0)
                    query_y = query_y.squeeze(0)

                    loss, _ = model(support_x, support_y, query_x, query_y)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(1, val_batches)
            print(f"[Result] Epoch {epoch}: val loss = {avg_val_loss:.4f}")

    # Save the model
    ckpt_dir = Path(cfg.runtime.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"protonet_{arch_name}_epoch{epoch}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
