from __future__ import annotations

import torch
from torch.optim import AdamW
from tqdm import tqdm

from preprocessing.dataloaders import make_fewshot_dataloaders
from utils.config import Config
from utils.logger import setup_logger
from archs.v1.arch import ProtoNet


def main():
    """
    Training the model.
    """

    config = Config()

    logger = setup_logger(config, name="proto")
    # Make the few shot dataloaders
    train_loader, val_loader = make_fewshot_dataloaders(
        config=config,
    )

    logger.info("Training the model ...")
    logger.info("Training and Validation dataloaders created successfully")

    logger.info("Instantiating the model ...")
    model = ProtoNet(emb_dim=config.EMBEDDING_DIM, distance=config.DISTANCE).to(config.DEVICE)

    logger.info(f"Setting Optimizer: AdamW (lr={config.LEARNING_RATE}, weight_decay={config.WEIGHT_DECAY})")
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    logger.info(f"Number of epochs: {config.MAX_EPOCHS}")

    for epoch in range(1, config.MAX_EPOCHS + 1):

        model.train()

        running_loss = 0.0
        batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            support_x, support_y, query_x, query_y = batch

            # Because batch_size=1 , first dim is "episode"
            support_x = support_x.squeeze(0)
            # (Ns, 1, n_mels, T)
            support_y = support_y.squeeze(0)
            # (Ns,)
            query_x = query_x.squeeze(0)
            # (Nq, 1, n_mels, T)
            query_y = query_y.squeeze(0)
            # (Nq,)

            optimizer.zero_grad()
            loss, _ = model(support_x, support_y, query_x, query_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1
        
        avg_loss = running_loss / max(1, batches)
        logger.info(f"Epoch {epoch}: train loss = {avg_loss:.4f}")

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
            logger.info(f"[Result] Epoch {epoch}: val loss = {avg_val_loss:.4f}")

    # Save the model 
    ckpt_dir = config.CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"protonet_v1_epoch{epoch}.pt"
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
