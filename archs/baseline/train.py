# train.py

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import Adam
from rich.progress import track

from data.loaders import make_fewshot_dataloaders
from model.prototype_network import ProtoNet


def main():
    # Project root = /export/home/4prasad/uhh_project_dsai
    project_root = Path(__file__).resolve().parent

    # Your symlinked data lives under: baseline/data_root
    data_root = project_root / "data_root"

    # Training and validation roots (based on your `ls` output)
    train_root = data_root / "Training_Set"
    val_root = data_root / "Validation_Set_DSAI_2025_2026"

    # Collect all annotation CSVs under those folders
    train_ann = [str(p) for p in train_root.rglob("*.csv")]
    val_ann = [str(p) for p in val_root.rglob("*.csv")]

    print(f"[INFO] Using train_root = {train_root}")
    print(f"[INFO] Using val_root   = {val_root}")
    print(f"[INFO] Found {len(train_ann)} training CSVs")
    print(f"[INFO] Found {len(val_ann)} validation CSVs")

    if not train_ann:
        raise RuntimeError(
            "No training CSVs found under Training_Set – check the path."
        )
    if not val_ann:
        raise RuntimeError(
            "No validation CSVs found under Validation_Set_DSAI_2025_2026 – check the path."
        )

    # Build few-shot dataloaders
    train_loader, val_loader = make_fewshot_dataloaders(
        train_root=train_root,  # root_dir for training audio/CSV
        train_annotation_files=train_ann,
        val_root=val_root,  # root_dir for validation audio/CSV
        val_annotation_files=val_ann,
        k_way=2,  # 5-way classification per episode
        n_shot=5,  # 5-shot support examples per class
        n_query=3,  # 10 query examples per class
        batch_size=1,  # 1 episode per batch
        num_workers=0,
    )

    # Model + optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    model = ProtoNet(emb_dim=128, distance="euclidean").to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    num_epochs = 10  # you can increase once it works

    for epoch in range(1, num_epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        running_loss = 0.0
        batches = 0

        for batch in track(train_loader, description=f"Epoch {epoch} [train]"):
            support_x, support_y, query_x, query_y = batch

            # because batch_size=1, first dim is "episode"
            support_x = support_x.squeeze(0)  # (Ns, 1, n_mels, T)
            support_y = support_y.squeeze(0)  # (Ns,)
            query_x = query_x.squeeze(0)  # (Nq, 1, n_mels, T)
            query_y = query_y.squeeze(0)  # (Nq,)

            optimizer.zero_grad()
            loss, logits = model(support_x, support_y, query_x, query_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        avg_loss = running_loss / max(1, batches)
        print(f"[RESULT] Epoch {epoch}: train loss = {avg_loss:.4f}")

        # ---------- VALIDATION ----------
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in track(val_loader, description=f"Epoch {epoch} [val]"):
                    support_x, support_y, query_x, query_y = batch
                    support_x = support_x.squeeze(0)
                    support_y = support_y.squeeze(0)
                    query_x = query_x.squeeze(0)
                    query_y = query_y.squeeze(0)

                    loss, logits = model(support_x, support_y, query_x, query_y)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(1, val_batches)
            print(f"[RESULT] Epoch {epoch}: val loss = {avg_val_loss:.4f}")

    # ---------- SAVE CHECKPOINT ----------
    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "protonet_baseline_py312.pt"
    torch.save(model.state_dict(), ckpt_path)
    print("[INFO] Saved model checkpoint to", ckpt_path)


if __name__ == "__main__":
    main()
