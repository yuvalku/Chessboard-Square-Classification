import sys
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dataset import SquareDataset
from src.model import ResNet18Classifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def compute_class_weights(gt_csv: Path, num_classes: int = 13) -> torch.Tensor:
    df = pd.read_csv(gt_csv)
    counts = np.zeros(num_classes, dtype=np.float64)
    vc = df["label_id"].value_counts()
    for k, v in vc.items():
        k = int(k)
        if 0 <= k < num_classes:
            counts[k] = float(v)

    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.mean()
    w = np.clip(w, 0.25, 6.0)
    return torch.tensor(w, dtype=torch.float32)


def main():
    seed = 42
    set_seed(seed)

    data_root = ROOT / "dataset_out" / "squares_multi"
    gt_csv = data_root / "gt.csv"
    if not gt_csv.exists():
        raise FileNotFoundError(f"Missing gt.csv at {gt_csv}")

    train_ds = SquareDataset(str(data_root), split="train", seed=seed, val_ratio=0.2)
    val_ds = SquareDataset(str(data_root), split="val", seed=seed, val_ratio=0.2)

    print("train samples:", len(train_ds))
    print("val samples:", len(val_ds))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=128 if device.type == "cuda" else 64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256 if device.type == "cuda" else 64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = ResNet18Classifier(num_classes=13, pretrained=True).to(device)

    class_w = compute_class_weights(gt_csv, num_classes=13).to(device)
    crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)

    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    epochs = 15
    scheduler = CosineAnnealingLR(opt, T_max=epochs)

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    best_path = ckpt_dir / "resnet18_multi_best.pt"
    last_path = ckpt_dir / "resnet18_multi_last.pt"

    best_val_acc = -1.0
    patience = 4
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, y) * bs
            n += bs

        train_loss = total_loss / max(1, n)
        train_acc = total_acc / max(1, n)

        model.eval()
        vloss = 0.0
        vacc = 0.0
        vn = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                loss = crit(logits, y)

                bs = x.size(0)
                vloss += loss.item() * bs
                vacc += accuracy(logits, y) * bs
                vn += bs

        val_loss = vloss / max(1, vn)
        val_acc = vacc / max(1, vn)

        lr = opt.param_groups[0]["lr"]
        print(
            f"epoch {epoch:02d}: "
            f"lr {lr:.2e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        torch.save(model.state_dict(), last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            bad_epochs = 0
            print("  saved new best:", best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("early stop")
                break

        scheduler.step()

    print("best val acc:", best_val_acc)
    print("best ckpt:", best_path)
    print("last ckpt:", last_path)


if __name__ == "__main__":
    main()
