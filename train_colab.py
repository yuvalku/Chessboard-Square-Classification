import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# Detect if running in Colab notebook (only then drive.mount is possible)
try:
    from google.colab import drive  # type: ignore
    from IPython import get_ipython  # type: ignore
    IN_COLAB_NOTEBOOK = get_ipython() is not None
except Exception:
    IN_COLAB_NOTEBOOK = False


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def train_model(resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # --- Persistent save directory ---
    drive_root = "/content/drive/MyDrive"
    default_drive_save_dir = os.path.join(drive_root, "chessboard_models")
    local_save_dir = os.path.join(BASE_DIR, "models")

    if os.path.isdir(drive_root):
        save_dir = default_drive_save_dir
    elif IN_COLAB_NOTEBOOK:
        drive.mount("/content/drive", force_remount=False)
        save_dir = default_drive_save_dir if os.path.isdir(drive_root) else local_save_dir
    else:
        save_dir = local_save_dir

    os.makedirs(save_dir, exist_ok=True)

    # --- Timestamped filenames (never overwrite) ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = os.path.join(save_dir, f"chess_model_{run_id}.pth")
    ckpt_path = os.path.join(save_dir, f"checkpoint_last_{run_id}.pth")
    curves_path = os.path.join(save_dir, f"learning_curves_{run_id}.png")
    log_path = os.path.join(save_dir, f"train_log_{run_id}.txt")

    # Optional: also keep "latest" copies for convenience
    SAVE_LATEST = False
    latest_model_path = os.path.join(save_dir, "chess_model_latest.pth")
    latest_ckpt_path = os.path.join(save_dir, "checkpoint_latest.pth")
    latest_curves_path = os.path.join(save_dir, "learning_curves_latest.png")
    latest_log_path = os.path.join(save_dir, "train_log_latest.txt")

    # Redirect stdout/stderr to a per-run log (and optionally a latest log)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    run_log_file = open(log_path, "a", encoding="utf-8")

    if SAVE_LATEST:
        latest_log_file = open(latest_log_path, "a", encoding="utf-8")
        sys.stdout = Tee(original_stdout, run_log_file, latest_log_file)
        sys.stderr = Tee(original_stderr, run_log_file, latest_log_file)
    else:
        sys.stdout = Tee(original_stdout, run_log_file)
        sys.stderr = Tee(original_stderr, run_log_file)

    start_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 90)
    print(f"Run started: {start_stamp} (run_id={run_id})")
    print(f"Saving to folder: {save_dir}")
    print(f"Model: {model_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Curves: {curves_path}")
    print(f"Log: {log_path}")
    print(f"Device: {device}")
    print("=" * 90)

    try:
        # Transforms
        data_transforms = {
            "train": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            "val": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        }

        # Dataset paths (absolute)
        train_root = os.path.join(BASE_DIR, "final_dataset", "train")
        val_root = os.path.join(BASE_DIR, "final_dataset", "val")

        if not os.path.isdir(train_root) or not os.path.isdir(val_root):
            raise FileNotFoundError(
                "Dataset folders not found.\n"
                f"Expected:\n  {train_root}\n  {val_root}\n"
                "Make sure final_dataset is inside /content/chessboard-project (same folder as train.py)."
            )

        image_datasets = {
            "train": datasets.ImageFolder(train_root, transform=data_transforms["train"]),
            "val": datasets.ImageFolder(val_root, transform=data_transforms["val"]),
        }

        dataloaders = {
            "train": DataLoader(image_datasets["train"], batch_size=32, shuffle=True),
            "val": DataLoader(image_datasets["val"], batch_size=32, shuffle=False),
        }

        # Class weights
        class_counts = np.array([
            sum(
                os.path.isfile(os.path.join(train_root, c, f))
                for f in os.listdir(os.path.join(train_root, c))
            )
            for c in image_datasets["train"].classes
        ])
        class_counts = np.clip(class_counts, 1, None)

        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(weights)
        weights_tensor = torch.FloatTensor(weights).to(device)

        # Model (no deprecation warning)
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_classes = len(image_datasets["train"].classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        num_epochs = 20
        start_epoch = 0

        # Resume: by default resume from "latest" checkpoint if it exists
        if resume and SAVE_LATEST and os.path.exists(latest_ckpt_path):
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            history = ckpt.get("history", history)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"Resuming from epoch {start_epoch} (loaded {latest_ckpt_path})")
        elif resume:
            print("resume=True but no latest checkpoint found, starting fresh")

        for epoch in range(start_epoch, num_epochs):
            for phase in ["train", "val"]:
                model.train() if phase == "train" else model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])

                if phase == "train":
                    history["train_loss"].append(epoch_loss)
                    history["train_acc"].append(epoch_acc.item())
                else:
                    history["val_loss"].append(epoch_loss)
                    history["val_acc"].append(epoch_acc.item())

                print(f"Epoch {epoch+1}/{num_epochs} | {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save checkpoint (timestamped)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "history": history,
                    "classes": image_datasets["train"].classes,
                },
                ckpt_path,
            )

            # Also save/update "latest" checkpoint if desired
            if SAVE_LATEST:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optim_state": optimizer.state_dict(),
                        "history": history,
                        "classes": image_datasets["train"].classes,
                    },
                    latest_ckpt_path,
                )

            print(f"Saved checkpoint: {ckpt_path}")

        # Final model save (timestamped)
        torch.save(model.state_dict(), model_path)
        if SAVE_LATEST:
            torch.save(model.state_dict(), latest_model_path)

        print(f"Training Complete. Model saved as {model_path}")

        # Plot and save (timestamped)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history["train_acc"], label="Train Acc")
        plt.plot(history["val_acc"], label="Val Acc")
        plt.title("Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(curves_path, dpi=200, bbox_inches="tight")
        if SAVE_LATEST:
            plt.savefig(latest_curves_path, dpi=200, bbox_inches="tight")

        plt.show()
        plt.close()

        print(f"Saved learning curves: {curves_path}")
        print(f"Saved run log: {log_path}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        run_log_file.close()
        if "latest_log_file" in locals():
            latest_log_file.close()


if __name__ == "__main__":
    train_model(resume=False)
