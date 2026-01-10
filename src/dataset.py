from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SquareDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        seed: int = 42,
        val_ratio: float = 0.2,
    ):
        self.root = Path(root_dir)
        self.img_dir = self.root / "images"
        self.df = pd.read_csv(self.root / "gt.csv")

        # deterministic split by frame so squares from same frame stay together
        frames = sorted(self.df["frame"].unique())
        rng = np.random.default_rng(seed)
        rng.shuffle(frames)

        n_val = int(len(frames) * val_ratio)
        val_frames = set(frames[:n_val])
        train_frames = set(frames[n_val:])

        if split == "train":
            self.df = self.df[self.df["frame"].isin(train_frames)].reset_index(drop=True)
        elif split == "val":
            self.df = self.df[self.df["frame"].isin(val_frames)].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train' or 'val'")

        # ImageNet normalization (ResNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["image_name"]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # to float in [0,1]
        img = img.astype(np.float32) / 255.0

        # CHW
        img = np.transpose(img, (2, 0, 1))  # (3,H,W)

        # normalize
        img = (img - self.mean) / self.std

        label_id = int(row["label_id"])

        x = torch.tensor(img, dtype=torch.float32)
        y = torch.tensor(label_id, dtype=torch.long)
        return x, y
