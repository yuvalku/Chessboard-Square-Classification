import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dataset_tools.fen_utils import fen_to_grid_ids
from src.predict import predict_board


def main():
    frames_root = ROOT / "dataset_out" / "frames_eval"
    img_dir = frames_root / "images"
    gt_csv = frames_root / "gt.csv"

    if not gt_csv.exists():
        raise FileNotFoundError(f"Missing: {gt_csv}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing: {img_dir}")

    df = pd.read_csv(gt_csv)
    required = {"image_name", "fen"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"gt.csv missing columns {required}. Found: {list(df.columns)}")

    total_frames = 0
    total_square_correct = 0
    total_square_count = 0
    total_board_exact = 0

    # group by game inferred from image_name prefix
    # image_name looks like: game2_per_frame_frame_000200.jpg
    def game_key(name: str) -> str:
        s = str(name)
        i = s.find("_frame_")
        return s[:i] if i != -1 else "unknown_game"

    df["game"] = df["image_name"].map(game_key)

    for game, gdf in df.groupby("game"):
        g_square_correct = 0
        g_square_total = 0
        g_board_exact = 0
        g_used = 0

        for _, row in gdf.iterrows():
            img_path = img_dir / row["image_name"]
            if not img_path.exists():
                continue

            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            fen = row["fen"]
            gt = fen_to_grid_ids(fen)  # (8,8) ids 0..12

            pred = predict_board(rgb).cpu().numpy().astype(np.int64)  # (8,8) ids 0..13

            correct = int(np.sum(pred == gt))
            total = int(gt.size)

            g_square_correct += correct
            g_square_total += total
            g_board_exact += int(correct == total)
            g_used += 1

        if g_used == 0:
            print(f"{game}: frames 0 | square_acc 0.0000 | board_exact 0.0000")
            continue

        g_square_acc = g_square_correct / g_square_total
        g_board_acc = g_board_exact / g_used

        print(f"{game}: frames {g_used} | square_acc {g_square_acc:.4f} | board_exact {g_board_acc:.4f}")

        total_frames += g_used
        total_square_correct += g_square_correct
        total_square_count += g_square_total
        total_board_exact += g_board_exact

    if total_frames > 0:
        overall_square_acc = total_square_correct / total_square_count
        overall_board_exact = total_board_exact / total_frames
    else:
        overall_square_acc = 0.0
        overall_board_exact = 0.0

    print("\nOVERALL")
    print("frames:", total_frames)
    print(f"square_acc: {overall_square_acc:.4f}")
    print(f"board_exact: {overall_board_exact:.4f}")


if __name__ == "__main__":
    main()
