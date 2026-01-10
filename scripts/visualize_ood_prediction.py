import os
import cv2
import numpy as np
import torch

from dataset_tools.extract_squares import extract_square_crops_warped
from src.predict import predict_board   # adjust import if needed


def visualize_ood_prediction(
    image_path: str,
    out_dir: str = "ood_debug",
    color=(0, 0, 255),   # red in BGR
    alpha: float = 0.4
):
    """
    Run predict_board on a single image and overlay OOD/unknown squares (value=12).

    Saves an image with highlighted squares.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict board
    board = predict_board(image_rgb)   # (8,8)
    board_np = board.cpu().numpy()

    # Extract square crops again to get square geometry
    squares, meta = extract_square_crops_warped(
        image_rgb,
        out_size=64,
        return_meta=True
    )

    # meta should contain bounding boxes per square
    # meta[i]["bbox"] = (x0, y0, x1, y1)
    overlay = image.copy()

    for idx in range(64):
        r = idx // 8
        c = idx % 8

        if board_np[r, c] == 12:   # unknown / OOD
            bbox = meta[idx]["bbox"]
            x0, y0, x1, y1 = map(int, bbox)

            cv2.rectangle(
                overlay,
                (x0, y0),
                (x1, y1),
                color,
                thickness=-1
            )

    # Alpha blend
    vis = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    out_path = os.path.join(out_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, vis)

    print(f"[OOD VIS] saved to {out_path}")
