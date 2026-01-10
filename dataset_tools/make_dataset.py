import io
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dataset_tools.fen_utils import fen_to_grid_ids
from dataset_tools.extract_squares import extract_square_crops_warped as extract_square_crops

ROOT = Path(__file__).resolve().parents[1]

DATA_ZIP = ROOT / "data.zip"  # container zip in repo root

OUT_ROOT = ROOT / "dataset_out"

# square-crop dataset (your current one)
OUT_DIR = OUT_ROOT / "squares_multi"
IMG_DIR = OUT_DIR / "images"

# frame-level dataset (required format: images/ + gt.csv with fen + view)
FRAMES_OUT = OUT_ROOT / "frames_eval"
FRAMES_IMG_DIR = FRAMES_OUT / "images"

IMG_EXTS = (".jpg", ".jpeg", ".png")
JPEG_QUALITY = 95


def extract_frame_number(path_in_zip: str):
    name = Path(path_in_zip).name
    m = re.search(r"frame_(\d+)", name.lower())
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    if not nums:
        return None
    return int(nums[-1])


def read_zip_csv(z: zipfile.ZipFile) -> pd.DataFrame | None:
    csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not csvs:
        return None

    # prefer shallow csv path
    csvs.sort(key=lambda s: (s.count("/"), len(s)))
    with z.open(csvs[0]) as f:
        df = pd.read_csv(f)

    required = {"from_frame", "to_frame", "fen"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"CSV missing required columns {required}. Found: {list(df.columns)}")

    df = df.copy()
    df["from_frame"] = df["from_frame"].astype(int)
    df["to_frame"] = df["to_frame"].astype(int)

    bad = df[df["from_frame"] != df["to_frame"]]
    if len(bad) > 0:
        ex = bad.iloc[0].to_dict()
        raise RuntimeError(f"Spec violation: from_frame != to_frame. Example: {ex}")

    return df


def list_game_zip_members(container: zipfile.ZipFile):
    members = [n for n in container.namelist() if n.lower().endswith(".zip")]
    game_like = [n for n in members if "game" in n.lower() and "per_frame" in n.lower()]
    game_like.sort()
    if game_like:
        return game_like
    members.sort()
    return members


def is_image_path_in_images_folder(path_in_zip: str) -> bool:
    p = path_in_zip.replace("\\", "/").lower()
    if not p.endswith(IMG_EXTS):
        return False
    # only accept paths that include "/images/"
    return "/tagged_images/" in p or p.startswith("tagged_images/")


def infer_view_placeholder() -> str:
    # The official dataset expects something like:
    # "white_closer" or "black_closer".
    # We cannot infer it reliably from FEN alone, so we store a placeholder.
    # If later you decide to detect view, only this function needs to change.
    return "unknown"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    FRAMES_OUT.mkdir(parents=True, exist_ok=True)
    FRAMES_IMG_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_ZIP.exists():
        raise RuntimeError(f"Missing data.zip at: {DATA_ZIP}")

    square_rows = []
    frame_rows = []

    total_games = 0
    total_frames_used = 0

    with zipfile.ZipFile(DATA_ZIP, "r") as container:
        game_zip_members = list_game_zip_members(container)
        if not game_zip_members:
            raise RuntimeError(f"No inner game zips found inside {DATA_ZIP}")

        for member in game_zip_members:
            game_zip_bytes = container.read(member)
            game_zip_name = Path(member).name
            stem = Path(game_zip_name).stem.replace(" ", "_")

            with zipfile.ZipFile(io.BytesIO(game_zip_bytes), "r") as z:
                df = read_zip_csv(z)
                if df is None:
                    print(f"SKIP (no CSV): {game_zip_name}")
                    continue

                frame_to_fen = dict(zip(df["from_frame"], df["fen"]))

                imgs = sorted([n for n in z.namelist() if is_image_path_in_images_folder(n)])

                used_in_zip = 0

                for p in imgs:
                    fr = extract_frame_number(p)
                    if fr is None:
                        continue

                    fen = frame_to_fen.get(fr)
                    if fen is None:
                        continue

                    data = z.read(p)
                    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

                    # -------------------------
                    # 1) Save frame-level sample
                    # -------------------------
                    frame_out_name = f"{stem}_frame_{fr:06d}.jpg"
                    frame_out_path = FRAMES_IMG_DIR / frame_out_name
                    cv2.imwrite(
                        str(frame_out_path),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                    )

                    frame_rows.append(
                        {
                            "image_name": frame_out_name,
                            "fen": fen,
                            "view": infer_view_placeholder(),
                        }
                    )

                    # -------------------------
                    # 2) Save 64 square crops + per-square labels
                    # -------------------------
                    label_grid = fen_to_grid_ids(fen)  # (8,8), ids 0..12

                    squares = extract_square_crops(img, out_size=64)
                    if len(squares) != 64:
                        continue

                    idx = 0
                    for r in range(8):
                        for c in range(8):
                            crop = squares[idx]
                            label_id = int(label_grid[r, c])
                            if not (0 <= label_id <= 12):
                                raise RuntimeError(
                                    f"Label out of range {label_id} in {game_zip_name} frame {fr}"
                                )

                            out_name = f"{stem}_frame_{fr:06d}_r{r}c{c}.jpg"
                            out_path = IMG_DIR / out_name
                            cv2.imwrite(
                                str(out_path),
                                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                            )

                            square_rows.append(
                                {
                                    "image_name": out_name,
                                    "label_id": label_id,
                                    "game_zip": game_zip_name,
                                    "frame": fr,
                                    "row": r,
                                    "col": c,
                                }
                            )
                            idx += 1

                    used_in_zip += 1

                total_games += 1
                total_frames_used += used_in_zip
                print(f"OK {game_zip_name}: used frames {used_in_zip}")

    # write square dataset csv (your current one)
    gt_squares = pd.DataFrame(square_rows)
    gt_squares_path = OUT_DIR / "gt.csv"
    gt_squares.to_csv(gt_squares_path, index=False)

    # write frame-level dataset csv (required format)
    gt_frames = pd.DataFrame(frame_rows).drop_duplicates(subset=["image_name"])
    gt_frames_path = FRAMES_OUT / "gt.csv"
    gt_frames.to_csv(gt_frames_path, index=False)

    print("\nDONE")
    print("games processed:", total_games)
    print("frames used:", total_frames_used)
    print("square crops saved:", len(square_rows))
    print("square gt.csv:", gt_squares_path)
    print("frame-level dataset:", FRAMES_OUT)
    print("frame gt.csv:", gt_frames_path)


if __name__ == "__main__":
    main()
