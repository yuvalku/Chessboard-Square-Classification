from __future__ import annotations
import argparse
import csv
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_zip", type=str, default="raw_data.zip", help="Path to raw_data.zip")
    p.add_argument("--data_dir", type=str, default="Data", help="Where to unpack per-game zips")
    p.add_argument("--out_dir", type=str, default="generated_dataset", help="Output dataset_root with images/ and gt.csv")
    p.add_argument("--default_view", type=str, default="unknown", help="Fallback view if not present in CSV")
    return p.parse_args()


def unzip_to(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def find_images_dir(game_root: Path) -> Path:
    candidates = [
        game_root / "images",
        game_root / "tagged_games" / "images",
        game_root / "tagged_games",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            if (c / "images").exists():
                return c / "images"
            img_files = list(c.glob("*.jpg")) + list(c.glob("*.png")) + list(c.glob("*.jpeg"))
            if img_files:
                return c
    raise FileNotFoundError(f"Could not find images folder under {game_root}")


def find_csv(game_root: Path) -> Path:
    candidates = list(game_root.glob("*.csv")) + list((game_root/"tagged_games").glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found under {game_root}")
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def parse_frame_number_from_filename(name: str) -> Optional[int]:
    m = re.search(r"(\d{3,})", name)
    if not m:
        return None
    return int(m.group(1))


def index_images_by_frame(images_dir: Path) -> dict[int, Path]:
    idx: dict[int, Path] = {}
    for p in images_dir.iterdir():
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        n = parse_frame_number_from_filename(p.name)
        if n is None:
            continue
        idx.setdefault(n, p)
    return idx


def read_csv_rows(csv_path: Path) -> List[dict]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "fen" not in df.columns:
        raise ValueError(f"CSV missing 'fen' column: {csv_path} columns={list(df.columns)}")
    return df.to_dict("records")


def main():
    args = parse_args()
    raw_zip = Path(args.raw_zip)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    if not raw_zip.exists():
        raise FileNotFoundError(f"raw_zip not found: {raw_zip}")
    unzip_to(raw_zip, data_dir)

    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / "gt.csv"

    game_zips = sorted([p for p in data_dir.glob("*.zip") if p.is_file()])
    if not game_zips:
        raise FileNotFoundError(f"No game zip files found in {data_dir}. Expected game*_per_frame.zip")

    rows_out = []
    for gz in tqdm(game_zips, desc="games"):
        game_name = gz.stem
        game_root = data_dir / game_name
        if game_root.exists():
            shutil.rmtree(game_root)
        unzip_to(gz, game_root)

        images_dir = find_images_dir(game_root)
        csv_path = find_csv(game_root)

        img_index = index_images_by_frame(images_dir)
        records = read_csv_rows(csv_path)

        for rec in records:
            frame = None
            for k in ("from_frame", "to_frame", "frame", "frame_id"):
                if k in rec and rec[k] == rec[k]:
                    try:
                        frame = int(rec[k])
                        break
                    except Exception:
                        pass
            if frame is None:
                continue

            fen = str(rec["fen"])
            view = str(rec.get("view", args.default_view))

            if frame not in img_index:
                for delta in (1, -1, 2, -2):
                    if (frame + delta) in img_index:
                        frame = frame + delta
                        break

            img_path = img_index.get(frame)
            if img_path is None:
                continue

            new_name = f"{game_name}_frame_{frame:06d}{img_path.suffix.lower()}"
            shutil.copy2(img_path, img_out / new_name)
            rows_out.append({"image_name": new_name, "fen": fen, "view": view})

    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_name", "fen", "view"])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"Done. Wrote {len(rows_out)} rows to {gt_path} and images to {img_out}")


if __name__ == "__main__":
    main()
