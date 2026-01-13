import os
import zipfile
import random
import pandas as pd
import numpy as np
from PIL import Image

# --- Helper functions ---

def discover_games(base_raw_dir="raw_games"):
    """Discover available game names under `base_raw_dir`.

    Supports:
    - Zips named like `<game>_per_frame.zip`
    - Already extracted folders containing `<game>.csv`
    """
    games = set()

    if not os.path.isdir(base_raw_dir):
        return []

    for name in os.listdir(base_raw_dir):
        full_path = os.path.join(base_raw_dir, name)

        if os.path.isfile(full_path) and name.endswith("_per_frame.zip"):
            games.add(name[: -len("_per_frame.zip")])
            continue

        if os.path.isdir(full_path):
            game_name = name
            csv_path = os.path.join(full_path, f"{game_name}.csv")
            if os.path.isfile(csv_path):
                games.add(game_name)

    return sorted(games)


def unzip_folder(game_name, base_raw_dir="raw_games", remove_zip=True):
    """Extracts a zipped game folder into `base_raw_dir/game_name`.

    Expects a zip named like `{game_name}_per_frame.zip` inside `base_raw_dir`.
    Extraction is skipped if the destination already appears extracted.
    """
    zip_path = os.path.join(base_raw_dir, f"{game_name}_per_frame.zip")
    dest_dir = os.path.join(base_raw_dir, game_name)

    expected_csv = os.path.join(dest_dir, f"{game_name}.csv")
    expected_images_dir = os.path.join(dest_dir, "tagged_images")

    if os.path.isdir(dest_dir) and os.path.isfile(expected_csv) and os.path.isdir(expected_images_dir):
        if remove_zip and os.path.isfile(zip_path):
            try:
                os.remove(zip_path)
            except OSError as e:
                print(f"Could not remove zip {zip_path}: {e}")
        return

    if not os.path.isfile(zip_path):
        print(f"Zip not found for {game_name}: {zip_path}")
        return

    os.makedirs(dest_dir, exist_ok=True)

    dest_real = os.path.realpath(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target_path = os.path.realpath(os.path.join(dest_real, member.filename))
            if not (target_path == dest_real or target_path.startswith(dest_real + os.sep)):
                raise ValueError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(dest_dir)

    if remove_zip:
        try:
            os.remove(zip_path)
        except OSError as e:
            print(f"Could not remove zip {zip_path}: {e}")


def parse_fen_to_matrix(fen):
    """Convert a FEN string into an 8x8 matrix of class names, including handling for 'm'."""
    board_part = fen.split(" ")[0]
    board_part = board_part.replace("m", "r")  # Fix 'm' -> 'r' (rook)

    rows = board_part.split("/")
    mapping = {
        "P": "white_pawn",
        "N": "white_knight",
        "B": "white_bishop",
        "R": "white_rook",
        "Q": "white_queen",
        "K": "white_king",
        "p": "black_pawn",
        "n": "black_knight",
        "b": "black_bishop",
        "r": "black_rook",
        "q": "black_queen",
        "k": "black_king",
    }

    matrix = []
    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(["empty"] * int(char))
            else:
                board_row.append(mapping.get(char, "empty"))
        matrix.append(board_row)
    return np.array(matrix)


def slice_image_with_overlap(
    game_name,
    image_path,
    output_root_for_split,
    board_matrix,
    overlap_percent=0.7,
    final_size=(224, 224),
):
    """
    Advanced slicing: crops with overlap and saves into split/class folders.
    output_root_for_split should be something like: final_dataset/train
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return
    except Exception as e:
        print(f"Could not open image {image_path}: {e}")
        return

    img_width, img_height = img.size
    stride_w = img_width / 8.0
    stride_h = img_height / 8.0

    crop_w = stride_w * (1.0 + overlap_percent)
    crop_h = stride_h * (1.0 + overlap_percent)

    name_only = os.path.splitext(os.path.basename(image_path))[0]

    for r in range(8):
        for c in range(8):
            center_x = (c * stride_w) + (stride_w / 2.0)
            center_y = (r * stride_h) + (stride_h / 2.0)

            left = center_x - (crop_w / 2.0)
            upper = center_y - (crop_h / 2.0)
            right = center_x + (crop_w / 2.0)
            lower = center_y + (crop_h / 2.0)

            tile = img.crop((left, upper, right, lower))
            tile = tile.resize(final_size, Image.Resampling.LANCZOS)

            label = board_matrix[r, c]
            class_dir = os.path.join(output_root_for_split, label)
            os.makedirs(class_dir, exist_ok=True)

            tile_filename = f"{game_name}_{name_only}_r{r}_c{c}.png"
            tile.save(os.path.join(class_dir, tile_filename))


def gather_all_frames(base_raw_dir="raw_games"):
    """
    Returns a list of dicts:
      { game, frame_num, fen, img_path }
    Each row corresponds to a labeled frame (not tiles).
    """
    games = discover_games(base_raw_dir)
    if not games:
        print(f"No games found under: {base_raw_dir}")
        return []

    all_items = []
    for game in games:
        # unzip_folder(game, base_raw_dir)
        game_path = os.path.join(base_raw_dir, game)
        csv_path = os.path.join(game_path, f"{game}.csv")
        images_path = os.path.join(game_path, "tagged_images")

        if not os.path.exists(csv_path):
            print(f"Skipping {game}, CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "from_frame" not in df.columns or "fen" not in df.columns:
            print(f"Skipping {game}, missing columns in CSV (need from_frame, fen): {csv_path}")
            continue

        for _, row in df.iterrows():
            frame_num = int(row["from_frame"])
            fen = row["fen"]
            img_path = os.path.join(images_path, f"frame_{frame_num:06d}.jpg")
            if not os.path.exists(img_path):
                continue

            all_items.append(
                {
                    "game": game,
                    "frame_num": frame_num,
                    "fen": fen,
                    "img_path": img_path,
                }
            )

    # Deduplicate by image path (frame-level uniqueness)
    seen = set()
    dedup = []
    for item in all_items:
        p = item["img_path"]
        if p in seen:
            continue
        seen.add(p)
        dedup.append(item)

    return dedup


def split_frames_frame_level(items, seed=42, ratio=(0.8, 0.1, 0.1)):
    """
    Frame-level split.
    Returns: train_items, val_items, test_items
    """
    assert abs(sum(ratio) - 1.0) < 1e-9

    rng = random.Random(seed)
    items_shuffled = items[:]
    rng.shuffle(items_shuffled)

    n = len(items_shuffled)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])
    n_test = n - n_train - n_val

    train_items = items_shuffled[:n_train]
    val_items = items_shuffled[n_train : n_train + n_val]
    test_items = items_shuffled[n_train + n_val :]

    return train_items, val_items, test_items


def build_dataset_from_splits(train_items, val_items, test_items, out_root="final_dataset"):
    """
    Slices frames into tiles and writes directly to:
      final_dataset/train/<class>/*.png
      final_dataset/val/<class>/*.png
      final_dataset/test/<class>/*.png
    """
    splits = [("train", train_items), ("val", val_items), ("test", test_items)]

    for split_name, split_items in splits:
        split_root = os.path.join(out_root, split_name)
        os.makedirs(split_root, exist_ok=True)
        print(f"Building split: {split_name}, frames: {len(split_items)}")

        for item in split_items:
            board_matrix = parse_fen_to_matrix(item["fen"])
            slice_image_with_overlap(
                game_name=item["game"],
                image_path=item["img_path"],
                output_root_for_split=split_root,
                board_matrix=board_matrix,
            )


# --- Final run ---

if __name__ == "__main__":
    base_raw_dir = "raw_games"
    out_root = "final_dataset"
    seed = 42
    ratio = (0.8, 0.1, 0.1)

    items = gather_all_frames(base_raw_dir)
    if not items:
        raise SystemExit("No labeled frames found. Check raw_games structure.")

    train_items, val_items, test_items = split_frames_frame_level(items, seed=seed, ratio=ratio)

    print(f"Total frames: {len(items)}")
    print(f"Train frames: {len(train_items)} | Val frames: {len(val_items)} | Test frames: {len(test_items)}")

    build_dataset_from_splits(train_items, val_items, test_items, out_root=out_root)

    print(f"All done! Frame-level dataset is ready in '{out_root}'")
