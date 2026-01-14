import os
import zipfile
import random
import shutil
import re
import pandas as pd
import numpy as np
from PIL import Image
import chess.pgn

PRINT_EVERY_DIFF = 500          # progress while computing motion scores
PRINT_EVERY_TILING = 50         # progress while slicing frames into tiles
PRINT_EVERY_GATHER = 200        # progress while reading CSV rows

# PGN controls (to avoid exploding dataset size)
PGN_SAMPLE_EVERY = 20           # was 5 (too many). Increase for long videos.
PGN_SKIP_MARGIN = 10            # skip frames near move boundaries
PGN_MIN_GAP = 10                # minimum spacing between chosen motion peaks
PGN_MAX_PER_SEGMENT = 25        # cap frames per PGN state segment

# Motion scoring controls (faster)
MOTION_DOWNSAMPLE = 20          # was 10. Bigger -> faster for motion detection


# --- Helper functions ---

def discover_games(base_raw_dir="raw_games"):
    """
    Supports:
    - Zips named like `<game>_per_frame.zip`
    - Extracted folders with `<game>.csv` + `tagged_images/`
    - PGN-only folders with `<game>.pgn` + `images/`
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
            pgn_path = os.path.join(full_path, f"{game_name}.pgn")
            tagged_dir = os.path.join(full_path, "tagged_images")
            images_dir = os.path.join(full_path, "images")

            if os.path.isfile(csv_path) and os.path.isdir(tagged_dir):
                games.add(game_name)
                continue

            if os.path.isfile(pgn_path) and os.path.isdir(images_dir):
                games.add(game_name)
                continue

    return sorted(games)


def unzip_folder(game_name, base_raw_dir="raw_games", remove_zip=False):
    """
    Extracts `{game_name}_per_frame.zip` into `raw_games/game_name`.
    Skips if already extracted.
    """
    zip_path = os.path.join(base_raw_dir, f"{game_name}_per_frame.zip")
    dest_dir = os.path.join(base_raw_dir, game_name)

    expected_csv = os.path.join(dest_dir, f"{game_name}.csv")
    expected_images_dir = os.path.join(dest_dir, "tagged_images")

    if os.path.isdir(dest_dir) and os.path.isfile(expected_csv) and os.path.isdir(expected_images_dir):
        return

    if not os.path.isfile(zip_path):
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
    board_part = fen.split(" ")[0]
    board_part = board_part.replace("m", "r")  # Fix 'm' -> 'r' (rook)

    rows = board_part.split("/")
    mapping = {
        "P": "white_pawn", "N": "white_knight", "B": "white_bishop",
        "R": "white_rook", "Q": "white_queen", "K": "white_king",
        "p": "black_pawn", "n": "black_knight", "b": "black_bishop",
        "r": "black_rook", "q": "black_queen", "k": "black_king",
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
    try:
        img = Image.open(image_path)
    except Exception:
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
    CSV-labeled frames only.
    """
    games = discover_games(base_raw_dir)
    if not games:
        print(f"No games found under: {base_raw_dir}")
        return []

    all_items = []
    for game in games:
        # If you still have zipped csv games, keep this enabled:
        unzip_folder(game, base_raw_dir)

        game_path = os.path.join(base_raw_dir, game)
        csv_path = os.path.join(game_path, f"{game}.csv")
        images_path = os.path.join(game_path, "tagged_images")

        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if "from_frame" not in df.columns or "fen" not in df.columns:
            print(f"[CSV] Skipping {game}, missing columns: {csv_path}")
            continue

        before = len(all_items)
        for idx, row in enumerate(df.itertuples(index=False), 1):
            frame_num = int(getattr(row, "from_frame"))
            fen = getattr(row, "fen")
            img_path = os.path.join(images_path, f"frame_{frame_num:06d}.jpg")
            if os.path.exists(img_path):
                all_items.append({"game": game, "frame_num": frame_num, "fen": fen, "img_path": img_path})

            if idx % PRINT_EVERY_GATHER == 0:
                print(f"[CSV] {game}: read {idx}/{len(df)} rows")

        added = len(all_items) - before
        print(f"[CSV] {game}: collected {added} labeled frames")

    # Deduplicate by image path
    seen = set()
    dedup = []
    for item in all_items:
        p = item["img_path"]
        if p in seen:
            continue
        seen.add(p)
        dedup.append(item)

    print(f"[CSV] Total labeled frames after dedup: {len(dedup)}")
    return dedup


def split_frames_frame_level(items, seed=42, ratio=(0.8, 0.1, 0.1)):
    assert abs(sum(ratio) - 1.0) < 1e-9
    rng = random.Random(seed)
    items_shuffled = items[:]
    rng.shuffle(items_shuffled)

    n = len(items_shuffled)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])

    train_items = items_shuffled[:n_train]
    val_items = items_shuffled[n_train:n_train + n_val]
    test_items = items_shuffled[n_train + n_val:]
    return train_items, val_items, test_items


def build_dataset_from_splits(train_items, val_items, test_items, out_root="final_dataset"):
    splits = [("train", train_items), ("val", val_items), ("test", test_items)]

    for split_name, split_items in splits:
        split_root = os.path.join(out_root, split_name)
        os.makedirs(split_root, exist_ok=True)
        print(f"[TILES] Building split: {split_name}, frames: {len(split_items)}")

        for idx, item in enumerate(split_items, 1):
            board_matrix = parse_fen_to_matrix(item["fen"])
            slice_image_with_overlap(
                game_name=item["game"],
                image_path=item["img_path"],
                output_root_for_split=split_root,
                board_matrix=board_matrix,
            )
            if idx % PRINT_EVERY_TILING == 0 or idx == len(split_items):
                print(f"[TILES] {split_name}: {idx}/{len(split_items)} frames processed")


# --- PGN support ---

def pgn_to_fens(pgn_path):
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        game = chess.pgn.read_game(f)
        if game is None:
            return []
    board = game.board()
    fens = [board.fen()]
    for mv in game.mainline_moves():
        board.push(mv)
        fens.append(board.fen())
    return fens


def list_frames_in_dir(images_dir):
    """
    Robust: accept any .jpg/.png and sort by last number in the filename if present.
    """
    frames = []
    if not os.path.isdir(images_dir):
        return frames

    for name in os.listdir(images_dir):
        n = name.lower()
        if n.endswith((".jpg", ".jpeg", ".png")):
            frames.append(os.path.join(images_dir, name))

    def key_fn(p):
        base = os.path.basename(p)
        m = re.findall(r"\d+", base)
        return int(m[-1]) if m else base

    frames.sort(key=key_fn)
    return frames


def pick_top_peaks(scores, k, min_gap=10):
    if k <= 0:
        return []
    idxs = np.argsort(scores)[::-1]
    chosen = []
    blocked = np.zeros(len(scores), dtype=bool)

    for i in idxs:
        if len(chosen) >= k:
            break
        if blocked[i]:
            continue
        chosen.append(int(i))
        lo = max(0, i - min_gap)
        hi = min(len(scores) - 1, i + min_gap)
        blocked[lo:hi + 1] = True

    chosen.sort()
    return chosen


def compute_motion_scores_cached(frames, downsample=MOTION_DOWNSAMPLE, print_every=PRINT_EVERY_DIFF, tag=""):
    """
    Faster motion scoring:
    - reads each frame once
    - compares to previous
    """
    if len(frames) < 2:
        return []

    def load_small_gray(path):
        img = Image.open(path).convert("L")
        if downsample > 1:
            img = img.resize((max(1, img.size[0] // downsample), max(1, img.size[1] // downsample)))
        return np.asarray(img, dtype=np.int16)

    scores = []
    prev = None
    for i, p in enumerate(frames):
        try:
            cur = load_small_gray(p)
        except Exception:
            cur = None

        if prev is not None and cur is not None:
            scores.append(float(np.mean(np.abs(cur - prev))))
        elif prev is not None:
            scores.append(-1.0)

        prev = cur

        if i > 0 and (i % print_every == 0):
            print(f"[PGN] {tag} diff progress {i}/{len(frames)-1}")

    return scores


def build_items_from_pgn_only(
    game_name,
    game_path,
    sample_every=PGN_SAMPLE_EVERY,
    skip_margin=PGN_SKIP_MARGIN,
    min_gap=PGN_MIN_GAP,
    max_per_segment=PGN_MAX_PER_SEGMENT,
):
    pgn_path = os.path.join(game_path, f"{game_name}.pgn")
    if not os.path.isfile(pgn_path):
        return []

    images_dir = os.path.join(game_path, "images")
    frames = list_frames_in_dir(images_dir)
    print(f"[PGN] {game_name}: found {len(frames)} frames in {images_dir}")

    if len(frames) < 2:
        return []

    fens = pgn_to_fens(pgn_path)
    print(f"[PGN] {game_name}: PGN states (FENs) = {len(fens)}")
    if len(fens) < 2:
        return []

    num_moves = len(fens) - 1
    print(f"[PGN] {game_name}: computing motion scores for {len(frames)-1} pairs (downsample={MOTION_DOWNSAMPLE})...")
    scores = compute_motion_scores_cached(frames, downsample=MOTION_DOWNSAMPLE, print_every=PRINT_EVERY_DIFF, tag=game_name)

    boundaries = pick_top_peaks(scores, k=min(num_moves, len(scores)), min_gap=min_gap)
    print(f"[PGN] {game_name}: moves={num_moves} | picked boundaries={len(boundaries)} | min_gap={min_gap}")

    cut_points = [-1] + boundaries + [len(frames) - 1]

    items = []
    for seg_idx in range(len(cut_points) - 1):
        fen_idx = seg_idx
        if fen_idx >= len(fens):
            break

        start = cut_points[seg_idx] + 1
        end = cut_points[seg_idx + 1]

        start2 = min(end, start + (skip_margin if seg_idx > 0 else 0))
        end2 = max(start2, end - (skip_margin if seg_idx < len(cut_points) - 2 else 0))
        if end2 < start2:
            continue

        count = 0
        for j in range(start2, end2 + 1, sample_every):
            items.append({"game": game_name, "fen": fens[fen_idx], "img_path": frames[j]})
            count += 1
            if max_per_segment is not None and count >= max_per_segment:
                break

    print(
        f"[PGN] {game_name}: generated {len(items)} labeled frames "
        f"(sample_every={sample_every}, skip_margin={skip_margin}, max_per_segment={max_per_segment})"
    )
    return items


# --- Final run ---

if __name__ == "__main__":
    base_raw_dir = "raw_games"
    out_root = "final_dataset"
    seed = 42
    ratio = (0.8, 0.1, 0.1)

    games = discover_games(base_raw_dir)
    print(f"[INFO] Discovered games: {games}")

    # Optional: skip reprocessing if dataset already exists
    if os.path.isdir(out_root):
        print(f"[INFO] '{out_root}' already exists. Delete it if you want to rebuild.")
        # If you want auto-rebuild, uncomment:
        # shutil.rmtree(out_root)

    # 1) CSV-based labeled frames
    items = gather_all_frames(base_raw_dir)
    if not items:
        print("[WARN] No CSV labeled frames found. Val/test may be empty.")

    train_items, val_items, test_items = split_frames_frame_level(items, seed=seed, ratio=ratio)

    # 2) Add PGN-only games to TRAIN only
    pgn_added = 0
    for game in games:
        game_path = os.path.join(base_raw_dir, game)
        csv_path = os.path.join(game_path, f"{game}.csv")

        if not os.path.isfile(csv_path):
            pgn_items = build_items_from_pgn_only(game, game_path)
            if pgn_items:
                train_items.extend(pgn_items)
                pgn_added += len(pgn_items)

    # 3) Deduplicate train by img_path
    seen = set()
    train_dedup = []
    for it in train_items:
        p = it["img_path"]
        if p in seen:
            continue
        seen.add(p)
        train_dedup.append(it)
    train_items = train_dedup

    print(f"[INFO] Total CSV frames: {len(items)}")
    print(f"[INFO] PGN-only frames added to train: {pgn_added}")
    print(f"[INFO] Train frames: {len(train_items)} | Val frames: {len(val_items)} | Test frames: {len(test_items)}")

    # 4) Build tiles
    build_dataset_from_splits(train_items, val_items, test_items, out_root=out_root)
    print(f"[DONE] Dataset is ready in '{out_root}'")
