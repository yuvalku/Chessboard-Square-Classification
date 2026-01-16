"""Build the `final_dataset/` image classification dataset.

This script supports two labeling sources:
- CSV-labeled games: `raw_games/<game>/<game>.csv` + `tagged_images/`
- PGN-only games: `raw_games/<game>/<game>.pgn` + `images/`

For each selected frame, it slices the board image into an 8x8 grid of overlapping
square crops and saves them under `final_dataset/{train,val,test}/<class>/`.
"""

import os
import shutil
import re
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageStat
import chess.pgn
import scipy.signal
import scipy.ndimage
import cv2

# Motion/selection configuration
MOTION_DOWNSAMPLE = 32           # Higher values ignore small motion
MIN_MOVE_DURATION_FRAMES = 5     # Minimum length for a "stable" segment
CENTER_CHECK_RADIUS = 0.20       # Center ROI size for occupancy validation

# Manual orientation overrides.
# True = black at bottom (flipped), False = white at bottom (standard).
MANUAL_ORIENTATION_MAP = {
    "game2": False,
    "game5": False,
    "game6": False,
    "game7": False,
    "game8": False,
    "game9": False,
    "game10": False,
    "game11": False,
    "game12": False,
    "game13": True,
}

# --- Helpers ---

def discover_games(base_raw_dir="raw_games"):
    games = set()
    if not os.path.isdir(base_raw_dir): return []
    for name in os.listdir(base_raw_dir):
        if os.path.isdir(os.path.join(base_raw_dir, name)):
            games.add(name)
    return sorted(list(games))

def pgn_to_fens(pgn_path):
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        game = chess.pgn.read_game(f)
    if game is None: return []
    board = game.board()
    fens = [board.fen()]
    for mv in game.mainline_moves():
        board.push(mv)
        fens.append(board.fen())
    return fens

def list_frames_in_dir(images_dir):
    frames = []
    if not os.path.isdir(images_dir): return frames
    for name in os.listdir(images_dir):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            frames.append(os.path.join(images_dir, name))
    frames.sort(key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[-1]))
    return frames

def get_motion_profile(frames, downsample=MOTION_DOWNSAMPLE):
    scores = []
    prev_array = None
    print(f"   [Analysis] Computing motion profile for {len(frames)} frames...")
    for p in frames:
        try:
            with Image.open(p) as img:
                small = img.convert("L").resize((img.width // downsample, img.height // downsample))
                curr_array = np.asarray(small, dtype=np.float32)
                if prev_array is not None:
                    diff = np.mean(np.abs(curr_array - prev_array))
                    scores.append(diff)
                else:
                    scores.append(0.0)
                prev_array = curr_array
        except Exception:
            scores.append(0.0)
    scores = np.array(scores)
    if scores.max() > 0: scores = scores / scores.max()
    return scores

def parse_fen_to_matrix(fen, flip_board=False):
    board_part = fen.split(" ")[0].replace("m", "r") 
    rows = board_part.split("/")
    mapping = {"P": "white_pawn", "N": "white_knight", "B": "white_bishop", "R": "white_rook", "Q": "white_queen", "K": "white_king",
               "p": "black_pawn", "n": "black_knight", "b": "black_bishop", "r": "black_rook", "q": "black_queen", "k": "black_king"}
    matrix = []
    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit(): board_row.extend(["empty"] * int(char))
            else: board_row.append(mapping.get(char, "empty"))
        matrix.append(board_row)
    
    np_matrix = np.array(matrix)
    
    if flip_board:
        np_matrix = np.rot90(np_matrix, 2)
        
    return np_matrix

# --- Orientation ---

def detect_board_orientation(image_path, game_name):
    """Return whether the board should be treated as flipped.

    Currently this uses `MANUAL_ORIENTATION_MAP` when available. If a game is not in
    the map, it falls back to standard orientation.
    """
    if game_name in MANUAL_ORIENTATION_MAP:
        forced_val = MANUAL_ORIENTATION_MAP[game_name]
        print(f"   [Orientation] {game_name}: Manual Override -> {'FLIPPED' if forced_val else 'STANDARD'}")
        return forced_val

    return False

# --- PROCESSING LOGIC ---

def process_game_robust(game_name, base_raw_dir):
    game_path = os.path.join(base_raw_dir, game_name)
    pgn_path = os.path.join(game_path, f"{game_name}.pgn")
    
    if not os.path.exists(pgn_path): return []
    fens = pgn_to_fens(pgn_path)
    target_count = len(fens)
    print(f"\n[PGN] {game_name}: Expecting {target_count} moves.")

    images_dir = os.path.join(game_path, "images")
    frames = list_frames_in_dir(images_dir)
    if len(frames) < 50: return []

    # 1. FIND PAUSES
    motion_scores = get_motion_profile(frames)
    noise_floor = np.median(motion_scores)
    if noise_floor == 0: noise_floor = 0.005
    
    candidate_segments = []
    for threshold_mult in [1.5, 2.0, 3.0, 5.0, 8.0]:
        threshold = noise_floor * threshold_mult
        is_stable = motion_scores <= threshold
        structure = np.ones(MIN_MOVE_DURATION_FRAMES)
        is_stable = scipy.ndimage.binary_opening(is_stable, structure)
        labeled_array, num_features = scipy.ndimage.label(is_stable)
        
        if num_features >= target_count:
            current_segments = []
            for i in range(1, num_features + 1):
                indices = np.where(labeled_array == i)[0]
                start, end = indices[0], indices[-1]
                duration = end - start
                current_segments.append((start, end, duration))
            candidate_segments = current_segments
            break 
            
    if len(candidate_segments) < target_count:
        print(f"[SKIP] {game_name}: Unstable video (Found {len(candidate_segments)}/{target_count}).")
        return []

    # 2. SELECT LONGEST PAUSES
    candidate_segments.sort(key=lambda x: x[2], reverse=True)
    best_segments = candidate_segments[:target_count] 
    best_segments.sort(key=lambda x: x[0]) 

    # 3. DETECT ORIENTATION (Improved)
    first_frame_path = frames[best_segments[0][0]]
    is_flipped = detect_board_orientation(first_frame_path, game_name)
    
    if is_flipped:
        print(f"   [Orientation] Result: BLACK perspective (flipped).")
    else:
        print(f"   [Orientation] Result: WHITE perspective (standard).")

    valid_items = []
    for i, (start, end, duration) in enumerate(best_segments):
        mid_idx = (start + end) // 2
        img_path = frames[mid_idx]
        fen = fens[i]
        
        valid_items.append({
                    "game": game_name,
                    "fen": fen,
                    "img_path": img_path,
                    "flipped": is_flipped 
                })

    print(f"[Result] {game_name}: Kept {len(valid_items)} / {target_count} clean frames.")
    return valid_items

def validate_center_occupancy(pil_crop, label):
    try:
        img_arr = np.array(pil_crop.convert("L"))
        h, w = img_arr.shape
        roi_h = int(h * CENTER_CHECK_RADIUS)
        roi_w = int(w * CENTER_CHECK_RADIUS)
        y1 = (h - roi_h) // 2
        x1 = (w - roi_w) // 2
        center_crop = img_arr[y1:y1+roi_h, x1:x1+roi_w]
        
        blurred = cv2.GaussianBlur(center_crop, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edge_pixels = np.count_nonzero(edges)
        std_dev = np.std(center_crop)
        
        is_label_occupied = (label != "empty")
        
        if not is_label_occupied:
            is_visually_occupied = (edge_pixels > 10) and (std_dev > 20)
            if is_visually_occupied: return False 
        else:
            is_visually_empty = (edge_pixels < 5) and (std_dev < 15)
            if is_visually_empty: return False 

        return True
    except Exception:
        return True

def slice_image_with_overlap(game_name, image_path, output_root, board_matrix, overlap_percent=0.7, final_size=(224, 224)):
    try:
        with Image.open(image_path).convert("RGB") as img:
            # Normalize input size so training and inference use the same grid geometry.
            img = img.resize((800, 800))
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
                    label = board_matrix[r, c]
                    
                    # Skip tiles that look inconsistent with the label (helps reduce noisy labels).
                    if validate_center_occupancy(tile, label):
                        tile = tile.resize(final_size, Image.Resampling.LANCZOS)
                        class_dir = os.path.join(output_root, label)
                        os.makedirs(class_dir, exist_ok=True)
                        
                        # Save via PIL to preserve RGB channel order.
                        save_path = os.path.join(class_dir, f"{game_name}_{name_only}_r{r}_c{c}.jpg")
                        tile.save(save_path, "JPEG", quality=95)
                        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    """Entry point: build `final_dataset/` from `raw_games/`."""
    base_raw_dir = "raw_games"
    out_root = "final_dataset"
    if os.path.exists(out_root): shutil.rmtree(out_root)
    
    games = discover_games(base_raw_dir)
    all_valid_items = []
    
    for game in games:
        csv_path = os.path.join(base_raw_dir, game, f"{game}.csv")
        
        if os.path.exists(csv_path):
            print(f"Processing {game} via CSV...")
            df = pd.read_csv(csv_path)
            images_path = os.path.join(base_raw_dir, game, "tagged_images")
            for _, row in df.iterrows():
                frame_num = int(row['from_frame'])
                img_p = os.path.join(images_path, f"frame_{frame_num:06d}.jpg")
                if os.path.exists(img_p):
                    all_valid_items.append({"game": game, "fen": row['fen'], "img_path": img_p, "flipped": False})
        else:
            items = process_game_robust(game, base_raw_dir)
            all_valid_items.extend(items)

    print(f"\nTotal Valid Frames Collected: {len(all_valid_items)}")
    
    random.seed(42)
    random.shuffle(all_valid_items)
    
    n_total = len(all_valid_items)
    n_val = int(n_total * 0.1)
    n_test = int(n_total * 0.1)
    n_train = n_total - n_val - n_test
    
    train_set = all_valid_items[:n_train]
    val_set = all_valid_items[n_train:n_train+n_val]
    test_set = all_valid_items[n_train+n_val:]
    
    print(f"Split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    
    for split_name, items in [("train", train_set), ("val", val_set), ("test", test_set)]:
        print(f"Generating {split_name} dataset...")
        split_root = os.path.join(out_root, split_name)
        for it in items:
            mat = parse_fen_to_matrix(it["fen"], flip_board=it["flipped"])
            slice_image_with_overlap(it["game"], it["img_path"], split_root, mat)

    print("Done.")

if __name__ == "__main__":
    main()