"""
Chessboard square classifier inference.

This script:
- Loads a ResNet18 square classifier.
- Slices a chessboard image into an 8x8 grid of overlapping crops.
- Predicts a class per square and returns the required (8, 8) integer matrix.
- Uses Unicode glyphs with stroke/fill for the professional piece look.
- Neighborhood consensus removed; 2/4 corner detection logic active.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2

# --- BALANCED THRESHOLDS ---
CONF_THRESHOLD = 0.85      
ENTROPY_THRESHOLD = 0.6    
LAPLACIAN_THRESHOLD = 30   
CANNY_THRESHOLD = 110      
OVERLAP_PERCENT = 0.7      

MAPPING_TO_API = {
    'white_pawn': 0, 'white_rook': 1, 'white_knight': 2, 'white_bishop': 3,
    'white_queen': 4, 'white_king': 5, 'black_pawn': 6, 'black_rook': 7,
    'black_knight': 8, 'black_bishop': 9, 'black_queen': 10, 'black_king': 11,
    'empty': 12, 'ood': 13
}

CLASSES = ['black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
           'empty', 'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook']

CLASS_TO_FEN_MAP = {
    'white_pawn': 'P', 'white_knight': 'N', 'white_bishop': 'B', 'white_rook': 'R', 'white_queen': 'Q', 'white_king': 'K',
    'black_pawn': 'p', 'black_knight': 'n', 'black_bishop': 'b', 'black_rook': 'r', 'black_queen': 'q', 'black_king': 'k',
    'empty': '1'
}

GLYPH_MAP = {
    "P": "\u2659", "R": "\u2656", "N": "\u2658", "B": "\u2657", "Q": "\u2655", "K": "\u2654",
    "p": "\u265F", "r": "\u265C", "n": "\u265E", "b": "\u265D", "q": "\u265B", "k": "\u265A",
}

_inference_model = None

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model_state") or checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 13)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

def is_visually_valid(pil_tile):
    try:
        tile_np = np.array(pil_tile)
        img_cv = cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(img_cv, cv2.CV_64F).var()
        if variance < LAPLACIAN_THRESHOLD:
            return False

        dst = cv2.cornerHarris(img_cv, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        corners = np.argwhere(dst > 0.01 * dst.max())
        
        h, w = img_cv.shape
        margin = 20 
        zones = [(0, margin, 0, margin), (0, margin, w - margin, w),
                 (h - margin, h, 0, margin), (h - margin, h, w - margin, w)]
        
        corners_found = 0
        for (y_min, y_max, x_min, x_max) in zones:
            for cy, cx in corners:
                if y_min <= cy <= y_max and x_min <= cx <= x_max:
                    corners_found += 1
                    break 
        
        if corners_found >= 2:
            return True

        edges = cv2.Canny(img_cv, 50, CANNY_THRESHOLD)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, 30, 10)
        if lines is not None:
            has_h, has_v = False, False
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if (30 < angle < 60) or (120 < angle < 150): return False
                if angle < 20 or angle > 160: has_h = True
                elif 70 < angle < 110: has_v = True
                if has_h and has_v: return True
        return False
    except:
        return True

def predict_board(image: np.ndarray, debug_compare: bool = False) -> torch.Tensor:
    global _inference_model
    device = next(_inference_model.parameters()).device
    img_pil = Image.fromarray(image).convert("RGB").resize((800, 800))
    stride = 100 
    crop_size = stride * (1 + OVERLAP_PERCENT)
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    output_tensor = torch.zeros((8, 8), dtype=torch.int64)

    for r in range(8):
        for c in range(8):
            left = (c * stride) - (stride * OVERLAP_PERCENT / 2)
            upper = (r * stride) - (stride * OVERLAP_PERCENT / 2)
            tile = img_pil.crop((left, upper, left + crop_size, upper + crop_size))
            
            if not is_visually_valid(tile):
                output_tensor[r, c] = 13
                continue

            tile_tensor = preprocess(tile).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = _inference_model(tile_tensor)
                probs = F.softmax(outputs[0], dim=0)
                conf, pred_idx = torch.max(probs, 0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            
            if conf.item() < CONF_THRESHOLD or entropy > ENTROPY_THRESHOLD:
                output_tensor[r, c] = 13
            else:
                output_tensor[r, c] = MAPPING_TO_API[CLASSES[pred_idx]]

    final_output = output_tensor.to("cpu")

    if debug_compare:
        pred_img = visual_board_from_matrix(final_output, out_path=None)
        canvas = Image.new('RGB', (1600, 800))
        canvas.paste(img_pil, (0, 0))
        canvas.paste(pred_img.resize((800, 800)), (800, 0))
        os.makedirs('./results', exist_ok=True)
        canvas.save('./results/debug_comparison.png')

    return final_output

def matrix_to_fen(matrix_tensor):
    REVERSE_API = {v: k for k, v in MAPPING_TO_API.items()}
    fen_rows = []
    for r in range(8):
        fen_row = ""
        empty_count = 0
        for c in range(8):
            val = matrix_tensor[r, c].item()
            if val == 12 or val == 13:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                label = REVERSE_API[val]
                fen_row += CLASS_TO_FEN_MAP[label]
        if empty_count > 0: fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)

def visual_board_from_matrix(
    matrix_tensor,
    square_size: int = 80,
    border: int = 20,
    light_color=(240, 217, 181),
    dark_color=(181, 136, 99),
    piece_scale: float = 0.8,
    out_path: str = './results/output_visual.png'
):
    size = 8 * square_size + 2 * border
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    REVERSE_API = {v: k for k, v in MAPPING_TO_API.items()}

    for r in range(8):
        for c in range(8):
            x0 = border + c * square_size
            y0 = border + r * square_size
            x1, y1 = x0 + square_size, y0 + square_size
            color = light_color if (r + c) % 2 == 0 else dark_color
            draw.rectangle([x0, y0, x1, y1], fill=color)

            val = matrix_tensor[r, c].item()
            if val == 13: # Red X
                m = square_size * 0.2
                draw.line([x0+m, y0+m, x1-m, y1-m], fill=(255, 0, 0), width=5)
                draw.line([x1-m, y0+m, x0+m, y1-m], fill=(255, 0, 0), width=5)
            elif val != 12:
                label = REVERSE_API[val]
                piece_char = CLASS_TO_FEN_MAP[label]
                glyph = GLYPH_MAP.get(piece_char, piece_char)
                font_size = int(square_size * piece_scale)
                font_path = os.path.join("assets", "fonts", "DejaVuSans.ttf")
                font = ImageFont.truetype(font_path, font_size) if os.path.exists(font_path) else ImageFont.load_default()
                
                cx, cy = x0 + square_size/2, y0 + square_size/2
                bbox = draw.textbbox((0, 0), glyph, font=font)
                w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                
                # REVERTED PIECE STYLING
                fill = (255, 255, 255) if piece_char.isupper() else (0, 0, 0)
                stroke = (0, 0, 0) if piece_char.isupper() else (255, 255, 255)
                draw.text((cx - w/2, cy - h/2 - 5), glyph, font=font, fill=fill, stroke_width=2, stroke_fill=stroke)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)
    return img

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _inference_model = load_model("chess_model.pth", device)
    
    input_folder = "test_predict"
    output_base_dir = "./results"
    os.makedirs(output_base_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in img_files:
        img_path = os.path.join(input_folder, img_name)
        raw_img = np.array(Image.open(img_path).convert("RGB"))
        
        # Predict and show side-by-side
        board_tensor = predict_board(raw_img, debug_compare=True)
        
        # Output unique files for batch testing
        new_debug_path = os.path.join(output_base_dir, f"debug_{img_name}")
        if os.path.exists('./results/debug_comparison.png'):
            if os.path.exists(new_debug_path): os.remove(new_debug_path)
            os.rename('./results/debug_comparison.png', new_debug_path)
            
        print(f"Processed: {img_name}")
        print(f"FEN: {matrix_to_fen(board_tensor)}\n")