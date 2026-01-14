import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

# --- הגדרות ומפות ---
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

# --- פונקציות עזר ---

def load_model(model_path, device):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 13)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_and_predict(image_path, model, device, conf_threshold=0.75, overlap_percent=0.7):
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (800, 800))
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    stride = 800 / 8
    crop_size = stride * (1 + overlap_percent)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    board_matrix = []
    for r in range(8):
        current_row = []
        for c in range(8):
            center_x, center_y = (c * stride) + (stride / 2), (r * stride) + (stride / 2)
            left, upper = center_x - (crop_size / 2), center_y - (crop_size / 2)
            right, lower = center_x + (crop_size / 2), center_y + (crop_size / 2)
            
            tile = img_pil.crop((left, upper, right, lower))
            tile_tensor = preprocess(tile).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(tile_tensor)
                probs = F.softmax(outputs[0], dim=0)
                conf, pred_idx = torch.max(probs, 0)
            
            if conf.item() < conf_threshold:
                current_row.append('X')
            else:
                label = CLASSES[pred_idx]
                current_row.append(CLASS_TO_FEN_MAP[label])
        board_matrix.append(current_row)
    return board_matrix

def matrix_to_fen(matrix):
    fen_rows = []
    for row in matrix:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == '1': empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0: fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)

# --- פונקציית הויזואליזציה המשופרת ---

def visual_board_from_matrix(
    matrix,
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

    # ציור הלוח
    for r in range(8):
        for c in range(8):
            x0 = border + c * square_size
            y0 = border + r * square_size
            x1, y1 = x0 + square_size, y0 + square_size
            color = light_color if (r + c) % 2 == 0 else dark_color
            draw.rectangle([x0, y0, x1, y1], fill=color)

            piece_char = matrix[r][c]
            
            # מקרה של OOD - איקס אדום
            if piece_char == 'X':
                m = square_size * 0.2
                draw.line([x0+m, y0+m, x1-m, y1-m], fill=(255, 0, 0), width=5)
                draw.line([x1-m, y0+m, x0+m, y1-m], fill=(255, 0, 0), width=5)
            
            # מקרה של כלי שחמט
            elif piece_char != '1':
                glyph = GLYPH_MAP.get(piece_char, piece_char)
                font_size = int(square_size * piece_scale)
                font_path = os.path.join("assets", "fonts", "DejaVuSans.ttf")
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    print(f"Warning: Font not found at {font_path}, using default.")
                    font = ImageFont.load_default()
                
                # חישוב מרכז
                cx, cy = x0 + square_size/2, y0 + square_size/2
                bbox = draw.textbbox((0, 0), glyph, font=font)
                w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                
                fill = (255, 255, 255) if piece_char.isupper() else (0, 0, 0)
                stroke = (0, 0, 0) if piece_char.isupper() else (255, 255, 255)
                
                draw.text((cx - w/2, cy - h/2 - 5), glyph, font=font, fill=fill, 
                          stroke_width=2, stroke_fill=stroke)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return img

# --- הרצה ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('chess_model.pth', device)
    
    img_path = 'test_predict/game2_per_frame_frame_001740.jpg'
    
    matrix = process_and_predict(img_path, model, device)
    fen = matrix_to_fen(matrix)
    
    print(f"Predicted FEN: {fen}")
    visual_board_from_matrix(matrix, out_path='./results/final_board.png')