import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import cv2
import numpy as np

# --- Settings and classes ---
# Make sure this order matches the folder order in final_dataset/train exactly
CLASSES = ['black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
           'empty', 'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook']

# Map class name back to FEN character
CLASS_TO_FEN = {
    'white_pawn': 'P', 'white_knight': 'N', 'white_bishop': 'B', 'white_rook': 'R', 'white_queen': 'Q', 'white_king': 'K',
    'black_pawn': 'p', 'black_knight': 'n', 'black_bishop': 'b', 'black_rook': 'r', 'black_queen': 'q', 'black_king': 'k',
    'empty': '1'
}

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 13)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def get_fen_from_matrix(matrix):
    """Convert a character matrix into a standard FEN string."""
    fen_rows = []
    for row in matrix:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == '1':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)

def process_image(image_path, model, device, overlap_percent=0.7):
    # Load image for drawing (OpenCV) and processing (PIL)
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (800, 800))  # Normalize to a fixed size
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    img_width, img_height = img_pil.size
    stride_w, stride_h = img_width / 8, img_height / 8
    crop_w, crop_h = stride_w * (1 + overlap_percent), stride_h * (1 + overlap_percent)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    board_matrix = []
    visual_output = img_cv.copy()

    for r in range(8):
        current_row = []
        for c in range(8):
            # Compute overlap crop exactly like in training
            center_x, center_y = (c * stride_w) + (stride_w / 2), (r * stride_h) + (stride_h / 2)
            left, upper = center_x - (crop_w / 2), center_y - (crop_h / 2)
            right, lower = center_x + (crop_w / 2), center_y + (crop_h / 2)
            
            tile = img_pil.crop((left, upper, right, lower))
            tile_tensor = preprocess(tile).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(tile_tensor)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
                conf, pred_idx = torch.max(probs, 0)
            
            label = CLASSES[pred_idx]
            current_row.append(CLASS_TO_FEN[label])

            # --- Requirement: mark red X on low-confidence squares ---
            if conf.item() < 0.75:  # Confidence threshold (tunable)
                x_start, y_start = int(c * stride_w), int(r * stride_h)
                x_end, y_end = int((c + 1) * stride_w), int((r + 1) * stride_h)
                cv2.line(visual_output, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
                cv2.line(visual_output, (x_end, y_start), (x_start, y_end), (0, 0, 255), 2)
        
        board_matrix.append(current_row)

    # Save visualization output
    cv2.imwrite('output_visual.png', visual_output)
    
    return get_fen_from_matrix(board_matrix)

if __name__ == "__main__":
    model, device = load_model('chess_model.pth')
    test_img = 'path_to_test_image.jpg'  # The image the instructor will test
    fen_result = process_image(test_img, model, device)
    print(f"Predicted FEN: {fen_result}")