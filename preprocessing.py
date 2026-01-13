import os
import pandas as pd
import numpy as np
from PIL import Image
import splitfolders

# --- פונקציות עזר ---

def parse_fen_to_matrix(fen):
    """ הופך FEN למטריצה 8x8 של שמות מחלקות, כולל טיפול ב-'m' """
    board_part = fen.split(' ')[0]
    board_part = board_part.replace('m', 'r') # תיקון m ל-r (Rook)
    
    rows = board_part.split('/')
    mapping = {
        'P': 'white_pawn', 'N': 'white_knight', 'B': 'white_bishop', 
        'R': 'white_rook', 'Q': 'white_queen', 'K': 'white_king',
        'p': 'black_pawn', 'n': 'black_knight', 'b': 'black_bishop', 
        'r': 'black_rook', 'q': 'black_queen', 'k': 'black_king'
    }
    
    matrix = []
    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(['empty'] * int(char))
            else:
                board_row.append(mapping.get(char, 'empty'))
        matrix.append(board_row)
    return np.array(matrix)

def slice_image_with_overlap(game_name, image_path, output_folder, board_matrix, 
                             overlap_percent=0.7, final_size=(224, 224)):
    """
    הפונקציה המתקדמת ששלחת: חותכת עם חפיפה ושומרת בתיקיות זמניות
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        return

    img_width, img_height = img.size
    stride_w = img_width / 8
    stride_h = img_height / 8

    # חישוב גודל החיתוך עם ה-Overlap
    crop_w = stride_w * (1 + overlap_percent)
    crop_h = stride_h * (1 + overlap_percent)

    for r in range(8):
        for c in range(8):
            # מציאת המרכז
            center_x = (c * stride_w) + (stride_w / 2)
            center_y = (r * stride_h) + (stride_h / 2)

            # חישוב גבולות הקופסה
            left = center_x - (crop_w / 2)
            upper = center_y - (crop_h / 2)
            right = center_x + (crop_w / 2)
            lower = center_y + (crop_h / 2)

            # חיתוך (PIL מטפל ב-Padding אוטומטית אם חורגים מהגבולות)
            tile = img.crop((left, upper, right, lower))
            tile = tile.resize(final_size, Image.Resampling.LANCZOS)

            # קבלת התווית
            label = board_matrix[r, c]
            
            # יצירת תיקייה למחלקה בתוך temp_dataset
            class_dir = os.path.join(output_folder, label)
            os.makedirs(class_dir, exist_ok=True)

            # שמירה
            name_only = os.path.splitext(os.path.basename(image_path))[0]
            tile_filename = f"{game_name}_{name_only}_r{r}_c{c}.png"
            tile.save(os.path.join(class_dir, tile_filename))

# --- התהליך המרכזי שסורק את כל 5 המשחקים ---

def process_all_data(base_raw_dir, temp_dir):
    games = ['game2', 'game4', 'game5', 'game6', 'game7']
    
    for game in games:
        game_path = os.path.join(base_raw_dir, game)
        csv_path = os.path.join(game_path, f"{game}.csv")
        images_path = os.path.join(game_path, "tagged_images")
        
        if not os.path.exists(csv_path):
            print(f"Skipping {game}, CSV not found.")
            print(f"{csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        print(f"Processing pieces from {game}...")

        for _, row in df.iterrows():
            frame_num = row['from_frame']
            img_path = os.path.join(images_path, f"frame_{int(frame_num):06d}.jpg")
            
            if os.path.exists(img_path):
                board_matrix = parse_fen_to_matrix(row['fen'])
                slice_image_with_overlap(game, img_path, temp_dir, board_matrix)
            else:
                print(f"Image not found: {img_path}")

# --- הרצה סופית ---

if __name__ == "__main__":
    # 1. יצירת תיקייה עם כל המשבצות (בתוך 13 תיקיות מחלקות)
    process_all_data('raw_games', 'temp_dataset')
    
    # 2. חלוקה ל-Train, Val, Test בתוך תיקייה סופית
    # וודאי שהתקנת: pip install split-folders
    splitfolders.ratio('temp_dataset', output='final_dataset', 
                       seed=42, ratio=(.8, .1, .1))
    
    print("All done! Dataset is ready in 'final_dataset'")