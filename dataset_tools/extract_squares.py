import cv2
import numpy as np
from dataset_tools.board_warp import warp_board

def extract_board_region(img: np.ndarray) -> np.ndarray:
    """
    Baseline board crop.
    Assumes the chessboard occupies the central square region of the image.
    """
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    board = img[y0:y0+side, x0:x0+side]
    return board


def extract_square_crops(img: np.ndarray, out_size: int = 64):
    """
    Given a full image, return a list of 64 square crops
    ordered row-major: top-left -> bottom-right.
    """
    board = extract_board_region(img)
    board = cv2.resize(board, (out_size * 8, out_size * 8))

    squares = []
    step = out_size

    for row in range(8):
        for col in range(8):
            y0 = row * step
            x0 = col * step
            sq = board[y0:y0+step, x0:x0+step]
            squares.append(sq)

    return squares


def extract_square_crops_warped(img: np.ndarray, out_size: int = 64):
    """
    Warp the board to top-down view then slice into squares.
    """
    board = warp_board(img, out_size=out_size * 8)
    squares = []
    step = out_size

    for row in range(8):
        for col in range(8):
            y0 = row * step
            x0 = col * step
            sq = board[y0:y0+step, x0:x0+step]
            squares.append(sq)

    return squares
