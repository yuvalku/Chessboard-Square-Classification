import cv2
import numpy as np

def order_points(pts):
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def warp_board(img_rgb: np.ndarray, out_size: int = 512) -> np.ndarray:
    """
    Try to find the chessboard outer contour and warp to a square.
    Returns warped RGB image of shape (out_size, out_size, 3).
    Falls back to center-crop square if contour not found.
    """
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # preprocess
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # close gaps
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    quad = None
    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break

    h, w = img_rgb.shape[:2]

    if quad is None:
        # fallback: center square crop
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        cropped = img_rgb[y0:y0+side, x0:x0+side]
        return cv2.resize(cropped, (out_size, out_size))

    rect = order_points(quad)

    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_rgb, M, (out_size, out_size))
    return warped
