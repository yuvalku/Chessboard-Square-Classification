import numpy as np
import torch
import torch.nn.functional as F
import cv2  # <-- added

from src.model import ResNet18Classifier
from dataset_tools.extract_squares import extract_square_crops_warped as extract_square_crops

_MODEL = None

# Post-processing OOD (confidence / ambiguity)
CONF_REJECT_TAU = 0.65
CONF_REJECT_MARGIN = 0.15

# Pre-processing OOD (simple image heuristics on the square crop)
PRE_OOD_DARK_MEAN = 35.0
PRE_OOD_BRIGHT_MEAN = 220.0
PRE_OOD_EDGE_FRAC = 0.25

ENABLE_PRE_OOD = True
ENABLE_POST_OOD = True

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # Model predicts 13 classes: 0..12 (12 is Empty)
    model = ResNet18Classifier(num_classes=13, pretrained=False)
    state = torch.load("checkpoints/resnet18_multi_best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _MODEL = model
    return _MODEL


def _square_is_occluded(rgb_sq: np.ndarray) -> bool:
    """
    Pre-processing OOD heuristic. Returns True if the crop looks unreliable/occluded.
    Uses only cheap signals: brightness + edge density.
    """
    gray = cv2.cvtColor(rgb_sq, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    edges = cv2.Canny(gray, 50, 150)
    edge_frac = float(edges.mean()) / 255.0

    return (
        mean < PRE_OOD_DARK_MEAN
        or mean > PRE_OOD_BRIGHT_MEAN
        or edge_frac > PRE_OOD_EDGE_FRAC
    )


def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.

    Input:
      image: np.ndarray (H,W,3), RGB, uint8

    Output:
      torch.Tensor (8,8), dtype int64, on CPU
      Values in [0..13] with:
        0..11 = pieces
        12    = Empty / Unknown (MANDATORY fallback for uncertain/occluded)
        13    = Internal OOD flag (used only internally; mapped to 12 before return)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H,W,3)")
    if image.dtype != np.uint8:
        raise ValueError("image dtype must be uint8")

    model = _load_model()

    # 64 square crops (RGB uint8)
    squares = extract_square_crops(image, out_size=64)
    if len(squares) != 64:
        # If extraction fails completely, return all unknown/empty
        return torch.full((8, 8), 12, dtype=torch.int64)

    # -------- Pre-processing OOD (before model) -> mark as 13 internally --------
    pre_ood = np.array([_square_is_occluded(sq) for sq in squares], dtype=bool)

    # Build batch tensor [64,3,64,64]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    batch = []
    for sq in squares:
        x = sq.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        x = (x - mean) / std
        batch.append(x)

    x = torch.tensor(np.stack(batch, axis=0), dtype=torch.float32)

    with torch.no_grad():
        logits = model(x)  # (64, 13) => classes 0..12
        probs = F.softmax(logits, dim=1)
        conf, preds = torch.max(probs, dim=1)  # (64,), (64,)

        # Post-processing OOD (confidence + optional margin)
        if CONF_REJECT_MARGIN > 0.0:
            top2 = torch.topk(probs, k=2, dim=1).values  # (64, 2)
            margin = top2[:, 0] - top2[:, 1]
            reject = (conf < CONF_REJECT_TAU) | (margin < CONF_REJECT_MARGIN)
        else:
            reject = conf < CONF_REJECT_TAU

        preds = preds.to(torch.int64)

        # Apply post-processing rejection -> internal OOD (13)
        if ENABLE_POST_OOD  :
            preds[reject] = 13

        # Apply pre-processing OOD -> internal OOD (13)
        if ENABLE_PRE_OOD:
            preds[torch.from_numpy(pre_ood)] = 13

        # -------- Final mapping for evaluator: internal OOD (13) -> Unknown (12) --------
        preds[preds == 13] = 12

    board = preds.view(8, 8).cpu().to(torch.int64)
    board = torch.clamp(board, 0, 13)
    return board
