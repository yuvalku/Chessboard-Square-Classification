# Chessboard Square Classification

This project trains a CNN (ResNet18) to classify each chessboard square into one of 13 classes (12 pieces + empty), and can predict a FEN string from a board image.

## Setup (Conda, Windows)

### 1) Create and activate a conda environment

```powershell
conda create -n chessboard-squares python=3.10 -y
conda activate chessboard-squares
```

### 2) Install Python requirements

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) (Optional) PyTorch GPU install

By default, `pip install -r requirements.txt` will install PyTorch from PyPI (often CPU-only).
If you want GPU acceleration, install the CUDA-enabled build for your CUDA version using the official selector:

- https://pytorch.org/get-started/locally/

Then re-run `pip install -r requirements.txt` if needed.

### 4) Quick sanity check

```powershell
python -c "import torch; import torchvision; import cv2; import numpy; import pandas; import PIL; import sklearn; import matplotlib; import seaborn; import splitfolders; import chess; print('OK')"
```

## Project scripts

On Windows/PowerShell, prefer running scripts as `python <script>.py` (for example `python preprocessing.py`).
Running `./preprocessing.py` or `./preprocessing` may do nothing if `.py` file associations are not configured.

### Data preprocessing / dataset creation

#### Download the `raw_games/` data (required)

The training/validation/test dataset is generated from game folders under `raw_games/`.
This repo does not include the full `raw_games/` directory in Git (it's large), so you must download it separately.

- Open the Google Drive folder link in `raw_games_link.txt`
- Download the `raw_games` folder (or zip) from Drive
- Place it at the repo root so the path is exactly: `raw_games/<game>/...`

After downloading, you should have a structure similar to:

```
raw_games/
	game2/
		game2.csv
		tagged_images/
	game10/
		game10.pgn
		images/
	...
```

`preprocessing.py`:
- Scans `raw_games/` for games in two supported layouts:
	- CSV-labeled games: `<game>.csv` + `tagged_images/`
	- PGN-only games: `<game>.pgn` + `images/` (frames are auto-aligned to moves using simple motion peaks)
- Builds `final_dataset/{train,val,test}/<class>/*.png` directly by slicing each frame into 8×8 overlapping tiles
- Adds PGN-only generated frames to **train only** (keeps val/test as strictly CSV-labeled)

Run:

```powershell
python preprocessing.py
```

Expected inputs:
- CSV-labeled games:
	- `raw_games/<game>/<game>.csv` (must include columns `from_frame` and `fen`)
	- `raw_games/<game>/tagged_images/frame_XXXXXX.jpg`

- PGN-only games:
	- `raw_games/<game>/<game>.pgn`
	- `raw_games/<game>/images/frame_XXXXXX.(jpg|png)`

Notes:
- If you have a zip like `raw_games/<game>_per_frame.zip`, extract it to `raw_games/<game>/...` first. The helper `unzip_folder()` exists in `preprocessing.py`, and there's also `unzip.py` you can use for safe extraction.
- The script prints per-game progress (CSV frame counts and PGN diff progress).

### Training

Prerequisite: run `python preprocessing.py` first so you have:
- `final_dataset/train/<class>/*.png`
- `final_dataset/val/<class>/*.png`

`train.py` trains a ResNet18 classifier (ImageNet-pretrained backbone) on `final_dataset/train` and `final_dataset/val`, then saves:
- `chess_model.pth` (PyTorch `state_dict`)
- `learning_curves.png`

Notes:
- The first time you run training, torchvision may download ResNet18 pretrained weights (requires internet) unless they are already cached.
- Class order comes from `torchvision.datasets.ImageFolder` (alphabetical by folder name). If you rename class folders, retrain.
- Hyperparameters like `num_epochs`, `batch_size`, and learning rate are currently set inside `train.py`.

Pretrained weights in this repo (optional):
- `chess_model_with_pgn.pth`
- `chess_model_without_pgn.pth`
- `chess_model_koral.pth`

Model file formats:
- `train.py` saves a plain `state_dict`.
- Some provided `.pth` files may be *checkpoint dicts* that include metadata keys like `epoch`, `model_state`, `optim_state`, `history`, `classes`.
- `predict.py` was updated to handle both formats (it extracts `model_state`/`state_dict` automatically).

Compatibility note:
- `evaluate.py` currently expects a plain `state_dict` file (like `chess_model.pth`). If you want to evaluate a checkpoint dict, either use `chess_model.pth` or update `evaluate.py` to extract `model_state` similarly to `predict.py`.

Run:

```powershell
python train.py
```

### Evaluation + OOD (low-confidence) count

`evaluate.py` evaluates the model on `final_dataset/test` and shows a confusion matrix.

Run:

```powershell
python evaluate.py
```

### Predict FEN from an image

`predict.py` loads a model, slices an input image into 8×8 tiles (same overlap idea as training), predicts a class per tile, and outputs a FEN string.

By default it will auto-search for a model in this order (first match wins):
- `models/chess_model.pth`, `chess_model.pth`
- `models/chess_model_without_pgn.pth`, `chess_model_without_pgn.pth`
- `models/chess_model_with_pgn.pth`, `chess_model_with_pgn.pth`
- `models/chess_model_koral.pth`, `chess_model_koral.pth`

1) Edit the image path inside the script (`img_path` near the bottom).
2) Run (recommended on Windows):

```powershell
python .\predict.py
```

Output:
- Printed `Predicted FEN: ...`
- `results/final_board_<image_name>.png` (board visualization with red X marks on low-confidence squares)

## Notes

- If you see import errors in VS Code/Pylance, make sure VS Code is using the same conda environment (`chessboard-squares`).
- Folder order matters: `predict.py` expects `CLASSES` to match the class folder ordering used by `torchvision.datasets.ImageFolder` (usually alphabetical by folder name).
