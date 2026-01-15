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
- Some provided `.pth` files may be _checkpoint dicts_ that include metadata keys like `epoch`, `model_state`, `optim_state`, `history`, `classes`.
- `predict.py` was updated to handle both formats (it extracts `model_state`/`state_dict` automatically).

#### Training on Google Colab (GPU) using `train_colab.py`

If you want faster training (GPU), use `train_colab.py`. It is designed to:

- Run on Colab with a GPU runtime
- Save outputs persistently to Google Drive (so you don't lose them when the session ends)

1. Open a Colab notebook and enable GPU:

- **Runtime → Change runtime type → Hardware accelerator: GPU**

2. Get the project code into Colab (one option):

```bash
git clone https://github.com/yuvalku/Chessboard-Square-Classification.git
cd Chessboard-Square-Classification
git checkout yuval_new
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Make sure `final_dataset/` exists in the repo folder on Colab.

`train_colab.py` expects these folders to exist **next to the script**:

- `final_dataset/train/...`
- `final_dataset/val/...`

If you created `final_dataset/` locally, upload/copy it into the Colab workspace repo folder.

5. Run training:

```bash
python train_colab.py
```

Outputs:

- If Drive is available, files are saved under: `/content/drive/MyDrive/chessboard_models/`
- Filenames are timestamped, e.g. `chess_model_YYYYMMDD_HHMMSS.pth`, `checkpoint_last_YYYYMMDD_HHMMSS.pth`, `learning_curves_YYYYMMDD_HHMMSS.png`

Notes:

- `checkpoint_last_*.pth` is a checkpoint dict (contains `epoch`, `model_state`, `optim_state`, `history`, `classes`).
- `chess_model_*.pth` is a plain `state_dict`.

Run:

```powershell
python train.py
```

### Predict FEN from an image

`predict.py` loads a model, slices an input image into 8×8 tiles (same overlap idea as training), predicts a class per tile, and outputs a FEN string.

1. Edit the model path inside the script (`model_path` near the bottom).

   - Default: `models/chess_model_with_pgn.pth`
   - You can point it to any `.pth` file you trained/downloaded.
   - The loader supports both:
     - a plain PyTorch `state_dict`, and
     - a checkpoint dict containing keys like `model_state`, `epoch`, `optim_state`, `history`, `classes`.

2. Edit the image path inside the script (`img_path` near the bottom).
3. Run (recommended on Windows):

```powershell
python .\predict.py
```

Output:

- Printed `Predicted FEN: ...`
- `results/final_board_<image_name>.png` (board visualization with red X marks on low-confidence squares)

## Notes

- If you see import errors in VS Code/Pylance, make sure VS Code is using the same conda environment (`chessboard-squares`).
- Folder order matters: `predict.py` expects `CLASSES` to match the class folder ordering used by `torchvision.datasets.ImageFolder` (usually alphabetical by folder name).
