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

### 3) Quick sanity check

```powershell
python -c "import torch; import torchvision; import cv2; import numpy; import pandas; import PIL; import sklearn; import matplotlib; import seaborn; import chess; print('OK')"
```

## Project scripts

On Windows/PowerShell, run the scripts as `python .\<script>.py` (for example `python .\preprocessing.py`).

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

`python .\preprocessing.py`:

- Scans `raw_games/` for games in two supported layouts:
  - CSV-labeled games: `<game>.csv` + `tagged_images/`
  - PGN-only games: `<game>.pgn` + `images/` (frames are auto-aligned to moves using simple motion peaks)
- Builds `final_dataset/{train,val,test}/<class>/*.png` directly by slicing each frame into 8×8 overlapping tiles
- Adds PGN-only generated frames to **train only** (keeps val/test as strictly CSV-labeled)

Run:

```powershell
python .\preprocessing.py
```

Expected inputs:

- CSV-labeled games:

  - `raw_games/<game>/<game>.csv` (must include columns `from_frame` and `fen`)
  - `raw_games/<game>/tagged_images/frame_XXXXXX.jpg`

- PGN-only games:
  - `raw_games/<game>/<game>.pgn`
  - `raw_games/<game>/images/frame_XXXXXX.(jpg|png)`

Notes:

- Make sure your `raw_games/<game>/...` folders are already extracted/unzipped (the preprocessing script expects folders on disk).
- The script prints per-game progress (CSV frame counts and PGN diff progress).

### Training

Prerequisite: run `python .\preprocessing.py` first so you have:

- `final_dataset/train/<class>/*.png`
- `final_dataset/val/<class>/*.png`

`train.py` trains a ResNet18 classifier (ImageNet-pretrained backbone) on `final_dataset/train` and `final_dataset/val`, then saves:

- Timestamped artifacts under `models/` (or Google Drive on Colab), including:
  - `chess_model_<run_id>.pth` (PyTorch `state_dict`)
  - `checkpoint_last_<run_id>.pth` (checkpoint dict with epoch/optimizer/history/classes)
  - `learning_curves_<run_id>.png`
  - `train_log_<run_id>.txt`

Notes:

- The first time you run training, torchvision may download ResNet18 pretrained weights (requires internet) unless they are already cached.
- Class order comes from `torchvision.datasets.ImageFolder` (alphabetical by folder name). If you rename class folders, retrain.
- Hyperparameters like `num_epochs`, `batch_size`, and learning rate are currently set inside `train.py`.

#### Training results (report)

The latest training figures and summary text used for the report are stored in `for_report/`:

- Learning curves:

<img src="for_report/learning_curves.png" alt="Training learning curves" width="700" />

- Confusion matrix:

<img src="for_report/evaluation_confusion_matrix.png" alt="Evaluation confusion matrix" width="600" />

Model file formats:

- `train.py` saves a plain `state_dict`.
- Some provided `.pth` files may be _checkpoint dicts_ that include metadata keys like `epoch`, `model_state`, `optim_state`, `history`, `classes`.
- `predict.py` was updated to handle both formats (it extracts `model_state`/`state_dict` automatically).

#### Training on Google Colab (GPU) using `train_model.ipynb`

If you want faster training (GPU), use the Colab notebook `train_model.ipynb`. It unzips a prepared project folder from Google Drive and runs training.

This repo’s recommended Colab workflow is **Drive-first**:

1. Prepare a folder locally: `chessboard-project-colab/`

Create a folder named exactly:

```
chessboard-project-colab/
```

Inside it, put:

- Open the `chessboard-project-colab/` folder.
- Copy `train.py` from this repo into `chessboard-project-colab/`.
- Place your prepared `final_dataset/` folder inside `chessboard-project-colab/`.

So the structure looks like:

```
chessboard-project-colab/
  train.py
  final_dataset/
    train/
      <class_name>/
        *.png
    val/
      <class_name>/
        *.png
    test/            # optional for training
      <class_name>/
        *.png
```

`<class_name>` must be the same 13 folders used by this project (12 pieces + `empty`), for example:

```
black_bishop, black_king, black_knight, black_pawn, black_queen, black_rook,
white_bishop, white_king, white_knight, white_pawn, white_queen, white_rook,
empty
```

2. Create `final_dataset/`

You have two options:

- Recommended: generate it locally by running:

```powershell
python .\preprocessing.py
```

This creates `final_dataset/train`, `final_dataset/val`, `final_dataset/test` automatically.

- Manual: create the folder structure shown above and place the square images yourself.

3. Zip and upload to Google Drive

- Zip the whole `chessboard-project-colab/` folder (so the zip contains the folder, not just its contents).
- Upload the zip to your Google Drive (for example to `MyDrive/`).

4. Train in Colab using `train_model.ipynb`

Open `train_model.ipynb` in Google Colab. The notebook mounts Drive, unzips `chessboard-project-colab.zip` into `/content/`, and runs the trainer.

- Mount Drive
- Unzip `chessboard-project-colab.zip` into `/content/`
- Install dependencies
- Run `python /content/chessboard-project-colab/train.py`

Make sure GPU is enabled:

- **Runtime → Change runtime type → Hardware accelerator: GPU**

Outputs:

- Models and curves are saved to Drive (default): `/content/drive/MyDrive/chessboard_models/`
- Filenames are timestamped, e.g. `chess_model_YYYYMMDD_HHMMSS.pth`, `checkpoint_last_YYYYMMDD_HHMMSS.pth`, `learning_curves_YYYYMMDD_HHMMSS.png`

Notes:

- `checkpoint_last_*.pth` is a checkpoint dict (contains keys like `epoch`, `model_state`, `optim_state`, `history`, `classes`).
- `chess_model_*.pth` is a plain PyTorch `state_dict`.

### Predict FEN from an image

`predict.py` loads a model, slices an input image into 8×8 tiles (same overlap idea as training), predicts a class per tile, and outputs a FEN string.

#### Expected input image

For best results, the input should be a **single chessboard** image where the board is clearly visible.

- Recommended: **top-down (or near top-down)** view of the board.
- The board should be **roughly centered** and **not heavily perspective-skewed**.
- Avoid hands/pieces-in-motion/occlusions when capturing the image.
- Lighting should be consistent (avoid heavy glare/shadows).

Important: the current pipeline does a simple 8×8 grid crop after resizing the image to `800×800`. It does **not** detect corners / do perspective correction.
So if your photo includes a lot of background or the board is rotated/skewed, grid cells may not line up with real squares.

#### Usage

1. Put your input images into `image_predict_folder/` (the script processes all `*.jpg`/`*.png` files in that folder).
2. Edit the model path inside the script (look for `load_model("chess_model.pth", device)`).
   - You can point it to any `.pth` file you trained/downloaded.
   - The loader supports both:
     - a plain PyTorch `state_dict`, and
     - a checkpoint dict containing keys like `model_state`, `epoch`, `optim_state`, `history`, `classes`.
3. Run (recommended on Windows):

```powershell
python .\predict.py
```

Output:

- Printed predicted board matrix and FEN string for each processed image.

#### `debug_compare` (side-by-side debug output)

`predict.py` supports a `debug_compare` flag when calling `predict_board(...)`.

- If `debug_compare=False`: saves **only the predicted board render** (no input image) to `results/`.
  - Example: `results/pred_game2_frame_002472.png`

<img src="assets/readme/pred_game2_frame_002472.png" alt="debug_compare=False (predicted board only) example" width="350" />

- If `debug_compare=True`: saves a **side-by-side** image (input on the left, predicted board render on the right) to `results/`.
  - Example: `results/debug_game2_frame_002472.jpg`

<img src="assets/readme/debug_game2_frame_002472.jpg" alt="debug_compare=True (side-by-side) example" width="650" />

The predicted board render marks rejected/uncertain squares with a red X.

#### OOD handling (out-of-distribution squares)

During inference, `predict.py` can mark a square as **OOD** (out-of-distribution) when it is likely not a valid/clean chessboard square crop.

A square is marked as OOD (internal label id `13`) when either:

- The crop fails basic visual validity checks (blur/low texture, not enough corner structure / edges), or
- The classifier is not confident enough (low softmax confidence) or is too uncertain (high entropy).

Effects:

- In the **rendered board image**, OOD squares are drawn with a **red X**.
- In the **printed FEN**, OOD squares are treated like **empty squares** (they contribute to the digit counts), so the FEN may be less reliable if many squares are OOD.

You can tune this behavior by adjusting thresholds at the top of `predict.py` (for example `CONF_THRESHOLD` and `ENTROPY_THRESHOLD`).

Tip: if many squares get a red X, try a cleaner/cropped top-down board image.

## Notes

- If you see import errors in VS Code/Pylance, make sure VS Code is using the same conda environment (`chessboard-squares`).
- Folder order matters: `predict.py` expects `CLASSES` to match the class folder ordering used by `torchvision.datasets.ImageFolder` (usually alphabetical by folder name).
