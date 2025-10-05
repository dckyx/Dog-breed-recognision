This repository implements a dog breed classifier in Python with **PyTorch** (ResNet‑18 backbone) in a modular way:
- `downloader.py`: downloads the Kaggle dataset (70 dog breeds) and copies it to `dane/pieski/`.
- `loader.py`: loads images from class folders into NumPy arrays (`IMG_SIZE=128`), returns `(X, y, categories)`.
- `model.py`: `ResNetCustom(num_classes)` (ResNet‑18, last FC changed to `num_classes`).
- `trainer.py`: training loop with frozen backbone + new FC, saves weights to `modele/cnn_model.pth`.
- `evaluator.py`: accuracy on a test set and a helper to visualize predictions.
- `predictator.py`: single‑image prediction + matplotlib preview.

## Requirements
Python 3.9+ and:
```
torch torchvision
numpy scikit-learn opencv-python pillow matplotlib
kagglehub
```
(Install with `pip install -r requirements.txt` or add the above manually.)

> Note: `model.py` uses `resnet18(pretrained=True)` which is deprecated in Torch 2.x. Prefer `weights=models.ResNet18_Weights.DEFAULT`.

## Data
`downloader.pobierz_dane()` downloads **gpiosenka/70-dog-breedsimage-data-set** via KaggleHub to a temp dir and copies it to `dane/pieski/` preserving the folder structure expected by `loader.zaladuuj()` (class‑per‑folder).


## Important notes & fixes
- **Imports**: `loader.py` imports `from src.downloader import pobierz_dane`. If your files are flat (no `src/` pkg), change it to `from downloader import pobierz_dane`.
- **Normalization**: `loader.py` scales images to `[0,1]` but does **not** apply ImageNet normalization. Keep prediction transforms consistent (avoid `Normalize(ImageNet)` or add the same normalization in both train & predict).
- **Torch 2.x**: replace `resnet18(pretrained=True)` with:
  ```python
  from torchvision import models
  m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
  ```
  and then replace the final `fc` layer.
- **Saved weights**: `trainer.py` saves to `modele/cnn_model.pth`. Use the same path when loading for eval/predict.
- **Device**: all modules select `mps` (Apple), then `cuda`, then `cpu` automatically.
- **Label mapping**: `loader.zaladuuj` returns `categories` (sorted class names). Use this list to map model outputs back to labels.

## Known gaps in uploaded files
- Several files include literal `...` placeholders, so those spots must be filled for the code to run (e.g., training loop details, evaluation loop body, prediction transforms). Fix these before running.
- In `model.py`, the old `pretrained=True` API may warn/fail on latest Torch. Switch to `weights=` as noted.
- Ensure `IMG_SIZE=128` is used consistently in train & predict.
