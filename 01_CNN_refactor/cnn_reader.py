"""Phone-photo bingo sheet reader using the whole-cell CNN.

Pipeline:
  EXIF-correct decode -> exposure-normalize (color-preserving) ->
  teal-band geometry (sheet_to_cells_teal) -> per-cell 75-way logits
  from cell_classifier_phone.pth -> column-constrained decoding.

Column constraints (true for every bingo card):
  B:1-15, I:16-30, N:31-45, G:46-60, O:61-75, with 5 UNIQUE numbers
  per column.  Decoding maximizes the sum of log-softmax scores subject
  to uniqueness via linear assignment in each column.
"""

import io
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageOps
from torchvision import transforms

import photo_sheet_cells

COLUMN_RANGES = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75)]
FREE = (2, 2)
_SCAN_WIDTH = 1257

# ImageFolder sorted classes lexicographically ("1","10","11",...), so the
# model's output index does NOT equal the number.  Map between them.
_CLASS_STR = sorted(str(n) for n in range(1, 76))
IDX_TO_NUM = [int(c) for c in _CLASS_STR]
NUM_TO_IDX = {int(c): i for i, c in enumerate(_CLASS_STR)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cell_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def _normalize(img):
    """Downscale to scan scale; stretch luminance so paper is bright while
    preserving hue (teal band detection needs color)."""
    H, W = img.shape[:2]
    scale = _SCAN_WIDTH / W
    r = img
    if abs(scale - 1.0) > 0.02:
        r = cv2.resize(img, (_SCAN_WIDTH, int(round(H * scale))),
                       interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(r, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(np.float32)
    lo, hi = np.percentile(y, (2, 98))
    y2 = np.clip((y - lo) * (255.0 / max(1.0, hi - lo)), 0, 255)
    yuv[:, :, 0] = y2.astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


class _ConvStack(nn.Module):
    """Feature extractor identical to predict_digit.DigitClassifier."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )

    def forward(self, x):
        return self.features(x)


class CellClassifier(_ConvStack):
    def __init__(self, ncls=76):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9, 64),
            nn.ReLU(),
            nn.Linear(64, ncls),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(path=None):
    """Load the whole-cell phone reader (76 classes: index 0 unused)."""
    path = Path(path or Path(__file__).parent / "cell_classifier_phone.pth")
    model = CellClassifier(76).to(DEVICE)
    model.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def load_photo_bytes(data):
    """EXIF-correct decode of raw JPEG bytes -> BGR."""
    im = ImageOps.exif_transpose(
        Image.open(io.BytesIO(data))).convert("RGB")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def cells_from(sheet_bgr):
    """Full-sheet BGR (normalized by caller) -> {cid: {(r, c): cell}}."""
    items = photo_sheet_cells.sheet_to_cells_teal(sheet_bgr)
    by = {}
    for cid, r, c, cell in items:
        by.setdefault(cid, {})[(r, c)] = cell
    return by


def cell_logits(model, cell_bgr):
    """(75,) float32 logits for one BGR cell."""
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(gray)
    x = cell_transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)[0].cpu().numpy().astype(np.float64)
    return logits


def decode_cell_args(logits, col):
    """Raw argmax number for a cell (1..75), no constraints."""
    idx = int(np.argmax(logits))
    n = IDX_TO_NUM[idx] if idx < len(IDX_TO_NUM) else -1
    lo, hi = COLUMN_RANGES[col]
    valid = lo <= n <= hi
    return n, valid


def decode_card(logits_grid, conf_threshold=0.0):
    """logits_grid[5][5] -> (numbers[5][5], conf[5][5]).

    Per column, solve a linear assignment between the 5 non-free cells and
    the 15 legal numbers so each number is used at most once, maximizing
    total log-probability.  Empty dict placeholders decode to None.
    """
    from scipy.optimize import linear_sum_assignment

    numbers = [[None] * 5 for _ in range(5)]
    confs = [[0.0] * 5 for _ in range(5)]
    for c in range(5):
        lo, hi = COLUMN_RANGES[c]
        cand = list(range(lo, hi + 1))
        rows = [r for r in range(5) if (r, c) != FREE and
                logits_grid.get((r, c)) is not None]
        if not rows:
            continue
        logit_for = {}
        for r in rows:
            lg = logits_grid[(r, c)]
            logit_for[r] = np.array([lg[NUM_TO_IDX[n]] for n in cand])
        cost = np.stack([logit_for[r] for r in rows])
        r_idx, n_idx = linear_sum_assignment(-cost)
        for ri, ni in zip(r_idx, n_idx):
            r = rows[ri]
            numbers[r][c] = cand[ni]
            z = logit_for[r] - logit_for[r].max()
            probs = np.exp(z)
            probs /= probs.sum()
            confs[r][c] = float(probs[ni])
    numbers[2][2] = None
    confs[2][2] = 0.0
    return numbers, confs


def card_result(numbers, confs):
    """Shape into the server /scan-card per-cell dict format."""
    grid = []
    for r in range(5):
        row = []
        for c in range(5):
            if (r, c) == FREE:
                row.append({"value": None, "is_free_space": True,
                            "valid": True, "raw": "FREE",
                            "confidence": 100.0})
            else:
                v = numbers[r][c]
                row.append({"value": v, "is_free_space": False,
                            "valid": v is not None,
                            "raw": str(v) if v is not None else None,
                            "confidence": round(100 * confs[r][c], 1)})
        grid.append(row)
    needs_review = any(cell["value"] is None
                       for row in grid for cell in row
                       if not cell["is_free_space"])
    return {"grid": grid, "needs_review": needs_review}


def read_sheet_bytes(model, data):
    """Read a full photo (raw bytes) -> server-shaped /scan-card payload."""
    return _read_sheet(model, _normalize(load_photo_bytes(data)))


def read_sheet_path(model, path):
    """Read a full photo from disk -> server-shaped /scan-card payload."""
    return _read_sheet(model, _normalize(load_photo_bytes(
        Path(path).read_bytes())))


def _read_sheet(model, sheet_bgr):
    """sheet_bgr (normalized full-sheet BGR) -> /scan-card payload."""
    by_card = cells_from(sheet_bgr)
    if not by_card:
        return {"cards": [], "error": "no teal bands detected"}
    cards = []
    for cid in sorted(by_card):
        lgrid = {}
        for (r, c), cell in by_card[cid].items():
            lgrid[(r, c)] = cell_logits(model, cell)
        numbers, confs = decode_card(lgrid)
        cards.append(card_result(numbers, confs))
    return {"cards": cards}


if __name__ == "__main__":
    model = load_model()
    for arg in sys.argv[1:]:
        result = read_sheet_path(model, arg)
        result.pop("cards", None)
        print(json.dumps(result, indent=2))