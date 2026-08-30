"""Whole-cell -> bingo-number (75-class) phone reader.

Phone cells routinely have fused digit pairs, so digit-splitting is
unreliable; instead classify each cell directly as its 1..75 number.
This script
  1. builds a labeled whole-cell set from phone_sheets/ by cross-referencing
     the scan ground truth (blank cells are dropped),
  2. trains a 75-way CNN (DigitClassifier arch, last layer 75),
  3. reports per-class accuracy on the held-out sheet.

Because bingo numbers are sparse per class, expect modest accuracy at low
data volumes; scaling photo count (30 sheets) is the plan.
"""

import re
import sys
from pathlib import Path

import cv2
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent))

from predict_digit import DigitClassifier  # noqa: E402
from evaluate_photo_sheets import load_photo, normalize  # noqa: E402
from evaluate_scan_cells import parse_truth  # noqa: E402
from photo_sheet_cells import sheet_to_cells_teal  # noqa: E402

PHONE = Path(__file__).parent / "phone_sheets"
PHONE3 = Path(__file__).parent / "phone_sheets3"
DST = Path("/tmp/phone_cells_v2")
FREE = (2, 2)
SPLITS = {"train": range(1, 25), "valid": range(25, 31)}
# (source dir, sheet->include pairs to skip, prefix for filenames)
BATCHES = [(PHONE, {}, "A"), (PHONE3, {1}, "B")]

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.06, 0.06),
                            scale=(0.92, 1.08)),
    transforms.RandomResizedCrop((48, 48), scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
test_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def build():
    import shutil
    shutil.rmtree(DST, ignore_errors=True)
    truth = parse_truth()
    stats = {"train": {"ok": 0, "blank": 0, "no_truth": 0},
             "valid": {"ok": 0, "blank": 0, "no_truth": 0}}
    for src, skip, pre in BATCHES:
        for path in sorted(src.glob("*.jpeg")):
            m = re.search(r"(\d+)(?=\.jpe?g$)", path.name)
            if not m:
                continue
            s = int(m.group(1))
            if s in skip:
                print(f"== SKIP {path.name} (bad frame)")
                continue
            split = next((k for k, nums in SPLITS.items() if s in nums), None)
            cells = sheet_to_cells_teal(normalize(load_photo(path)))
            by_card = {}
            for cid, r, c, cell in cells:
                by_card.setdefault(cid, {})[(r, c)] = cell
            print(f"== {path.name} [{pre}]: sheet {s} -> {split}, cards {sorted(by_card)}")
            for cid in sorted(by_card):
                gt = truth.get((s, cid))
                if gt is None:
                    print(f"  card{cid}: no truth")
                    continue
                for r in range(5):
                    for c in range(5):
                        if (r, c) == FREE:
                            continue
                        cell = by_card[cid].get((r, c))
                        if cell is None:
                            stats[split]["blank"] += 1
                            continue
                        g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                        if (g < 150).mean() < 0.01:
                            stats[split]["blank"] += 1
                            continue
                        num = gt[r][c]
                        d = DST / split / str(num)
                        d.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(d / f"{pre}_{path.stem}_c{cid}_r{r}c{c}.jpg"),
                                    cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY))
                        stats[split]["ok"] += 1
    for k, v in stats.items():
        print(f"{k}: {v}")


def main():
    build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class CellClassifier(DigitClassifier):
        def __init__(self, ncls):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 9, 64),
                nn.ReLU(),
                nn.Linear(64, ncls),
            )

    class TwoHeadClassifier(DigitClassifier):
        def __init__(self):
            super().__init__()
            self.tens_head = nn.Linear(64, 8)
            self.ones_head = nn.Linear(64, 10)

        def forward(self, x):
            f = self.features(x)
            f = f.flatten(1)
            f = nn.functional.relu(self.bottleneck(f))
            return self.tens_head(f), self.ones_head(f)

        @property
        def bottleneck(self):
            return self.classifier[1]

    two_head = False
    saved = Path(__file__).parent / "cell_classifier_phone.pth"
    if two_head:
        model = TwoHeadClassifier().to(device)
    else:
        for split in SPLITS:
            if not (DST / split).exists():
                continue
            ds = ImageFolder(DST / split, test_tf)
            classes = sorted(int(c) for c in ds.classes)
            print(f"{split}: {len(ds)} cells, class range {min(classes)}..{max(classes)}")
            break
        model = CellClassifier(ncls=max(int(c)
                                        for c in ImageFolder(
                                            DST / "train").classes) + 1).to(device)
    train_ds = ImageFolder(DST / "train", train_tf)
    valid_ds = ImageFolder(DST / "valid", test_tf)
    tl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    vl = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    for epoch in range(60):
        model.train()
        tot = corr = 0
        for imgs, labs in tl:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            if two_head:
                t, o = model(imgs)
                loss = (nn.functional.cross_entropy(t, labs // 10)
                        + nn.functional.cross_entropy(o, labs % 10))
            else:
                loss = nn.functional.cross_entropy(model(imgs), labs)
            loss.backward()
            opt.step()
            with torch.no_grad():
                if two_head:
                    t, o = model(imgs)
                    pred = torch.where(t.argmax(1) == 0, o.argmax(1),
                                       10 * t.argmax(1) + o.argmax(1))
                    corr += (pred == labs).sum().item()
                else:
                    corr += (model(imgs).argmax(1) == labs).sum().item()
            tot += labs.size(0)
        sch.step()
        print(f"epoch {epoch + 1:2d} train acc={corr / tot:.4f}")
    model.eval()
    ys, pr = [], []
    with torch.no_grad():
        for imgs, labs in vl:
            if two_head:
                t, o = model(imgs.to(device))
                pred = torch.where(t.argmax(1) == 0, o.argmax(1),
                                   10 * t.argmax(1) + o.argmax(1))
            else:
                pred = model(imgs.to(device)).argmax(1)
            pr += pred.cpu().tolist()
            ys += labs.tolist()
    print(classification_report(ys, pr, zero_division=0))
    torch.save(model.state_dict(), saved)
    print(f"saved -> {saved}")


if __name__ == "__main__":
    main()