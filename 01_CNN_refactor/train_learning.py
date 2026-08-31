"""Fine-tune the phone-cell CNN on user-confirmed cells.

The scanning server (bingo_scan.py) pairs each reviewed/corrected photo
scan's cell crops with the confirmed numbers and persists one labeled
image per cell into ../learning_cells/<number>/.  This script:
  1. rebuilds the base dataset (train_phone_cells.build),
  2. merges the learning cells into the train split,
  3. fine-tunes cell_classifier_phone.pth (a few epochs, low LR so the
     base model's knowledge isn't destroyed),
  4. saves the updated checkpoint (previous one kept as .bak).

The ImageFolder class ordering matches how cnn_reader maps logits back
to numbers: ImageFolder.classes is lexicographically sorted, so class
index i corresponds to the i-th smallest label, with index 0 unused.
"""

import datetime as _dt
import shutil
import sys
import tarfile
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).parent))

from train_phone_cells import (  # noqa: E402
    CellClassifier,
    DST,
    build,
    test_tf,
    train_tf,
)

LEARNING = Path(__file__).parent / "learning_cells"
BACKUP_DIR = Path(__file__).parent / "learning_backups"
CKPT = Path(__file__).parent / "cell_classifier_phone.pth"
EPOCHS = 30
LR = 1e-4
BATCH = 32
# Stop early when valid accuracy hasn't improved in this many consecutive
# epochs.  Fine-tuning on a small confirmed corpus, the model plateaus after
# ~15-20 epochs and longer runs risk overfitting; this makes epoch count
# self-tuning instead of a hand-picked number.
PATIENCE = 6
VALID_EVERY = 1


def backup_learning():
    """Snapshot learning_cells into learning_backups/learning_<ts>.tar.gz
    so the confirmed corpus can be restored/reused later, e.g. for longer
    training runs.  The backup archives the labels via the directory
    structure (learning_cells/<number>/<cell>.jpg).  Keeps the last 20."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not LEARNING.is_dir():
        return None
    cells = list(LEARNING.rglob("*.jpg"))
    if not cells:
        return None
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"learning_{ts}.tar.gz"
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(LEARNING, arcname=LEARNING.name)
    arch = sorted(BACKUP_DIR.glob("learning_*.tar.gz"))
    for old in arch[:-20]:
        old.unlink()
    return dest


def merge_learning():
    """Copy confirmed cells into DST/train/<number>/.  Filenames embed the
    scan id, so re-runs are idempotent and different scans never clash."""
    n = 0
    for num_dir in sorted(LEARNING.iterdir()):
        if not num_dir.is_dir():
            continue
        dst = DST / "train" / num_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in num_dir.iterdir():
            if f.suffix.lower() != ".jpg":
                continue
            target = dst / f.name
            if not target.exists():
                shutil.copy(f, target)
                n += 1
    return n


def main():
    # Line-buffer stdout so that when piped (e.g. nohup ... &> log), each
    # epoch's progress line is flushed immediately instead of at process
    # exit.  This makes long CPU retrains observable/monitorable.
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CKPT.exists():
        raise SystemExit(f"no checkpoint to fine-tune from: {CKPT}")

    build()
    backup = backup_learning()
    print(f"learning cells backed up -> {backup}")
    merged = merge_learning()
    if merged == 0:
        print("WARNING: no confirmed learning cells yet — training is a no-"
              "op, just re-saving the current checkpoint.")

    ncls = max(int(c) for c in ImageFolder(DST / "train").classes) + 1
    model = CellClassifier(ncls).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))

    train_ds = ImageFolder(DST / "train", train_tf)
    valid_ds = ImageFolder(DST / "valid", test_tf)
    print(f"train {len(train_ds)} cells (incl. {merged} from confirmations), "
          f"valid {len(valid_ds)}")
    tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    vl = DataLoader(valid_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    def valid_accuracy():
        model.eval()
        corr = tot = 0
        with torch.no_grad():
            for imgs, labs in vl:
                pr = model(imgs.to(device)).argmax(1).cpu()
                corr += (pr == labs).sum().item()
                tot += labs.size(0)
        model.train()
        return corr / tot if tot else 0.0

    best_acc = 0.0
    best_state = None
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot = corr = 0
        for imgs, labs in tl:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(imgs), labs)
            loss.backward()
            opt.step()
            with torch.no_grad():
                corr += (model(imgs).argmax(1) == labs).sum().item()
            tot += labs.size(0)
        sch.step()
        train_acc = corr / tot

        if epoch % VALID_EVERY == 0 or epoch == EPOCHS:
            va = valid_accuracy()
            tag = ""
            if va > best_acc:
                best_acc = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                no_improve = 0
                tag = "  <-- best"
            else:
                no_improve += 1
            print(f"epoch {epoch:2d} train acc={train_acc:.4f} "
                  f"valid acc={va:.4f}{tag}")
            if no_improve >= PATIENCE:
                print(f"no valid-acc improvement for {PATIENCE} epochs "
                      f"(best {best_acc:.4f} @ epoch {best_epoch}); stopping early")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best valid acc {best_acc:.4f} @ epoch {best_epoch}")

    model.eval()
    ys, pr = [], []
    with torch.no_grad():
        for imgs, labs in vl:
            pr += model(imgs.to(device)).argmax(1).cpu().tolist()
            ys += labs.tolist()
    print(classification_report(ys, pr, zero_division=0))

    if CKPT.exists():
        shutil.copy(CKPT, CKPT.with_suffix(".pth.bak"))
    torch.save(model.state_dict(), CKPT)
    print(f"saved -> {CKPT}")


if __name__ == "__main__":
    main()