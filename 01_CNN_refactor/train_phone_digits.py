"""Train the phone-domain digit CNN (predict_digit.py arch).

Input: /tmp/phone_digits_dir/{train,valid}/{0..9} (built by
build_phone_digits.py, same 28x28 white-canvas representation as the
inference-time _render_glyph path). Same augmentation as train_scan_digits.
Saves digit_classifier_phone.pth so the scan-domain model stays intact.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from predict_digit import DigitClassifier  # noqa: E402


train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((36, 36)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.06, 0.06),
                            scale=(0.92, 1.08)),
    transforms.RandomResizedCrop((28, 28), scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

test_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--init", type=str, default=None,
                    help="path to .pth to initialize from")
    ap.add_argument("--out", type=str, default="digit_classifier_phone.pth")
    ap.add_argument("--data", type=str, default="/tmp/phone_digits_dir")
    args = ap.parse_args()

    train_ds = ImageFolder(Path(args.data) / "train", train_tf)
    test_ds = ImageFolder(Path(args.data) / "valid", test_tf)
    print("classes:", train_ds.class_to_idx)
    print(f"train={len(train_ds)} valid={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitClassifier().to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"initialized from {args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    crit = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        loss_tot = corr = tot = 0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labs)
            loss.backward()
            opt.step()
            loss_tot += loss.item() * imgs.size(0)
            corr += (out.argmax(1) == labs).sum().item()
            tot += labs.size(0)
        sch.step()
        print(f"epoch {epoch + 1:2d}/{args.epochs} loss={loss_tot/tot:.4f} "
              f"acc={corr/tot:.4f} lr={sch.get_last_lr()[0]:.1e}")

    model.eval()
    ys, pr = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            pr += model(imgs.to(device)).argmax(1).cpu().tolist()
            ys += labs.tolist()
    print("\n" + classification_report(ys, pr, target_names=train_ds.classes))
    print(confusion_matrix(ys, pr))
    acc = sum(a == b for a, b in zip(ys, pr)) / len(ys)
    print(f"\nVALID ACCURACY: {acc:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()