"""Train the small scan-domain digit CNN (matches predict_digit.py arch).

Glyph crops live in /tmp/scan_digits_dir/{train,valid}/{0..9} at native
resolutions (tens/ones bbox crops). Uses the same 28x28 resize + normalize
as predict_digit.py / recognize_cards._render_glyph so scans run through
the same preprocessing as training.
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix


class DigitClassifier(nn.Module):
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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
    train_ds = ImageFolder("/tmp/scan_digits_dir/train", train_tf)
    test_ds = ImageFolder("/tmp/scan_digits_dir/valid", test_tf)
    print("classes:", train_ds.class_to_idx)
    print(f"train={len(train_ds)} valid={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    epochs = 25
    for epoch in range(epochs):
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
        print(f"epoch {epoch + 1:2d}/{epochs} loss={loss_tot/tot:.4f} "
              f"acc={corr/tot:.4f}")

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

    torch.save(model.state_dict(), "digit_classifier.pth")
    print("saved digit_classifier.pth")


if __name__ == "__main__":
    main()