import sys

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms


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
        x = self.features(x)
        return self.classifier(x)


test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    ),
])


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = DigitClassifier().to(device)

model.load_state_dict(
    torch.load(
        "digit_classifier.pth",
        map_location=device
    )
)

model.eval()


if __name__ == "__main__":
    filename = sys.argv[1]

    image = Image.open(filename)

    image_tensor = test_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = probabilities.max(dim=1)

    print("File:", filename)
    print("Predicted digit:", prediction.item())
    print("Confidence:", f"{confidence.item():.3f}")
