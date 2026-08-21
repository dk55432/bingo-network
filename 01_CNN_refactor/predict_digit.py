import sys

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
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
