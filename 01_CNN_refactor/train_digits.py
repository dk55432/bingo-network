import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import confusion_matrix, classification_report


# ------------------------------------------------------------
# 1. Image preprocessing
# ------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomRotation(8),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    ),
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    ),
])


# ------------------------------------------------------------
# 2. Load the image datasets
# ------------------------------------------------------------

train_dataset = ImageFolder(
    "digits/train",
    transform=train_transform
)

test_dataset = ImageFolder(
    "digits/test",
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


# ------------------------------------------------------------
# 3. Define the neural network
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 4. Set up training
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)
print("Classes:", train_dataset.class_to_idx)

model = DigitClassifier().to(device)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ------------------------------------------------------------
# 5. Train the model
# ------------------------------------------------------------

number_of_epochs = 10

for epoch in range(number_of_epochs):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{number_of_epochs} "
        f"- loss: {average_loss:.4f} "
        f"- accuracy: {accuracy:.4f}"
    )


# ------------------------------------------------------------
# 6. Test the model
# ------------------------------------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total

print(f"Test accuracy: {test_accuracy:.4f}")

# ------------------------------------------------------------
# Evaluation code
# ------------------------------------------------------------
all_labels = []
all_predictions = []

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

print("\nClassification report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=train_dataset.classes
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)

# ------------------------------------------------------------
# 7. Save the trained model
# ------------------------------------------------------------

torch.save(model.state_dict(), "digit_classifier.pth")

print("Saved model to digit_classifier.pth")
