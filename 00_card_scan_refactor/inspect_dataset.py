from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from digit_preprocess import preprocess


DATASET_DIR = Path("digits")
SAMPLES_TO_SHOW = 25


def load_dataset(directory):
    images = []
    labels = []

    for digit in range(10):
        digit_dir = directory / str(digit)

        files = sorted(digit_dir.glob("*.jpg"))

        print(f"Digit {digit}: {len(files)} images")

        for filename in files:
            image = preprocess(filename)

            # Verify the preprocessing produced what we expect.
            if image.size != (28, 28):
                raise ValueError(
                    f"{filename} produced image size {image.size}"
                )

            if image.mode != "L":
                raise ValueError(
                    f"{filename} produced mode {image.mode}"
                )

            # Convert PIL image to NumPy array.
            pixels = np.array(image)

            images.append(pixels)
            labels.append(digit)

    return images, labels


def main():
    print("Loading training dataset...\n")

    images, labels = load_dataset(DATASET_DIR / "train")

    print("\nDataset summary")
    print("----------------")
    print(f"Total images: {len(images)}")

    # Convert the list of 28x28 arrays into one 3-D NumPy array.
    images_array = np.array(images)

    print(f"Array shape:   {images_array.shape}")
    print(f"Data type:     {images_array.dtype}")
    print(f"Pixel minimum: {images_array.min()}")
    print(f"Pixel maximum: {images_array.max()}")

    # Show a random sample.
    sample_count = min(SAMPLES_TO_SHOW, len(images))

    indices = random.sample(range(len(images)), sample_count)

    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    axes = axes.flatten()

    for ax, index in zip(axes, indices):
        ax.imshow(images[index], cmap="gray")
        ax.set_title(f"Actual: {labels[index]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
