from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


FILES = [
    Path("digits/test/3/row0_col0.jpg"),
    Path("digits/test/5/row3_col3.jpg"),
    Path("digits/test/6/row3_col4 copy.jpg"),
    Path("digits/test/9/row1_col0.jpg"),
]

def inspect(filename):
    image = Image.open(filename).convert("L")
    pixels = np.array(image)

    # Same threshold used by digit_preprocess.py
    mask = pixels < 150

    rows, cols = np.where(mask)

    if len(rows) == 0:
        raise ValueError(f"Couldn't find digit in {filename}")

    top = rows.min()
    bottom = rows.max() + 1
    left = cols.min()
    right = cols.max() + 1

    # Make a copy of the original and draw the bounding box on it.
    boxed = image.copy()
    boxed_pixels = np.array(boxed)

    # Draw the bounding box in black.
    boxed_pixels[top, left:right] = 0
    boxed_pixels[bottom - 1, left:right] = 0
    boxed_pixels[top:bottom, left] = 0
    boxed_pixels[top:bottom, right - 1] = 0

    boxed = Image.fromarray(boxed_pixels)

    # Reproduce the final preprocessing.
    digit = image.crop((left, top, right, bottom))
    digit.thumbnail((22, 22))

    result = Image.new("L", (28, 28), 255)

    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2

    result.paste(digit, (x, y))

    return image, mask, boxed, result, (left, top, right, bottom)


def main():

    for filename in FILES:

        image, mask, boxed, result, bbox = inspect(filename)

        print()
        print(filename)
        print(f"Bounding box: {bbox}")
        print(
            f"Bounding box size: "
            f"{bbox[2] - bbox[0]} x {bbox[3] - bbox[1]}"
        )

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Threshold mask")
        axes[1].axis("off")

        axes[2].imshow(boxed, cmap="gray")
        axes[2].set_title("Bounding box")
        axes[2].axis("off")

        axes[3].imshow(result, cmap="gray")
        axes[3].set_title("Final 28×28")
        axes[3].axis("off")

        plt.suptitle(filename.name)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()