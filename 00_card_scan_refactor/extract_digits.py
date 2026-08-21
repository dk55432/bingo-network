from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


# Change this to wherever your cell images currently live.
INPUT_DIR = Path("cells")

# Extracted digits will go here.
OUTPUT_DIR = Path("extracted_digits")


def find_digit_components(image):
    """
    Find connected components that are likely to be digits.

    Returns components sorted from left to right.
    """

    pixels = np.array(image)

    # Threshold dark pixels.
    mask = pixels < 150

    # Remove small isolated noise.
    mask = ndimage.binary_opening(
        mask,
        structure=np.ones((3, 3))
    )

    # Label connected components.
    labeled, num_features = ndimage.label(mask)

    if num_features == 0:
        return []

    components = []

    for label_number in range(1, num_features + 1):

        rows, cols = np.where(labeled == label_number)

        if len(rows) == 0:
            continue

        top = rows.min()
        bottom = rows.max() + 1
        left = cols.min()
        right = cols.max() + 1

        width = right - left
        height = bottom - top
        area = len(rows)

        components.append({
            "label": label_number,
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": width,
            "height": height,
            "area": area,
        })

    # Sort from left to right.
    components.sort(key=lambda c: c["left"])

    return components


def crop_component(image, component):
    """Crop one connected component from the original grayscale image."""

    return image.crop(
        (
            component["left"],
            component["top"],
            component["right"],
            component["bottom"],
        )
    )


def save_digit(image, output_path):
    """Save an extracted digit."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(output_path)


def process_cell(filename):
    """Process one bingo-cell image."""

    image = Image.open(filename).convert("L")

    components = find_digit_components(image)

    # For now, only accept cells with one or two components.
    if len(components) not in (1, 2):
        print(
            f"WARNING: {filename.name}: "
            f"found {len(components)} components"
        )
        return

    # Get the actual number from the filename.
    #
    # For example:
    #     52.jpg → "52"
    #     8.jpg  → "8"
    number = filename.stem

    # Sanity check.
    if not number.isdigit():
        print(
            f"WARNING: {filename.name}: "
            f"filename doesn't look like a number"
        )
        return

    if len(number) != len(components):
        print(
            f"WARNING: {filename.name}: "
            f"filename says {number}, "
            f"but found {len(components)} components"
        )
        return

    # Save each digit.
    for position, component in enumerate(components):

        digit = number[position]

        cropped = crop_component(
            image,
            component
        )

        output_filename = (
            f"{number}_{position}_{digit}.jpg"
        )

        output_path = (
            OUTPUT_DIR
            / digit
            / output_filename
        )

        save_digit(
            cropped,
            output_path
        )

    print(
        f"{filename.name}: "
        f"extracted {len(components)} digits"
    )


def main():

    if not INPUT_DIR.exists():
        print(
            f"ERROR: Input directory does not exist: "
            f"{INPUT_DIR}"
        )
        return

    files = sorted(
        INPUT_DIR.glob("*.jpg")
    )

    print(
        f"Found {len(files)} cell images."
    )

    for filename in files:
        process_cell(filename)


if __name__ == "__main__":
    main()