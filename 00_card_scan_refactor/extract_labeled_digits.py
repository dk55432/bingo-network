from pathlib import Path
import csv

import numpy as np
from PIL import Image
from scipy import ndimage


LABEL_FILE = Path("labels.csv")
OUTPUT_DIR = Path("extracted_digits")


def find_components(image):
    """
    Find connected components that might be digits.

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

    # Find connected components.
    labeled, num_features = ndimage.label(mask)

    if num_features == 0:
        return []

    components = []

    for label_number in range(1, num_features + 1):

        rows, cols = np.where(
            labeled == label_number
        )

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
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": width,
            "height": height,
            "area": area,
        })

    # Sort left-to-right.
    components.sort(
        key=lambda c: c["left"]
    )

    return components


def crop_component(image, component):
    """Crop a component from the original grayscale image."""

    return image.crop(
        (
            component["left"],
            component["top"],
            component["right"],
            component["bottom"],
        )
    )


def preprocess_digit(digit):
    """
    Convert a cropped digit to the same 28x28 format
    used by our classifier.
    """

    # The component itself gives us the crop, so we
    # don't need to threshold again here.

    digit = digit.copy()

    # Resize while preserving aspect ratio.
    digit.thumbnail((22, 22))

    # White 28x28 canvas.
    result = Image.new(
        "L",
        (28, 28),
        255
    )

    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2

    result.paste(
        digit,
        (x, y)
    )

    return result


def make_output_name(source, position, digit):
    """
    Create a useful filename from the source path.

    Example:

    IMG_4901/card_0/row2_col0.jpg

    becomes:

    IMG_4901_card_0_row2_col0_0_5.jpg
    """

    parts = source.parts

    # Find the pieces we're interested in.
    try:
        cells_index = parts.index("cells")

        relevant = parts[
            cells_index - 2:
        ]

    except ValueError:
        relevant = parts

    stem = Path(
        "_".join(relevant)
    ).stem

    return (
        f"{stem}_digit{position}_{digit}.jpg"
    )


def main():

    if not LABEL_FILE.exists():
        print(
            f"ERROR: {LABEL_FILE} not found."
        )
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    processed = 0
    successful = 0
    warnings = 0

    with LABEL_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            source = Path(row["source"])
            number = row["digits"]

            processed += 1

            if not source.exists():
                print(
                    f"WARNING: source does not exist:"
                    f"\n  {source}"
                )
                warnings += 1
                continue

            image = Image.open(
                source
            ).convert("L")

            components = find_components(image)

            expected = len(number)

            if len(components) < expected:

                print(
                    f"WARNING: {source}"
                )

                print(
                    f"  Expected {expected} "
                    f"components for '{number}'"
                )

                print(
                    f"  Found only {len(components)}"
                )

                warnings += 1
                continue

            # Keep only the largest components.
            components = sorted(
                components,
                key=lambda c: c["area"],
                reverse=True
            )

            print(
                f"{source.name}: "
                f"expected={expected}, "
                f"components={len(components)}, "
                f"areas={[c['area'] for c in components[:5]]}"
            )

            components = components[:expected]

            components.sort(
                key=lambda c: c["left"]
            )

            # Components are left-to-right.
            for position, (
                digit,
                component
            ) in enumerate(
                zip(number, components)
            ):

                cropped = crop_component(
                    image,
                    component
                )

                processed_digit = (
                    preprocess_digit(cropped)
                )

                output_name = (
                    make_output_name(
                        source,
                        position,
                        digit
                    )
                )

                output_path = (
                    OUTPUT_DIR
                    / digit
                    / output_name
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                processed_digit.save(
                    output_path
                )

            successful += 1

    print()
    print("Extraction complete")
    print("-------------------")
    print(f"Cells processed: {processed}")
    print(f"Successful:      {successful}")
    print(f"Warnings:        {warnings}")


if __name__ == "__main__":
    main()