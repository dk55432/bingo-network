from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


LABEL_FILE = Path("labels.csv")


def find_components(image):
    pixels = np.array(image)

    mask = pixels < 150

    mask = ndimage.binary_opening(
        mask,
        structure=np.ones((3, 3))
    )

    labeled, num_features = ndimage.label(mask)

    components = []

    for label_number in range(
        1,
        num_features + 1
    ):

        rows, cols = np.where(
            labeled == label_number
        )

        if len(rows) == 0:
            continue

        components.append({
            "label": label_number,
            "left": cols.min(),
            "right": cols.max(),
            "top": rows.min(),
            "bottom": rows.max(),
            "area": len(rows),
        })

    components.sort(
        key=lambda c: c["area"],
        reverse=True
    )

    return mask, components


def inspect(filename, number):

    image = Image.open(filename).convert("L")

    mask, components = find_components(
        image
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    # Original
    axes[0].imshow(
        image,
        cmap="gray"
    )
    axes[0].set_title(
        f"Original\n{filename}"
    )
    axes[0].axis("off")

    # Mask
    axes[1].imshow(
        mask,
        cmap="gray"
    )
    axes[1].set_title(
        f"Threshold mask\n"
        f"Expected: {number}\n"
        f"Found: {len(components)}"
    )
    axes[1].axis("off")

    # Components
    axes[2].imshow(
        image,
        cmap="gray"
    )

    axes[2].set_title(
        "Connected components"
    )

    for index, component in enumerate(
        components
    ):

        x = component["left"]
        y = component["top"]
        w = (
            component["right"]
            - component["left"]
        )
        h = (
            component["bottom"]
            - component["top"]
        )

        rectangle = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            linewidth=2
        )

        axes[2].add_patch(
            rectangle
        )

        axes[2].text(
            x,
            y,
            str(index + 1),
            fontsize=14,
            backgroundcolor="white"
        )

    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def main():

    with LABEL_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        failures = []

        for row in reader:

            number = row["digits"]

            source = Path(
                row["source"]
            )

            if not source.exists():
                continue

            image = Image.open(
                source
            ).convert("L")

            _, components = find_components(
                image
            )

            if len(components) != len(number):

                failures.append(
                    (source, number)
                )

    print(
        f"Found {len(failures)} "
        f"failed extractions."
    )

    for index, (filename, number) in enumerate(
        failures,
        start=1
    ):

        print(
            f"\nFailure {index} "
            f"of {len(failures)}"
        )

        inspect(
            filename,
            number
        )

        answer = input(
            "Press Enter for next, "
            "or q to quit: "
        )

        if answer.lower() == "q":
            break


if __name__ == "__main__":
    main()