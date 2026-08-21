from pathlib import Path
import csv
import random
import subprocess

import matplotlib.pyplot as plt
from PIL import Image


INPUT_DIR = Path("../debug_output")
LABEL_FILE = Path("labels.csv")

SKIP_FILENAME = "row2_col2.jpg"


def find_cell_images():
    """Recursively find all JPG files inside directories named 'cells'."""

    files = []

    for cells_dir in INPUT_DIR.rglob("cells"):
        if not cells_dir.is_dir():
            continue

        for filename in cells_dir.glob("*.jpg"):
            if filename.name == SKIP_FILENAME:
                continue

            files.append(filename)

    return sorted(files)


def load_existing_labels():
    """Load labels.csv if it already exists."""

    labels = {}

    if not LABEL_FILE.exists():
        return labels

    with LABEL_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            labels[row["source"]] = row["digits"]

    return labels


def save_label(filename, digits):
    """Append one label to labels.csv."""

    file_exists = LABEL_FILE.exists()

    with LABEL_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["source", "digits"])

        writer.writerow([
            str(filename),
            digits
        ])


def display_image(filename, position, total):
    """Display one cell image and close it when a key is pressed."""

    image = Image.open(filename)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(image, cmap="gray")
    ax.axis("off")

    ax.set_title(
        f"{position} / {total}\n"
        f"{filename}\n\n"
        "Press any key to continue"
    )

    plt.tight_layout()

    def on_key(event):
        plt.close(fig)

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key
    )

    plt.show()

    # Return focus to macOS Terminal.
    subprocess.run([
        "osascript",
        "-e",
        'tell application "Terminal" to activate'
    ])
        

def main():

    if not INPUT_DIR.exists():
        print(
            f"ERROR: Input directory does not exist: "
            f"{INPUT_DIR}"
        )
        return

    files = find_cell_images()
    random.seed(42)
    random.shuffle(files)

    print(f"Found {len(files)} cell images.")

    if not files:
        return

    existing_labels = load_existing_labels()

    print(
        f"Already labeled: "
        f"{len(existing_labels)}"
    )

    remaining = [
        filename
        for filename in files
        if str(filename) not in existing_labels
    ]

    print(
        f"Remaining: {len(remaining)}"
    )

    print()
    print("Instructions:")
    print("  Enter the number shown in the cell.")
    print("  Examples: 52, 13, 7")
    print("  Enter 's' to skip an image.")
    print("  Enter 'q' to quit.")
    print()

    for count, filename in enumerate(
        remaining,
        start=1
    ):

        display_image(
            filename,
            count,
            len(remaining)
        )

        while True:

            answer = input(
                "Number in this cell: "
            ).strip()

            if answer.lower() == "q":
                print("Quitting.")
                return

            if answer.lower() == "s":
                print("Skipped.")
                break

            # Make sure the answer consists only
            # of digits.
            if not answer.isdigit():
                print(
                    "Please enter a number, "
                    "'s', or 'q'."
                )
                continue

            # Bingo numbers should be 1-75.
            number = int(answer)

            if not 1 <= number <= 75:
                print(
                    "Please enter a Bingo number "
                    "between 1 and 75."
                )
                continue

            save_label(
                filename,
                answer
            )

            print(
                f"Saved: {filename} → {answer}"
            )

            break

    print()
    print("Labeling complete.")


if __name__ == "__main__":
    main()