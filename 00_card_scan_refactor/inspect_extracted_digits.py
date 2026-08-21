from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

DIGITS_DIR = Path("extracted_digits")

COLUMNS = 4
ROWS = 3
IMAGES_PER_SHEET = COLUMNS * ROWS

BAD_FILE = Path("bad_extracted_digits.txt")


def short_filename(filename):
    """
    Make a readable filename for the contact sheet.

    Example:
        IMG_4901_card_0_cells_row3_col1_digit0_1.jpg

    becomes:
        row3_col1_d0.jpg
    """

    name = filename.stem
    parts = name.split("_")

    row = None
    col = None
    digit = None

    for part in parts:
        if part.startswith("row"):
            row = part
        elif part.startswith("col"):
            col = part
        elif part.startswith("digit"):
            digit = part.replace("digit", "d")

    if row and col and digit:
        return f"{row}_{col}_{digit}.jpg"

    return filename.name


def load_existing_bad_files():
    """Load previously recorded bad filenames."""

    bad_files = set()

    if not BAD_FILE.exists():
        return bad_files

    with BAD_FILE.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                bad_files.add(line)

    return bad_files


def save_bad_files(bad_files):
    """Write the bad-file list in a stable, sorted order."""

    with BAD_FILE.open("w") as f:
        for filename in sorted(bad_files):
            f.write(filename + "\n")


class ContactSheet:
    def __init__(self, digit, files, start_index, bad_files):
        self.digit = digit
        self.files = files
        self.start_index = start_index
        self.bad_files = bad_files

        self.selected = files[
            start_index:start_index + IMAGES_PER_SHEET
        ]

        self.fig, self.axes = plt.subplots(
            ROWS,
            COLUMNS,
            figsize=(9, 7)
        )

        self.axes = self.axes.flatten()

        for ax in self.axes:
            ax.axis("off")

        self.draw()

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key
        )

        self.fig.canvas.mpl_connect(
            "button_press_event",
            self.on_mouse_click
        )

    def draw(self):
        """Draw the current contact sheet."""

        for ax in self.axes:
            ax.clear()
            ax.axis("off")

        for offset, (ax, filename) in enumerate(
            zip(self.axes, self.selected)
        ):
            image = Image.open(filename)

            ax.imshow(
                image,
                cmap="gray"
            )

            ax.axis("off")

            index = self.start_index + offset

            title = (
                f"{index}: "
                f"{short_filename(filename)}"
            )

            # Mark previously identified bad images.
            if str(filename) in self.bad_files:
                title = "BAD  " + title

            ax.set_title(
                title,
                fontsize=8
            )

        first = self.start_index
        last = (
            self.start_index +
            len(self.selected) - 1
        )

        self.fig.suptitle(
            f"Digit {self.digit}  "
            f"(images {first}-{last} "
            f"of {len(self.files)})\n"
            f"[n] next   [p] previous   "
            f"[b] mark bad   [q] quit",
            fontsize=12
        )

        self.fig.tight_layout(
            rect=(0, 0, 1, 0.92)
        )

        self.fig.canvas.draw_idle()

    def current_filename_from_axis(self, ax):
        """Return the filename associated with an axis."""

        try:
            position = list(self.axes).index(ax)
        except ValueError:
            return None

        if position >= len(self.selected):
            return None

        return self.selected[position]

    def on_mouse_click(self, event):
        """
        Clicking an image marks/unmarks it as bad.

        This is intentionally active all the time, so after pressing
        'b' you can simply click the bad image. The 'b' key itself is
        also shown as a reminder that you're marking bad images.
        """

        if event.inaxes is None:
            return

        filename = self.current_filename_from_axis(
            event.inaxes
        )

        if filename is None:
            return

        filename_string = str(filename)

        if filename_string in self.bad_files:
            self.bad_files.remove(filename_string)
            print(f"UNMARKED: {filename}")
        else:
            self.bad_files.add(filename_string)
            print(f"BAD:      {filename}")

        save_bad_files(self.bad_files)
        self.draw()

    def on_key(self, event):
        """Handle keyboard commands."""

        key = event.key.lower() if event.key else ""

        if key == "n":
            self.next_sheet()

        elif key == "p":
            self.previous_sheet()

        elif key == "b":
            print()
            print(
                "BAD MODE: click any bad image(s) "
                "in the contact sheet."
            )
            print(
                "Click a bad image again to unmark it."
            )

        elif key == "q":
            plt.close(self.fig)

    def next_sheet(self):
        """Advance to the next contact sheet."""

        new_index = (
            self.start_index +
            IMAGES_PER_SHEET
        )

        if new_index >= len(self.files):
            print("Already on the last contact sheet.")
            return

        self.start_index = new_index

        self.selected = self.files[
            self.start_index:
            self.start_index + IMAGES_PER_SHEET
        ]

        self.draw()

    def previous_sheet(self):
        """Go back to the previous contact sheet."""

        new_index = (
            self.start_index -
            IMAGES_PER_SHEET
        )

        if new_index < 0:
            new_index = 0

        if new_index == self.start_index:
            print("Already on the first contact sheet.")
            return

        self.start_index = new_index

        self.selected = self.files[
            self.start_index:
            self.start_index + IMAGES_PER_SHEET
        ]

        self.draw()


def main():

    if not DIGITS_DIR.exists():
        print(
            f"ERROR: {DIGITS_DIR} does not exist."
        )
        return

    bad_files = load_existing_bad_files()

    if bad_files:
        print(
            f"Loaded {len(bad_files)} previously "
            f"marked bad images from {BAD_FILE}"
        )

    print()
    print("## Extracted digit dataset")
    print()

    files_by_digit = {}

    total = 0

    for digit in "0123456789":

        digit_dir = DIGITS_DIR / digit

        files = sorted(
            digit_dir.glob("*.jpg")
        )

        files_by_digit[digit] = files

        print(
            f"Digit {digit}: "
            f"{len(files)} images"
        )

        total += len(files)

    print()
    print(
        f"Total extracted digits: {total}"
    )

    print()
    print("Controls:")
    print("  n = next contact sheet")
    print("  p = previous contact sheet")
    print("  b = bad mode reminder")
    print("  click an image = mark/unmark it as bad")
    print("  q = quit")
    print()

    for digit in "0123456789":

        files = files_by_digit[digit]

        if not files:
            continue

        for start_index in range(
            0,
            len(files),
            IMAGES_PER_SHEET
        ):

            sheet = ContactSheet(
                digit,
                files,
                start_index,
                bad_files
            )

            plt.show()

            # If the window was closed with q or the window close
            # button, move on to the next digit/sheet.
            if not plt.fignum_exists(sheet.fig.number):
                continue

    print()
    print(
        f"Saved {len(bad_files)} bad images to:"
    )
    print(f"  {BAD_FILE}")


if __name__ == "__main__":
    main()
