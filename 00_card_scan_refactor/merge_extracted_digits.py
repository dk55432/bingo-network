from pathlib import Path
import shutil


SOURCE_DIR = Path("extracted_digits")
DEST_DIR = Path("digits/train")


# These images were inspected manually and should NOT
# be added to the training dataset.
BAD_FILES = {
    "extracted_digits/0/IMG_4901_card_0_cells_row3_col3_digit1_0.jpg",

    "extracted_digits/1/IMG_4901_card_0_cells_row3_col1_digit0_1.jpg",
    "extracted_digits/1/IMG_4902_card_0_cells_row0_col0_digit0_1.jpg",
    "extracted_digits/1/IMG_4904_card_0_cells_row0_col1_digit0_1.jpg",
    "extracted_digits/1/IMG_4904_card_0_cells_row2_col1_digit0_1.jpg",
    "extracted_digits/1/IMG_4904_card_0_cells_row3_col1_digit0_1.jpg",
    "extracted_digits/1/IMG_4907_card_0_cells_row1_col1_digit0_1.jpg",
    "extracted_digits/1/IMG_4947_card_1_cells_row4_col0_digit0_1.jpg",

    "extracted_digits/3/IMG_4947a_card_1_cells_row4_col2_digit0_3.jpg",

    "extracted_digits/4/IMG_4959_card_0_cells_row2_col3_digit0_4.jpg",

    "extracted_digits/5/IMG_4901_card_0_cells_row3_col3_digit0_5.jpg",

    "extracted_digits/7/IMG_4947a_card_1_cells_row4_col4_digit0_7.jpg",

    "extracted_digits/8/IMG_4947a_card_1_cells_row4_col0_digit0_8.jpg",

    "extracted_digits/9/IMG_4901_card_0_cells_row3_col1_digit1_9.jpg",
    "extracted_digits/9/IMG_4902_card_0_cells_row4_col3_digit1_9.jpg",
}


def main():

    print("Merging verified extracted digits...")
    print()

    # Make sure the destination exists.
    for digit in range(10):
        (DEST_DIR / str(digit)).mkdir(
            parents=True,
            exist_ok=True
        )

    total_added = 0
    total_skipped = 0

    print("Results")
    print("-------")

    for digit in range(10):

        source_digit_dir = SOURCE_DIR / str(digit)
        dest_digit_dir = DEST_DIR / str(digit)

        if not source_digit_dir.exists():
            print(f"Digit {digit}: source directory missing")
            continue

        files = sorted(
            source_digit_dir.glob("*.jpg")
        )

        added = 0
        skipped = 0

        for source_file in files:

            # Convert the path to the same form used
            # in BAD_FILES.
            relative_path = source_file.as_posix()

            if relative_path in BAD_FILES:
                print(
                    f"SKIP: {relative_path}"
                )
                skipped += 1
                total_skipped += 1
                continue

            # Prefix the filename so it is obvious that
            # this image came from the labeled dataset.
            destination_name = (
                f"labeled_{source_file.name}"
            )

            destination = (
                dest_digit_dir / destination_name
            )

            # Don't overwrite an existing file.
            if destination.exists():
                print(
                    f"SKIP (already exists): "
                    f"{destination}"
                )
                skipped += 1
                total_skipped += 1
                continue

            shutil.copy2(
                source_file,
                destination
            )

            added += 1
            total_added += 1

        print(
            f"Digit {digit}: "
            f"added={added}, "
            f"skipped={skipped}"
        )

    print()
    print("Merge complete")
    print("--------------")
    print(f"Images added:   {total_added}")
    print(f"Images skipped: {total_skipped}")


if __name__ == "__main__":
    main()