from pathlib import Path
from collections import defaultdict
import re


ROOT = Path("extracted_digits")


# Actual filename format:
#
# IMG_4901_card_0_cells_row0_col0_digit0_1.jpg
#
# Meaning:
#
#   image    = IMG_4901
#   card     = 0
#   row      = 0
#   col      = 0
#   position = 0
#   digit    = 1
#
PATTERN = re.compile(
    r"^(?P<image>.+?)"
    r"_card_(?P<card>\d+)"
    r"_cells"
    r"_row(?P<row>\d+)"
    r"_col(?P<col>\d+)"
    r"_digit(?P<position>\d+)"
    r"_(?P<digit>\d+)"
    r"\.jpg$"
)


# ------------------------------------------------------------
# Read extracted digit images
# ------------------------------------------------------------

cells = defaultdict(list)
parse_failures = []


for path in ROOT.rglob("*.jpg"):

    match = PATTERN.match(path.name)

    if not match:
        parse_failures.append(path)
        continue

    data = match.groupdict()

    key = (
        data["image"],
        int(data["card"]),
        int(data["row"]),
        int(data["col"]),
    )

    cells[key].append(
        {
            "position": int(data["position"]),
            "digit": int(data["digit"]),
            "path": str(path),
        }
    )


# ------------------------------------------------------------
# Overall summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("EXTRACTED DIGIT ANALYSIS")
print("=" * 70)

print(f"\nDirectory: {ROOT}")
print(f"Total cell groups: {len(cells)}")

total_images = sum(
    len(candidates)
    for candidates in cells.values()
)

print(f"Total digit images: {total_images}")
print(f"Files that failed to parse: {len(parse_failures)}")


# ------------------------------------------------------------
# Parse failures
# ------------------------------------------------------------

if parse_failures:

    print()
    print("=" * 70)
    print("PARSE FAILURES")
    print("=" * 70)

    for path in parse_failures:
        print(f"  {path}")


# ------------------------------------------------------------
# Candidate count per cell
# ------------------------------------------------------------

distribution = defaultdict(int)

for candidates in cells.values():
    distribution[len(candidates)] += 1


print()
print("=" * 70)
print("DIGIT COUNT PER CELL")
print("=" * 70)

for count in sorted(distribution):

    print(
        f"  {count} digit(s): "
        f"{distribution[count]} cells"
    )


# ------------------------------------------------------------
# Actual digits by cell
# ------------------------------------------------------------

print()
print("=" * 70)
print("DIGIT STRUCTURE")
print("=" * 70)


number_counts = defaultdict(int)

for key, candidates in cells.items():

    ordered = sorted(
        candidates,
        key=lambda x: x["position"]
    )

    number = "".join(
        str(candidate["digit"])
        for candidate in ordered
    )

    number_counts[number] += 1


print("\nMost common extracted numbers:")

for number, count in sorted(
    number_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:30]:

    print(
        f"  {number}: {count} cells"
    )


# ------------------------------------------------------------
# Digit frequency
# ------------------------------------------------------------

digit_counts = defaultdict(int)

for candidates in cells.values():

    for candidate in candidates:

        digit_counts[
            candidate["digit"]
        ] += 1


print()
print("=" * 70)
print("DIGIT FREQUENCY")
print("=" * 70)

for digit in range(10):

    print(
        f"  digit {digit}: "
        f"{digit_counts[digit]} images"
    )


# ------------------------------------------------------------
# Position frequency
# ------------------------------------------------------------

position_counts = defaultdict(int)

for candidates in cells.values():

    for candidate in candidates:

        position_counts[
            candidate["position"]
        ] += 1


print()
print("=" * 70)
print("DIGIT POSITION FREQUENCY")
print("=" * 70)

for position in sorted(position_counts):

    print(
        f"  position {position}: "
        f"{position_counts[position]} images"
    )


# ------------------------------------------------------------
# Digit frequency by position
# ------------------------------------------------------------

position_digit_counts = defaultdict(int)

for candidates in cells.values():

    for candidate in candidates:

        key = (
            candidate["position"],
            candidate["digit"],
        )

        position_digit_counts[key] += 1


print()
print("=" * 70)
print("DIGITS BY POSITION")
print("=" * 70)

for position in sorted(
    {key[0] for key in position_digit_counts}
):

    print()
    print(f"Position {position}:")

    for digit in range(10):

        count = position_digit_counts[
            (position, digit)
        ]

        if count:

            print(
                f"  digit {digit}: "
                f"{count}"
            )


# ------------------------------------------------------------
# Single-digit cells
# ------------------------------------------------------------

single_digit_cells = []

for key, candidates in cells.items():

    if len(candidates) == 1:

        candidate = candidates[0]

        single_digit_cells.append(
            (
                key,
                candidate["digit"]
            )
        )


print()
print("=" * 70)
print("SINGLE-DIGIT CELLS")
print("=" * 70)

print(
    f"Total: {len(single_digit_cells)}"
)

single_by_digit = defaultdict(int)

for key, digit in single_digit_cells:
    single_by_digit[digit] += 1

for digit in sorted(single_by_digit):

    print(
        f"  digit {digit}: "
        f"{single_by_digit[digit]} cells"
    )


# ------------------------------------------------------------
# Two-digit cells
# ------------------------------------------------------------

two_digit_cells = [
    (key, candidates)
    for key, candidates in cells.items()
    if len(candidates) == 2
]


print()
print("=" * 70)
print("TWO-DIGIT CELLS")
print("=" * 70)

print(
    f"Total: {len(two_digit_cells)}"
)


# ------------------------------------------------------------
# Validate positions
# ------------------------------------------------------------

bad_position_cells = []

for key, candidates in two_digit_cells:

    positions = sorted(
        candidate["position"]
        for candidate in candidates
    )

    if positions != [0, 1]:

        bad_position_cells.append(
            (key, positions)
        )


print()
print("=" * 70)
print("POSITION VALIDATION")
print("=" * 70)

if bad_position_cells:

    print(
        f"WARNING: "
        f"{len(bad_position_cells)} cells "
        f"have unexpected positions"
    )

    for key, positions in bad_position_cells[:20]:

        print(
            f"  {key}: positions={positions}"
        )

else:

    print(
        "All two-digit cells have "
        "positions 0 and 1."
    )


# ------------------------------------------------------------
# Sample cells
# ------------------------------------------------------------

print()
print("=" * 70)
print("SAMPLE CELLS")
print("=" * 70)

for key in sorted(cells)[:15]:

    image, card, row, col = key

    ordered = sorted(
        cells[key],
        key=lambda x: x["position"]
    )

    number = "".join(
        str(candidate["digit"])
        for candidate in ordered
    )

    print(
        f"\n{image} "
        f"card={card} "
        f"row={row} "
        f"col={col}"
    )

    print(
        f"    number={number}"
    )

    for candidate in ordered:

        print(
            f"    position={candidate['position']} "
            f"digit={candidate['digit']}"
        )


# ------------------------------------------------------------
# Cells with duplicate positions
# ------------------------------------------------------------

duplicate_position_cells = []

for key, candidates in cells.items():

    positions = [
        candidate["position"]
        for candidate in candidates
    ]

    if len(positions) != len(set(positions)):

        duplicate_position_cells.append(
            (key, positions)
        )


print()
print("=" * 70)
print("DUPLICATE DIGIT POSITIONS")
print("=" * 70)

if duplicate_position_cells:

    print(
        f"WARNING: "
        f"{len(duplicate_position_cells)} cells "
        f"have duplicate positions"
    )

    for key, positions in duplicate_position_cells[:20]:

        print(
            f"  {key}: positions={positions}"
        )

else:

    print("None")


# ------------------------------------------------------------
# Done
# ------------------------------------------------------------

print()
print("=" * 70)
print("DONE")
print("=" * 70)