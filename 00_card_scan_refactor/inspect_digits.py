from pathlib import Path
from PIL import Image


for dataset_name in ["digit_candidates", "digit_tests"]:

    print(f"\n=== {dataset_name} ===")

    dataset = Path(dataset_name)

    for digit in range(10):

        directory = dataset / str(digit)
        images = list(directory.glob("*"))

        sizes = set()
        modes = set()

        for filename in images:
            try:
                with Image.open(filename) as image:
                    sizes.add(image.size)
                    modes.add(image.mode)
            except Exception as e:
                print(f"Could not read {filename}: {e}")

        print(
            f"{digit}: "
            f"{len(images)} images, "
            f"sizes={sorted(sizes)}, "
            f"modes={sorted(modes)}"
        )