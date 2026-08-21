from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from digit_preprocess import preprocess

BAD_FILE = Path("bad_extracted_digits.txt")
DATASET_DIR = Path("digits")
K_NEIGHBORS = 7

def load_bad_files():
    if not BAD_FILE.exists():
        raise FileNotFoundError(f"{BAD_FILE} does not exist.")
    with BAD_FILE.open("r") as f:
        return [Path(line.strip()) for line in f if line.strip()]

def load_dataset(directory):
    images, labels, filenames = [], [], []
    for digit in range(10):
        for filename in sorted((directory / str(digit)).glob("*.jpg")):
            image = preprocess(filename)
            images.append(np.array(image).flatten())
            labels.append(digit)
            filenames.append(filename)
    return np.array(images), np.array(labels), filenames

def train_classifier():
    print("Loading training dataset...")
    X_train, y_train, train_filenames = load_dataset(DATASET_DIR / "train")
    print(f"Training images: {len(X_train)}")
    classifier = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    classifier.fit(X_train, y_train)
    return classifier, X_train, y_train, train_filenames

def show_bad_image(filename, classifier, X_train, y_train,
                   train_filenames, index, total):
    original = Image.open(filename).convert("L")
    processed = preprocess(filename)
    X = np.array(processed).flatten().reshape(1, -1)

    prediction = classifier.predict(X)[0]
    probabilities = classifier.predict_proba(X)[0]
    class_index = np.where(classifier.classes_ == prediction)[0][0]
    confidence = probabilities[class_index]

    distances, neighbor_indices = classifier.kneighbors(
        X, n_neighbors=K_NEIGHBORS
    )

    print()
    print("=" * 80)
    print(f"Bad image {index + 1} of {total}")
    print("=" * 80)
    print(f"File:       {filename}")
    print(f"Expected:   {filename.parent.name}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.1%}")
    print()
    print("Nearest training examples:")
    for position, (distance, neighbor_index) in enumerate(
        zip(distances[0], neighbor_indices[0]), start=1
    ):
        print(
            f"  {position}. label={y_train[neighbor_index]}  "
            f"distance={distance:.0f}  "
            f"{train_filenames[neighbor_index]}"
        )

    fig, axes = plt.subplots(1, 9, figsize=(18, 4))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title(f"ORIGINAL\nExpected: {filename.parent.name}")
    axes[0].axis("off")

    axes[1].imshow(processed, cmap="gray")
    axes[1].set_title(
        f"PREPROCESSED\nPrediction: {prediction}\n"
        f"Confidence: {confidence:.1%}"
    )
    axes[1].axis("off")

    for position, neighbor_index in enumerate(
        neighbor_indices[0], start=2
    ):
        axes[position].imshow(
            X_train[neighbor_index].reshape(28, 28), cmap="gray"
        )
        axes[position].set_title(
            f"TRAIN {position - 1}\n"
            f"Digit: {y_train[neighbor_index]}\n"
            f"Dist: {distances[0][position - 2]:.0f}",
            fontsize=9
        )
        axes[position].axis("off")

    fig.suptitle(
        f"BAD IMAGE {index + 1} OF {total}\n{filename}",
        fontsize=13
    )
    plt.tight_layout(rect=(0, 0, 1, 0.86))
    return fig

def main():
    print()
    print("## Bad Digit Diagnostic")
    print()

    bad_files = load_bad_files()
    print(f"Bad images found: {len(bad_files)}")

    if not bad_files:
        print("No bad images were found.")
        return

    classifier, X_train, y_train, train_filenames = train_classifier()

    print()
    print("Controls:")
    print("  n = next bad image")
    print("  p = previous bad image")
    print("  q = quit")
    print()

    current_index = 0

    while True:
        filename = bad_files[current_index]

        if not filename.exists():
            print(f"WARNING: File does not exist: {filename}")
            current_index += 1
            if current_index >= len(bad_files):
                print("Reached end of bad-image list.")
                return
            continue

        fig = show_bad_image(
            filename, classifier, X_train, y_train,
            train_filenames, current_index, len(bad_files)
        )

        pressed_key = {"value": None}

        def on_key(event):
            key = event.key.lower() if event.key else None
            if key in ("n", "p", "q"):
                pressed_key["value"] = key
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

        key = pressed_key["value"]

        if key == "q":
            print("Done.")
            return
        elif key == "n":
            if current_index < len(bad_files) - 1:
                current_index += 1
            else:
                print("Already at the last bad image.")
        elif key == "p":
            if current_index > 0:
                current_index -= 1
            else:
                print("Already at the first bad image.")

if __name__ == "__main__":
    main()
