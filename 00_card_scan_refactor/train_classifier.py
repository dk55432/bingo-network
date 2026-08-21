from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from digit_preprocess import preprocess


DATASET_DIR = Path("digits")


def load_dataset(directory):
    images = []
    labels = []
    filenames = []

    for digit in range(10):
        digit_dir = directory / str(digit)
        files = sorted(digit_dir.glob("*.jpg"))

        for filename in files:
            image = preprocess(filename)

            pixels = np.array(image).flatten()

            images.append(pixels)
            labels.append(digit)
            filenames.append(filename)

    return np.array(images), np.array(labels), filenames


def main():

    print("Loading training dataset...")

    X_train, y_train, train_filenames = load_dataset(
        DATASET_DIR / "train"
    )

    print("Loading test dataset...")

    X_test, y_test, test_filenames = load_dataset(
        DATASET_DIR / "test"
    )

    print()
    print("Dataset summary")
    print("----------------")
    print(f"Training images:  {len(X_train)}")
    print(f"Test images:      {len(X_test)}")
    print(f"Pixels per image: {X_train.shape[1]}")

    # Create and train the classifier.
    classifier = KNeighborsClassifier(
        n_neighbors=7
    )

    classifier.fit(
        X_train,
        y_train
    )

    # Predictions for the test images.
    predictions = classifier.predict(X_test)
    
    # Probability estimates for the test images.
    probabilities = classifier.predict_proba(X_test)
    
    # Confidence is the probability assigned to
    # the digit that the classifier predicted.
    vote_confidence = np.max(
        probabilities,
        axis=1
    )

    # for index in range(len(X_test)):

    #     print(
    #         f"Image {index:2d}: "
    #         f"actual={y_test[index]} "
    #         f"predicted={predictions[index]} "
    #         f"confidence={vote_confidence[index]:.1%}"
    #     )
    
    # Calculate accuracy.
    correct = np.sum(
        predictions == y_test
    )

    total = len(y_test)

    print()
    print("Results")
    print("-------")
    print(f"Correct: {correct} / {total}")
    print(
        f"Test accuracy: "
        f"{correct / total:.1%}"
    )

    # Find the incorrectly classified images.
    wrong_indices = np.where(
        predictions != y_test
    )[0]
    
    # Calculate a confidence-like score for every test image.
    #
    # For each test image, examine ALL training images and
    # find the closest example of each digit.

    confidence_values = []

    print()
    print("## Per-image confidence")

    for index in range(len(X_test)):

        predicted_digit = predictions[index]

        # Find the probability assigned to the predicted digit.
        predicted_class_index = np.where(
            classifier.classes_ == predicted_digit
        )[0][0]

        confidence = probabilities[
            index,
            predicted_class_index
        ]

        print(
            f"Image {index:2d}: "
            f"actual={y_test[index]} "
            f"predicted={predicted_digit} "
            f"confidence={confidence:.1%}"
        )
    
    # -------------------------------------------------
    # Display nearest neighbors for incorrect results.
    # -------------------------------------------------

    if len(wrong_indices) > 0:

        for index in wrong_indices:

            distances, neighbor_indices = (
                classifier.kneighbors(
                    X_test[index].reshape(1, -1),
                    n_neighbors=7
                )
            )

            fig, axes = plt.subplots(
                1,
                8,
                figsize=(16, 3)
            )

            axes[0].imshow(
                X_test[index].reshape(28, 28),
                cmap="gray"
            )

            axes[0].set_title(
                f"TEST\n"
                f"Actual: {y_test[index]}\n"
                f"Predicted: {predictions[index]}"
            )

            axes[0].axis("off")

            for position, (
                distance,
                neighbor_index
            ) in enumerate(
                zip(
                    distances[0],
                    neighbor_indices[0]
                ),
                start=1
            ):

                axes[position].imshow(
                    X_train[
                        neighbor_index
                    ].reshape(28, 28),
                    cmap="gray"
                )

                axes[position].set_title(
                    f"TRAIN\n"
                    f"Label: "
                    f"{y_train[neighbor_index]}\n"
                    f"Distance: "
                    f"{distance:.0f}"
                )

                axes[position].axis("off")

            plt.suptitle(
                f"Misclassified test image #{index}"
            )

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()