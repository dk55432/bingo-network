from PIL import Image
import numpy as np
from scipy import ndimage

def preprocess(filename):
    image = Image.open(filename).convert("L")

    pixels = np.array(image)

    # Find pixels that are substantially darker
    # than the paper/background.
    mask = pixels < 150

    # Remove small isolated noise.
    mask = ndimage.binary_opening(
        mask,
        structure=np.ones((3, 3))
    )

    # Find connected components.
    labeled, num_features = ndimage.label(mask)

    if num_features == 0:
        raise ValueError("Couldn't find the digit")

    # Find the largest connected component.
    component_sizes = ndimage.sum(
        mask,
        labeled,
        range(1, num_features + 1)
    )

    largest_component = (
        np.argmax(component_sizes) + 1
    )

    # Keep only the largest component.
    mask = labeled == largest_component
    
    rows, cols = np.where(mask)

    if len(rows) == 0:
        raise ValueError("Couldn't find the digit")

    top = rows.min()
    bottom = rows.max() + 1
    left = cols.min()
    right = cols.max() + 1

    digit = image.crop((left, top, right, bottom))

    # Resize while preserving aspect ratio.
    digit.thumbnail((22, 22))

    # Put it in the center of a 28x28 image.
    result = Image.new("L", (28, 28), 255)

    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2

    result.paste(digit, (x, y))

    return result


# image = preprocess("digit_candidates/2/row0_col1 copy 2.jpg")
# image.save("processed.png")