from PIL import Image
import numpy as np
import os


def nearest_of_three(value):
    possible_values = [0, 127, 255]
    return min(possible_values, key=lambda x: abs(x - value))


def clean():
    mask_dir = "512_data_with_augmentation/y"
    mask_paths = sorted([
        os.path.join(mask_dir, fname)
        for fname in os.listdir(mask_dir)
        if fname.endswith('.png') and fname.startswith("y")
    ])

    masks = [np.array(Image.open(path).convert("L")) for path in mask_paths]

    for i, mask in enumerate(masks):
        vectorized_nearest = np.vectorize(nearest_of_three)
        masks[i] = vectorized_nearest(mask)
        masks[i] = masks[i].astype(np.uint8)

        # Save the mask
        Image.fromarray(masks[i]).save(
            "512_data_with_augmentation/cleaned_y/" + mask_paths[i].split("/")[-1])

clean()
