from imgaug import augmenters as iaa
import imageio

original_images = [imageio.v2.imread(f'../512_data/x/x_{i}.png') for i in range(1, 15)]
masks = [imageio.v2.imread(f'../512_data/y/y_{i}.png') for i in range(1, 15)]

affine_and_rotation_flipping_augmentations = [
    iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    ]),  # Flip left-right, then apply complex affine transformation

    iaa.Sequential([
        iaa.Flipud(0.5),
        iaa.Affine(scale=(0.8, 1.2), rotate=(-10, 10), shear=(-8, 8))
    ]),  # Flip up-down, then apply scaling, rotation, and shearing

    iaa.Sequential([
        iaa.Fliplr(0.3),
        iaa.Flipud(0.3),
        iaa.Affine(rotate=(-30, 30))
    ]),  # Flip left-right, up-down with a probability and rotate

    iaa.Sequential([
        iaa.Affine(translate_px={"x": (-10, 10), "y": (-10, 10)}),
        iaa.Affine(rotate=(-45, 45)),
        iaa.Fliplr(1.0)
    ]),  # Translate by pixels, rotate, then flip horizontally

    iaa.Sequential([
        iaa.Affine(scale=(1.0, 1.2)),  # Uniform scale
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(translate_percent={"x": 0.1}, rotate=90)  # Translate 10% on x-axis and rotate 90 degrees
    ]),  # Scale, then flip, and apply a fixed rotation

    iaa.Sequential([
        iaa.Flipud(0.5),
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-90, 90), scale={"x": (0.8, 1), "y": (1, 1.2)})
    ]),  # Flip both ways, then rotate and apply non-uniform scaling
]


index = 267

for i, seq in enumerate(affine_and_rotation_flipping_augmentations, 1):
    seq_det = seq.to_deterministic()  # Call this once per batch
    augmented_images = seq_det.augment_images(original_images)
    augmented_masks = seq_det.augment_images(masks)

    for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks), 1):
        imageio.imwrite(f'../512_data_with_augmentation/x/x_{index}.png', image)
        imageio.imwrite(f'../512_data_with_augmentation/y/y_{index}.png', mask)
        index += 1
