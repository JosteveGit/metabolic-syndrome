from imgaug import augmenters as iaa
import imageio

original_images = [imageio.v2.imread(f'../512_data/x/x_{i}.png') for i in range(1, 15)]
masks = [imageio.v2.imread(f'../512_data/y/y_{i}.png') for i in range(1, 15)]

shear_transformations = [
    iaa.Sequential([
        iaa.Affine(shear=(-20, 20)),  # Shear by -20 to 20 degrees
        iaa.Affine(translate_percent={"x": (-0.1, 0.1)}),  # Horizontal translation
    ]),
    iaa.Sequential([
        iaa.Affine(shear=(-15, 15)),  # Shear by -15 to 15 degrees
        iaa.Affine(scale=(0.9, 1.1)),  # Uniform scaling
    ]),
    iaa.Sequential([
        iaa.Affine(shear=(-10, 10)),  # Shear by -10 to 10 degrees
        iaa.Affine(rotate=(-5, 5)),  # Rotation by -5 to 5 degrees
    ]),
    iaa.Sequential([
        iaa.Affine(shear=(-25, 25)),  # Shear by -25 to 25 degrees
        iaa.Affine(translate_px={"x": (-10, 10), "y": (-10, 10)}),  # Translation in pixels
    ]),
    iaa.Sequential([
        iaa.Affine(shear=(-30, 30)),  # Shear by -30 to 30 degrees
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),  # Non-uniform scaling
    ]),
    iaa.Sequential([
        iaa.Affine(shear=(-5, 5)),  # Shear by -5 to 5 degrees
        iaa.Fliplr(0.5),  # Horizontal flip 50% of the time
        iaa.Crop(percent=(0, 0.1)),  # Random cropping
    ])
]

index = 729

for i, seq in enumerate(shear_transformations, 1):
    seq_det = seq.to_deterministic()  # Call this once per batch
    augmented_images = seq_det.augment_images(original_images)
    augmented_masks = seq_det.augment_images(masks)

    for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks), 1):
        imageio.imwrite(f'../512_data_with_augmentation/x/x_{index}.png', image)
        imageio.imwrite(f'../512_data_with_augmentation/y/y_{index}.png', mask)
        index += 1
