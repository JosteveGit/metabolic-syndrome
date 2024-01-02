from imgaug import augmenters as iaa
import imageio

original_images = [imageio.v2.imread(f'../512_data/x/x_{i}.png') for i in range(1, 15)]
masks = [imageio.v2.imread(f'../512_data/y/y_{i}.png') for i in range(1, 15)]

affine_transformations = [
    iaa.Sequential([iaa.Affine(scale=(0.9, 1.1))]),  # Uniform scaling
    iaa.Sequential([iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})]),  # Translation
    iaa.Sequential([iaa.Affine(rotate=(-10, 10))]),  # Rotation
    iaa.Sequential([iaa.Affine(shear=(-16, 16))]),  # Shearing
    iaa.Sequential([iaa.Affine(translate_px={"x": (-5, 5), "y": (-5, 5)})]),  # Pixel translation
    iaa.Sequential([iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})])  # Non-uniform scaling
]

index = 15

for i, seq in enumerate(affine_transformations, 1):
    seq_det = seq.to_deterministic()  # Call this once per batch
    augmented_images = seq_det.augment_images(original_images)
    augmented_masks = seq_det.augment_images(masks)

    for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks), 1):
        imageio.imwrite(f'../512_data_with_augmentation/x/x_{index}.png', image)
        imageio.imwrite(f'../512_data_with_augmentation/y/y_{index}.png', mask)
        index += 1
