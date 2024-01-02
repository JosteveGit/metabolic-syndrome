from imgaug import augmenters as iaa
import imageio

original_images = [imageio.v2.imread(f'../512_data/x/x_{i}.png') for i in range(1, 15)]
masks = [imageio.v2.imread(f'../512_data/y/y_{i}.png') for i in range(1, 15)]

rotation_and_flipping_augmentations = [
    iaa.Sequential([iaa.Fliplr(1.0)]),  # Horizontally flip all images
    iaa.Sequential([iaa.Flipud(1.0)]),  # Vertically flip all images
    iaa.Sequential([iaa.Affine(rotate=90)]),  # Rotate images by 90 degrees
    iaa.Sequential([iaa.Affine(rotate=180)]),  # Rotate images by 180 degrees
    iaa.Sequential([iaa.Affine(rotate=270)]),  # Rotate images by 270 degrees
    iaa.Sequential([iaa.Fliplr(0.5), iaa.Affine(rotate=(-45, 45))]),
    # Random flip horizontally and rotate between -45 and 45 degrees
]

index = 99

for i, seq in enumerate(rotation_and_flipping_augmentations, 1):
    seq_det = seq.to_deterministic()  # Call this once per batch
    augmented_images = seq_det.augment_images(original_images)
    augmented_masks = seq_det.augment_images(masks)

    for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks), 1):
        imageio.imwrite(f'../512_data_with_augmentation/x/x_{index}.png', image)
        imageio.imwrite(f'../512_data_with_augmentation/y/y_{index}.png', mask)
        index += 1
