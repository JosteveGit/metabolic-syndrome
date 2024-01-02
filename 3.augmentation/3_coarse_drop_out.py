from imgaug import augmenters as iaa
import imageio

original_images = [imageio.v2.imread(f'../512_data/x/x_{i}.png') for i in range(1, 15)]
masks = [imageio.v2.imread(f'../512_data/y/y_{i}.png') for i in range(1, 15)]

cutout_augmentations = [
    iaa.Sequential([
        iaa.CoarseDropout((0.01, 0.2), size_percent=(0.02, 0.25)),  # Coarse dropout
        iaa.Affine(scale=(0.9, 1.1))  # Uniform scaling
    ]),
    iaa.Sequential([
        iaa.CoarseDropout((0.01, 0.1), size_percent=(0.02, 0.25)),  # Coarse dropout
        iaa.Fliplr(0.5),  # Horizontal flip
    ]),
    iaa.Sequential([
        iaa.CoarseDropout((0.01, 0.1), size_percent=(0.02, 0.15)),  # Coarse dropout
        iaa.Affine(rotate=(-45, 45)),  # Rotation
    ]),
    iaa.Sequential([
        iaa.CoarseDropout((0.01, 0.1), size_percent=(0.02, 0.25)),  # Coarse dropout
        iaa.Affine(shear=(-20, 20)),  # Shearing
        iaa.ElasticTransformation(alpha=50, sigma=5)  # Elastic transformation
    ]),
    iaa.Sequential([
        iaa.CoarseDropout((0.01, 0.1), size_percent=(0.02, 0.15)),  # Coarse dropout
        iaa.PerspectiveTransform(scale=(0.01, 0.1)),  # Perspective transformation
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})  # Translation
    ]),
    iaa.Sequential([
        iaa.CoarseDropout((0.01, 0.1), size_percent=(0.02, 0.15)),  # Coarse dropout
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),  # Non-uniform scaling
        iaa.CoarseDropout((0.01, 0.1), size_percent=(0.02, 0.25))  # Coarse dropout
    ])
]
index = 183

for i, seq in enumerate(cutout_augmentations, 1):
    seq_det = seq.to_deterministic()  # Call this once per batch
    augmented_images = seq_det.augment_images(original_images)
    augmented_masks = seq_det.augment_images(masks)

    for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks), 1):
        imageio.imwrite(f'../512_data_with_augmentation/x/x_{index}.png', image)
        imageio.imwrite(f'../512_data_with_augmentation/y/y_{index}.png', mask)
        index += 1
