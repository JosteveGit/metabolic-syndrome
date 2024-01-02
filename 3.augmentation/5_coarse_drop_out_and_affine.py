from imgaug import augmenters as iaa
import imageio

original_images = [imageio.v2.imread(f'../512_data/x/x_{i}.png') for i in range(1, 15)]
masks = [imageio.v2.imread(f'../512_data/y/y_{i}.png') for i in range(1, 15)]

augmentation_list = [
    iaa.Sequential([
        iaa.CoarseDropout(0.02, size_percent=0.5),
        iaa.Affine(scale=(0.9, 1.1), rotate=(-10, 10), translate_percent=(-0.1, 0.1))
    ]),  # CoarseDropout and Affine with scaling, rotation, and translation

    iaa.Sequential([
        iaa.CoarseDropout(0.02, size_percent=0.15),
        iaa.Affine(scale=(0.95, 1.05), rotate=(-15, 15))
    ]),  # CoarseDropout with per channel dropout and Affine with moderate scaling and rotation

    iaa.Sequential([
        iaa.CoarseDropout(0.05, size_percent=0.2),
        iaa.Affine(shear=(-20, 20), translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)})
    ]),  # CoarseDropout and Affine with shearing and translation

    iaa.Sequential([
        iaa.CoarseDropout((0.03, 0.07), size_percent=(0.02, 0.25)),
        iaa.Affine(rotate=(-30, 30), scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
    ]),  # CoarseDropout with variable size and Affine with rotation and non-uniform scaling

    iaa.Sequential([
        iaa.CoarseDropout(0.04, size_percent=0.1),
        iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)}, rotate=(-45, 45))
    ]),  # CoarseDropout with per channel dropout and Affine with pixel translation and larger rotation

    iaa.Sequential([
        iaa.CoarseDropout((0.01, 0.1), size_percent=(0.05, 0.1)),
        iaa.Affine(scale=(0.85, 1.15), translate_percent=(-0.2, 0.2))
    ])
]

index = 351

for i, seq in enumerate(augmentation_list, 1):
    seq_det = seq.to_deterministic()  # Call this once per batch
    augmented_images = seq_det.augment_images(original_images)
    augmented_masks = seq_det.augment_images(masks)

    for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks), 1):
        imageio.imwrite(f'../512_data_with_augmentation/x/x_{index}.png', image)
        imageio.imwrite(f'../512_data_with_augmentation/y/y_{index}.png', mask)
        index += 1

