from imgaug import augmenters as iaa
import imageio

original_images = [imageio.v2.imread(f'../512_data/x/x_{i}.png') for i in range(1, 15)]
masks = [imageio.v2.imread(f'../512_data/y/y_{i}.png') for i in range(1, 15)]

# Complex affine transformations without blur or noise
perspective_transformations = [
    iaa.Sequential([
        iaa.PerspectiveTransform(scale=(0.01, 0.1)),
        iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)})
    ]),  # Perspective transform followed by slight translation

    iaa.Sequential([
        iaa.PerspectiveTransform(scale=(0.01, 0.1), keep_size=True),
        iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),
        iaa.Affine(scale=(0.95, 1.05))
    ]),  # Perspective transform, elastic transformation, and uniform scaling

    iaa.Sequential([
        iaa.PerspectiveTransform(scale=(0.01, 0.1)),
        iaa.Affine(rotate=(-5, 5)),
    ]),  # Perspective transform with rotation and contrast adjustment

    iaa.Sequential([
        iaa.PerspectiveTransform(scale=(0.01, 0.1)),
        iaa.Affine(shear=(-5, 5)),
    ]),  # Perspective transform, shearing, and gamma contrast

    iaa.Sequential([
        iaa.PerspectiveTransform(scale=(0.01, 0.1), keep_size=True),
        iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}),
    ]),  # Perspective transform, non-uniform scaling, and Gaussian blur

    iaa.Sequential([
        iaa.PerspectiveTransform(scale=(0.01, 0.1)),
        iaa.Affine(translate_px={"x": (-10, 10), "y": (-10, 10)}),
    ]),  # Perspective transform, pixel translation, and sharpening
]

index = 561

for i, seq in enumerate(perspective_transformations, 1):
    seq_det = seq.to_deterministic()  # Call this once per batch
    augmented_images = seq_det.augment_images(original_images)
    augmented_masks = seq_det.augment_images(masks)

    for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks), 1):
        imageio.imwrite(f'../512_data_with_augmentation/x/x_{index}.png', image)
        imageio.imwrite(f'../512_data_with_augmentation/y/y_{index}.png', mask)
        index += 1
