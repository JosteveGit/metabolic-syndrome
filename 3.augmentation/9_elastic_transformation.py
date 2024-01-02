from imgaug import augmenters as iaa
import imageio

original_images = [imageio.v2.imread(f'../512_data/x/x_{i}.png') for i in range(1, 15)]
masks = [imageio.v2.imread(f'../512_data/y/y_{i}.png') for i in range(1, 15)]

# Complex affine transformations without blur or noise
elastic_transformations = [
    iaa.Sequential([
        iaa.ElasticTransformation(alpha=50, sigma=5),  # Intense elastic transformation
        iaa.Sequential([iaa.Fliplr(1.0)]),  # Horizontally flip all images
        iaa.Sequential([iaa.Flipud(1.0)]),
    ]),

    iaa.Sequential([
        iaa.ElasticTransformation(alpha=10, sigma=2),  # Mild elastic transformation
        iaa.Sequential([iaa.Fliplr(1.0)]),  # Horizontally flip all images
    ]),

    iaa.Sequential([
        iaa.ElasticTransformation(alpha=30, sigma=3),  # Moderate elastic transformation
        iaa.Sequential([iaa.Flipud(1.0)]),
    ]),

    iaa.Sequential([
        iaa.ElasticTransformation(alpha=20, sigma=4),  # Elastic transformation with different parameters
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})  # Translation
    ]),

    iaa.Sequential([
        iaa.ElasticTransformation(alpha=40, sigma=6),  # Strong elastic transformation
        iaa.Affine(scale=(0.9, 1.1))  # Uniform scaling
    ]),

    iaa.Sequential([
        iaa.ElasticTransformation(alpha=60, sigma=7),  # Very strong elastic transformation
        iaa.Affine(rotate=(-10, 10))  # Rotation
    ])
]

index = 645

for i, seq in enumerate(elastic_transformations, 1):
    seq_det = seq.to_deterministic()  # Call this once per batch
    augmented_images = seq_det.augment_images(original_images)
    augmented_masks = seq_det.augment_images(masks)

    for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks), 1):
        imageio.imwrite(f'../512_data_with_augmentation/x/x_{index}.png', image)
        imageio.imwrite(f'../512_data_with_augmentation/y/y_{index}.png', mask)
        index += 1
