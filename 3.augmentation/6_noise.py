from imgaug import augmenters as iaa
import imageio

original_images = [imageio.v2.imread(f'../512_data/x/x_{i}.png') for i in range(1, 15)]
masks = [imageio.v2.imread(f'../512_data/y/y_{i}.png') for i in range(1, 15)]

noise_augmentations = [
    iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=(0.01, 0.02)),
        iaa.AdditiveGaussianNoise(scale=(0.03, 0.04), per_channel=True)
    ]),  # Gaussian noise, with some channel-specific variations

    iaa.Sequential([
        iaa.Add((5, 10)),  # Increase pixel intensity
        iaa.Add((-10, -5))  # Decrease pixel intensity
    ]),  # Slight increase or decrease in the pixel values

    iaa.Sequential([
        iaa.Multiply((0.99, 1.01)),
        iaa.ContrastNormalization((0.98, 1.02))
    ]),  # Small changes in brightness and contrast

    iaa.Sequential([
        iaa.GaussianBlur(sigma=(0.5, 1.0)),
        iaa.Sharpen(alpha=(0.0, 0.1))
    ]),  # Mild blurring followed by slight sharpening
]

index = 435

for i, seq in enumerate(noise_augmentations, 1):
    seq_det = seq.to_deterministic()  # Call this once per batch
    augmented_images = seq_det.augment_images(original_images)
    augmented_masks = masks

    for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks), 1):
        imageio.imwrite(f'../512_data_with_augmentation/x/x_{index}.png', image)
        imageio.imwrite(f'../512_data_with_augmentation/y/y_{index}.png', mask)
        index += 1
