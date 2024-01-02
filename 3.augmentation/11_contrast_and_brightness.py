from imgaug import augmenters as iaa
import imageio

original_images = [imageio.v2.imread(f'../512_data/x/x_{i}.png') for i in range(1, 15)]
masks = [imageio.v2.imread(f'../512_data/y/y_{i}.png') for i in range(1, 15)]

contrast_and_brightness_augmentations = [
    iaa.Sequential([
        iaa.LinearContrast((0.8, 1.2)),  # Adjust contrast
        iaa.Multiply((0.9, 1.1))  # Adjust brightness
    ]),  # Simple contrast and brightness adjustment

    iaa.Sequential([
        iaa.GammaContrast((0.5, 1.5)),  # Adjust gamma contrast
        iaa.SigmoidContrast(gain=(5, 10), cutoff=(0.4, 0.6)),  # Sigmoid contrast adjustment
        iaa.Multiply((0.95, 1.05))  # Slight brightness adjustment
    ]),  # Gamma and sigmoid contrast adjustments with brightness

    iaa.Sequential([
        iaa.Multiply((0.8, 1.2)),  # Adjust brightness
        iaa.LinearContrast((0.95, 1.05))  # Fine-tune contrast
    ]),  # All channel histogram equalization with contrast and brightness

    iaa.Sequential([
        iaa.Alpha((0.5, 1.0), iaa.Multiply((0.8, 1.2)), per_channel=True),  # Adjust brightness with alpha blending
        iaa.ContrastNormalization((0.75, 1.25))  # Normalize contrast
    ])  # Alpha blending brightness and contrast normalization
]

index = 813

for i, seq in enumerate(contrast_and_brightness_augmentations, 1):
    seq_det = seq.to_deterministic()  # Call this once per batch
    augmented_images = seq_det.augment_images(original_images)
    augmented_masks = masks

    for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks), 1):
        imageio.imwrite(f'../512_data_with_augmentation/x/x_{index}.png', image)
        imageio.imwrite(f'../512_data_with_augmentation/y/y_{index}.png', mask)
        index += 1
