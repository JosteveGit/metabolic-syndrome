from imgaug import augmenters as iaa
import imageio

original_images = [imageio.v2.imread(f'../512_data/x/x_{i}.png') for i in range(1, 15)]
masks = [imageio.v2.imread(f'../512_data/y/y_{i}.png') for i in range(1, 15)]

blur_augmentations = [
    iaa.Sequential([iaa.GaussianBlur(sigma=(0.5, 1.5))]),  # Gaussian blur with variable sigma
    iaa.Sequential([iaa.AverageBlur(k=(2, 5))]),  # Average blur with variable kernel size
    iaa.Sequential([iaa.MedianBlur(k=(3, 7))]),  # Median blur with variable kernel size
    # Bilateral blur, preserving edges
    iaa.Sequential([iaa.MotionBlur(k=(3, 7), angle=[-45, 45])]),  # Motion blur with variable kernel size and angle
    iaa.Sequential([iaa.GaussianBlur(sigma=(1, 3)), iaa.AverageBlur(k=(2, 3)), iaa.MedianBlur(k=(3))]),
    # Combination of Gaussian, average, and median blurs
]

index = 491

for i, seq in enumerate(blur_augmentations, 1):
    seq_det = seq.to_deterministic()  # Call this once per batch
    augmented_images = seq_det.augment_images(original_images)
    augmented_masks = masks

    for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks), 1):
        imageio.imwrite(f'../512_data_with_augmentation/x/x_{index}.png', image)
        imageio.imwrite(f'../512_data_with_augmentation/y/y_{index}.png', mask)
        index += 1
