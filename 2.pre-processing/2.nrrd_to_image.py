import nibabel as nib
from PIL import Image
import numpy as np
import nrrd


class NrrdToImage:

    def __init__(self, nrrd_file_path, output_path, size: tuple = (512, 512)):
        self.nrrd_file_path = nrrd_file_path
        self.output_path = output_path
        self.size = size

    def process(self):
        # Read the NRRD file
        nrrd_data, header = nrrd.read(self.nrrd_file_path)

        # Convert to NIfTI
        nifti_img = nib.Nifti1Image(nrrd_data, np.eye(4))

        # Get the data from the file
        nifti_data = nifti_img.get_fdata()

        # Select the slice you are interested in
        slice_data = nifti_data[:, :, 0]  # First slice in the Z-direction

        # Normalize the slice to be between 0 and 255
        slice_normalized = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255.0
        slice_normalized = slice_normalized.astype(np.uint8)

        # Create an RGB image based on the normalized grayscale image
        rgb_slice = np.stack((slice_normalized,) * 3, axis=-1)

        # # Assign the color red to one class (let's assume gray corresponds to 127 in the normalized image)
        rgb_slice[slice_normalized == 127] = [255, 255, 255]
        #
        # # Assign the color green to another class (let's assume white corresponds to 255 in the normalized image)
        rgb_slice[slice_normalized == 255] = [127, 127, 127]

        slice_image = Image.fromarray(rgb_slice)
        slice_image = slice_image.resize(self.size, Image.LANCZOS)

        slice_image = slice_image.rotate(-90, expand=True)  # Rotate 90 degrees

        slice_image = slice_image.transpose(Image.FLIP_LEFT_RIGHT)

        slice_image.save(self.output_path)

        return self.output_path
