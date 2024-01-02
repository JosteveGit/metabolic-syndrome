import pydicom
from PIL import Image
import numpy as np


class DicomToImage:

    def __init__(self, dicom_file_path, output_path, standard_size):
        self.dicom_file_path = dicom_file_path
        self.output_path = output_path
        self.standard_size = standard_size

    def _read_dicom(self) -> tuple:
        dicom_file = pydicom.dcmread(self.dicom_file_path)

        new_image = dicom_file.pixel_array.astype(float)
        scaled_image = (np.maximum(new_image, 0) / new_image.max(initial=0)) * 255.0
        scaled_image = np.uint8(scaled_image)

        pixel_spacing = dicom_file.PixelSpacing
        slice_thickness = dicom_file.SliceThickness
        return scaled_image, pixel_spacing, slice_thickness,

    def _convert_and_resize(self, pixel_array) -> tuple:
        image = Image.fromarray(pixel_array)
        original_size = image.size
        resized_image = image.resize(self.standard_size, Image.LANCZOS)

        resized_image.save(self.output_path)
        return original_size, resized_image

    def process(self) -> tuple:
        pixel_array, pixel_spacing, slice_thickness = self._read_dicom()
        original_size, resized_image = self._convert_and_resize(pixel_array)
        return tuple(pixel_spacing), slice_thickness, original_size, resized_image


DicomToImage(
    dicom_file_path="/Users/josteveadekanbi/Documents/RESEARCH/MyDataset/1/FO-1105916004508578806.dcm",
    output_path="/Users/josteveadekanbi/Documents/RESEARCH/MyDataset/1/FO-1105916004508578806.png",
    standard_size=(512, 512)
).process()
