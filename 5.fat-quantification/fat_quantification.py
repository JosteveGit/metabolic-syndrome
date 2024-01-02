from PIL import Image
import numpy as np


class FatQuantification:

    def __init__(self, pixel_spacing: tuple, slice_thickness: float, original_size: tuple, resized_image: Image,
                 masked_image: np.ndarray):
        self.pixel_spacing = pixel_spacing
        self.slice_thickness = slice_thickness
        self.original_size = original_size
        self.resized_image = resized_image
        self.masked_image = masked_image

    def _get_visceral_pixel_count_from_mask(self) -> int:
        return np.count_nonzero(self.masked_image == 127)

    def _get_superficial_pixel_count_from_mask(self) -> int:
        return np.count_nonzero(self.masked_image == 255)

    def process(self) -> tuple:
        standard_size = self.resized_image.size
        original_size = self.original_size

        scaling_factor_x = original_size[0] / standard_size[0]
        scaling_factor_y = original_size[1] / standard_size[1]

        visceral_pixel_count = self._get_visceral_pixel_count_from_mask()
        superficial_pixel_count = self._get_superficial_pixel_count_from_mask()

        total_factor = scaling_factor_x * scaling_factor_y

        visceral_original_pixel_count = visceral_pixel_count * total_factor
        superficial_original_pixel_count = superficial_pixel_count * total_factor

        area = self.pixel_spacing[0] * self.pixel_spacing[1]

        physical_area_visceral_mm2 = visceral_original_pixel_count * area
        physical_area_superficial_mm2 = superficial_original_pixel_count * area

        visceral_volume_mm3 = physical_area_visceral_mm2 * self.slice_thickness
        superficial_volume_mm3 = physical_area_superficial_mm2 * self.slice_thickness

        visceral_volume_cm3 = visceral_volume_mm3 / 1000
        superficial_volume_cm3 = superficial_volume_mm3 / 1000

        return visceral_volume_cm3, superficial_volume_cm3
