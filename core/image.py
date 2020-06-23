from typing import Tuple

import numpy as np
from PIL.Image import Image
import cv2

ROICoordinates = Tuple[int, int, int, int]
ScaledROICoordinates = Tuple[float, float, float, float]

DEFAULT_COLOR_SPACE = 'RGB'


def img_to_array(image: Image, color_space: str = DEFAULT_COLOR_SPACE):
    array_image = np.array(image)
    if color_space == DEFAULT_COLOR_SPACE:
        return array_image
    cv2_color_code = getattr(cv2, f'COLOR_{DEFAULT_COLOR_SPACE}2{color_space}')
    return cv2.cvtColor(array_image, cv2_color_code)


class ROIImage:

    def __init__(self, image: Image, coordinates: ROICoordinates,
                 parent_image: Image = None):
        self.image = image
        self.coordinates = coordinates
        self.parent_image = parent_image


class ImageExtractor:

    def __init__(self, image: Image):
        self.image = image
        self.array_image = img_to_array(self.image)

    def extract_from_scale(self, scaled_coordinates: ScaledROICoordinates):
        roi_coordinates = self.scale_roi_coordinates(scaled_coordinates)
        return self.extract(roi_coordinates)

    def extract(self, roi_coordinates: ROICoordinates) -> ROIImage:
        coordinates = self._normalize_coordinates(roi_coordinates)
        roi_image = self.image.crop(coordinates)
        return ROIImage(roi_image, coordinates, self.image)

    def scale_roi_coordinates(self, roi_coordinates: ScaledROICoordinates) -> ROICoordinates:
        # compute the (x, y)-coordinates of the bounding box for the object
        roi_coordinates = roi_coordinates * np.array([
            self.image.width, self.image.height,
            self.image.width, self.image.height])
        return roi_coordinates.astype("int")

    def _normalize_coordinates(self, roi_coordinates: ROICoordinates) -> ROICoordinates:
        # ensure the bounding boxes fall within the dimensions of the frame
        norm_x = lambda x: int(min(self.image.width - 1, x))
        norm_y = lambda y: int(min(self.image.height - 1, y))
        start_x, start_y, end_x, end_y = roi_coordinates
        return norm_x(start_x), norm_y(start_y), norm_x(end_x), norm_y(end_y)
