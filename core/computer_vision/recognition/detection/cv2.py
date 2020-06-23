from typing import Tuple, Iterable

import cv2
from simple_settings import settings

from core.image import Image, ROIImage, ScaledROICoordinates, ImageExtractor, img_to_array

from .abstract import AbstractDetectionModel


class CV2DetectionModel(AbstractDetectionModel):

    MODEL: cv2.dnn_Net
    IMAGE_SIZE: Tuple[int, int] = None
    MEAN: Tuple[int, int, int] = None
    SCALEFACTOR: float = 1.0
    COLOR_SPACE = 'RGB'

    def __init__(self, min_confidence: float = settings.CONFIDENCE):
        self.min_confidence = min_confidence

    def detect(self, image: Image) -> Iterable[ROIImage]:
        image_extractor = ImageExtractor(image)
        detections = self._get_detections(image_extractor.array_image)
        detection_qt: int = detections.shape[2]
        for i in range(0, detection_qt):
            confidence: float = detections[0, 0, i, 2]
            scaled_roi_coordinates: ScaledROICoordinates = detections[0, 0, i, 3:7]
            if confidence > self.min_confidence:
                yield image_extractor.extract_from_scale(scaled_roi_coordinates)

    def _get_detections(self, array_image):
        array_image = self._transform_image(array_image)
        blob = cv2.dnn.blobFromImage(
            array_image,
            scalefactor=self.SCALEFACTOR,
            size=self.IMAGE_SIZE,
            mean=self.MEAN)
        self.MODEL.setInput(blob)
        detections = self.MODEL.forward()
        return detections

    def _transform_image(self, img: Image):
        return img_to_array(img, self.COLOR_SPACE)
