from typing import Union

from core.computer_vision.tracking import CentroidTraker
from core.image import ROIImage
from core.utils.functional import rgetattr


class DetectedObject:

    def __init__(self, roi_image: ROIImage,
                 centroid_traker: CentroidTraker = None,
                 tracker=None, classification=None):
        self.roi_image = roi_image
        self.centroid_traker = centroid_traker
        self.tracker = tracker
        self.classification = classification

    @property
    def id(self) -> Union[id, None]:
        return rgetattr(self, 'centroid_traker.id')

    def __str__(self):
        id_label = rgetattr(self, 'id', '')
        if self.classification:
            confidence_percentage = f"{self.classification.prediction * 100:.2f}%"
            return f"{self.classification.label} {id_label} - {confidence_percentage}"
        else:
            return id_label

    def __eq__(self, other: "DetectedObject"):
        return self.id == other.id

    def __ne__(self, other: "DetectedObject"):
        return not self == other
