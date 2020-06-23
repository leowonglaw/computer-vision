from typing import Iterable
from abc import ABC, abstractmethod

from core.image import Image, ROIImage


class AbstractVideoRecognition(ABC):

    @abstractmethod
    def recognize(self, frame: Image) -> Iterable[ROIImage]:
        pass


class AbstractVideoTrakingManager(ABC):

    @abstractmethod
    def update_trackers(self, frame: Image):
        pass
