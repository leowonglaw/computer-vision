from abc import ABC, abstractmethod
from typing import Iterable

from core.image import Image, ROIImage


class AbstractDetectionModel(ABC):

    @abstractmethod
    def detect(self, image: Image) -> Iterable[ROIImage]:
        pass
