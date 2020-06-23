from abc import ABC, abstractmethod
from typing import Iterable, Tuple
from operator import itemgetter

from core.image import Image

Prediction = Iterable[float]


class Classification:

    label: str
    prediction: float

    def __init__(self, classification_list: Iterable[Tuple[str, float]]):
        self.classification_list = classification_list
        self.label, self.prediction = max(classification_list, key=itemgetter(1))


class AbstractClasificationModel(ABC):

    @abstractmethod
    def classify(self, image: Image) -> Classification:
        pass

    def bulk_classify(self, images: Iterable[Image]) -> Iterable[Classification]:
        for img in images:
            yield self.classify(img)

    @abstractmethod
    def predict(self, image: Image) -> Prediction:
        pass

    def bulk_predict(self, images: Iterable[Image]) -> Iterable[Prediction]:
        for img in images:
            yield self.predict(img)
