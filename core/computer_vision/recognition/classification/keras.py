from typing import List, Tuple

from simple_settings import settings
import numpy as np
from tensorflow.keras.models import Model as KerasModel
import cv2

from core.image import Image, img_to_array
from .abstract import AbstractClasificationModel, Classification, Prediction


class KerasClassificationModel(AbstractClasificationModel):

    MODEL: KerasModel
    CLASES: List[str]
    IMAGE_SIZE: Tuple[int, int] = settings.IMAGE_SIZE
    COLOR_SPACE = 'RGB'

    def classify(self, image: Image) -> Classification:
        prediction_list = self.predict(image)
        return Classification(zip(self.CLASES, prediction_list))

    def predict(self, image: Image) -> Prediction:
        arr_img = self._transform_image(image)
        return self.MODEL.predict(arr_img)[0]

    def _transform_image(self, image: Image):
        arr_img = img_to_array(image, self.COLOR_SPACE)
        arr_img = cv2.resize(arr_img, self.IMAGE_SIZE)
        arr_img = np.expand_dims(arr_img, axis=0)
        return arr_img
