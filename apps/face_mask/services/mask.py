from simple_settings import settings

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from core.computer_vision.recognition.classification.keras import KerasClassificationModel
from core.utils.model_loader import ModelLoader
from core.image import Image


class MaskClassifier(KerasClassificationModel):

    MODEL = ModelLoader().from_keras(settings.MASK_DETECTOR_MODEL)
    CLASES = ['MASK', 'NO_MASK']

    def _transform_image(self, image: Image):
        arr_img = super()._transform_image(image)
        return preprocess_input(arr_img)
