from simple_settings import settings

from core.computer_vision.recognition.detection import CV2DetectionModel
from core.utils.model_loader import ModelLoader


class FaceDetection(CV2DetectionModel):

    MODEL = ModelLoader().from_cafe(*settings.FACE_DETECTOR_MODEL_CAFFE)
    IMAGE_SIZE = (300, 300)
    MEAN = settings.MEAN
    COLOR_SPACE = 'BGR'
