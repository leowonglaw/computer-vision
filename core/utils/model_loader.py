import logging
from typing import Dict, Hashable, Any

import cv2
from tensorflow.keras.models import load_model

from .singleton import SingletonMeta


LOG = logging.getLogger(__name__)


class ModelLoader(metaclass=SingletonMeta):

    _models: Dict[Hashable, Any] = dict()

    def from_keras(self, path):
        model = self._models.get(path)
        if not model:
            LOG.info("Loading keras model: %s", path)
            model = load_model(path)
            self._models[path] = model
        return model

    def from_cafe(self, prototxt_path, caffemodel_path):
        path = prototxt_path, caffemodel_path
        model = self._models.get(path)
        if not model:
            LOG.info("Loading caffe model: %s", path)
            model = cv2.dnn.readNetFromCaffe(*path)
            self._models[path] = model
        return model
