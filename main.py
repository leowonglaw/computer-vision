import logging.config
from simple_settings import settings

from core.app import GUIApp
from apps.face_mask.controllers import FaceMaskRecognition

logging.config.dictConfig(settings.LOGGING)


if __name__ == '__main__':
    with GUIApp():
        client = FaceMaskRecognition()
        client.open_window()
