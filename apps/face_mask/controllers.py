import logging
from simple_settings import settings

from core.video import VideoStreamer
from core.image import Image

from .services.face import FaceDetection
from .services.mask import MaskClassifier
from .services.video import VideoRecognitionTracker

from .views import FaceMaskRecognitionWindow

LOG = logging.getLogger(__name__)
VIDEO_SOURCE = settings.VIDEO_SOURCE
SKIP_FRAME = settings.SKIP_FRAME


class FaceMaskRecognition:

    SKIP_FRAME = SKIP_FRAME
    VIDEO_SOURCE = VIDEO_SOURCE

    def __init__(self):
        self._init_recognition()
        self.window = FaceMaskRecognitionWindow()
        self.video_source = VideoStreamer(source=self.VIDEO_SOURCE)

    def _init_recognition(self):
        detection_model = FaceDetection()
        classification_model = MaskClassifier()
        self.video_recognition = VideoRecognitionTracker(
            detection_model, classification_model)

    def open_window(self):
        self.window.open()
        frame = self.video_source.read_frame()
        self.window.resize(*frame.size)
        self._live_stream()

    def _live_stream(self):
        try:
            self._update_frame()
        except:
            LOG.exception('Unhandled exception')
        finally:
            self.window.after(1, self._live_stream)

    def _update_frame(self):
        frame = self.video_source.read_frame()
        self._process_frame(frame)
        self.window.render(frame, self.detected_objects)

    def _process_frame(self, frame: Image):
        if self.video_source.frame_count % self.SKIP_FRAME == 0:
            self.video_recognition.recognize(frame)
        else:
            self.video_recognition.update_trackers(frame)

    @property
    def detected_objects(self):
        return self.video_recognition.detected_objects
