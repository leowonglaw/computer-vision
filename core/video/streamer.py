import logging
from typing import Iterable

from imutils.video import FPS
import cv2
from PIL import Image as image_utils

from ..image import Image

LOG = logging.getLogger(__name__)


class VideoStreamer:

    def __init__(self, source=0, window_title="Frame"):
        self.window_title = window_title
        self.video_stream = cv2.VideoCapture(source)
        self.frame_count = 0
        self._current_frame = None
        self._is_streaming = False
        self.fps = FPS().start()

    def stream(self, display=True) -> Iterable[Image]:
        self._is_streaming = True
        while self._is_streaming:
            frame = self.read_frame()
            yield frame
            if display:
                self.display_frame()

    def stop(self):
        self._is_streaming = False
        self.video_stream.stop()
        self.fps.stop()
        LOG.info("streamer elapsed time: %.2f", self.fps.elapsed())
        LOG.info("streamer FPS: %.2f", self.fps.fps())

    def read_frame(self) -> Image:
        _, self._current_frame = self.video_stream.read()
        array_frame = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2RGB)
        self._update_fps()
        return image_utils.fromarray(array_frame)

    def display_frame(self):
        cv2.imshow(self.window_title, self._current_frame)

    def _update_fps(self):
        self.frame_count += 1
        self.fps.update()

    @property
    def engine(self):
        return cv2
