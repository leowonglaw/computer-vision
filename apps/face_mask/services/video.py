from typing import List, Iterable
import dlib

from core.computer_vision.recognition.detection import AbstractDetectionModel
from core.computer_vision.recognition.classification import AbstractClasificationModel
from core.computer_vision.tracking import CentroidManager
from core.computer_vision.video import AbstractVideoRecognition, AbstractVideoTrakingManager
from core.image import Image, img_to_array

from ..models import ROIImage, DetectedObject


class VideoRecognitionTracker(AbstractVideoRecognition, AbstractVideoTrakingManager):

    def __init__(self, detection_model: AbstractDetectionModel, classifier_model: AbstractClasificationModel):
        self.video_recognition = VideoRecognition(detection_model, classifier_model)
        self.video_traking_manager = VideoTrakingManager()

    def recognize(self, frame: Image) -> Iterable[DetectedObject]:
        detected_objects = self.video_recognition.recognize(frame)
        self.video_traking_manager.set_trackers(frame, detected_objects)
        return detected_objects

    def update_trackers(self, frame: Image):
        self.video_traking_manager.update_trackers(frame, self.detected_objects)

    @property
    def detected_objects(self):
        return self.video_recognition.detected_objects


class VideoRecognition(AbstractVideoRecognition):

    def __init__(self, detection_model: AbstractDetectionModel, classifier_model: AbstractClasificationModel):
        self.detected_objects: List[DetectedObject] = []
        self.classifier_model = classifier_model
        self.detection_model = detection_model

    def recognize(self, frame: Image) -> Iterable[DetectedObject]:
        self.detected_objects = self._get_detected_objects_in_frame(frame)
        self._set_classifications()
        return self.detected_objects

    def _set_classifications(self):
        images = [obj.roi_image.image for obj in self.detected_objects]
        classifications = self.classifier_model.bulk_classify(images)
        for obj, classification in zip(self.detected_objects, classifications):
            obj.classification = classification

    def _get_detected_objects_in_frame(self, frame: Image) -> List[DetectedObject]:
        roi_images = self.detection_model.detect(frame)
        return [DetectedObject(roi_image) for roi_image in roi_images]


class VideoTrakingManager(AbstractVideoTrakingManager):

    _array_frame: List
    detected_objects: List[DetectedObject]

    def __init__(self):
        self.centroid_tracker = CentroidManager()

    def update_trackers(self, frame: Image, detected_objects):
        self.detected_objects = detected_objects
        self.__update_image(frame)
        coordinates_list = list(self._update_coordinates())
        return self._update_centroid_trakers(coordinates_list)

    def set_trackers(self, frame: Image, detected_objects):
        self.__update_image(frame)
        for obj in detected_objects:
            obj.tracker = self._create_tracker(obj.roi_image)

    def _create_tracker(self, roi_image: ROIImage):
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(*roi_image.coordinates)
        tracker.start_track(self._array_frame, rect)
        return tracker

    def _update_coordinates(self):
        for obj in self.detected_objects:
            yield self.__update_object_coordinates(obj)

    def _update_centroid_trakers(self, coordinates_list):
        centroids = self.centroid_tracker.update(coordinates_list)
        for obj, centroid in zip(self.detected_objects, centroids):
            obj.centroid_traker = centroid
        return centroids

    def __update_object_coordinates(self, detected_object: DetectedObject):
        tracker = detected_object.tracker
        tracker.update(self._array_frame)
        obj, pos = detected_object, tracker.get_position()
        obj.roi_image.coordinates = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())
        return obj.roi_image.coordinates

    def __update_image(self, frame: Image):
        self._array_frame = img_to_array(frame)
