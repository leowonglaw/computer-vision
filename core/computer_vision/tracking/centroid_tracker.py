from typing import Set, Tuple, List, Dict
from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist
from simple_settings import settings

from core.image import ROICoordinates


Centroid = Tuple[int, int]
TrakerID = int


class CentroidTraker:

    def __init__(self, id: TrakerID, centroid: Centroid):
        self.id = id
        self.centroid = centroid
        self.disappeared_count = 0


class CentroidManager:

    def __init__(self, max_disappeared=settings.MAX_DISAPPEARED, max_distance=settings.MAX_DISTANCE):
        # initialize the next unique traker ID along with two ordered
        # dictionaries used to keep track of mapping a given traker
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.counter = 0
        self.trakers: Dict[CentroidTraker] = OrderedDict()

        # store the number of maximum consecutive frames a given
        # traker is allowed to be marked as "disappeared" until we
        # need to deregister the traker from tracking
        self.max_disappeared = max_disappeared

        # store the maximum distance between centroids to associate
        # an traker -- if the distance is larger than this maximum
        # distance we'll start to mark the traker as "disappeared"
        self.max_distance = max_distance
        # aux variable
        self._input_centroids = None

    def register(self, centroid: Centroid):
        # when registering an traker we use the next available traker
        # ID to store the centroid
        self.trakers[self.counter] = CentroidTraker(self.counter, centroid)
        self.counter += 1

    def deregister(self, traker_id: TrakerID):
        # to deregister an traker ID we delete the traker ID from
        # both of our respective dictionaries
        del self.trakers[traker_id]

    def update(self, rects: List[ROICoordinates]):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            self._deregister_max_desappeared_trakers()
            # return early as there are no centroids or tracking info
            # to update
            return self.trakers

        self._input_centroids = self._get_input_centroids(rects)

        # if we are currently not tracking any trakers take the input
        # centroids and register each of them
        if len(self.trakers) == 0:
            for i in range(0, len(self._input_centroids)):
                self.register(self._input_centroids[i])
        else:
            # otherwise, are are currently tracking trakers so we need to
            # try to match the input centroids to existing traker
            # centroids
            self.match_input_centroid_to_traker()
        # return the set of trakers
        return self._get_centroids()

    def _get_centroids(self):
        centroids = [None] * len(self._input_centroids)
        centroids_map = {
            tuple(cent): idx
            for idx, cent in enumerate(self._input_centroids)
        }
        for traker in self.trakers.values():
            idx = centroids_map.get(tuple(traker.centroid))
            if idx is not None:
                centroids[idx] = traker
        return centroids

    def match_input_centroid_to_traker(self):
        # compute the distance between each pair of traker
        # centroids and input centroids, respectively -- our
        # goal will be to match an input centroid to an existing
        # traker centroid
        centroids_list = [traker.centroid for traker in self.trakers.values()]
        distance_array = dist.cdist(centroids_list, self._input_centroids)
        unused_rows, unused_cols = self._get_unused_pixles(distance_array)
        # in the event that the number of traker centroids is
        # equal or greater than the number of input centroids
        # we need to check and see if some of these trakers have
        # potentially disappeared
        if distance_array.shape[0] >= distance_array.shape[1]:
            self._unregister_disappeared_trakers(unused_rows)
        # otherwise, if the number of input centroids is greater
        # than the number of existing traker centroids we need to
        # register each new input centroid as a traker
        else:
            self._register_appeared_trakers(unused_cols)

    def _get_input_centroids(self, rects: List[ROICoordinates]) -> List[Centroid]:
        # initialize an array of input centroids for the current frame
        input_centroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for i, (start_x, start_y, end_x, end_y) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (c_x, c_y)
        return input_centroids

    def _deregister_max_desappeared_trakers(self) -> None:
        # loop over any existing tracked trakers and mark them
        # as disappeared

        for traker in list(self.trakers.values()):
            traker.disappeared_count += 1

            # if we have reached a maximum number of consecutive
            # frames where a given traker has been marked as
            # missing, deregister it
            if traker.disappeared_count > self.max_disappeared:
                self.deregister(traker.id)

    def _get_unused_pixles(self, distance_array):
        used_rows, used_cols = self._get_used_indexes(distance_array)
        # compute both the row and column index we have NOT yet examined
        unused_rows = set(range(0, distance_array.shape[0])).difference(used_rows)
        unused_cols = set(range(0, distance_array.shape[1])).difference(used_cols)
        return unused_rows, unused_cols

    def _get_used_indexes(self, distance_array):
        # in order to perform this matching we must (1) find the
        # smallest value in each row and then (2) sort the row
        # indexes based on their minimum values so that the row
        # with the smallest value as at the *front* of the index
        # list
        rows = distance_array.min(axis=1).argsort()

        # next, we perform a similar process on the columns by
        # finding the smallest value in each column and then
        # sorting using the previously computed row index list
        cols = distance_array.argmin(axis=1)[rows]

        # in order to determine if we need to update, register,
        # or deregister an traker we need to keep track of which
        # of the rows and column indexes we have already examined
        used_rows: Set[int] = set()
        used_cols: Set[int] = set()

        traker_ids = list(self.trakers.keys())
        # loop over the combination of the (row, column) index
        # tuples
        for (row, col) in zip(rows, cols):
            # if the distance between centroids is greater than
            # the maximum distance, do not associate the two
            # centroids to the same traker
            if distance_array[row, col] > self.max_distance:
                continue

            # otherwise, grab the traker ID for the current row,
            # set its new centroid, and reset the disappeared
            # counter
            traker_id = traker_ids[row]
            traker = self.trakers[traker_id]
            traker.centroid = self._input_centroids[col]
            traker.disappeared_count = 0

            # indicate that we have examined each of the row and
            # column indexes, respectively
            used_rows.add(row)
            used_cols.add(col)
        return used_rows, used_cols

    def _unregister_disappeared_trakers(self, unused_rows):
        # grab the set of traker IDs and corresponding centroids
        traker_ids = list(self.trakers.keys())
        # loop over the unused row indexes
        for row in unused_rows:
            # grab the traker ID for the corresponding row
            # index and increment the disappeared counter
            traker_id = traker_ids[row]
            traker = self.trakers[traker_id]
            traker.disappeared_count += 1
            # check to see if the number of consecutive
            # frames the traker has been marked "disappeared"
            # for warrants deregistering the traker
            if traker.disappeared_count > self.max_disappeared:
                self.deregister(traker_id)

    def _register_appeared_trakers(self, unused_cols):
        for col in unused_cols:
            self.register(self._input_centroids[col])
