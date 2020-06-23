# Face mask recognition and tracking
Detects multiple people and verifies if each person is wearing a face mask or not in real time.
It uses a deep learning single shoot detector, a deep learning classifier and a centroid object tracker. 

From the tutorials:
- https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
- https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

## Solution
1) Uses a face detection model retrieve the faces.
2) The retrieved faces are classified by a face mask classifier whether or not is wearing a face mask.
3) The faces (independent of it's classification) are tracked by a centroid tracker.

## Characteristics
- Support for multiple faces and face masks.
- Real time recognition and tracking.
- Usage of python `tkinter` as GUI.
- The effective range is about 0.5 to 3m (not too close, not too far).
- While the face recognition support slanted images, the mask detection does not.
- The face detection may give an incorrect output when the face is out of the effectve range.
- The centroid tracker may not update the ID to the same face.
- The centroid tracker may not assign a centroid object correctly.
- The face net and mask net have different supported colors (RGB vs BGR) and images sizes for input.

## Settings

| Option               | Model       |
| -------------------- | ----------- |
| Face detection       | ResNet SSD  |
| Face mask classifier | MobilNetV2  |
| Object tracker       | Centroid tracker |
