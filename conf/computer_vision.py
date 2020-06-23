## TRAINER
LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 0.2

## DETECTION
IMAGE_SIZE = (224, 224)
MEAN = (104.0, 177.0, 123.0)

## OBJECT TRAKER
MAX_DISAPPEARED = 50
MAX_DISTANCE = 50

## OBJECT DETECTOR
CONFIDENCE = 0.5

## MODELS
MODELS_DIR = "data/models"
MASK_DETECTOR_MODEL = MODELS_DIR + "/mask_detector/mask_detector.model"
FACE_DETECTOR_MODEL_CAFFE = (MODELS_DIR + "/face_detector/deploy.prototxt",
                             MODELS_DIR + "/face_detector/ssd_mobilenet.caffemodel")
