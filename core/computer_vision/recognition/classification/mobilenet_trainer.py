# USAGE
# python train_mask_detector.py --dataset dataset

# import the necessary packages
import os
import argparse
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array, load_img,)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Dense, Input, AveragePooling2D,
    Dropout, Flatten)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from simple_settings import settings

IMAGE_SIZE = settings.IMAGE_SIZE
LEARNING_RATE = settings.LEARNING_RATE
EPOCHS = settings.EPOCHS
BATCH_SIZE = settings.BATCH_SIZE
TEST_SIZE = settings.TEST_SIZE

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


class MobileNetTrainer:
    ''' load the MobileNetV2 network, ensuring the head FC layer sets are
        left off
    '''

    def __init__(self, image_size=IMAGE_SIZE)):
        self.image_size = image_size
        self.base_model = MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,)
        self.head_model = self._init_head_model()
        self.model = Model(inputs=self.base_model.input, outputs=head_model)
        self.label_binarizer = LabelBinarizer()
        self.data = []
        self.labels = []

    def load_data(self, path):
        image_paths = list(paths.list_images(path))
        for image_path in image_paths:
            image, label = self._load_image(image_path)
            self.data.append(image)
            self.labels.append(label)

    def _transform_data(self):
        # convert the data and labels to NumPy arrays
        self.data = np.array(self.data, dtype="float32")
        self.labels = np.array(self.labels)

        # perform one-hot encoding on the labels
        self.labels = self.label_binarizer.fit_transform(self.labels)
        self.labels = to_categorical(self.labels)

    def _load_image(self, image_path):
        # extract the class label from the filename
        label = image_path.split(os.path.sep)[-2]

        image = load_img(image_path, target_size=self.image_size)
        image = img_to_array(image)
        image = preprocess_input(image)
        return image, label

    def _init_head_model(self):
        # construct the head of the model that will be placed on top of the
        # the base model
        head_model = self.base_model.output
        head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(128, activation="relu")(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(2, activation="softmax")(head_model)
        return head_model

    def _process_model(self):
        # place the head FC model on top of the base model (this will become
        # the actual model we will train)

        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the first training process
        for layer in self.base_model.layers:
            layer.trainable = False

    def compile_model(self):
        opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
        model.compile(
            loss="binary_crossentropy",
            optimizer=opt,
            metrics=["accuracy"])

    def train_head_model(self, image_data_generator: ImageDataGenerator):
        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        (train_x, test_x, train_y, test_Y) = train_test_split(data, labels,
            test_size=TEST_SIZE, stratify=labels, random_state=42)
        head = model.fit(
            image_data_generator.flow(train_x, train_y, batch_size = BATCH_SIZE),
            steps_per_epoch=len(train_x) // BATCH_SIZE,
            validation_data=(test_x, test_Y),
            validation_steps=len(test_x) // BATCH_SIZE,
            epochs=EPOCHS)

    def predict(self, test_set, train_set):
        # make predictions on the testing set
        print("[INFO] evaluating network...")
        pred_idx_list = model.predict(test_set, batch_size=BATCH_SIZE)

        # for each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        pred_idx_list = np.argmax(pred_idx_list, axis=1)

        # show a nicely formatted classification report
        print(classification_report(train_set.argmax(axis = 1), pred_idx_list,
            target_names = self.label_binarizer.classes_))

    def save_model(self, path, format="h5"):
        # serialize the model to disk
        print("[INFO] saving mask detector model...")
        model.save(path, save_format=format)

    def plot(self, title: str, filepath: str):
        plotter = MobileNetTrainerPlotter(title, self)
        plotter.plot(filepath)


class MobileNetTrainerPlotter:

    HISTORY_LABEL_LIST = [
        ("loss", "train_loss"),
        ("val_loss", "val_loss"),
        ("accuracy", "train_acc"),
        ("val_accuracy", "val_acc"),
    ]

    def __init__(self, title, trainer: MobileNetTrainer):
        self.title = title
        self.trainer = trainer
        plt.style.use("ggplot")

    def plot(self, filepath):
        self.draw()
        self.set_labels()
        self.save(filepath)

    def draw(self):
        for history, label in self.HISTORY_LABEL_LIST:
            plt.plot(np.arange(0, EPOCHS),
                    self.trainer.head.history[history],
                    label=label)
        plt.figure()

    def set_labels(self):
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc = "lower left")

    def save(self, filepath):
        plt.savefig(filepath)
