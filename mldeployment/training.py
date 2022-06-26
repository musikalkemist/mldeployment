"""This module trains a ConvNet model built with Keras, for classifying
digits using the MNIST dataset.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from tensorflow import keras


NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
NUM_EPOCHS = 1


def prepare_mnist_training_data() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Prepares training data for the model.

    Returns:
        A tuple of four numpy arrays:
            - The training images,
            - The training labels,
            - The test images,
            - The test labels.
    """
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], *INPUT_SHAPE).astype(
        np.float32
    )
    test_images = test_images.reshape(test_images.shape[0], *INPUT_SHAPE).astype(
        np.float32
    )

    # Normalize the images to the range [0, 1]
    train_images /= 255.0
    test_images /= 255.0

    # One-hot encode the labels
    train_labels = keras.utils.to_categorical(train_labels, NUM_CLASSES)
    test_labels = keras.utils.to_categorical(test_labels, NUM_CLASSES)

    return train_images, train_labels, test_images, test_labels


def build_convnet_model(input_shape: Tuple, num_classes: int) -> keras.Model:
    """Builds a simple ConvNet model.

    Args:
        input_shape: The shape of the input data.
        num_classes: The number of classes to classify.

    Returns:
        A Keras model.
    """
    model = keras.models.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_model(
    model: keras.Model,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    num_epochs: int,
) -> keras.Model:
    """Trains a model.

    Args:
        model: The model to train.
        train_images: The training images.
        train_labels: The training labels.
        test_images: The test images.
        test_labels: The test labels.
        num_epochs: The number of epochs to train for.
    Returns:
        The trained model.
    """
    model.fit(
        train_images,
        train_labels,
        epochs=num_epochs,
        validation_data=(test_images, test_labels),
    )
    return model


def save_model(model: keras.Model, model_save_path: Path) -> None:
    """Saves a model to a file.

    Args:
        model: The model to save.
        model_file: The file to save the model to.
    """
    model.save(model_save_path)


def main():
    """Trains a model for classifying digits using the MNIST dataset."""

    train_images, train_labels, test_images, test_labels = prepare_mnist_training_data()

    model = build_convnet_model(INPUT_SHAPE, NUM_CLASSES)

    model = train_model(
        model, train_images, train_labels, test_images, test_labels, NUM_EPOCHS
    )

    save_model(model, Path("model"))


if __name__ == "__main__":
    main()
