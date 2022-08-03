"""This module saves a Keras model to BentoML."""

from pathlib import Path

from tensorflow import keras
import bentoml


def load_model_and_save_it_to_bento(model_file: Path) -> None:
    """Loads a keras model from disk and saves it to BentoML."""
    model = keras.models.load_model(model_file)
    bento_model = bentoml.keras.save_model("keras_model", model)
    print(f"Bento model tag = {bento_model.tag}")


if __name__ == "__main__":
    load_model_and_save_it_to_bento(Path("model"))