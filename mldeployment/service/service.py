"""This module defines a BentoML service that uses a Keras model to classify
digits.
"""

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray


classifier_runner = bentoml.keras.get("keras_model:ko7zi6xt3cn4caku").to_runner()

mnist_service = bentoml.Service("mnist_classifier", runners=[classifier_runner])

@mnist_service.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data: np.ndarray) -> np.ndarray:
    return classifier_runner.predict.run(input_data)