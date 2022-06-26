from typing import Tuple
import json

import numpy as np
import requests
from mldeployment.training import prepare_mnist_training_data


SERVICE_URL = "http://localhost:3000/classify"


def sample_random_mnist_data_point() -> Tuple[np.ndarray, np.ndarray]:
    _, _, test_images, test_labels = prepare_mnist_training_data()
    random_index = np.random.randint(0, len(test_images))
    random_test_image = test_images[random_index]
    random_test_image = np.expand_dims(random_test_image, 0)
    return random_test_image, test_labels[random_index]


def make_request_to_bento_service(
    service_url: str, input_array: np.ndarray
) -> str:
    serialized_input_data = json.dumps(input_array.tolist())
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    return response.text


def main():
    input_data, expected_output = sample_random_mnist_data_point()
    prediction = make_request_to_bento_service(SERVICE_URL, input_data)
    print(f"Prediction: {prediction}")
    print(f"Expected output: {expected_output}")


if __name__ == "__main__":
    main()
