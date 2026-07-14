"""
test_api.py

Production API Tests

Author: Argha Sarkar Project
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import app


client = TestClient(app)


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def create_test_image():

    image = np.random.randint(

        0,

        255,

        (224, 224, 3),

        dtype=np.uint8,

    )

    image = Image.fromarray(image)

    buffer = io.BytesIO()

    image.save(

        buffer,

        format="PNG",

    )

    buffer.seek(0)

    return buffer


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

def test_root():

    response = client.get("/api/v1/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data

    assert "version" in data


# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------

def test_health():

    response = client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data

    assert "model_loaded" in data


# ---------------------------------------------------------
# Model Information
# ---------------------------------------------------------

def test_model_information():

    response = client.get("/api/v1/model")

    assert response.status_code == 200

    data = response.json()

    assert "model_name" in data

    assert "framework" in data


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def test_prediction():

    image = create_test_image()

    response = client.post(

        "/api/v1/predict",

        files={

            "file": (

                "sample.png",

                image,

                "image/png",

            )

        },

    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert "confidence" in data

    assert "probabilities" in data


# ---------------------------------------------------------
# Confidence
# ---------------------------------------------------------

def test_confidence():

    image = create_test_image()

    response = client.post(

        "/api/v1/confidence",

        files={

            "file": (

                "sample.png",

                image,

                "image/png",

            )

        },

    )

    assert response.status_code == 200

    data = response.json()

    assert "confidence" in data


# ---------------------------------------------------------
# Top5 Prediction
# ---------------------------------------------------------

def test_top5():

    image = create_test_image()

    response = client.post(

        "/api/v1/top5",

        files={

            "file": (

                "sample.png",

                image,

                "image/png",

            )

        },

    )

    assert response.status_code == 200

    data = response.json()

    assert "top5" in data

    assert len(data["top5"]) <= 5


# ---------------------------------------------------------
# Invalid Endpoint
# ---------------------------------------------------------

def test_invalid_endpoint():

    response = client.get(

        "/api/v1/invalid"

    )

    assert response.status_code == 404


# ---------------------------------------------------------
# Invalid Image
# ---------------------------------------------------------

def test_invalid_image():

    response = client.post(

        "/api/v1/predict",

        files={

            "file": (

                "invalid.txt",

                b"hello",

                "text/plain",

            )

        },

    )

    assert response.status_code in [

        400,

        422,

        500,

    ]


# ---------------------------------------------------------
# Stress Test
# ---------------------------------------------------------

@pytest.mark.parametrize(

    "index",

    range(5),

)

def test_multiple_predictions(

    index,

):

    image = create_test_image()

    response = client.post(

        "/api/v1/predict",

        files={

            "file": (

                f"image_{index}.png",

                image,

                "image/png",

            )

        },

    )

    assert response.status_code == 200


# ---------------------------------------------------------
# Performance
# ---------------------------------------------------------

def test_response_time():

    import time

    image = create_test_image()

    start = time.time()

    response = client.post(

        "/api/v1/predict",

        files={

            "file": (

                "performance.png",

                image,

                "image/png",

            )

        },

    )

    elapsed = time.time() - start

    assert response.status_code == 200

    assert elapsed < 5


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    pytest.main(
        [
            "-v",
            __file__,
        ]
    )