"""
routes.py

FastAPI Routes

Author: Argha Sarkar Project
"""

from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.api.predictor import APIPredictor
from src.api.schemas import (BatchPredictionResponse, HealthResponse,
                             ModelInfoResponse, PredictionResponse,
                             RootResponse)

router = APIRouter()

# Lazy singleton — instantiated once on first request
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = APIPredictor()
    return _predictor



# ---------------------------------------------------------
# Root
# ---------------------------------------------------------


@router.get(
    "/",
    response_model=RootResponse,
)
def root():

    return {
        "message": "ASL Vision API",
        "documentation": "/docs",
        "version": "1.0.0",
    }


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    status = get_predictor().health()

    return {
        "status": "OK",
        "model_loaded": status["model_loaded"],
        "version": "1.0.0",
    }


# ---------------------------------------------------------
# Model Information
# ---------------------------------------------------------


@router.get(
    "/model",
    response_model=ModelInfoResponse,
)
def model_info():

    status = get_predictor().health()

    return {
        "model_name": "ASL Vision",
        "framework": "TensorFlow",
        "input_shape": list(status["input_shape"]),
        "output_classes": status["num_classes"],
        "version": "1.0.0",
    }


# ---------------------------------------------------------
# Predict Single Image
# ---------------------------------------------------------


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict_image(
    file: UploadFile = File(...),
):

    try:

        image_bytes = await file.read()

        image = np.frombuffer(
            image_bytes,
            np.uint8,
        )

        image = cv2.imdecode(
            image,
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise HTTPException(
                status_code=400,
                detail="Invalid image.",
            )

        result = get_predictor().predict(image)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ---------------------------------------------------------
# Predict Folder
# ---------------------------------------------------------


@router.post(
    "/predict-folder",
    response_model=BatchPredictionResponse,
)
async def predict_folder(
    folder: str,
):

    folder = Path(folder)

    if not folder.exists():

        raise HTTPException(
            status_code=404,
            detail="Folder not found.",
        )

    predictions = get_predictor().predict_folder(folder)

    return {
        "total_images": len(predictions),
        "predictions": predictions,
    }


# ---------------------------------------------------------
# Supported Classes
# ---------------------------------------------------------


@router.get(
    "/classes",
)
def classes():

    p = get_predictor()

    if p.labels is None:

        return JSONResponse(
            {
                "classes": [],
            }
        )

    classes = p.labels.classes_.tolist()

    return {"classes": classes}


# ---------------------------------------------------------
# Confidence
# ---------------------------------------------------------


@router.post(
    "/confidence",
)
async def confidence(
    file: UploadFile = File(...),
):

    image_bytes = await file.read()

    image = np.frombuffer(
        image_bytes,
        np.uint8,
    )

    image = cv2.imdecode(
        image,
        cv2.IMREAD_COLOR,
    )

    result = get_predictor().predict(image)

    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
    }


# ---------------------------------------------------------
# Top-5 Prediction
# ---------------------------------------------------------


@router.post(
    "/top5",
)
async def top5(
    file: UploadFile = File(...),
):

    image_bytes = await file.read()

    image = np.frombuffer(
        image_bytes,
        np.uint8,
    )

    image = cv2.imdecode(
        image,
        cv2.IMREAD_COLOR,
    )

    p = get_predictor()

    result = p.predict(image)

    probabilities = np.array(result["probabilities"])

    indices = np.argsort(probabilities)[::-1][:5]

    top5 = []

    for index in indices:

        if p.labels is not None:

            label = p.labels.inverse_transform([index])[0]

        else:

            label = str(index)

        top5.append(
            {
                "class_index": int(index),
                "prediction": label,
                "confidence": float(probabilities[index]),
            }
        )

    return {"top5": top5}
