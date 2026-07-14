"""
schemas.py

FastAPI Request & Response Schemas

Author: Argha Sarkar Project
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------


class HealthResponse(BaseModel):

    status: str = "OK"

    model_loaded: bool

    version: str


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------


class PredictionResponse(BaseModel):

    prediction: str

    class_index: int

    confidence: float

    probabilities: List[float]


# ---------------------------------------------------------
# Batch Prediction
# ---------------------------------------------------------


class BatchPrediction(BaseModel):

    filename: str

    prediction: str

    class_index: int

    confidence: float


class BatchPredictionResponse(BaseModel):

    total_images: int

    predictions: List[BatchPrediction]


# ---------------------------------------------------------
# Model Information
# ---------------------------------------------------------


class ModelInfoResponse(BaseModel):

    model_name: str

    framework: str

    input_shape: List[int]

    output_classes: int

    version: str


# ---------------------------------------------------------
# Error Response
# ---------------------------------------------------------


class ErrorResponse(BaseModel):

    error: str

    detail: Optional[str] = None


# ---------------------------------------------------------
# Confidence Summary
# ---------------------------------------------------------


class ConfidenceResponse(BaseModel):

    mean_confidence: float

    minimum_confidence: float

    maximum_confidence: float

    median_confidence: float


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------


class MetricsResponse(BaseModel):

    accuracy: float

    precision: float

    recall: float

    f1_score: float


# ---------------------------------------------------------
# API Information
# ---------------------------------------------------------


class APIInfo(BaseModel):

    application: str

    version: str

    description: str

    author: str = "Argha Sarkar"


# ---------------------------------------------------------
# Upload Response
# ---------------------------------------------------------


class UploadResponse(BaseModel):

    filename: str

    message: str


# ---------------------------------------------------------
# Delete Response
# ---------------------------------------------------------


class DeleteResponse(BaseModel):

    filename: str

    deleted: bool


# ---------------------------------------------------------
# Root Response
# ---------------------------------------------------------


class RootResponse(BaseModel):

    message: str

    documentation: str

    version: str