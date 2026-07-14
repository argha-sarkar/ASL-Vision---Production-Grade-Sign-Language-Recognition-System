"""
utils.py

FastAPI Utility Functions

Author: Argha Sarkar Project
"""

from pathlib import Path
import shutil
import uuid

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from src.api.config import settings


# ---------------------------------------------------------
# File Validation
# ---------------------------------------------------------

def validate_extension(
    filename: str,
):

    extension = Path(filename).suffix.lower()

    if extension not in settings.ALLOWED_EXTENSIONS:

        raise HTTPException(

            status_code=400,

            detail=f"Unsupported file format: {extension}",

        )

    return extension


# ---------------------------------------------------------
# Generate Unique Filename
# ---------------------------------------------------------

def unique_filename(
    filename: str,
):

    extension = Path(filename).suffix

    return f"{uuid.uuid4().hex}{extension}"


# ---------------------------------------------------------
# Save Uploaded Image
# ---------------------------------------------------------

def save_upload(
    file: UploadFile,
):

    validate_extension(
        file.filename
    )

    filename = unique_filename(
        file.filename
    )

    save_path = (
        Path(settings.TEMP_DIR)
        / filename
    )

    with open(
        save_path,
        "wb",
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer,
        )

    return save_path


# ---------------------------------------------------------
# Delete Temporary File
# ---------------------------------------------------------

def delete_file(
    filepath,
):

    filepath = Path(filepath)

    if filepath.exists():

        filepath.unlink()


# ---------------------------------------------------------
# Read Image
# ---------------------------------------------------------

def read_image(
    image_path,
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise HTTPException(

            status_code=400,

            detail="Unable to read image.",

        )

    return image


# ---------------------------------------------------------
# Convert to RGB
# ---------------------------------------------------------

def to_rgb(
    image,
):

    if image.ndim == 2:

        image = cv2.cvtColor(

            image,

            cv2.COLOR_GRAY2RGB,

        )

    else:

        image = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2RGB,

        )

    return image


# ---------------------------------------------------------
# Resize Image
# ---------------------------------------------------------

def resize_image(
    image,
    size=(224, 224),
):

    return cv2.resize(

        image,

        size,

    )


# ---------------------------------------------------------
# Normalize Image
# ---------------------------------------------------------

def normalize_image(
    image,
):

    image = image.astype(
        np.float32
    )

    image /= 255.0

    return image


# ---------------------------------------------------------
# Convert Image to Tensor
# ---------------------------------------------------------

def image_to_tensor(
    image,
):

    image = np.expand_dims(

        image,

        axis=0,

    )

    return image


# ---------------------------------------------------------
# Complete Image Pipeline
# ---------------------------------------------------------

def prepare_image(
    image_path,
    image_size=(224, 224),
):

    image = read_image(
        image_path
    )

    image = to_rgb(
        image
    )

    image = resize_image(

        image,

        image_size,

    )

    image = normalize_image(
        image
    )

    image = image_to_tensor(
        image
    )

    return image


# ---------------------------------------------------------
# Allowed File Check
# ---------------------------------------------------------

def is_allowed_file(
    filename,
):

    extension = Path(
        filename
    ).suffix.lower()

    return extension in settings.ALLOWED_EXTENSIONS


# ---------------------------------------------------------
# Create Directory
# ---------------------------------------------------------

def create_directory(
    directory,
):

    directory = Path(directory)

    directory.mkdir(

        parents=True,

        exist_ok=True,

    )

    return directory


# ---------------------------------------------------------
# Folder Images
# ---------------------------------------------------------

def image_files(
    folder,
):

    folder = Path(folder)

    images = []

    for file in sorted(

        folder.iterdir()

    ):

        if is_allowed_file(

            file.name

        ):

            images.append(file)

    return images