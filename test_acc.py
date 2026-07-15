import sys
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
sys.path.append(str(Path("f:/Deep_Learning_Practice/day3/ASL-Vision")))

from src.api.predictor import APIPredictor

predictor = APIPredictor()

df = pd.read_csv("f:/Deep_Learning_Practice/day3/ASL-Vision/data/raw/sign_mnist_test.csv")
y_true = df['label'].values
x_test = df.drop('label', axis=1).values

# Test first 100 images
correct = 0
for i in range(100):
    # MNIST images are 28x28
    img = x_test[i].reshape(28, 28).astype(np.uint8)
    
    # Predictor expects a 3-channel image typically because of cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # But let's see, cvtColor is called if image.ndim == 3
    # img is 2D here, so predictor will just use it.
    
    # Wait, the predictor.preprocess expects the image to be resized to 28x28.
    # It does cv2.resize(image, (28, 28)) which works on 2D images too.
    
    res = predictor.predict(img)
    pred_idx = res["class_index"]
    
    if pred_idx == y_true[i]:
        correct += 1

print(f"Accuracy on 100 test images: {correct}%")
