import sys
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

sys.path.append(str(Path("f:/Deep_Learning_Practice/day3/ASL-Vision")))

from src.api.predictor import APIPredictor
predictor = APIPredictor()

df = pd.read_csv("f:/Deep_Learning_Practice/day3/ASL-Vision/data/raw/sign_mnist_test.csv")
y_true = df['label'].values
x_test = df.drop('label', axis=1).values

correct = 0
total = 100
for i in range(total):
    img = x_test[i].reshape(28, 28).astype(np.uint8)
    # The Predictor expects BGR if it's 3-channel, but for 1-channel it skips cvtColor
    res = predictor.predict(img)
    if res["class_index"] == y_true[i]:
        correct += 1

print(f"Accuracy on 1 channel MNIST: {correct / total * 100}%")

# Let's test if passing 3 channel image breaks it
correct = 0
for i in range(total):
    img = x_test[i].reshape(28, 28).astype(np.uint8)
    # create a fake BGR image
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    res = predictor.predict(img_bgr)
    if res["class_index"] == y_true[i]:
        correct += 1

print(f"Accuracy on 3 channel BGR MNIST: {correct / total * 100}%")

