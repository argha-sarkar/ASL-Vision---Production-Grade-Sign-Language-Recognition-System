import numpy as np

from src.evaluation.predictor import Predictor

predictor = Predictor()

dummy = np.random.rand(
    28,
    28,
    1,
).astype("float32")

result = predictor.predict_single(dummy)

print(result)
