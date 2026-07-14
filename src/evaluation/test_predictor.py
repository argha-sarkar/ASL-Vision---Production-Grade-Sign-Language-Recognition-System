from src.evaluation.predictor import Predictor

import numpy as np

predictor = Predictor()

dummy = np.random.rand(

    28,

    28,

    1,

).astype("float32")

result = predictor.predict_single(dummy)

print(result)