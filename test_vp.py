import sys
from pathlib import Path
import numpy as np
import cv2
sys.path.append(str(Path("f:/Deep_Learning_Practice/day3/ASL-Vision")))

# Mock av
import unittest.mock as mock
sys.modules['av'] = mock.MagicMock()

import importlib.util
spec = importlib.util.spec_from_file_location("Webcam", "f:/Deep_Learning_Practice/day3/ASL-Vision/pages/2_Webcam.py")
webcam = importlib.util.module_from_spec(spec)
sys.modules["Webcam"] = webcam
spec.loader.exec_module(webcam)
VideoProcessor = webcam.VideoProcessor

# Dummy frame
class DummyFrame:
    def to_ndarray(self, format):
        return np.zeros((480, 640, 3), dtype=np.uint8)

frame = DummyFrame()
vp = VideoProcessor()
import time
for i in range(10):
    vp.recv(frame)
    time.sleep(0.3)
print("Finished without errors! Sentence:", vp.sentence)
