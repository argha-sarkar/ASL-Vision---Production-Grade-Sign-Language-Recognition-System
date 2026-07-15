"""
2_Webcam.py

Real-Time Webcam Prediction

Author: Argha Sarkar Project
"""

import time

try:
    import av
    from streamlit_webrtc import (
        RTCConfiguration,
        VideoProcessorBase,
        webrtc_streamer,
    )
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.api.predictor import APIPredictor

st.set_page_config(
    page_title="Webcam Prediction",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Real-Time ASL Recognition")

st.markdown("---")


# ---------------------------------------------------------
# Load Predictor
# ---------------------------------------------------------


@st.cache_resource
def load_predictor():

    return APIPredictor()


predictor = load_predictor()


# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------

CLASS_NAMES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
]


# ---------------------------------------------------------
# Video Processor
# ---------------------------------------------------------

if not WEBRTC_AVAILABLE:
    st.error(
        "streamlit-webrtc is not installed.\n\n"
        "Run: `pip install streamlit-webrtc av` to enable webcam support."
    )
else:

    class VideoProcessor(VideoProcessorBase):

        def __init__(self):

            self.predictor = predictor

            self.last_prediction = ""

            self.last_confidence = 0.0

            self.last_time = 0
            
            self.sentence = ""
            
            self.current_stable_char = ""
            
            self.stable_frames = 0
            
            self.last_appended = False
            
            self.low_conf_frames = 0

        def recv(
            self,
            frame,
        ):

            image = frame.to_ndarray(format="bgr24")

            current = time.time()
            
            h, w, _ = image.shape
            roi_size = 300
            x1 = max(0, (w - roi_size) // 2)
            y1 = max(0, (h - roi_size) // 2)
            x2 = min(w, x1 + roi_size)
            y2 = min(h, y1 + roi_size)
            
            roi_image = image[y1:y2, x1:x2]
            
            # Flip horizontally to correct for webcam mirroring
            roi_image = cv2.flip(roi_image, 1)
            
            # Match color space of 1_Predict_Image.py (RGB)
            roi_image_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

            if current - self.last_time > 0.25:

                try:

                    result = self.predictor.predict(roi_image_rgb)

                    prediction = result["prediction"]

                    confidence = result["confidence"]

                    if isinstance(
                        prediction,
                        int,
                    ):
                        prediction = CLASS_NAMES[prediction]
                    elif isinstance(prediction, str) and prediction.isdigit():
                        prediction = CLASS_NAMES[int(prediction)]

                    self.last_prediction = prediction

                    self.last_confidence = confidence

                    if confidence > 0.7:
                        self.low_conf_frames = 0
                        if prediction == self.current_stable_char:
                            self.stable_frames += 1
                        else:
                            self.current_stable_char = prediction
                            self.stable_frames = 1
                            self.last_appended = False

                        if self.stable_frames >= 4 and not self.last_appended:
                            self.sentence += prediction
                            self.last_appended = True
                    else:
                        self.current_stable_char = ""
                        self.stable_frames = 0
                        self.last_appended = False
                        
                        self.low_conf_frames += 1
                        if self.low_conf_frames == 8:
                            if len(self.sentence) > 0 and self.sentence[-1] != " ":
                                self.sentence += " "

                    if len(self.sentence) > 35:
                        self.sentence = self.sentence[-35:]

                except Exception as e:
                    import traceback
                    traceback.print_exc()

                self.last_time = current

            cv2.rectangle(
                image,
                (10, 10),
                (350, 90),
                (0, 255, 0),
                -1,
            )
            
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
            )
            
            # Sentence background at the bottom
            cv2.rectangle(
                image,
                (0, h - 60),
                (w, h),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                image,
                f"Prediction: {self.last_prediction}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )

            cv2.putText(
                image,
                f"Confidence: {self.last_confidence*100:.2f}%",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
            
            cv2.putText(
                image,
                f"Sentence: {self.sentence}",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24",
            )


    # ---------------------------------------------------------
    # WebRTC
    # ---------------------------------------------------------

    rtc = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_streamer(
        key="asl-webcam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("Webcam")

st.sidebar.info(
    """
Start your webcam and show an ASL hand sign.

The prediction updates every 250 ms.
"""
)


# ---------------------------------------------------------
# Supported Classes
# ---------------------------------------------------------

st.markdown("---")

st.subheader("Supported Classes")

classes = pd.DataFrame(
    {
        "Index": range(len(CLASS_NAMES)),
        "Letter": CLASS_NAMES,
    }
)

st.dataframe(
    classes,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Notes
# ---------------------------------------------------------

st.markdown("---")

st.info(
    """
Tips

• Keep your hand centered.

• Use a plain background.

• Ensure good lighting.

• Keep only one hand visible.

• Hold the sign steady for better accuracy.
"""
)
