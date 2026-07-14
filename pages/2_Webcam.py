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

        def recv(
            self,
            frame,
        ):

            image = frame.to_ndarray(format="bgr24")

            current = time.time()

            if current - self.last_time > 0.25:

                try:

                    result = self.predictor.predict(image)

                    prediction = result["prediction"]

                    confidence = result["confidence"]

                    if isinstance(
                        prediction,
                        int,
                    ):

                        prediction = CLASS_NAMES[prediction]

                    self.last_prediction = prediction

                    self.last_confidence = confidence

                except Exception:

                    pass

                self.last_time = current

            cv2.rectangle(
                image,
                (10, 10),
                (350, 90),
                (0, 255, 0),
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
