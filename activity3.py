import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import os
from collections import Counter
from datetime import datetime


cv2.setNumThreads(0)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


st.title("🎥 Live Object Detection & Tracking")
st.write("Point your camera at objects to identify them in real-time.")

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

alert_object = st.sidebar.selectbox(
    "Alert Object",
    ["person", "cell phone", "bottle", "chair", "dog", "cat"]
)

save_frames = st.sidebar.checkbox("Save Detected Frames", True)

os.makedirs("saved_frames", exist_ok=True)


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")


    results = model.track(
        img,
        persist=True,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    counts = Counter()
    alert = False

    # Detect objects
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            counts[label] += 1

            if label == alert_object:
                alert = True


    annotated = result.plot()


    y = 30
    for obj, cnt in counts.items():
        cv2.putText(
            annotated,
            f"{obj}: {cnt}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        y += 25


    if alert:
        cv2.putText(
            annotated,
            f"ALERT: {alert_object.upper()} DETECTED!",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )


    if save_frames and len(counts) > 0:
        filename = f"saved_frames/frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, annotated)

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")



webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)