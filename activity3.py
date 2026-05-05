import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
import logging
import os


logging.getLogger("aioice").setLevel(logging.ERROR)


st.set_page_config(page_title="Live Object Detection", layout="wide")
st.title("🎥 Live Object Detection (YOLOv8)")
st.write("Real-time detection using webcam")


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  

model = load_model()

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

  
        img = cv2.resize(img, (640, 480))


        results = model(img)


        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")



rtc_configuration = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

try:
    from twilio.rest import Client

    account_sid = st.secrets["twilio"]["account_sid"]
    auth_token = st.secrets["twilio"]["auth_token"]

    client = Client(account_sid, auth_token)
    token = client.tokens.create()

    rtc_configuration = {
        "iceServers": token.ice_servers
    }

    st.success("✅ TURN server connected")

except Exception as e:
    st.warning("⚠️ TURN server not available (fallback to STUN only)")
    st.caption(str(e))


webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
)
