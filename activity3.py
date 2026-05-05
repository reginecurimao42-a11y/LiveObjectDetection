import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import os
import time
from datetime import datetime


st.set_page_config(
    page_title="Advanced Object Detection",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Live Object Detection and Tracing")
st.write("Point your camera at objects to identify them in real-time.")


st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.1, 1.0, 0.40, 0.05
)

alert_object = st.sidebar.selectbox(
    "Trigger Alert For",
    ["person","cell phone","bottle","chair","dog","cat"]
)

save_frames = st.sidebar.checkbox(
    "Save Detected Frames",
    True
)


os.makedirs("detected_frames", exist_ok=True)



@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()



frame_counter = 0
last_save = time.time()



def video_frame_callback(frame):

    global frame_counter
    global last_save

    img = frame.to_ndarray(format="bgr24")


    results = model.track(
        img,
        persist=True,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    counts = {}
    alert_found = False


    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            counts[label] = counts.get(label,0)+1


            if label == alert_object:
                alert_found = True


        y_pos = 30
        for obj,count in counts.items():

            text = f"{obj}: {count}"

            cv2.putText(
                img,
                text,
                (20,y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            y_pos += 35



    if alert_found:

        cv2.putText(
            img,
            f"ALERT: {alert_object.upper()} DETECTED!",
            (30,450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )



    annotated = result.plot()


    img = annotated



    if save_frames:


        if len(counts)>0 and time.time()-last_save > 5:

            filename = datetime.now().strftime(
                "detected_frames/frame_%Y%m%d_%H%M%S.jpg"
            )

            cv2.imwrite(filename,img)
            last_save = time.time()


    frame_counter += 1

    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )



webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers":[
            {"urls":["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video":True,
        "audio":False
    },
)

