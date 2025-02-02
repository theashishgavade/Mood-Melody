import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import time
import os

# Load the model and labels
model_path = "model.h5"
labels_path = "labels.npy"
emotion_path = "emotion.npy"

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error("Model file not found! Please check 'model.h5'.")
    st.stop()

if os.path.exists(labels_path):
    label = np.load(labels_path)
else:
    st.error("Labels file not found! Please check 'labels.npy'.")
    st.stop()

# Initialize MediaPipe components
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("🎵 Emotion-Based Music Recommender 🎵")

if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Load previously detected emotion if exists
if os.path.exists(emotion_path):
    try:
        emotion = np.load(emotion_path)[0]
    except:
        emotion = ""
else:
    emotion = ""

st.session_state["run"] = "true" if not emotion else "false"

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.label = label

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Flip the frame for a mirror effect
        frm = cv2.flip(frm, 1)

        # Convert frame to RGB for Mediapipe
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)

            # Predict emotion
            pred = self.label[np.argmax(self.model.predict(lst))]

            # Display detected emotion
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            # Save detected emotion
            np.save(emotion_path, np.array([pred]))

        # Draw landmarks
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Inputs for language and singer
lang = st.text_input("Language (e.g., English, Hindi, Spanish)")
singer = st.text_input("Singer (Optional)")

# Start the video stream only if needed
if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(
        key="key",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

# Button to recommend songs
btn = st.button("🎶 Recommend me songs 🎶")

if btn:
    if not emotion:
        st.warning("⚠️ Please let me capture your emotion first!")
        st.session_state["run"] = "true"
    else:
        search_query = f"{lang} {emotion} song {singer}"
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
        np.save(emotion_path, np.array([""]))
        st.session_state["run"] = "false"
