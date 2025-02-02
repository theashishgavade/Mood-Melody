import cv2
import numpy as np
import mediapipe as mp
import webbrowser
import time
import streamlit as st
from tensorflow.keras.models import load_model

# Load Model & Labels
try:
    model = load_model("model.h5")
    labels = np.load("labels.npy")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Global variables
detected_emotion = ""
emotion_captured = False  # Flag to freeze after 5 seconds
start_time = None
cap = None

# Function to process frames and detect emotion
def detect_emotion():
    global detected_emotion, emotion_captured, start_time, cap
    if cap is None:
        cap = cv2.VideoCapture(0)
        start_time = time.time()

    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        if not emotion_captured:  # Run only if emotion is not frozen
            features = []
            if results.face_landmarks:
                for i in results.face_landmarks.landmark:
                    features.append(i.x - results.face_landmarks.landmark[1].x)
                    features.append(i.y - results.face_landmarks.landmark[1].y)

                if results.left_hand_landmarks:
                    for i in results.left_hand_landmarks.landmark:
                        features.append(i.x - results.left_hand_landmarks.landmark[8].x)
                        features.append(i.y - results.left_hand_landmarks.landmark[8].y)
                else:
                    features.extend([0.0] * 42)

                if results.right_hand_landmarks:
                    for i in results.right_hand_landmarks.landmark:
                        features.append(i.x - results.right_hand_landmarks.landmark[8].x)
                        features.append(i.y - results.right_hand_landmarks.landmark[8].y)
                else:
                    features.extend([0.0] * 42)

                features = np.array(features).reshape(1, -1)

                try:
                    detected_emotion = labels[np.argmax(model.predict(features))]
                except Exception as e:
                    st.error(f"Emotion detection error: {e}")

        # Draw face landmarks on frame
        mp.solutions.drawing_utils.draw_landmarks(frame, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)
        # mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        # mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Display emotion on frame
        cv2.putText(frame, f"Emotion: {detected_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Show live video feed in Streamlit
        stframe.image(frame, channels="BGR", use_container_width=True)

        # Stop detection after 5 seconds
        if time.time() - start_time > 5:
            emotion_captured = True
            break

    cap.release()
    return detected_emotion

# Streamlit UI
st.markdown("<h2 style='text-align: center; color: white;  border-radius: 10px;'>🎵 Emotion-Based Music Recommendation</h2>", unsafe_allow_html=True)

# Layout: Two Columns
col1, col2 = st.columns([1, 1])  # Equal width columns

with col1:
    st.subheader("🎤 Music Preferences")
    
    # language = st.text_input("Enter language (e.g., English, Hindi, Spanish)", "")
    language = st.selectbox("Select Language", ["Hindi", "English", "Marathi","Punjabi", "Sanskrit", "Assamese", "Bengali", "Gujarati", "Kannada", "Konkani", "Maithili", "Malayalam",  "Nepali", "Odia",  "Sindhi", "Tamil", "Telugu", "Urdu", "Spanish", "French", "German", "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Chinese", "Arabic", "Dutch", "Turkish", "Swedish", "Polish", "Greek", "Hebrew", "Thai", "Vietnamese"])
        
        # st.stop()

    singer = st.text_input("Enter Singer's name (Optional)", "")

    platform = st.selectbox("Select Platform", ["YouTube", "YT Music","Spotify Web","Spotify Phone", "Gaana", "Apple Music", "Amazon Music", "JioSaavn", "Wynk", "Hungama", "SoundCloud", "Tidal", "Deezer"])


with col2:
    st.subheader("📷 Live Camera Feed")
    stframe = st.empty()  # Placeholder for webcam feed

    if st.button("🎵 Recommend Songs"):
        detected_emotion = detect_emotion()

        if not language:
            st.warning("Language is required to proceed.")
            st.stop()
        if not singer:
            st.success(f"Searching for: {detected_emotion} songs in {language} on {platform}")
        else:
            st.success(f"Searching for: {detected_emotion} songs in {language} by {singer} on {platform}")
        
        if platform == "Spotify Web":
            search_query = f"https://open.spotify.com/search/{language}+{detected_emotion}+song+{singer}/tracks"
            webbrowser.open(search_query)
        if platform == "Spotify Phone":
            search_query = f"https://open.spotify.com/search/{language}+{detected_emotion}+song+{singer}"
            webbrowser.open(search_query)
        if platform == "YouTube":
            search_query = f"https://www.youtube.com/results?search_query={language}+{detected_emotion}+song+{singer}"
            webbrowser.open(search_query)
        if platform == "YT Music":
            search_query = f"https://music.youtube.com/search?q={language}+{detected_emotion}+song+{singer}"
            webbrowser.open(search_query)
        if platform == "Gaana":
            search_query = f"https://gaana.com/search/{language}+{detected_emotion}+song+{singer}"
            webbrowser.open(search_query)
        if platform == "Apple Music":
            search_query = f"https://music.apple.com/search?term={language}+{detected_emotion}+song+{singer}"
            webbrowser.open(search_query)
        if platform == "Amazon Music":
            search_query = f"https://music.amazon.in/search/{language}+{detected_emotion}+song+{singer}"
            webbrowser.open(search_query)
        if platform == "JioSaavn":
            search_query = f"https://www.jiosaavn.com/search/{language}+{detected_emotion}+song+{singer}"
            webbrowser.open(search_query)
        if platform == "Wynk":
            search_query = f"https://wynk.in/music/search/{language}+{detected_emotion}+song+{singer}"
            webbrowser.open(search_query)
        if platform == "Hungama":
            search_query = f"https://www.hungama.com/music/{language}+{detected_emotion}+song+{singer}"
            webbrowser.open(search_query)
        if platform == "SoundCloud":
            search_query = f"https://soundcloud.com/search?q={language}+{detected_emotion}+song+{singer}"
            webbrowser.open(search_query)
        if platform == "Tidal":
            search_query = f"https://listen.tidal.com/search/{language}+{detected_emotion}+song+{singer}"
            webbrowser.open(search_query)
        if platform == "Deezer":
            search_query = f"https://www.deezer.com/search/{language}+{detected_emotion}+song+{singer}"
            webbrowser.open(search_query)

        # search_query = f"https://www.{platform.lower()}.com/results?search_query={language}+{detected_emotion}+song+{singer}"
        # webbrowser.open(search_query)

    if st.button("🔄 Reset Detection"):
        emotion_captured = False
        start_time = None
        detected_emotion = "Neutral"
        if cap:
            cap.release()
        cap = None
        st.warning("Emotion detection reset.")


