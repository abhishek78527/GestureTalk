import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model
import requests
import json
import time

# --- 0. Set Page Config ---
st.set_page_config(layout="wide")

# --- 1. Cached Model Loading Function ---
@st.cache_resource
def load_static_model():
    """Loads and caches the static letter model."""
    try:
        static_model = load_model('keypoint_model.keras')
        static_actions = np.load('keypoints_label_map.npy')
        return static_model, static_actions
    except Exception as e:
        st.error(f"FATAL ERROR: Could not load 'keypoint_model.keras'. Error: {e}")
        st.stop()

# Load the model
static_model, static_actions = load_static_model()

# --- 2. Initialize MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- 3. Initialize TTS Engine ---
# --- 3. Initialize TTS Engine ---
# Disabled on macOS because pyttsx3 causes objc issues
engine = None

# --- 4. Initialize Session State ---
if "recording" not in st.session_state:
    st.session_state.recording = False
if "start_time" not in st.session_state:
    st.session_state.start_time = 0
if "letter_sequence" not in st.session_state:
    st.session_state.letter_sequence = []
if "last_stable_letter" not in st.session_state:
    st.session_state.last_stable_letter = ""
if "ai_response" not in st.session_state:
    st.session_state.ai_response = ""

# --- 5. Helper Functions ---
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
        return keypoints
    return None # Returns None if no hand is found

# --- GOOGLE GEMINI AI FUNCTION (FIXED) ---
def get_ai_sentence(letters):
    """
    Sends a sequence of letters to the Google Gemini API to form a sentence.
    """
    apiKey = st.secrets.get("GEMINI_API_KEY") 
    if not apiKey:
        return "AI Error: GEMINI_API_KEY not found in secrets.toml."

    # Using the supported model: gemini-2.5-flash
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={apiKey}"
    
    letter_string = "-".join(letters)
    system_prompt = (
        "You are a helpful assistant. The user has spelled out a message using sign language letters. "
        "The sequence of letters is: " + letter_string +
        "\nThe spelling might have errors. Correct any spelling mistakes and "
        "form a single, coherent sentence from these letters. "
        "Respond ONLY with the final, corrected sentence."
    )
    payload = {"contents": [{"parts": [{"text": "Decode this"}]}], "systemInstruction": {"parts": [{"text": system_prompt}]}}
    
    try:
        # --- FIX: Increased timeout from 15 to 30 seconds ---
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if 'candidates' in result and result['candidates']:
            if 'finishReason' in result['candidates'][0] and result['candidates'][0]['finishReason'] != 'STOP':
                return f"AI Error: Generation stopped for reason: {result['candidates'][0]['finishReason']}"
            return result['candidates'][0]['content']['parts'][0]['text'].strip()

        if 'promptFeedback' in result and 'blockReason' in result['promptFeedback']:
             return f"AI Error: The prompt was blocked. Reason: {result['promptFeedback']['blockReason']}"
        return "AI could not process the letters."
    
    except requests.exceptions.HTTPError as e:
        error_message = ""
        try:
            error_details = e.response.json()
            error_message = error_details.get('error', {}).get('message', 'An unknown HTTP error occurred.')
        except requests.exceptions.JSONDecodeError:
            error_message = e.response.text
        st.error(f"API HTTP Error: {error_message}")
        return "AI API Error: Check your API key. The model name might be wrong."
        
    except requests.exceptions.Timeout:
        # Specific handler for the timeout error you experienced
        st.error("API Request Error: The request timed out after 30 seconds.")
        return "AI call failed: Request timed out. Please try a shorter message or run again."

    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return "AI call failed: A connection error occurred. Please check your internet connection."

# --- 6. Streamlit UI ---
st.sidebar.title("About GestureTalk")
st.sidebar.info(
    "This app is a spelling translator. It's always in 'Live Mode', "
    "recognizing letters. Press 'Start Spelling' to record those "
    "letters for 1 minute to form a sentence."
)
st.sidebar.success(f"**Static Model:** {len(static_actions)} letters")

st.title("GestureTalk – Spelling Translator")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Your Webcam")
    frame_placeholder = st.empty()

with col2:
    st.header("Live Letter")
    live_letter_placeholder = st.empty()
    
    st.header("Spelling Window")
    status_placeholder = st.empty()
    spelled_sequence_placeholder = st.empty()
    ai_response_placeholder = st.empty()

# Add the button
if st.button("Start 1-Minute Spelling Window", disabled=st.session_state.recording): 
    st.session_state.recording = True
    st.session_state.start_time = time.time()
    st.session_state.letter_sequence = []
    st.session_state.ai_response = ""
    st.session_state.last_stable_letter = ""
    st.toast("Recording window has begun!", icon="🟢")

# --- 7. Real-Time Loop ---
RECORD_DURATION = 60 # 60 seconds = 1 minute
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Camera access denied. Enable camera permission for VS Code and Terminal in macOS Settings.")
    st.stop()

prediction_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    
    # Color Fix
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    predicted_action = ""
    keypoints = extract_keypoints(results) # This will be None or a NumPy array
    
    if keypoints is not None:
        # --- Always-On Letter Recognition ---
        static_res = static_model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
        if np.max(static_res) > 0.90: # High confidence
            predicted_action = static_actions[np.argmax(static_res)]
        
        if predicted_action:
            prediction_buffer.append(predicted_action)

            # Check for a stable prediction (5 frames in a row)
            if len(prediction_buffer) > 5 and np.all(np.array(prediction_buffer[-5:]) == predicted_action):
                
                # Check if it's different from the last stable letter
                if predicted_action != st.session_state.last_stable_letter:
                    st.session_state.last_stable_letter = predicted_action 
                    
                    # Speak the letter
                    if engine:
                        engine.say(predicted_action)
                        engine.runAndWait()
                    
                    # If recording, add to the sequence
                    if st.session_state.recording:
                        st.session_state.letter_sequence.append(predicted_action)
                
                prediction_buffer = []
    
    else:
        # No hand detected, clear buffer and reset last letter
        prediction_buffer = []
        st.session_state.last_stable_letter = ""

    # --- Recording Logic ---
    if st.session_state.recording:
        elapsed_time = time.time() - st.session_state.start_time
        time_left = RECORD_DURATION - elapsed_time
        
        if time_left <= 0:
            # --- TIME'S UP! ---
            st.session_state.recording = False
            status_placeholder.warning("Time's up! Decoding with Google AI...")
            
            if st.session_state.letter_sequence:
                st.session_state.ai_response = get_ai_sentence(st.session_state.letter_sequence)
            else:
                st.session_state.ai_response = "No letters were detected."
            
            if engine and st.session_state.ai_response:
                engine.say(st.session_state.ai_response)
                engine.runAndWait()
        
        else:
            # --- STILL RECORDING ---
            status_placeholder.info(f"Recording... Time left: {int(time_left)} seconds")
    
    else:
        # --- IDLE STATE ---
        status_placeholder.write("Idle. Press the button to start spelling.")

    # --- Update UI ---
    frame_placeholder.image(image_rgb)
    
    # Update "Live Letter" box
    if keypoints is None: 
        live_letter_placeholder.markdown("<h3 style='text-align: center; color: gray;'>...</h3>", unsafe_allow_html=True)
    elif not st.session_state.last_stable_letter:
        live_letter_placeholder.markdown("<h3 style='text-align: center; color: gray;'>Analyzing...</h3>", unsafe_allow_html=True)
    else:
        live_letter_placeholder.markdown(f"<h1 style='text-align: center;'>{st.session_state.last_stable_letter}</h1>", unsafe_allow_html=True)

    # Update "Spelling Window" boxes
    spelled_sequence_placeholder.write(st.session_state.letter_sequence)
    ai_response_placeholder.write(st.session_state.ai_response)

cap.release()