import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Setup ---
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data_Dynamic') 

# Actions to collect
actions = np.array(['hello', 'thankyou', 'okay', 'no', 'great'])

# Thirty videos of data for each action
no_sequences = 30

# Videos are 30 frames in length
sequence_length = 30

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Helper Function ---
def extract_keypoints(results):
    """Converts MediaPipe landmarks into a 1D numpy array."""
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
        return keypoints
    return np.zeros(21 * 3) # Return zeros if no hand is detected # Return zeros if no hand is detected

# --- Folder Creation ---
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass # Folder already exists

# --- Collection Loop ---
cap = cv2.VideoCapture(0)

print("Starting data collection...")

# Loop through actions
for action in actions:
    # Loop through sequences (videos)
    for sequence in range(no_sequences):
        # Loop through frames in the sequence
        for frame_num in range(sequence_length):

            ret, frame = cap.read()
            frame = cv2.flip(frame, 1) # Flip for selfie view

            # Process with MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
            # --- UI Text ---
            if frame_num == 0: 
                # Show "STARTING COLLECTION" for the first frame
                cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Collecting sequence {sequence} for {action}', (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(2000) # Wait 2 seconds
            else: 
                # Show the action name
                cv2.putText(image, f'Collecting sequence {sequence} for {action}', (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

            # --- Save Keypoints ---
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")