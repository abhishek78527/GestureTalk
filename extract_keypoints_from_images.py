import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from tqdm import tqdm # A library to show a nice progress bar
import pandas as pd # To read the CSV for labels if needed (optional)

# --- 1. Initialize MediaPipe ---
mp_hands = mp.solutions.hands
hands_processor = mp_hands.Hands(
    static_image_mode=True, # We are processing static images
    max_num_hands=1,        # We only care about one hand per image
    min_detection_confidence=0.5
)

# --- 2. Helper function to extract keypoints ---
def extract_keypoints(image_path):
    """
    Reads an image, processes it with MediaPipe, 
    and returns the 63 keypoints.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None # Skip if image can't be read

    # Convert BGR to RGB (MediaPipe requirement)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = hands_processor.process(image_rgb)
    
    # Extract keypoints
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        # Get coordinates [x, y, z] for all 21 keypoints and flatten
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        return keypoints
    else:
        # If no hand is detected, return an array of zeros
        return None

# --- 3. Define Data Path and Loop Through Images ---
# This path points to the 'asl_alphabet_train' folder inside the unzipped dataset
IMAGE_DATA_PATH = os.path.join('ASL_Alphabet_Images', 'asl_alphabet_train')

# Discover the actual letter folders. The dataset sometimes contains an extra
# nested folder named 'asl_alphabet_train' (i.e. ASL_Alphabet_Images/asl_alphabet_train/asl_alphabet_train/A/...)
try:
    entries = [e for e in os.listdir(IMAGE_DATA_PATH) if not e.startswith('.')]
    if len(entries) == 0:
        raise FileNotFoundError

    # If there is a single nested folder that itself contains the letters, use that
    if len(entries) == 1 and os.path.isdir(os.path.join(IMAGE_DATA_PATH, entries[0])):
        nested = os.path.join(IMAGE_DATA_PATH, entries[0])
        nested_entries = [e for e in os.listdir(nested) if not e.startswith('.')]
        # If nested entries look like letter directories, treat them as actions
        if any(os.path.isdir(os.path.join(nested, ne)) for ne in nested_entries):
            IMAGE_DATA_PATH = nested
            actions = [a for a in nested_entries if os.path.isdir(os.path.join(IMAGE_DATA_PATH, a))]
        else:
            # fallback: treat the entries themselves as actions
            actions = [a for a in entries if os.path.isdir(os.path.join(IMAGE_DATA_PATH, a))]
    else:
        # Normal case: entries are letter directories (A, B, C...) directly inside IMAGE_DATA_PATH
        actions = [a for a in entries if os.path.isdir(os.path.join(IMAGE_DATA_PATH, a))]

    # If we still have no directory-like actions, treat files in IMAGE_DATA_PATH as images (single action)
    if not actions:
        # Could be that images are directly under IMAGE_DATA_PATH
        files = [f for f in entries if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            actions = ['.']  # placeholder single action
        else:
            print(f"Error: No letter directories or images found under {IMAGE_DATA_PATH}")
            exit()

    print(f"Found {len(actions)} actions (letters): {actions}")
except FileNotFoundError:
    print(f"Error: Directory not found at {IMAGE_DATA_PATH}")
    print("Please make sure you have unzipped the 'asl-alphabet.zip' file into 'ASL_Alphabet_Images'")
    exit()


# --- 4. Main Extraction Loop ---
sequences = [] # This will hold all the keypoint arrays
labels = []    # This will hold the corresponding labels (as numbers)

# --- CLI / limits ---
parser = argparse.ArgumentParser(description='Extract hand keypoints from ASL images')
parser.add_argument('--max-images', type=int, default=None,
                    help='If set, stop after processing this many images (useful for quick tests)')
args = parser.parse_args()
max_images = args.max_images

# Using tqdm for a visual progress bar
processed_count = 0
for label_index, action in enumerate(tqdm(actions, desc="Processing Actions")):
    # action might be '.' (images directly under IMAGE_DATA_PATH) or a letter folder
    if action == '.':
        action_path = IMAGE_DATA_PATH
    else:
        action_path = os.path.join(IMAGE_DATA_PATH, action)

    if not os.path.exists(action_path):
        continue

    # Collect image file paths. Be defensive in case there are extra nested subfolders.
    image_paths = []
    for item in os.listdir(action_path):
        if item.startswith('.'):
            continue
        item_path = os.path.join(action_path, item)
        if os.path.isdir(item_path):
            # gather images inside this subdirectory
            for f in os.listdir(item_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(item_path, f))
        else:
            if item.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(action_path, item))

    for image_path in image_paths:
        # respect max_images if provided
        if max_images is not None and processed_count >= max_images:
            break

        keypoints = extract_keypoints(image_path)

        if keypoints is not None:
            sequences.append(keypoints)
            labels.append(label_index) # We use the index (0 for 'A', 1 for 'B', etc.)
            processed_count += 1

    # break outer loop if we hit the max
    if max_images is not None and processed_count >= max_images:
        break

print(f"\nSuccessfully processed {len(sequences)} images.")

# --- 5. Save the new dataset ---
X = np.array(sequences)
y = np.array(labels)

np.save('keypoints_X.npy', X)
np.save('keypoints_y.npy', y)
np.save('keypoints_label_map.npy', actions) # Save the label names

print("\nKeypoint data saved!")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")