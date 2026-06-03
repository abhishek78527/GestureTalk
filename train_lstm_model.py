import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

# --- 1. Load Data ---
DATA_PATH = os.path.join('MP_Data_Dynamic')

# Get action names from folder names
actions = np.array([d for d in os.listdir(DATA_PATH) if not d.startswith('.')])

# 30 sequences per action, 30 frames per sequence
no_sequences = 30
sequence_length = 30

sequences, labels = [], []
for action_index, action in enumerate(actions):
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window) # X data
        labels.append(action_index) # y data

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"X shape: {X.shape}") # Should be (90, 30, 63) for 3 actions
print(f"y shape: {y.shape}") # Should be (90, 3)

# Save the label map
np.save('dynamic_actions_map.npy', actions)

# --- 2. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# --- 3. Define the LSTM Model ---
model = Sequential([
    # Input shape is (30, 63) - 30 frames, 63 keypoints each
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)),
    Dropout(0.2),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    # Output layer with 3 units (hello, thankyou, welcome)
    Dense(actions.shape[0], activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 4. Train the Model ---
print("\n--- Starting LSTM Model Training ---")
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

print("\n--- Training Complete ---")

# --- 5. Save the Model ---
model.save('dynamic_model.keras')
print("Model saved as 'dynamic_model.keras'")