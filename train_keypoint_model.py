import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- 1. Load Data ---
print("Loading keypoint data...")
try:
    X = np.load('keypoints_X.npy')
    y = np.load('keypoints_y.npy')
    actions = np.load('keypoints_label_map.npy')
except FileNotFoundError:
    print("Error: keypoints_X.npy or keypoints_y.npy not found.")
    print("Please run extract_keypoints_from_images.py first.")
    exit()

print(f"X shape: {X.shape}") # Should be (87000, 63)
print(f"y shape: {y.shape}") # Should be (87000,)

# --- 2. Preprocess Labels ---
# We need to one-hot encode the labels for categorical_crossentropy
y_categorical = to_categorical(y)

num_classes = len(actions)
print(f"Found {num_classes} classes.")

# --- 3. Split Data ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_categorical, 
    test_size=0.2,  # 20% for testing
    random_state=42,
    stratify=y_categorical # Ensures balanced classes in train/test splits
)

# --- 4. Define the Model (Dense Network) ---
model = Sequential([
    # Input layer: 63 keypoints
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    # Output layer: 'num_classes' neurons (29), one for each sign
    Dense(num_classes, activation='softmax')
])



# --- 5. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 6. Train the Model ---
print("\n--- Starting Model Training ---")
history = model.fit(
    X_train,
    y_train,
    epochs=30, # 30 epochs is a good start
    validation_data=(X_test, y_test)
)

print("\n--- Training Complete ---")

# --- 7. Save the Model ---
# We use the .keras format as recommended by the warning you saw earlier
model.save('keypoint_model.keras')
print("Model saved as 'keypoint_model.keras'")