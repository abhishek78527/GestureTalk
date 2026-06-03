import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("Loading preprocessed data...")
# --- 1. Load the data ---
try:
    X_train = np.load('mnist_train_X.npy')
    y_train = np.load('mnist_train_y.npy')
    X_val = np.load('mnist_val_X.npy')
    y_val = np.load('mnist_val_y.npy')
except FileNotFoundError:
    print("Error: .npy files not found. Please run preprocess_mnist.py first.")
    exit()

print("Data loaded successfully.")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")


# --- 2. Define the CNN Model ---
# This is a simple but effective CNN architecture for image classification
model = Sequential([
    # Input layer. The shape is (28, 28, 1)
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(), # Flattens the 2D image data into a 1D vector
    
    # Dense (fully connected) layers for classification
    Dense(128, activation='relu'),
    Dropout(0.5), # Dropout helps prevent overfitting
    
    # Output layer. 
    # y_train.shape[1] is 25 (the number of classes)
    # 'softmax' ensures the output is a probability distribution
    Dense(y_train.shape[1], activation='softmax') 
])

# 

# --- 3. Compile the Model ---
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', # Use this loss for one-hot encoded labels
    metrics=['accuracy']
)

model.summary()


# --- 4. Train the Model ---
print("\n--- Starting Model Training ---")
history = model.fit(
    X_train, 
    y_train, 
    epochs=15, # 15 epochs is a good starting point
    validation_data=(X_val, y_val)
)

print("\n--- Training Complete ---")

# --- 5. Save the Trained Model ---
model.save('static_sign_cnn_model.h5')
print("Model saved as 'static_sign_cnn_model.h5'")