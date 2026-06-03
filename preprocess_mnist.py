import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- 1. Load the Dataset ---
try:
    data = pd.read_csv('SignLanguageMNIST/sign_mnist_train.csv')
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {data.shape}")
except FileNotFoundError:
    print("Error: sign_mnist_train.csv not found in the 'SignLanguageMNIST' folder.")
    print("Please make sure you have downloaded and unzipped the dataset correctly.")
    exit()

# --- 2. Separate Labels (Y) and Pixel Data (X) ---
# The first column 'label' is our target (Y)
labels = data['label'].values

# Drop the 'label' column to get our pixel data (X)
pixels = data.drop('label', axis=1).values

print(f"Pixel data shape (flat): {pixels.shape}")
print(f"Label data shape: {labels.shape}")

# --- 3. Preprocessing ---

# A. Normalize Pixel Data
# Divide by 255.0 to scale values between 0.0 and 1.0
X_normalized = pixels.astype('float32') / 255.0

# B. Reshape Pixel Data
# The Keras model will expect a 4D array: (num_samples, height, width, color_channels)
# For grayscale, color_channels is 1.
X_reshaped = X_normalized.reshape(-1, 28, 28, 1)

# C. One-Hot Encode Labels
# Converts a label '3' into an array like [0, 0, 0, 1, 0, ...]
# This is required for categorical_crossentropy loss.
y_categorical = to_categorical(labels)

print(f"\n--- Preprocessing Complete ---")
print(f"Reshaped X shape: {X_reshaped.shape}")
print(f"Categorical Y shape: {y_categorical.shape}")


# --- 4. Create Training and Validation Sets ---
# It's crucial to test your model on data it hasn't seen.
# We'll split the loaded data into 80% for training and 20% for validation.
X_train, X_val, y_train, y_val = train_test_split(
    X_reshaped, 
    y_categorical, 
    test_size=0.2, 
    random_state=42 # for reproducible results
)

print(f"\n--- Data Split ---")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_validation shape: {X_val.shape}")
print(f"y_validation shape: {y_val.shape}")

# You can now save these or use them directly in a training script
# For example, to save them:
np.save('mnist_train_X.npy', X_train)
np.save('mnist_train_y.npy', y_train)
np.save('mnist_val_X.npy', X_val)
np.save('mnist_val_y.npy', y_val)

print("\nProcessed data saved to .npy files!")