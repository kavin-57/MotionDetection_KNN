import cv2
import numpy as np
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import os
import matplotlib.pyplot as plt

# Define edge detection function
def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# Define HOG feature extraction function using OpenCV
def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image)
    return hog_features.flatten()

# Define function for FFT on extracted features
def compute_fft(features):
    return np.abs(fft(features))

# Directory containing images
image_folder = "C:/Users/KAVIN/Desktop/New folder/images/"

# List to store features and labels
motion_data = []
labels = []

# Process each image in the directory
for image_name in sorted(os.listdir(image_folder)):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Warning: Could not load image {image_name}. Skipping...")
        continue  # Skip if the image can't be loaded

    # Resize image to a fixed size (optional, adjust as needed)
    image = cv2.resize(image, (128, 128))

    # Determine motion label from the filename
    if 'walking' in image_name.lower():
        label = "Walking"
    elif 'standing' in image_name.lower():
        label = "Standing"
    elif 'sitting' in image_name.lower():
        label = "Sitting"
    elif 'running' in image_name.lower():
        label = "Running"
    else:
        print(f"Warning: No valid motion label found for {image_name}. Skipping...")
        continue  # Skip if no valid label is found

    # Detect edges as pre-processing step
    edges = detect_edges(image)
    # Extract HOG features from the edge-detected image
    hog_features = extract_hog_features(edges)

    if hog_features.size == 0:
        print(f"Warning: No HOG features extracted from {image_name}. Skipping...")
        continue  # Skip if no features are extracted

    # Apply FFT to HOG features for frequency analysis
    fft_features = compute_fft(hog_features)

    # Append the features and the corresponding label
    motion_data.append(fft_features)
    labels.append(label)

# Check if any data was collected
if not motion_data or not labels:
    print("No valid motion data was collected. Exiting...")
else:
    # Convert motion data to NumPy array
    X = np.array(motion_data)
    y = np.array(labels)

    # Check number of samples
    n_samples = X.shape[0]
    if n_samples < 2:
        print(f"Only {n_samples} samples collected. Not enough to split into training, testing, and validation sets.")
        # Use all for training and set test/val as empty
        X_train, y_train = X, y
        X_test, y_test = np.array([]), np.array([])  # No test set
        X_val, y_val = np.array([]), np.array([])    # No validation set
    else:
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))

        # If any class has less than 2 samples, handle it
        if any(count < 2 for count in class_counts.values()):
            print("Warning: Some classes have less than 2 samples. Adjusting the dataset for training.")
            # Use all available data for training and skip stratified split
            X_train, y_train = X, y
            # Create empty test and validation sets
            X_test, y_test = np.array([]), np.array([])
            X_val, y_val = np.array([]), np.array([])
        else:
            # Split data into sample, training, testing, and validation sets
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

            # Split temp set into test and validation sets
            X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Initialize and train Random Forest classifier if there are enough samples
    if len(X_train) > 0:
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)

        # Evaluate model on training, testing, and validation sets
        train_accuracy = rf_model.score(X_train, y_train)
        test_accuracy = rf_model.score(X_test, y_test) if len(X_test) > 0 else 0.0
        val_accuracy = rf_model.score(X_val, y_val) if len(X_val) > 0 else 0.0

        # Generate predictions for confusion matrix
        y_pred = rf_model.predict(X_test) if len(X_test) > 0 else []

        # Prepare confusion matrix if there are test samples
        if len(y_pred) > 0:
            cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
            disp.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.show()

        # Prepare data for CSV output
        data = {
            'Motion Pattern': labels,
            'Sample Set': ['Sample'] * len(X),
            'Train Set': ['Train' if i < len(X_train) else '' for i in range(len(X))],
            'Test Set': ['Test' if len(X_train) <= i < len(X_train) + len(X_test) else '' for i in range(len(X))],
            'Validation Set': ['Validation' if i >= len(X_train) + len(X_test) else '' for i in range(len(X))]
        }

        # Save data to CSV
        output_df = pd.DataFrame(data)
        output_df.to_csv('motion_detection_output.csv', index=False)

        # Print performance metrics
        print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    else:
        print("No training data available. Exiting...")

# Generate joint angle comparison graph
# Sample data creation for joint angles
# Replace this with actual data loading from your CSV file
joint_data = {
    'Motion': ['Sitting', 'Sitting', 'Standing', 'Standing', 'Walking', 'Walking', 'Running', 'Running'],
    'Joint': ['Hip', 'Knee', 'Hip', 'Knee', 'Hip', 'Knee', 'Hip', 'Knee'],
    'Angle': [30, 80, 60, 70, 45, 85, 50, 90]  # Example angles
}

# Convert to DataFrame
joint_df = pd.DataFrame(joint_data)

# Pivot the data for easier plotting
pivot_joint_df = joint_df.pivot(index='Motion', columns='Joint', values='Angle')

# Create a bar plot for hip and knee joint angles
plt.figure(figsize=(10, 6))
ax = pivot_joint_df.plot(kind='bar')
plt.title('Hip and Knee Joint Angle Comparison')
plt.xlabel('Motion')
plt.ylabel('Joint Angle (degrees)')
plt.xticks(rotation=0)
plt.legend(title='Joint')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()
