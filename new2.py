import cv2
import numpy as np
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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

# Directory containing video files
video_folder = "C:/Users/KAVIN/Desktop/New folder/videos/"  # Change this path to your folder

# List to store features and labels
motion_data = []
labels = []

# Process each video in the directory
for video_name in sorted(os.listdir(video_folder)):
    video_path = os.path.join(video_folder, video_name)
    cap = cv2.VideoCapture(video_path)

    # Determine motion label from the filename
    if 'walking' in video_name.lower():
        label = "Walking"
    elif 'standing' in video_name.lower():
        label = "Standing"
    elif 'sitting' in video_name.lower():
        label = "Sitting"
    elif 'running' in video_name.lower():
        label = "Running"
    else:
        print(f"Warning: No valid motion label found for {video_name}. Skipping...")
        continue  # Skip if no valid label is found

    # Extract and process frames from the video
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no frames are left

        # Resize frame to a fixed size (optional, adjust as needed)
        frame = cv2.resize(frame, (128, 128))

        # Detect edges as pre-processing step
        edges = detect_edges(frame)

        # Extract HOG features from the edge-detected frame
        hog_features = extract_hog_features(edges)

        if hog_features.size == 0:
            print(f"Warning: No HOG features extracted from frame {frame_count} of {video_name}. Skipping...")
            continue

        # Apply FFT to HOG features for frequency analysis
        fft_features = compute_fft(hog_features)

        # Append the features and the corresponding label
        motion_data.append(fft_features)
        labels.append(label)

        frame_count += 1

    cap.release()

# Check if any data was collected
if not motion_data or not labels:
    print("No valid motion data was collected. Exiting...")
else:
    # Convert motion data to NumPy array
    X = np.array(motion_data)
    y = np.array(labels)

    # Train/Test split with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Further split into Test and Validation sets
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Initialize and train Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate model on training, testing, and validation sets
    train_accuracy = rf_model.score(X_train, y_train)
    test_accuracy = rf_model.score(X_test, y_test)
    val_accuracy = rf_model.score(X_val, y_val)

    # Generate predictions and confusion matrix
    y_pred = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # Display accuracy of each individual classification tree in Random Forest
    individual_accuracies = [tree.score(X_test, y_test) for tree in rf_model.estimators_]
    plt.figure()
    plt.plot(individual_accuracies, label="Individual Tree Accuracies")
    plt.axhline(y=test_accuracy, color='r', linestyle='--', label="Random Forest Accuracy")
    plt.xlabel("Tree Index")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy of Individual Trees vs Random Forest")
    plt.show()

    # Hierarchical Clustering with dendrogram
    hc = AgglomerativeClustering(n_clusters=len(np.unique(y)), metric='euclidean', linkage='ward')
    labels_pred = hc.fit_predict(X)

    # Linkage matrix for plotting dendrogram
    Z = linkage(X, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(Z, labels=y, leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index or Label')
    plt.ylabel('Distance')
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
