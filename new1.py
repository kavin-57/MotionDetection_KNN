import cv2
import numpy as np
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter

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
        continue

    # Resize image to a fixed size
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
        continue

    # Detect edges and extract HOG features
    edges = detect_edges(image)
    hog_features = extract_hog_features(edges)

    if hog_features.size == 0:
        print(f"Warning: No HOG features extracted from {image_name}. Skipping...")
        continue

    # Apply FFT to HOG features
    fft_features = compute_fft(hog_features)

    # Append features and labels
    motion_data.append(fft_features)
    labels.append(label)

# Check if any data was collected
if not motion_data or not labels:
    print("No valid motion data was collected. Exiting...")
else:
    # Convert motion data to NumPy array
    X = np.array(motion_data)
    y = np.array(labels)

    # Check class distribution
    class_counts = Counter(y)
    if all(count >= 2 for count in class_counts.values()):
        # If each class has at least two samples, use stratified split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    else:
        # Use a non-stratified split if any class has fewer than two samples
        print("Warning: Some classes have fewer than 2 samples. Using non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize Random Forest and individual decision trees
    rf_model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracy = rf_model.score(X_test, y_test)

    # Individual tree accuracies
    tree_accuracies = []
    for i, tree in enumerate(rf_model.estimators_):
        y_tree_pred = tree.predict(X_test)
        tree_accuracy = accuracy_score(y_test, y_tree_pred)
        tree_accuracies.append(tree_accuracy)
        print(f"Accuracy of Decision Tree {i+1}: {tree_accuracy * 100:.2f}%")

    # Overall Random Forest accuracy
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

    # Confusion matrix
    y_pred = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # Bar plot of tree accuracies
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(tree_accuracies) + 1), tree_accuracies, color='skyblue')
    plt.axhline(y=rf_accuracy, color='red', linestyle='--', label=f'RF Accuracy: {rf_accuracy * 100:.2f}%')
    plt.xlabel('Tree Index')
    plt.ylabel('Accuracy')
    plt.title('Individual Decision Tree Accuracies')
    plt.legend()
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

    # Save results to CSV
    output_data = {
        'Sample': labels,
        'Predicted': rf_model.predict(X),
        'RF Accuracy': [rf_accuracy] * len(y),
    }
    output_df = pd.DataFrame(output_data)
    output_df.to_csv('motion_classification_results.csv', index=False)

    # Print performance metrics
    print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
