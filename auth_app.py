import os
import cv2
import dlib
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load pre-trained face detection model
detector = dlib.get_frontal_face_detector()

# Function to extract face features
def extract_face_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        print("No face detected in the image.")
        return None
    
    # Assuming only one face is present in the image
    face = faces[0]
    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
    face_roi = gray[y:y+h, x:x+w]
    
    # Check if the face ROI is empty or too small
    if face_roi.size == 0 or min(face_roi.shape) < 100:
        print("Face ROI is too small or empty.")
        return None
    
    # Resize the face ROI to a fixed size (e.g., 100x100)
    face_roi = cv2.resize(face_roi, (100, 100))
    
    # Flatten the face ROI to a 1D array (feature vector)
    feature_vector = face_roi.flatten()
    
    return feature_vector

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
            else:
                print("Unable to read image:", path)
        except Exception as e:
            print("Error loading image:", path)
            print(e)
    return images

# Function to extract features from images
def extract_features_from_images(images):
    features = []
    for image in images:
        feature_vector = extract_face_features(image)
        if feature_vector is not None:
            features.append(feature_vector)
    return features

# Function to divide dataset into training and testing sets
def divide_dataset(images):
    num_images = len(images)
    num_training = num_images // 2
    training_images = images[:num_training]
    testing_images = images[num_training:]
    return training_images, testing_images

# Function to compute similarity between two feature vectors
def compute_similarity(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)

# Select threshold value for authentication (you can define this function)
def select_threshold(training_features, training_labels, testing_features):
    # Placeholder for threshold selection logic
    return 0.5  # Dummy threshold value

# Load genuine and imposter images from folders
genuine_folder = r"Original Images/Akshay Kumar"
imposter_folder = r"Faces"

genuine_images = load_images_from_folder(genuine_folder)
imposter_images = load_images_from_folder(imposter_folder)

# Divide genuine and imposter datasets into training and testing sets
training_genuine, testing_genuine = divide_dataset(genuine_images)
training_imposter, testing_imposter = divide_dataset(imposter_images)

# Extract features from training images
training_features_genuine = extract_features_from_images(training_genuine)
training_features_imposter = extract_features_from_images(training_imposter)

# Combine training features and labels
training_features = training_features_genuine + training_features_imposter
training_labels = [1] * len(training_features_genuine) + [0] * len(training_features_imposter)

# Extract features from testing images
testing_features_genuine = extract_features_from_images(testing_genuine)
testing_features_imposter = extract_features_from_images(testing_imposter)

# Combine testing features and labels
testing_features = testing_features_genuine + testing_features_imposter
testing_labels = [1] * len(testing_features_genuine) + [0] * len(testing_features_imposter)

# Select threshold for authentication
threshold = select_threshold(training_features, training_labels, testing_features)
print("Selected Threshold:", threshold)

# Compute similarity between testing and training features separately
similarities = []

for test_feature, test_label in zip(testing_features, testing_labels):
    test_similarities = [compute_similarity(test_feature, train_feature) for train_feature in training_features]
    similarities.extend(test_similarities)

# Print the length of similarities
print("Length of similarities:", len(similarities))

# Compute False Match Rate (FMR) and False Non-Match Rate (FNMR)
fmr_count = 0
fnmr_count = 0

for sim, label in zip(similarities, testing_labels):
    if sim >= threshold:
        if label == 0:
            fmr_count += 1
    else:
        if label == 1:
            fnmr_count += 1

fmr = fmr_count / len(testing_features_imposter)
fnmr = fnmr_count / len(testing_features_genuine)

print("False Match Rate (FMR):", fmr)
print("False Non-Match Rate (FNMR):", fnmr)

# Plot ROC curve
fpr, tpr, _ = roc_curve(testing_labels, similarities)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
