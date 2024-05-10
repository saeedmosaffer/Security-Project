import cv2
import dlib
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()

shape_predictor_path = "patht"
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_recognizer = dlib.face_recognition_model_v1("")

def extract_face_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    face = faces[0]
    shape = shape_predictor(gray, face)
    face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
    
    return np.array(face_descriptor)

def compute_similarity(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)


image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")

feature1 = extract_face_features(image1)
feature2 = extract_face_features(image2)

similarity = compute_similarity(feature1, feature2)

print("Similarity Score:", similarity)

fpr, tpr, thresholds = roc_curve([0, 1, 1, 0], [0.1, 0.9, 0.7, 0.3])
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
