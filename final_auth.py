import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve

train_dir="Original Images/"
generator = ImageDataGenerator()
train_ds = generator.flow_from_directory(train_dir,target_size=(224, 224),batch_size=32)
classes = list(train_ds.class_indices.keys())

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(len(classes),activation='softmax'))

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ["accuracy"])
model.summary()

history = model.fit(train_ds,epochs= 2, batch_size=32)

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.xlabel('Time')
plt.legend(['accuracy', 'loss'])
plt.show()

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224,224,3))
    plt.imshow(img)
    plt.show()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pred = model.predict(images, batch_size=32)
    print("Actual: "+(image_path.split("/")[-1]).split("_")[0])
    print("Predicted: "+classes[np.argmax(pred)])

predict_image("Original Images/Brad Pitt/Brad Pitt_102.jpg")
predict_image("Original Images/Charlize Theron/Charlize Theron_26.jpg")
predict_image("Original Images/Henry Cavill/Henry Cavill_28.jpg")
predict_image("Original Images/Tom Cruise/Tom Cruise_27.jpg")
predict_image("Original Images/Robert Downey Jr/Robert Downey Jr_106.jpg")
predict_image("Original Images/Natalie Portman/Natalie Portman_25.jpg")
predict_image("Original Images/Lisa Kudrow/Lisa Kudrow_34.jpg")
predict_image("Original Images/Ellen Degeneres/Ellen Degeneres_20.jpg")
predict_image("Original Images/Dwayne Johnson/Dwayne Johnson_29.jpg")
predict_image("Original Images/Elizabeth Olsen/Elizabeth Olsen_11.jpg")

# Function to calculate FMR and FNMR
def calculate_rates(threshold, y_true, y_pred):
    genuine_accepts = np.sum((y_pred >= threshold) & (y_true == 1))
    impostor_accepts = np.sum((y_pred >= threshold) & (y_true == 0))
    genuine_rejects = np.sum((y_pred < threshold) & (y_true == 1))
    impostor_rejects = np.sum((y_pred < threshold) & (y_true == 0))

    fmr = impostor_accepts / (impostor_accepts + genuine_rejects)
    fnmr = genuine_accepts / (genuine_accepts + impostor_rejects)

    return fmr, fnmr

# Predict probabilities for test images
test_images = ["Original Images/Brad Pitt/Brad Pitt_102.jpg", 
               "Original Images/Charlize Theron/Charlize Theron_26.jpg",
               "Original Images/Henry Cavill/Henry Cavill_28.jpg",
               "Original Images/Tom Cruise/Tom Cruise_27.jpg",
               "Original Images/Robert Downey Jr/Robert Downey Jr_106.jpg",
               "Original Images/Natalie Portman/Natalie Portman_25.jpg",
               "Original Images/Lisa Kudrow/Lisa Kudrow_34.jpg",
               "Original Images/Ellen Degeneres/Ellen Degeneres_20.jpg",
               "Original Images/Dwayne Johnson/Dwayne Johnson_29.jpg",
               "Original Images/Elizabeth Olsen/Elizabeth Olsen_11.jpg"]

true_labels = np.array([0, 1, 1, 0, 0, 1, 0, 1, 0, 1])  # Assuming 0 for impostor, 1 for genuine

predicted_probs = []
for image_path in test_images:
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred_prob = model.predict(x)[0, 1]  # Assuming 1 for genuine class
    predicted_probs.append(pred_prob)

# Calculate FMR and FNMR for different thresholds
thresholds = np.linspace(0, 1, 100)
fmrs = []
fnmrs = []
for threshold in thresholds:
    fmr, fnmr = calculate_rates(threshold, true_labels, predicted_probs)
    fmrs.append(fmr)
    fnmrs.append(fnmr)

# Plot ROC curve
plt.plot(fmrs, fnmrs)
plt.xlabel('False Match Rate (FMR)')
plt.ylabel('False Non-Match Rate (FNMR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.grid(True)
plt.show()

# Calculate Equal Error Rate (EER)
eer_threshold = thresholds[np.argmin(np.abs(np.array(fmrs) - np.array(fnmrs)))]
eer = (fmrs[np.argmin(np.abs(np.array(fmrs) - np.array(fnmrs)))] + fnmrs[np.argmin(np.abs(np.array(fmrs) - np.array(fnmrs)))]) / 2

print(f"Equal Error Rate (EER): {eer:.4f} at threshold {eer_threshold:.4f}")