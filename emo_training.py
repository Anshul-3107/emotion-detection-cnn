"""
Emotion Detection Project

Pipeline:
- Loads FER2013 dataset from local folders (downloaded via Kaggle)
- Preprocesses and trains a CNN
- Saves model to emotion_model.h5 and emotion_model.tflite
- Runs real-time webcam emotion detection

Usage:
    python emo_detection.py
Then type 'train' to train or 'detect' to run webcam detection.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = 48
BATCH_SIZE = 64
NUM_CLASSES = 7
EPOCHS = 15
MODEL_PATH = "emotion_model.h5"
TFLITE_PATH = "emotion_model.tflite"
EMOTION_LABELS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# -----------------------------
# Build CNN model
# -----------------------------
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# -----------------------------
# Train model using Kaggle FER2013 dataset
# -----------------------------
def train_model():
    train_dir = "dataset/train"
    val_dir = "dataset/test"

    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(
        train_dir, target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale", class_mode="categorical",
        batch_size=BATCH_SIZE
    )
    val_gen = datagen.flow_from_directory(
        val_dir, target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale", class_mode="categorical",
        batch_size=BATCH_SIZE
    )

    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    # Save legacy HDF5 model
    model.save(MODEL_PATH)

    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Model trained and saved to {MODEL_PATH} and {TFLITE_PATH}")

# -----------------------------
# Real-time detection
# -----------------------------
def realtime_detection():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file '{MODEL_PATH}' not found. Run training first (choose 'train').")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    print("🎥 Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray_frame[y:y+h, x:x+w]
            try:
                roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            except Exception:
                continue
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=-1)  # channel
            roi = np.expand_dims(roi, axis=0)   # batch

            preds = model.predict(roi, verbose=0)
            label = EMOTION_LABELS[int(np.argmax(preds))]

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Options: 'train' to train model; 'detect' to run webcam detection.")
    choice = input("Enter choice (train/detect): ").strip().lower()
    if choice == "train":
        train_model()
    elif choice == "detect":
        realtime_detection()
    else:
        print("Invalid choice. Exiting.")