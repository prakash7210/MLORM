import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
class AutomatedSystem:
    def __init__(self, img_size=128, batch_size=4, epochs=5):
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.model = None

    def preprocess_image(self, path):  
        path = str(path)
        if not os.path.isfile(path):
            print(f"Warning: File does not exist: {path}")
            return None
        
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: cv2.imread failed: {path}")
            return None
        
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=(0, -1))  
        return img

    def load_dataset(self, keys_folder, assignments_folders):
        images, labels = [], []

        for f in os.listdir(keys_folder):
            path = os.path.join(keys_folder, f)
            img = self.preprocess_image(path)
            if img is not None:
                images.append(img)
                labels.append(1)

        for version in assignments_folders:
            folder = os.path.join("data/assignments", version)
            if not os.path.exists(folder):
                print(f"Warning: Folder not found: {folder}")
                continue
            for f in os.listdir(folder):
                path = os.path.join(folder, f)
                img = self.preprocess_image(path)
                if img is not None:
                    images.append(img)
                    labels.append(0)

        if not images:
            raise ValueError("No images found in dataset!")
        
        images = np.vstack(images)
        labels = np.array(labels, dtype=np.int32)
        return images, labels

    def build_model(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(self.IMG_SIZE, self.IMG_SIZE, 1)),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])
        self.model.compile(optimizer='adam', 
                           loss='sparse_categorical_crossentropy', 
                           metrics=['accuracy'])

    def train(self, X, y):
        if self.model is None:
            self.build_model()
        self.model.fit(X, y, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE)

    def predict_label(self, img_path):
        img = self.preprocess_image(img_path)
        if img is None:
            return "File not found or invalid", 0.0
        
        pred = self.model.predict(img, verbose=0)[0]
        label = "Correct" if np.argmax(pred) == 1 else "Incorrect"
        score = float(np.max(pred) * 100)
        return label, score

if __name__ == "__main__":
    keys_folder = "data/keys"
    assignment_versions = ["A", "B"]

    omr = AutomatedSystem()
    X, y = omr.load_dataset(keys_folder, assignment_versions)
    print(f"Dataset loaded: {X.shape}, Labels: {y.shape}")

    omr.train(X, y)
    print("Training completed!")

    with open("omr_model.pkl", "wb") as f:
        pickle.dump(omr, f)
    print("OMRModel instance saved as omr_model.pkl")


app = Flask(__name__)
CORS(app)
with open("omr_model.pkl", "rb") as f:
    omr_system = pickle.load(f)
    if omr_system.model is None and os.path.exists("omr_cnn.h5"):
        omr_system.model = tf.keras.models.load_model("omr_cnn.h5")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    temp_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(temp_path)

    label, score = omr_system.predict_label(temp_path)

    os.remove(temp_path)

    return jsonify({ "score": round(score, 2)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
