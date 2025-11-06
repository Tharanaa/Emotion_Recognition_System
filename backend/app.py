from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load the pre-trained model
model = load_model("model.h5")

# Load face detection classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to process image and predict emotion
def detect_emotion(image_data):
    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(",")[1])
    image = Image.open(BytesIO(image_bytes))
    image = np.array(image.convert("RGB"))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return "No Face Detected"

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        prediction = model.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        return label

    return "No Face Detected"

# API Endpoint for Emotion Detection
@app.route("/detect-emotion", methods=["POST"])
def detect():
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        emotion = detect_emotion(image_data)
        return jsonify({"emotion": emotion})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
