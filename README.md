🎭 Emotion Recognition System

🧠 Project Overview

The Emotion Recognition System is a deep learning-based web application that detects and classifies human emotions in real time using a webcam feed. It captures facial expressions, processes them using a Convolutional Neural Network (CNN) model, and displays the detected emotion on the screen.
This project combines computer vision and deep learning to demonstrate human–computer interaction through facial emotion analysis.

🚀 Key Features

🎥 Real-time Emotion Detection – Captures live video feed via webcam and detects facial expressions instantly.

🧠 CNN-based Emotion Classification – Classifies emotions such as Happy, Sad, Angry, Surprise, Neutral, etc.

🌐 Flask Web Deployment – Lightweight Flask server to host the model and manage webcam streaming.

💾 Efficient Face Detection – Utilizes OpenCV’s Haar cascade classifier for accurate face region extraction.

🔧 Modular Design – Easy to update model, integrate APIs, or extend to other recognition systems.

🔄 Project Workflow

Capture Input: The webcam captures live frames of the user.

Preprocess Image: The frame is converted to grayscale and cropped to extract the face region.

Model Prediction: The CNN model processes the face and predicts the corresponding emotion.

Display Output: The recognized emotion is overlaid on the live video stream in real-time.

Web Interface: Flask handles the backend and serves the video feed to the browser.


🛠️ Technologies Used

Programming Language: Python

Libraries:

OpenCV (for image capture and face detection)

TensorFlow / Keras (for CNN model)

NumPy & Pandas (for data handling)

Flask (for deployment and web streaming)

Tools: Jupyter Notebook, VS Code


⚙️ Installation and Setup

1️⃣ Clone the Repository
git clone https://github.com/<your-username>/Emotion_Recognition_System.git
cd EmotionRecognitionSystem

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
python app.py

📊 Model Details

CNN architecture trained on facial emotion dataset (e.g., FER-2013).

Optimized for real-time inference with webcam input.

Supports emotion classes like Happy, Sad, Angry, Surprise, Neutral, Fear, Disgust.
