# Facial-Emotion-Recognition
This project detects human facial emotions in real-time using a web🔍 Facial Emotion Recognition with Real-Time Webcam
This project detects human emotions in real-time using your webcam and a deep learning model trained on the FER-2013 dataset. It uses OpenCV for face detection and a CNN model built with TensorFlow/Keras to classify emotions.

🎯 Features:
Real-time facial emotion detection via webcam

Pre-trained CNN model (emotion_model.keras or .h5)

Trained on FER-2013 dataset (7 emotion classes)

Uses OpenCV for face capture and display

🧠 Technologies Used:
Python

TensorFlow / Keras

OpenCV

NumPy

📁 Folder Structure:
bash
Copy
Edit
emotion_project/
├── emotion_model.h5          # Trained Keras model
├── real_time_emotion.py      # Python script for webcam emotion detection
└── README.md
🚀 How to Run:
Install dependencies:

bash
Copy
Edit
pip install tensorflow opencv-python numpy
Run the emotion detector:

bash
Copy
Edit
python real_time_emotion.pycam
