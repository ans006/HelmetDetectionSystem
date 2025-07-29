![Preview](https://img.shields.io/badge/YOLOv5-Helmet%20Detection-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


# 🛡️ Helmet Detection System using YOLOv5

This project detects whether a person is wearing a helmet or not using real-time webcam video and a YOLOv5 object detection model.

## 🔍 Features

- Real-time helmet and no-helmet detection
- Sound alert for violations
- Trained on custom dataset
- Easy to run with a webcam

## 📦 Requirements

Install dependencies using:
pip install -r requirements.txt


## 🚀 How to Run
python helmet_detect.py

## 🧠 Model
The model is trained on a custom dataset with YOLOv5. Replace best.pt with your own trained weights.

## 📂 Project Structure

HelmetDetectionSystem/
├── yolov5/               # YOLOv5 models and utils
├── helmet_data/          # Training images and labels
├── requirements.txt
├── helmet_detect.py      # Main script
├── README.md
└── LICENSE

## 📜 License
This project is licensed under the MIT License – see the LICENSE file for details.


## 🌟 Future Plans
Upload violations to server
Add SMS/Email alerts
Add helmet color classification

## 👩‍💻 Author
Developed by Arpita N Sheelvanth
Final Year B.E. – Information Science & Engineering
Passionate about AI, Computer Vision, and Full-Stack Development.
