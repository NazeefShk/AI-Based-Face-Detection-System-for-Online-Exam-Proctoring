# AI-Based Face Detection System for Online Exam Proctoring

## Overview
This project is a real-time AI-based face detection system designed to monitor candidates during online examinations using a webcam. The system helps detect suspicious activities such as absence of a face or the presence of multiple faces, supporting integrity in remote assessments.

## Features
- Real-time face detection using webcam
- Continuous monitoring of candidate presence
- Detection of multiple faces to identify suspicious behavior
- Lightweight and easy to run locally

## Technologies Used
- Python
- OpenCV
- Computer Vision
- YOLOv3 (for detection)

## Project Structure
.
├── proctoring.py
├── yolov3.cfg
├── coco.names
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md

csharp
Copy code

## How It Works
The system captures live video from a webcam and processes each frame using computer vision techniques.  
Face detection is applied in real time to:
- Check if a face is present
- Detect multiple faces in the frame

Based on these checks, the system can flag potential suspicious activity during an online examination.

## Model Weights
This project uses **YOLOv3** for detection.

Due to GitHub file size limitations, the `yolov3.weights` file is **not included** in this repository.

### Download the weights from:
https://pjreddie.com/media/files/yolov3.weights

After downloading, place the file in the project root directory.

## Installation & Setup

Of course 👍
Here is the **Installation & Setup section** in a **clean, copy-paste–ready form**, exactly as you can drop into your `README.md`.

---

````markdown
## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/ai-exam-proctoring-system.git
cd ai-exam-proctoring-system
````

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download YOLOv3 weights:

* Visit: [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
* Place the downloaded `yolov3.weights` file in the project root directory

4. Run the application:

```bash
python proctoring.py
```

> Ensure that a webcam is connected before running the program.

Use Cases

Online examinations

Remote academic assessments

Exam proctoring systems

Academic integrity monitoring

Disclaimer

This project is developed for educational and academic purposes and demonstrates the application of computer vision techniques in online exam monitoring.

Author

Nazeef Shaikh

