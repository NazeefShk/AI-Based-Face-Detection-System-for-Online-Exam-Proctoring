import cv2
import numpy as np
import time

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

warning_count = 0
prev_face_x = None
frame_count = 0
phone_detected = False

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        warning_count += 1

    if len(faces) > 1:
        cv2.putText(frame, "MULTIPLE FACES DETECTED", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        warning_count += 1

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if prev_face_x is not None:
            movement = abs(x - prev_face_x)
            if movement > 60:
                cv2.putText(frame, "HEAD MOVEMENT", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                warning_count += 1

        prev_face_x = x

    if frame_count % 15 == 0 and len(faces) > 0:
        phone_detected = False

        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False
        )
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.6 and classes[class_id] == "cell phone":
                    phone_detected = True
                    warning_count += 1

    if phone_detected:
        cv2.putText(frame, "PHONE DETECTED", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps}", (500, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Warnings: {warning_count}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("AI Exam Proctoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
