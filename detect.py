from ultralytics import YOLO
import cv2
import time

print("Loading model...")
model = YOLO("yolov8n.pt")
print("Model loaded!")

cap = cv2.VideoCapture(0)

time.sleep(2)

print("Camera opened:", cap.isOpened())

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        continue

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()