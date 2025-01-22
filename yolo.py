from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

img = cv2.imread('test.jpg')
results = model(img)[0]

for result in results.boxes.data.tolist():

    x1, y1, x2, y2, score, class_id = result
    class_name = results.names[int(class_id)]

    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 4))