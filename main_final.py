from ultralytics import YOLO
import cv2
from sort import Sort
import numpy as np
import time

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture("Untitled design.mp4")

mask = cv2.imread('mask.png')
mask = cv2.resize(mask, (1280, 720))

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

prev_positions = {}
prev_time = time.time()

pixels_per_meter = 8  

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1280, 720))
    imRegion = cv2.bitwise_and(img, mask)

    results = model(imRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ['car', 'truck', 'bus', 'motorcycle']:
                current = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current))

    resultsTracker = tracker.update(detections)

    current_time = time.time()
    time_diff = current_time - prev_time

    for x1, y1, x2, y2, _id in resultsTracker:
        x1, y1, x2, y2, _id = map(int, (x1, y1, x2, y2, _id))

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        speed = 0

        if _id in prev_positions:
            px, py = prev_positions[_id]

            distance_pixels = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            distance_meters = distance_pixels / pixels_per_meter

            speed = (distance_meters / time_diff) * 3.6  

        prev_positions[_id] = (cx, cy)

        
        if speed > 50:
            color = (0, 0, 255)  # RED
        else:
            color = (0, 255, 0)  # GREEN

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"ID:{_id} {int(speed)} km/h",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    prev_time = current_time

    cv2.imshow("Speed Detection", img)

    cv2.waitKey(1) 
        

cap.release()
cv2.destroyAllWindows()