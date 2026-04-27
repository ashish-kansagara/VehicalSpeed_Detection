from ultralytics import YOLO
import cv2
from sort import Sort
import numpy as np

model = YOLO('yolov10n.pt')

cap=cv2.VideoCapture("13105476_3840_2160_30fps.mp4")
mask=cv2.imread('mask.png')# for masking video for sertain place
mask = cv2.resize(mask, (1280, 720))


while True:
    sucess,img=cap.read()
    success, img = cap.read()
    if not success:
        break
    img = cv2.resize(img, (1280, 720))
    imRegion=cv2.bitwise_and(img,mask)
    results=model(imRegion,stream=True)
    tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
    #results=model(img,stream=True)
    detections=np.empty((0,5))
    
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            conf=float(box.conf[0])
            cls=int(box.cls[0])
            label=model.names[cls]
            
            if label in ['car','truck','bus','motorcycle']:
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,1,255),2)
                cv2.putText(img,f"{label} {conf:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
                #current=np.array([x1,y1,x2,y2,conf])
                #detections=np.vstack((detections, current))
            
   
    cv2.imshow("webcam",img)
    cv2.waitKey(1)