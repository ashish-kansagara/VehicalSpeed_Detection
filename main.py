from ultralytics import YOLO
import cv2
from sort import Sort
import numpy as np

model = YOLO('yolov8n.pt')

# img = cv2.imread('car.jpg')

# results = model(img, show=False)

# annotated_img = results[0].plot()

# cv2.imshow('YOLOv8 Detection', annotated_img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap=cv2.VideoCapture("154195-807166827.mp4")
mask=cv2.imread('Untitled_design.png')# for masking video for sertain place
mask = cv2.resize(mask, (1280, 720))

while True:
    sucess,img=cap.read()
    success, img = cap.read()
    if not success:
        break
    img = cv2.resize(img, (1280, 720))
    imRegion=cv2.bitwise_and(img,mask)
    results=model(imRegion,stream=True)
    #results=model(img,stream=True)
    tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
    detections=np.empty((0,5))
    
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            conf=float(box.conf[0])
            cls=int(box.cls[0])
            label=model.names[cls]
            
            if label in ['car','truck','bus','motorbike','bike']:
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,1,255),2)
                cv2.putText(img,f"{label} {conf:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
                #current=np.array([x1,y1,x2,y2,conf])
                #detections=np.vstack((detections, current))
            
    # resultTracker=tracker.update(detections)
    # for x1,y1,x2,y2, _id in  resultTracker:
    #     x1,y1,x2,y2,_id=map(int,(x1,y1,x2,y2, _id))
    #     cx=int((x1+x2)/2)  
    #     cy=int((y1+y2)/2)  
        
    #     cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)
        
    #     cv2.putText(img,f"Id:{_id}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
        
    #     cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        
    cv2.imshow("webcam",img)
    cv2.waitKey(1)