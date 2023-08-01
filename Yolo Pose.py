import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt') 

cap =cv2.VideoCapture(0)

while cap.isOpened():
    success,frame=cap.read()

    if success:
        results =model(frame,save = True)
        annotated = results[0].plot() 

        cv2.imshow("YPOLO",annotated)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    
    else : 
        print("Error")
        break
cap.release()