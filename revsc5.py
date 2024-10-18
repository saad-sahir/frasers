from ultralytics import YOLO
import cv2 as cv
from collections import defaultdict

def carpark(video):
    model = YOLO('weights/yolov9t.pt')
    cap = cv.VideoCapture(video)
    start_time = None
    trigger = False
    fps = cap.get(cv.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        current_time = current_frame / fps
        if current_frame % 3 == 0:
            h, w= frame.shape[:2]
            results = model.track(frame, persist=True, verbose=True)
            detections = results[0].boxes.data
            cars = []
            for detection in detections:
                if len(detection) == 7:
                    x_min, y_min, x_max, y_max, id, confidence, class_id = detection
                elif len(detection) == 6:
                    x_min, y_min, x_max, y_max, confidence, class_id = detection
                if confidence >=  0.7 and class_id == 2:
                    cars.append(detection)
                    car_pos = (int((x_max+x_min)//2), int(y_max))
                    x, y = car_pos
                    if y > int(h*0.55) and x < int(w*0.6):
                        cv.rectangle(
                                frame, 
                                (int(x_min), int(y_min)), 
                                (int(x_max), int(y_max)), 
                                (0, 0, 255), 4
                            )
                    else:
                        cv.rectangle(
                                frame, 
                                (int(x_min), int(y_min)), 
                                (int(x_max), int(y_max)), 
                                (0, 255, 0), 4
                            )
            cv.putText(
                frame,
                f"Number of cars: {len(cars)}",
                (25, 10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )
            if len(cars) > 8:
                if not trigger:
                    trigger = True
                    start_time = current_time
                elapsed_time = current_time - start_time
                text = f"Duration : {int(elapsed_time)}s"
                cv.putText(
                    frame,
                    text,
                    (50, 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)
                )
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'): 
                break
    cap.release()
    cv.destroyAllWindows()
