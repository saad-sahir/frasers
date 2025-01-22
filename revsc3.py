import datetime
import cv2
import base64
from ultralytics import YOLO
from collections import defaultdict

def loading_bay(video):
    model = YOLO('weights/yolov9t.pt')
    cap = cv2.VideoCapture(video)
    notifications = defaultdict(lambda: False)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    api_post_template = {
        "building": "AP",
        "camera_number": "DO01",
        "event_date": None,
        "event_type": "SC02",
        "image": None,
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = current_frame / fps
        if current_frame % 3 == 0:
            
            height, width = frame.shape[:2]            
            ## width < 0.8
            ## height > 0.35
            
            results = model.track(frame, persist=True, verbose=False)
            detections = results[0].boxes.data
            no_cars = len([d for d in detections if d[-1] == 2])
            cv2.putText(
                frame,
                f"Number of cars: {no_cars}",
                (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4
            )
            for detection in detections:
                if len(detection) == 7:
                    x_min, y_min, x_max, y_max, id, confidence, class_id = detection
                else:
                    x_min, y_min, x_max, y_max, confidence, class_id = detection
                    id = None
                if id and int(class_id) == 2:
                    
                    car_pos = (int((x_max+x_min)//2), int(y_max))
                    x, y = car_pos
                    
                    if x < width * 0.8 and y > height * 0.35:
                        car_id = int(id)
                        cv2.putText(
                            frame,
                            f"({int(id)})",
                            (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )
                        cv2.rectangle(
                            frame,
                            (int(x_min), int(y_min)),
                            (int(x_max), int(y_max)),
                            (255, 0, 0), 4
                        )
                        if not notifications[car_id]:
                            api_post = api_post_template
                            api_post['event_date'] = datetime.datetime.now().isoformat()
                            resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                            _, buffer = cv2.imencode('.jpg', resized_frame)
                            api_post["image"] = base64.b64encode(buffer).decode('utf-8')
                            print(api_post)
                            notifications[car_id] = True
                            
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    video = 'revsc3.mp4'
    results = loading_bay(video)
