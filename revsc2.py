import base64
import cv2
import datetime
from collections import defaultdict
from ultralytics import YOLO

def person_loitering(video):
    model = YOLO("weights/yolov9t.pt")
    cap = cv2.VideoCapture(video)
    start_times = defaultdict(lambda: None)
    triggers = defaultdict(lambda: False)
    notifications = defaultdict(lambda: False)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # api_post_template = {
    #     "building": "AP",
    #     "camera_number": "DO01",
    #     "event_date": None,
    #     "event_type": "SC02",
    #     "image": None,
    # }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = current_frame / fps
        if current_frame % 1 == 0:
            results = model.track(frame, persist=True, verbose=False)
            detections = results[0].boxes.data
            for detection in detections:
                if len(detection) == 7:
                    x_min, y_min, x_max, y_max, id, confidence, class_id = detection
                else:
                    x_min, y_min, x_max, y_max, confidence, class_id = detection
                    id = None
                if id and int(class_id) == 0:
                    person_id = int(id)
                    cv2.putText(
                        frame,
                        f"({int(id)})",
                        (int(x_min), int(y_min)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
                    cv2.rectangle(
                        frame,
                        (int(x_min), int(y_min)),
                        (int(x_max), int(y_max)),
                        (255, 0, 0), 4
                    )
                    if not triggers[person_id]:
                        triggers[person_id] = True
                        start_times[person_id] = current_time
                    elapsed_time = current_time - start_times[person_id]
                    text = f"Duration : {int(elapsed_time)}s"
                    cv2.putText(
                        frame,
                        text,
                        (int(x_max) - 100, int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)
                    )
                    if int(elapsed_time) >= 10 and not notifications[person_id]: 
                        # api_post = api_post_template
                        # api_post["event_date"] = datetime.datetime.now().isoformat()
                        # resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                        # _, buffer = cv2.imencode('.jpg', resized_frame)
                        # api_post["image"] = base64.b64encode(buffer).decode('utf-8')
                        print("Notification triggered")
                        notifications[person_id] = True
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video = 'cctv/dropoff_loitering.mp4'
    results = person_loitering(video)