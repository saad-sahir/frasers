import base64
import cv2
import datetime
from collections import defaultdict
from ultralytics import YOLO 

def dropoff_car(video):
    model = YOLO("weights/yolov9t.pt")
    cap = cv2.VideoCapture(video)
    
    start_times = defaultdict(lambda: None)
    triggers = defaultdict(lambda: False)
    notifications = defaultdict(lambda: False)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    r = {}
    cars = []
    
    maximum_allowed_time = 10 # set to 10 seconds for testing

    api_post_template = {
        "building": "AP",
        "camera_number": "DO01",
        "event_date": None,
        "event_type": "SC02",
        "image": None,
    }

    api_url = "https://example.com/api_endpoint"  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = current_frame // fps
        if current_frame % 3 == 0:
            cv2.putText(frame, f"Time: {current_time}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            results = model.track(frame, persist=True, verbose=False)
            detections = results[0].boxes.data
            track_ids = [int(d[4]) for d in detections if d[5] > 0.8 and d[-1] == 2]
            for detection in detections:
                if len(detection) == 7:
                    x_min, y_min, x_max, y_max, id, confidence, class_id = detection
                else:
                    x_min, y_min, x_max, y_max, confidence, class_id = detection
                    id = None
                if confidence > 0.8 and int(class_id) == 2:
                    cars.append(detection)
                    car_id = int(id)
                    cv2.putText(frame, f"({int(id)})",(int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 4)
                    if not triggers[car_id]:
                        triggers[car_id] = True
                        start_times[car_id] = current_time
                    elapsed_time = current_time - start_times[car_id]
                    cv2.putText(frame, f"Duration: {int(elapsed_time)}s", (int(x_max)-100, int(y_min)-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                    if int(elapsed_time) >= maximum_allowed_time and not notifications[car_id]:
                        api_post = api_post_template
                        api_post["event_date"] = datetime.datetime.now().isoformat()
                        resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                        print(api_post_template)
                        # Convert the resized frame to base64
                        _, buffer = cv2.imencode('.jpg', resized_frame)
                        api_post["image"] = base64.b64encode(buffer).decode('utf-8')
                        # try:
                        #     response = requests.post(api_url, json=api_post)
                        #     response.raise_for_status()
                        #     print(f"API call successful: {response.text}")
                        # except requests.exceptions.RequestException as e:
                        #     print(f"API call failed: {e}")
                        notifications[car_id] = True
            cv2.imshow('frame', frame)
            r[current_frame] = cars
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    return r

if __name__ == '__main__':
    # video = sys.argv[1]
    video = 'cctv/dropoff_loitering.mp4'
    results = dropoff_car(video)