from ultralytics import YOLO
import cv2 as cv
import datetime
import base64
from collections import defaultdict

class ActivityMonitor:
    def __init__(self, model_path, max_time=10, confidence_thresh=0.7, roi=None):
        self.model = YOLO(model_path)
        self.max_time = max_time
        self.confidence_thresh = confidence_thresh
        self.roi = roi
    
    def process_video(self, video, object_class, event_type, api_url):
        cap = cv.VideoCapture(video)
        start_times = defaultdict(lambda: None)
        triggers = defaultdict(lambda: False)
        notifications = defaultdict(lambda: False)
        fps = cap.get(cv.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            current_time = current_frame / fps
            if current_frame % 3 == 0:
                results = self.model.track(frame, persist=True, verbose=False)
                detections = results[0].boxes.data
                
                for detection in detections:
                    x_min, y_min, x_max, y_max, *rest = detection
                    confidence, class_id = rest[-2:]
                    obj_id = int(rest[0]) if len(rest) == 3 else None
                    
                    if confidence >= self.confidence_thresh and int(class_id) == object_class:
                        if self.roi:
                            if not (self.roi[0] < (x_min + x_max) / 2 < self.roi[2] and self.roi[1] < y_max < self.roi[3]):
                                continue
                        
                        cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 4)
                        
                        if not triggers[obj_id]:
                            triggers[obj_id] = True
                            start_times[obj_id] = current_time
                        
                        elapsed_time = current_time - start_times[obj_id]
                        cv.putText(frame, f"Duration: {int(elapsed_time)}s", (int(x_max)-100, int(y_min)-50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                        
                        if elapsed_time >= self.max_time and not notifications[obj_id]:
                            self.send_alert(frame, event_type, api_url)
                            notifications[obj_id] = True
                
                cv.imshow('frame', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv.destroyAllWindows()
    
    def send_alert(self, frame, event_type, api_url):
        api_post = {
            "building": "AP",
            "camera_number": "DO01",
            "event_date": datetime.datetime.now().isoformat(),
            "event_type": event_type,
            "image": None,
        }
        
        _, buffer = cv.imencode('.jpg', cv.resize(frame, (0, 0), fx=0.5, fy=0.5))
        api_post["image"] = base64.b64encode(buffer).decode('utf-8')
        # Uncomment to send request
        # requests.post(api_url, json=api_post)
        print(f"Alert Sent: {api_post}")

if __name__ == '__main__':
    monitor = ActivityMonitor("weights/yolov9m.pt", max_time=10, confidence_thresh=0.7, roi=(0, 0, 600, 550))
    monitor.process_video("cctv/carpark.mp4", object_class=2, event_type="SC05", api_url="https://example.com/api_endpoint")
