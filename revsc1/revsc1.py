import os
import base64
import cv2
import datetime
import json
from collections import defaultdict
from ultralytics import YOLO
import threading

# Disable GUI warnings
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Load the camera configuration
def load_camera_config(scenario_id):
    with open("/app/cameraConfig.json", "r") as f:
        config = json.load(f)
    return [cam for cam in config["cameras"] if scenario_id in cam["scenario_ids"]]

# Load the threshold configuration
def load_threshold_config(scenario_id):
    with open("/app/thresholdConfig.json", "r") as f:
        config = json.load(f)
    return config.get(scenario_id, {})

# Function to process a single camera stream
def process_camera(camera, model, maximum_allowed_time):
    rtsp_url = camera["rtsp_url"]
    camera_id = camera["camera_id"]
    building = camera["building"]

    print(f"Starting camera: {camera_id} at URL: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"Failed to connect to camera: {camera_id}")
        return

    start_times = defaultdict(lambda: None)
    triggers = defaultdict(lambda: False)
    notifications = defaultdict(lambda: False)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 30  # Default FPS if unable to fetch

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Stream ended or failed for camera: {camera_id}")
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = current_frame // fps

        if current_frame % 3 == 0:  # Process every third frame
            results = model.track(frame, persist=True, verbose=False)
            detections = results[0].boxes.data
            for detection in detections:
                if len(detection) == 7:
                    x_min, y_min, x_max, y_max, id, confidence, class_id = detection
                else:
                    x_min, y_min, x_max, y_max, confidence, class_id = detection
                    id = None

                if confidence > 0.8 and int(class_id) == 2:  # Class 2: Cars
                    car_id = int(id) if id else -1
                    if not triggers[car_id]:
                        triggers[car_id] = True
                        start_times[car_id] = current_time

                    elapsed_time = current_time - start_times[car_id]

                    if int(elapsed_time) >= maximum_allowed_time and not notifications[car_id]:
                        # Draw bounding boxes and labels on the frame
                        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                        cv2.putText(
                            frame,
                            f"Car {car_id}",
                            (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                        # Prepare API payload
                        api_post = {
                            "building": building,
                            "camera_number": camera_id,
                            "event_date_and_time": datetime.datetime.now().isoformat(),
                            "event_type": "DOP",
                            "image": None,
                        }

                        # Convert the processed frame (with bounding boxes) to base64
                        _, buffer = cv2.imencode(".jpg", frame)
                        api_post["image"] = base64.b64encode(buffer).decode("utf-8")

                        print(f"API Payload: {api_post}")
                        notifications[car_id] = True

    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    scenario_id = "DOP"  # Update for this scenario
    cameras = load_camera_config(scenario_id)

    # Load thresholds for this scenario
    threshold_config = load_threshold_config(scenario_id)
    maximum_allowed_time = threshold_config.get("maximum_allowed_time", 100)  # Default to 100 seconds

    model = YOLO("./weights/yolov9t.pt")

    threads = []
    for camera in cameras:
        thread = threading.Thread(target=process_camera, args=(camera, model, maximum_allowed_time))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()