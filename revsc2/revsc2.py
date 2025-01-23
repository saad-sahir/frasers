import base64
import cv2
import datetime
import json
import requests
from collections import defaultdict
from ultralytics import YOLO

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

# Function to process video for person/package detection
def person_or_package(camera, maximum_allowed_time):
    rtsp_url = camera["rtsp_url"]
    camera_id = camera["camera_id"]
    building = camera["building"]
    api_url = "https://dimitar-playground-dev.bettywebblocks.com/va_demo"  # Replace with the actual API endpoint

    model = YOLO("./weights/yolov9t.pt")
    package_model = YOLO("./weights/package.pt")
    cap = cv2.VideoCapture(rtsp_url)
    
    start_times = defaultdict(lambda: None)
    triggers = defaultdict(lambda: False)
    notifications = defaultdict(lambda: False)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = current_frame / fps

        if current_frame % 1 == 0:  # Process every frame
            results = model.track(frame, persist=True, verbose=False)
            results2 = package_model.track(frame, persist=True, verbose=False)
            detections = results[0].boxes.data
            packages = results2[0].boxes.data

            # Handle person detections
            for detection in detections:
                if len(detection) == 7:
                    x_min, y_min, x_max, y_max, id, confidence, class_id = detection
                else:
                    x_min, y_min, x_max, y_max, confidence, class_id = detection
                    id = None

                if id and int(class_id) == 0:  # Class 0 for people
                    person_id = int(id)
                    if not triggers[person_id]:
                        triggers[person_id] = True
                        start_times[person_id] = current_time

                    elapsed_time = current_time - start_times[person_id]

                    if int(elapsed_time) >= maximum_allowed_time and not notifications[person_id]:
                        # Prepare API payload
                        api_post = {
                            "building": building,
                            "camera_number": camera_id,
                            "event_date_and_time": datetime.datetime.now().isoformat(),
                            "event_type": "PPS",
                            "image": None,
                        }

                        # Convert the frame to base64
                        _, buffer = cv2.imencode(".jpg", frame)
                        api_post["image"] = base64.b64encode(buffer).decode("utf-8")
                        
                        # Send the payload to the API
                        try:
                            response = requests.post(api_url, json=api_post)
                            response.raise_for_status()
                            print(f"API call successful: {response.text}")
                        except requests.exceptions.RequestException as e:
                            print(f"API call failed: {e}")

                        notifications[person_id] = True

    cap.release()

# Main function
if __name__ == "__main__":
    scenario_id = "PPS"  # Scenario ID for this script
    cameras = load_camera_config(scenario_id)
    
    # Load the threshold for this scenario
    threshold_config = load_threshold_config(scenario_id)
    maximum_allowed_time = threshold_config.get("maximum_allowed_time", 10)  # Default to 10 seconds

    for camera in cameras:
        person_or_package(camera, maximum_allowed_time)