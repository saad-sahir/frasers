from ultralytics import YOLO
import cv2 as cv
import json
import datetime
import base64
import requests
from collections import defaultdict

# Load the camera configuration
def load_camera_config(scenario_id):
    with open("/app/cameraConfig.json", "r") as f:  # Adjust path for Docker
        config = json.load(f)
    return [cam for cam in config["cameras"] if scenario_id in cam["scenario_ids"]]

# Load the threshold configuration
def load_threshold_config(scenario_id):
    with open("/app/thresholdConfig.json", "r") as f:  # Adjust path for Docker
        config = json.load(f)
    return config.get(scenario_id, {})

# Function to process carpark video
def carpark(camera, maximum_allowed_time):
    rtsp_url = camera["rtsp_url"]
    camera_id = camera["camera_id"]
    building = camera["building"]
    api_url = "https://dimitar-playground-dev.bettywebblocks.com/va_demo"  # Replace with actual endpoint

    model = YOLO('./weights/yolov9t.pt')
    cap = cv.VideoCapture(rtsp_url)
    notifications = defaultdict(lambda: False)
    fps = cap.get(cv.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        current_time = current_frame / fps

        if current_frame % 3 == 0:
            h, w = frame.shape[:2]
            results = model.track(frame, persist=True, verbose=False)
            detections = results[0].boxes.data
            cars = []

            for detection in detections:
                if len(detection) == 7:
                    x_min, y_min, x_max, y_max, id, confidence, class_id = detection
                elif len(detection) == 6:
                    x_min, y_min, x_max, y_max, confidence, class_id = detection
                else:
                    continue

                if confidence >= 0.7 and class_id == 2:
                    cars.append(detection)
                    car_pos = (int((x_max + x_min) // 2), int(y_max))
                    x, y = car_pos

                    # Check if the car is blocking the road
                    if y > int(h * 0.55) and x < int(w * 0.6):
                        cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 4)
                    else:
                        cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 4)

            # Display car count on the frame
            cv.putText(
                frame,
                f"Number of cars: {len(cars)}",
                (25, 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Trigger if too many cars block the car park
            if len(cars) > 8:
                if not notifications["jam"]:
                    notifications["jam"] = True
                    start_time = current_time

                elapsed_time = current_time - start_time
                if elapsed_time >= maximum_allowed_time and not notifications["alert_sent"]:
                    # Prepare API payload
                    api_post = {
                        "building": building,
                        "camera_number": camera_id,
                        "event_date_and_time": datetime.datetime.now().isoformat(),
                        "event_type": "CIJ",  # Carpark Internal Jam
                        "image": None,
                    }

                    # Convert frame to base64
                    _, buffer = cv.imencode(".jpg", frame)
                    api_post["image"] = base64.b64encode(buffer).decode("utf-8")

                    # Send the payload to the API
                    try:
                        response = requests.post(api_url, json=api_post)
                        response.raise_for_status()
                        print(f"API call successful: {response.text}")
                    except requests.exceptions.RequestException as e:
                        print(f"API call failed: {e}")

                    notifications["alert_sent"] = True

        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

# Main function
if __name__ == "__main__":
    scenario_id = "CIJ"  # Scenario ID for carpark internal jam
    threshold_config = load_threshold_config(scenario_id)
    maximum_allowed_time = threshold_config.get("maximum_allowed_time", 60)  # Default to 60 seconds

    cameras = load_camera_config(scenario_id)

    for camera in cameras:
        carpark(camera, maximum_allowed_time)