from ultralytics import YOLO
import datetime
import cv2 as cv
import base64
import json
import requests

# Load the threshold configuration
def load_threshold_config(scenario_id):
    with open("/app/thresholdConfig.json", "r") as f:  # Adjust path for Docker
        config = json.load(f)
    return config.get(scenario_id, {})

# Load the camera configuration
def load_camera_config(scenario_id):
    with open("/app/cameraConfig.json", "r") as f:  # Adjust path for Docker
        config = json.load(f)
    return [cam for cam in config["cameras"] if scenario_id in cam["scenario_ids"]]

def nightwatch(camera, current_hour_cutoff):
    rtsp_url = camera["rtsp_url"]
    camera_id = camera["camera_id"]
    building = camera["building"]
    api_url = "https://dimitar-playground-dev.bettywebblocks.com/va_demo"  # Replace with the actual API endpoint

    model = YOLO('./weights/yolov9t.pt')
    cap = cv.VideoCapture(rtsp_url)
    notification = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        current_time = datetime.datetime.now()
        current_hour = current_time.hour
        time_str = current_time.strftime("%H:%M")

        if current_frame % 3 == 0:  # Process every third frame
            results = model.track(frame, persist=True, verbose=False)
            detections = results[0].boxes.data

            # Display current time on the frame
            cv.putText(
                frame,
                f"Current time: {time_str}",
                (50, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            for detection in detections:
                if len(detection) == 7:
                    x_min, y_min, x_max, y_max, id, confidence, class_id = detection
                else:
                    x_min, y_min, x_max, y_max, confidence, class_id = detection
                    id = None

                if confidence > 0.55 and int(class_id) == 0:  # Class 0 for people
                    person_id = int(id) if id else None
                    color = (255, 0, 0) if current_hour < current_hour_cutoff else (0, 0, 255)

                    cv.putText(
                        frame,
                        f"({person_id})" if person_id else "(Unknown)",
                        (int(x_min), int(y_min) - 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    cv.rectangle(
                        frame,
                        (int(x_min), int(y_min)),
                        (int(x_max), int(y_max)),
                        color,
                        4,
                    )

                    if current_hour >= current_hour_cutoff:  # Intruder detection
                        cv.putText(
                            frame,
                            "Intruder Detected",
                            (int(x_max) - 100, int(y_min) - 10),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
                        if not notification:
                            # Prepare API payload
                            api_post = {
                                "building": building,
                                "camera_number": camera_id,
                                "event_date_and_time": current_time.isoformat(),
                                "event_type": "AOH",  # After-hours detection
                                "image": None,
                            }

                            # Convert frame to base64
                            _, buffer = cv.imencode('.jpg', frame)
                            api_post["image"] = base64.b64encode(buffer).decode('utf-8')

                            # Send the payload to the API
                            try:
                                response = requests.post(api_url, json=api_post)
                                response.raise_for_status()
                                print(f"API call successful: {response.text}")
                            except requests.exceptions.RequestException as e:
                                print(f"API call failed: {e}")

                            notification = True

        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

# Main function
if __name__ == "__main__":
    scenario_id = "AOH"  # Scenario ID for after-hours detection
    threshold_config = load_threshold_config(scenario_id)
    current_hour_cutoff = threshold_config.get("current_hour", 18)  # Default to 18:00 (6 PM)

    cameras = load_camera_config(scenario_id)

    for camera in cameras:
        nightwatch(camera, current_hour_cutoff)