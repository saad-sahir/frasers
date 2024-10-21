from ultralytics import YOLO
import datetime
import cv2 as cv
import base64

def nightwatch(video):
    model = YOLO('weights/yolov9t.pt')
    cap = cv.VideoCapture(video)
    current_hour, time_str = int(datetime.datetime.now().hour), datetime.datetime.now().strftime("%H:%m")
    # current_hour, time_str = 19, "19:04"
    notification = False

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
        current_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        if current_frame % 3 == 0:
            results = model.track(frame, persist=True, verbose=False)
            detections = results[0].boxes.data
            cv.putText(
                frame,
                f"Current time: {time_str}",
                (50, 50),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            for detection in detections:
                if len(detection) == 7:
                    x_min, y_min, x_max, y_max, id, confidence, class_id = detection
                else:
                    x_min, y_min, x_max, y_max, confidence, class_id = detection
                    id = None
                if confidence > 0.55 and int(class_id) == 0:
                    person_id = int(id)
                    color = (255, 0, 0) if current_hour < 18 else (0, 0, 255) 
                    cv.putText(
                        frame,
                        f"({int(id)})",
                        (int(x_min), int(y_min) - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
                    cv.rectangle(
                        frame,
                        (int(x_min), int(y_min)),
                        (int(x_max), int(y_max)),
                        color, 4
                    )
                    if current_hour > 18: # assuming 6pm is the cutoff for when everyone should leave
                        cv.putText(
                            frame,
                            "intruder",
                            (int(x_max) - 100, int(y_min) - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )
                        if not notification: 
                            api_post = api_post_template
                            api_post["event_date"] = datetime.datetime.now().isoformat()
                            resized_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
                            _, buffer = cv.imencode('.jpg', resized_frame)
                            api_post["image"] = base64.b64encode(buffer).decode('utf-8')
                            print(api_post_template)
                            notification = True
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv.destroyAllWindows()
