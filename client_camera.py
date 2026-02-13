import cv2
import requests
import numpy as np
import threading

url = "http://127.0.0.1:5000/process"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

processed_img = None
frame_count = 0
lock = threading.Lock()


def send_frame(frame):
    global processed_img

    try:
        _, img_encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        )

        response = requests.post(
            url,
            files={"image": img_encoded.tobytes()},
            timeout=1
        )

        if response.status_code == 200:
            img = cv2.imdecode(
                np.frombuffer(response.content, np.uint8),
                cv2.IMREAD_GRAYSCALE
            )
            with lock:
                processed_img = img

    except:
        pass


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize to reduce load
    frame = cv2.resize(frame, (640, 480))

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Frame drop → every 5th frame
    if frame_count % 5 == 0:
        threading.Thread(target=send_frame, args=(frame.copy(),)).start()

    cv2.imshow("Camera + Faces", frame)

    with lock:
        if processed_img is not None:
            cv2.imshow("Processed Edges", processed_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
