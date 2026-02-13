import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def calculate_symmetry(landmarks, w, h):
    points = []

    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    # Burun ucu landmark id = 1
    center_x = points[1][0]

    diffs = []

    for (x, y) in points:
        mirrored_x = 2 * center_x - x
        diffs.append(abs(x - mirrored_x))

    avg_diff = np.mean(diffs)

    # Skor normalize (deneysel katsayı)
    score = max(0, 100 - avg_diff * 0.1)
    return round(score, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS
            )

            score = calculate_symmetry(face_landmarks.landmark, w, h)

            cv2.putText(
                frame,
                f"Symmetry Score: {score}%",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Face Symmetry Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
