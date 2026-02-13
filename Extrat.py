import cv2
import os

video_path = "Cola.mp4"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % 5 == 0:  # her 5 frame'de 1 al
        cv2.imwrite(f"{output_folder}/frame_{count}.jpg", frame)

    count += 1

cap.release()
print("Frames extracted.")
