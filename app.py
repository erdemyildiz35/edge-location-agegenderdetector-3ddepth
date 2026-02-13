from flask import Flask, request, send_file
import cv2
import numpy as np
import io

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_image():
    file = request.files["image"]

    image_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)

    edges = cv2.Canny(img, 100, 200)

    _, buffer = cv2.imencode(".png", edges)

    return send_file(
        io.BytesIO(buffer),
        mimetype="image/png"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
