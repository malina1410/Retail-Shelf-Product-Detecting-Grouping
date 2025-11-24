from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import uuid
import os
import sys
import torch.serialization

# Add detect.py location to path
sys.path.append('.')
from detect import detect_from_image

# Initialize YOLOv5 model once globally
torch.serialization.add_safe_globals(['numpy.core.multiarray._reconstruct'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = 'last_yolov5s_results.pt'
model_data = torch.load(weights, map_location=device, weights_only=False)
model = model_data['model'].float()
model.to(device).eval()
half = device.type != 'cpu'

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_np is None:
        return jsonify({"error": "Uploaded file could not be decoded as an image"}), 400

    os.makedirs('static', exist_ok=True)

    # Save original image (used later for grouping)
    original_filename = f"static/original_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(original_filename, img_np)

    # Run detection – returns a saved file path + detection metadata
    detected_img_path, detections = detect_from_image(
        img_np.copy(), model=model, device=device, half=half
    )

    print(f"[DEBUG] Type of detected_img_path: {type(detected_img_path)}")
    print(f"[DEBUG] Sample detection: {detections[:2]}")

    if not os.path.exists(detected_img_path):
        return jsonify({"error": "Detection failed – image file not saved"}), 500

    return jsonify({
        "detections": detections,
        "image_path": os.path.abspath(original_filename),        # for grouping
        "detected_image_path": os.path.abspath(detected_img_path)  # bounding boxes drawn
    })

if __name__ == '__main__':
    app.run(port=5001, debug=True)
