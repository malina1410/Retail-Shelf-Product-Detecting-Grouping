from flask import Flask, request, jsonify
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import torch
import cv2
import random
import json
import open_clip
from PIL import Image

app = Flask(__name__)

# Load CLIP model (ViT-B/32)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_model = clip_model.to(device).eval()

@app.route('/group', methods=['POST'])
def group_products():
    data = request.json
    return process_grouping(data)


def process_grouping(data):
    detections = data.get("detections", [])
    image_path = data.get("image_path", "")

    if not detections or not image_path:
        return {"error": "Missing data"}, 400

    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Image not found: {image_path}"}, 404

    embeddings = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        crop = image[y1:y2, x1:x2]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            embeddings.append(np.zeros(512))
            continue

        try:
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = clip_model.encode_image(img_tensor).squeeze().cpu().numpy()
            embeddings.append(emb)
        except Exception as e:
            print(f"[WARN] Skipping crop due to error: {e}")
            embeddings.append(np.zeros(512))

    # Normalize and compute cosine distance
    embeddings = normalize(embeddings)
    dist_matrix = pairwise_distances(embeddings, metric='cosine')
    clustering = DBSCAN(eps=0.25, min_samples=2, metric='precomputed').fit(dist_matrix)

    labels = clustering.labels_
    grouped = {}
    for idx, group_id in enumerate(labels):
        key = f"group_{group_id}"
        if key not in grouped:
            grouped[key] = {
                "color": [random.randint(0, 255) for _ in range(3)],
                "items": []
            }
        grouped[key]["items"].append(detections[idx])

    result = {
        "groups": [
            {"group_id": k, "items": v["items"], "color": v["color"]}
            for k, v in grouped.items()
        ]
    }

    return result


def manual_group(detection_json_path, output_json_path="group_output.json"):
    with open(detection_json_path) as f:
        data = json.load(f)

    result = process_grouping(data)
    if isinstance(result, tuple):  # error case
        error, code = result
        print(f"❌ Error {code}: {error}")
        return

    with open(output_json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Grouping complete. Saved to {output_json_path}")


if __name__ == '__main__':
    # To run as a server
    app.run(port=5002, debug=True)
