from flask import Flask, request, jsonify, render_template
import requests
import uuid
import os

# Import the visualize function
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../visualization')))
from visualize import visualize_groups

app = Flask(__name__)
STATIC_DIR = os.path.join(os.getcwd(), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Step 1: Send image to detection microservice
        image_file = request.files['image']
        files = {'image': image_file}
        detection_response = requests.post("http://localhost:5001/detect", files=files)

        if detection_response.status_code != 200:
            return jsonify({"status": "error", "message": "Detection microservice failed"}), 500

        detection_json = detection_response.json()
        detections = detection_json["detections"]
        image_path = detection_json["image_path"]  # Clean image (not annotated)

        # Step 2: Send detections + clean image path to grouping microservice
        group_response = requests.post(
            "http://localhost:5002/group",
            json={
                "detections": detections,
                "image_path": image_path
            }
        )

        if group_response.status_code != 200:
            return jsonify({"status": "error", "message": "Grouping microservice failed"}), 500

        groups = group_response.json()["groups"]

        # Step 3: Visualize grouping results
        grouped_output_path = os.path.join(STATIC_DIR, f"grouped_{uuid.uuid4().hex}.jpg")
        visualize_groups(image_path, groups, grouped_output_path)

        return jsonify({
            "status": "success",
            "detections": detections,
            "groups": groups,
            "image_path": image_path,
            "grouped_image_path": grouped_output_path
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        image_file = request.files['image']
        response = requests.post("http://localhost:5000/process", files={'image': image_file})

        try:
            result = response.json()
        except Exception:
            result = {"status": "error", "message": "Invalid response from server."}
            return render_template("upload.html", result=result)

        if result["status"] == "success":
            grouped_filename = visualize_groups(
                result['image_path'],  # clean image
                result['groups']       # grouped box data
            )
            result["grouped_image_url"] = f"/static/{grouped_filename}"

        return render_template("upload.html", result=result)

    return render_template("upload.html")



if __name__ == '__main__':
    app.run(port=5000, debug=True)
