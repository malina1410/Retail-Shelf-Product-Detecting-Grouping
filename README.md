# 🚀 Retail Shelf Product Detection & Grouping

A complete **AI-powered microservice system** for detecting, grouping, and visualizing retail shelf products.

This project demonstrates an end-to-end workflow combining **object detection (YOLOv5)**, **feature-based grouping (CLIP + DBSCAN)**, and **visual analytics**, deployed using clean, isolated microservices.

---

## 💡 Project Overview

Modern retail analytics requires identifying products on shelves and understanding visual similarity between them.

This project implements a **three-stage microservice pipeline**:

1.  **Detection Service (YOLOv5)**
    * Detects all products on a retail shelf using a **YOLOv5** model trained on **FMCG** datasets.
    * Outputs bounding boxes in a structured **JSON** format.
2.  **Grouping Service (CLIP + DBSCAN)**
    * Extracts image crops from detections, converts them to **CLIP (ViT-B/32) embeddings**, and uses **DBSCAN** clustering to group visually similar products.
3.  **Visualization Service**
    * Draws **color-coded bounding boxes** for each cluster and returns the final annotated image.

Each microservice runs in its own **virtual environment**, preventing dependency conflicts between YOLOv5, CLIP, and clustering libraries.

---

## 🏗️ Architecture

The pipeline is orchestrated by a Flask UI, chaining the microservices sequentially:

```rust
Flask UI ---> Detection ---> Grouping
             |
             |
             V
        Visualization


✨ Features

    ✔️ Accurate FMCG product detection (YOLOv5)

    ✔️ Unsupervised grouping using CLIP + DBSCAN

    ✔️ Microservice-based architecture

    ✔️ Clean JSON-based communication

    ✔️ Color-coded visualization output

    ✔️ Separate virtual environments for each service

    ✔️ Easy-to-use Flask upload interface


📁 Project Structure

INFLECT_A1/
├── detection/         # YOLOv5 detection microservice
├── grouping/          # CLIP embedding + DBSCAN clustering
├── visualization/     # Draw colored bounding boxes and return final image
├── flask/             # UI + end-to-end pipeline orchestrator
├── docker-compose.yml
└── README.md


⚙️ Setup & Installation

1. Clone the Repository
git clone [https://github.com/yourusername/Retail-Shelf-Detection-Grouping.git](https://github.com/yourusername/Retail-Shelf-Detection-Grouping.git)
cd Retail-Shelf-Detection-Grouping

2. Set Up Individual Microservices
Each microservice has its own venv and requirements.txt.

Detection Service
cd detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# python detect.py (Example Run)

Grouping Service
cd grouping
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# python group.py (Example Run)

Visualization Service
cd visualization
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# python visualize.py (Example Run)

Flask UI
cd flask
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

▶️ Run the Complete Pipeline

Start the services in this order. Each will run on a dedicated port:
1. python detection/detect.py      (port 5001)
2. python grouping/group.py        (port 5002)
3. python visualization/visualize.py (port 5003)
4. python flask/app.py             (port 5000)

Then visit the Flask UI: http://localhost:5000/upload

Response:
{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.91
    }
  ]
}

Grouping Service – /group (POST)
Accepts detections and returns grouped clusters.
Request:
{
  "detections": [
    {
      "bbox": [..., ...],
      "crop_path": "path/to/crop.jpg"
    }
  ]
}

Response:
{
  "groups": [
    {
      "group_id": 0,
      "items": [0, 2, 4]
    }
  ]
}


Visualization Service – /visualize (POST)
<img width="261" height="680" alt="image" src="https://github.com/user-attachments/assets/e381fc7e-dacc-4fcb-a220-696fa7ec70ae" />

<img width="302" height="793" alt="image" src="https://github.com/user-attachments/assets/1a32e0d0-4182-4346-af3f-a4bf9501da23" />


Draws cluster-based color-coded bounding boxes on the image
