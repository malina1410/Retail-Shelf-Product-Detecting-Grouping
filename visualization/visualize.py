# visualization/visualize.py
import os
import uuid
import cv2
import json


def visualize_groups(image_path, group_data, output_dir='static'):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or cannot be loaded.")

    for group in group_data:
        color = tuple(group['color'])  # BGR format
        label = group['group_id']
        for item in group['items']:
            x1, y1, x2, y2 = item['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    output_filename = f"grouped_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, image)
    return output_filename
