import argparse
import os
import platform
import random
import shutil
import time
from pathlib import Path
import json

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import uuid
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

opt = None
from utils import google_utils
from utils.datasets import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='last_yolov5s_results.pt', help='model path(s)')
parser.add_argument('--source', type=str, default='inference/images', help='source')
parser.add_argument('--output', type=str, default='inference/output', help='output folder')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.29, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
opt = parser.parse_args(args=[])

def detect_from_image(image_np, model=None, device=None, half=False, save_img=False):
    img0 = image_np.copy()
    img = letterbox(img0, new_shape=opt.img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=opt.augment)[0]
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    det_img = img0.copy()
    detection_list = []

    for i, det in enumerate(pred):
        gn = torch.tensor(det_img.shape)[[1, 0, 1, 0]]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], det_img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, det_img, label=label, color=colors[int(cls)], line_thickness=3)

                x1, y1, x2, y2 = [int(coord.item()) for coord in xyxy]
                detection_list.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class": names[int(cls)]
                })

    image_filename = f"{uuid.uuid4().hex}.jpg"
    output_dir = "static"  # or any other persistent directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, image_filename)
    cv2.imwrite(output_path, det_img)

    print("[DEBUG] det_img type:", type(det_img))
    print("[DEBUG] det_img shape:", det_img.shape if isinstance(det_img, np.ndarray) else "Not an array")

    #  Return full absolute path for grouping
    return os.path.abspath(output_path), detection_list


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    half = device.type != 'cpu'

    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device, weights_only=False)['model'].float()
    model.to(device).eval()
    imgsz = check_img_size(imgsz, s=model.model[-1].stride.max())
    if half:
        model.half()

    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])
        modelc.to(device).eval()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = Path(out) / Path(p).name
            txt_path = Path(out) / Path(p).stem
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            detection_list = []

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, names[int(c)])

                for *xyxy, conf, cls in det:
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        with open(str(txt_path) + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))

                    if save_img or view_img:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    x1, y1, x2, y2 = [int(coord.item()) for coord in xyxy]
                    detection_list.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                        "class": names[int(cls)]
                    })

            print('%sDone. (%.3fs)' % (s, t2 - t1))

            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration

            if save_img:
                cv2.imwrite(str(save_path), im0)

                # ✨ Save detection JSON alongside the image
                json_path = Path(out) / (Path(p).stem + "_detections.json")
                json_data = {
                    "detections": detection_list,
                    "image_path": str(save_path)
                }
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=2)
                print(f"✅ Saved JSON to {json_path}")

    print('Done. (%.3fs)' % (time.time() - t0))
    return model, device, half

def load_model():
    device = torch_utils.select_device(opt.device)
    model = torch.load(opt.weights, map_location=device, weights_only=False)['model'].float()
    model.to(device).eval()
    half = device.type != 'cpu'
    if half:
        model.half()
    return model, device, half


def detect_from_bytes(image_bytes, model, device, half):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    filename = f"{uuid.uuid4().hex}.jpg"
    clean_path = os.path.join(STATIC_DIR, f"clean_{filename}")
    annotated_path = os.path.join(STATIC_DIR, f"annotated_{filename}")

    cv2.imwrite(clean_path, img_np.copy())  # Save clean copy

    # Inference
    det_img, detections = detect_from_image(img_np, model=model, device=device, half=half, save_img=False)
    cv2.imwrite(annotated_path, det_img)

    return {
        "detections": detections,
        "image_path": clean_path,       # For grouping
        "annotated_path": annotated_path  # Optional: for visual display
    }


if __name__ == '__main__':
    print(opt)
    with torch.no_grad():
        model, device, half = detect()
