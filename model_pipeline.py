import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------- Load model once (do this at module/API startup, not per-request) ----------
MODEL_PATH = "runs/detect/yolo11n_gpu_run1/weights/best.pt"   # MODEL path
model = YOLO(MODEL_PATH)

# ---------- Severity + explanation config ----------
TYPE_WEIGHT = {"pothole": 1.0, "unpaved_road": 0.7, "speed_breaker": 0.3}

DAMAGE_DESCRIPTIONS = {
    "pothole": "a depression in the road surface that can damage vehicles and pose safety risks",
    "unpaved_road": "a stretch of road lacking proper paving, causing dust, instability, and wear",
    "speed_breaker": "a raised road hazard that may be unmarked or poorly visible, risking accidents",
}


# ---------- Wrapper needed for Grad-CAM to work with YOLO's tuple output ----------
class YOLOWrapper(torch.nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output


def compute_severity(damage_class, confidence, bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
    raw_score = (0.5 * min(area_ratio * 5, 1.0)
                 + 0.2 * confidence
                 + 0.3 * TYPE_WEIGHT.get(damage_class, 0.5))

    if raw_score < 0.35:
        severity = "Low"
    elif raw_score < 0.55:
        severity = "Medium"
    elif raw_score < 0.75:
        severity = "High"
    else:
        severity = "Critical"
    return severity, round(raw_score, 3), round(area_ratio, 4)


def generate_explanation(damage_class, confidence, area_ratio, severity):
    desc = DAMAGE_DESCRIPTIONS.get(damage_class, "a road surface irregularity")
    return (f"Detected {damage_class.replace('_', ' ')} with {confidence*100:.1f}% confidence, "
            f"covering approximately {area_ratio*100:.1f}% of the visible area. "
            f"This is {desc}. Classified as {severity} severity based on size, "
            f"damage type, and model confidence.")


def generate_heatmap(img, model):
    """img: already-loaded BGR numpy array (from cv2.imread)"""
    wrapped_model = YOLOWrapper(model.model)
    wrapped_model.eval()

    target_layers = [model.model.model[-2]]
    cam = EigenCAM(wrapped_model, target_layers)

    img_resized = cv2.resize(img, (640, 640))
    rgb_img = img_resized[:, :, ::-1] / 255.0

    device = next(model.model.parameters()).device
    input_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    grayscale_cam = cam(input_tensor)[0]
    cam_image = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)
    return cam_image


def detect_damage(image_path, conf_threshold=0.2, output_dir="output", imgsz=640):
    """
    Full pipeline: detect -> save boxed image -> generate heatmap -> severity + explanation.
    Returns a structured dict; also saves 2 image files to output_dir.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"OpenCV could not read image (corrupt or unsupported format): {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # ---------- 1. Run detection ----------
    results = model(image_path, conf=conf_threshold, imgsz=imgsz)[0]
    img_h, img_w = results.orig_shape

    detections = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        confidence = round(float(box.conf), 4)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

        severity, severity_score, area_ratio = compute_severity(
            cls_name, confidence, (x1, y1, x2, y2), img_w, img_h
        )
        explanation = generate_explanation(cls_name, confidence, area_ratio, severity)

        detections.append({
            "class": cls_name,
            "confidence": confidence,
            "bbox": bbox,
            "bbox_normalized": [x1/img_w, y1/img_h, x2/img_w, y2/img_h],
            "area_ratio": area_ratio,
            "severity": severity,
            "severity_score": severity_score,
            "explanation": explanation
        })

    # ---------- 2. Save boxed image ----------
    boxed_path = os.path.join(output_dir, f"{base_name}_boxes.jpg")
    results.save(filename=boxed_path)

    # ---------- 3. Save heatmap image ----------
    heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.jpg")
    try:
        heatmap_img = generate_heatmap(img, model)
        cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
        heatmap_success = True
    except Exception as e:
        print(f"Heatmap generation failed for {image_path}: {e}")
        heatmap_path = None
        heatmap_success = False

    # ---------- 4. Return full structured result ----------
    return {
        "image_path": image_path,
        "image_width": img_w,
        "image_height": img_h,
        "damage_found": len(detections) > 0,
        "num_detections": len(detections),
        "detections": detections,
        "boxed_image_path": boxed_path,
        "heatmap_image_path": heatmap_path,
        "heatmap_generated": heatmap_success
    }





demo_images =  ['UnPavedRoad__44.jpg'] # ADD IMAGES NAMES IN THIS LIST
for img in demo_images:
    path = f'data/images/test/{img}' # ADD PATH WHERE IMAGES ARE SAVED
    try:
        result = detect_damage(path, output_dir="presentation_assets")
        print(f"✓ {img}: {result['num_detections']} detections")
        print(f"\nImage: {result['image_path']}")
        print(f"Damage found: {result['damage_found']} ({result['num_detections']} detection(s))")
        print(f"Boxed image saved to: {result['boxed_image_path']}")
        print(f"Heatmap saved to: {result['heatmap_image_path']}")
        
        for i, det in enumerate(result["detections"], 1):
            print(f"\n  Detection {i}:")
            print(f"    Class: {det['class']}")
            print(f"    Confidence: {det['confidence']*100:.1f}%")
            print(f"    Severity: {det['severity']} (score: {det['severity_score']})")
            print(f"    Explanation: {det['explanation']}")
    except Exception as e:
        print(f"✗ {img} failed: {e}")
    