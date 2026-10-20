import json
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]
LABEL_DIRS = [
    BASE_DIR / "raw_xbd" / "train" / "labels",
    BASE_DIR / "raw_xbd" / "test" / "labels",
    BASE_DIR / "raw_xbd" / "tier3" / "labels",
]
MODEL_PATH = BASE_DIR / "models" / "disaster_damage_mobilenetv2.keras"

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["No_Damage", "Minor_Damage", "Major_Damage", "Destroyed"]
CONTEXT_SCALE = 3.0
BATCH_SIZE = 32
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def find_matching_json(image_name):
    stem = Path(image_name).stem
    if not stem.endswith("_post_disaster"):
        raise ValueError(
            "Please upload an original xBD post-disaster image whose filename ends with '_post_disaster'."
        )

    # Match the frontend's deterministic search order: train, test, then tier3.
    # Return the first exact filename match; never select a different stem.
    target_name = f"{stem}.json"
    for label_dir in LABEL_DIRS:
        if not label_dir.is_dir():
            continue
        for candidate in label_dir.rglob(target_name):
            return candidate
    raise FileNotFoundError(f"No matching xBD JSON found for '{image_name}'.")


def parse_wkt_polygon(wkt):
    if not isinstance(wkt, str):
        return []
    match = re.search(r"POLYGON\s*\(\((.*?)\)\)", wkt, flags=re.IGNORECASE)
    if not match:
        return []

    points = []
    for pair in match.group(1).split(","):
        values = pair.strip().split()
        if len(values) >= 2:
            try:
                points.append((float(values[0]), float(values[1])))
            except ValueError:
                pass
    return points


def extract_buildings(data):
    buildings = []
    for feature in data.get("features", {}).get("xy", []):
        props = feature.get("properties", {})
        if props.get("feature_type") != "building":
            continue

        polygon = parse_wkt_polygon(feature.get("wkt", ""))
        if len(polygon) < 3:
            continue

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]

        buildings.append({
            "uid": props.get("uid", ""),
            "polygon": polygon,
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
            "xbd_subtype": props.get("subtype", ""),
        })
    return buildings


def normalize_buildings(buildings, image_size):
    width, height = image_size
    if not buildings:
        return buildings

    max_x = max(max(p[0] for p in b["polygon"]) for b in buildings)
    max_y = max(max(p[1] for p in b["polygon"]) for b in buildings)

    sx = width / max_x if max_x > width and max_x > 0 else 1.0
    sy = height / max_y if max_y > height and max_y > 0 else 1.0

    if sx == 1.0 and sy == 1.0:
        return buildings

    normalized = []
    for b in buildings:
        polygon = [(x * sx, y * sy) for x, y in b["polygon"]]
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        item = dict(b)
        item["polygon"] = polygon
        item["bbox"] = (min(xs), min(ys), max(xs), max(ys))
        normalized.append(item)
    return normalized


def context_crop(image, bbox):
    width, height = image.size
    x1, y1, x2, y2 = bbox

    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    crop_w = bw * CONTEXT_SCALE
    crop_h = bh * CONTEXT_SCALE

    left = max(0, int(round(cx - crop_w / 2)))
    top = max(0, int(round(cy - crop_h / 2)))
    right = min(width, int(round(cx + crop_w / 2)))
    bottom = min(height, int(round(cy + crop_h / 2)))

    if right <= left:
        right = min(width, left + 2)
    if bottom <= top:
        bottom = min(height, top + 2)

    return image.crop((left, top, right, bottom))


def predict_batch(images):
    model = get_model()
    arrays = [
        np.asarray(im.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32)
        for im in images
    ]
    batch = np.stack(arrays, axis=0)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    predictions = model.predict(batch, verbose=0, batch_size=BATCH_SIZE)

    output = []
    for prediction in predictions:
        index = int(np.argmax(prediction))
        output.append({
            "damage": CLASS_NAMES[index],
            "confidence": float(prediction[index] * 100),
            "probabilities": {
                CLASS_NAMES[i]: float(prediction[i] * 100)
                for i in range(len(CLASS_NAMES))
            },
        })
    return output


def get_priority(damage):
    return {
        "Destroyed": ("CRITICAL", 4),
        "Major_Damage": ("HIGH", 3),
        "Minor_Damage": ("MODERATE", 2),
        "No_Damage": ("LOW", 1),
    }.get(damage, ("LOW", 1))


def get_severity(damage):
    return {
        "Destroyed": "Severe",
        "Major_Damage": "High",
        "Minor_Damage": "Moderate",
        "No_Damage": "None",
    }.get(damage, "Unknown")


def analyze_xbd_image(image_path, original_filename):
    json_path = find_matching_json(original_filename)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    image = Image.open(image_path).convert("RGB")
    buildings = normalize_buildings(extract_buildings(data), image.size)

    if not buildings:
        raise ValueError("No building polygons found in the matching xBD JSON.")

    crops = [context_crop(image, b["bbox"]) for b in buildings]
    predictions = predict_batch(crops)

    results = []
    for i, (building, pred) in enumerate(zip(buildings, predictions), start=1):
        damage = pred["damage"]
        priority, rank = get_priority(damage)
        results.append({
            "building_id": f"B{i:03d}",
            "damage": damage,
            "confidence": round(pred["confidence"], 2),
            "probabilities": {k: round(v, 2) for k, v in pred["probabilities"].items()},
            "severity": get_severity(damage),
            "priority": priority,
            "priority_rank": rank,
            "polygon": building["polygon"],
            "bbox": building["bbox"],
            "xbd_uid": building["uid"],
            "xbd_subtype": building["xbd_subtype"],
        })

    counts = {name: 0 for name in CLASS_NAMES}
    priorities = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}

    for item in results:
        counts[item["damage"]] += 1
        priorities[item["priority"]] += 1

    affected = counts["Minor_Damage"] + counts["Major_Damage"] + counts["Destroyed"]
    impact = affected / len(results) * 100 if results else 0

    return {
        "status": "success",
        "image": original_filename,
        "matched_json": str(json_path),
        "building_count": len(results),
        "summary": {
            "damage_counts": counts,
            "priority_counts": priorities,
            "impact_ratio": round(impact, 2),
        },
        "buildings": results,
    }
