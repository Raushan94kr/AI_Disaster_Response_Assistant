import os
import io
import re
import hashlib
import json
import csv
import requests
import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as PDFImage, PageBreak
)

# CONFIGURATION

st.set_page_config(
    page_title="AI Disaster Response Assistant",
    page_icon="🚨",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "disaster_damage_mobilenetv2.keras"
)
XBD_LABEL_DIRS = [
    os.path.join(BASE_DIR, "raw_xbd", "train", "labels"),
    os.path.join(BASE_DIR, "raw_xbd", "test", "labels"),
    os.path.join(BASE_DIR, "raw_xbd", "tier3", "labels"),
]
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["No_Damage", "Minor_Damage", "Major_Damage", "Destroyed"]

# Display colors only; model probabilities are not altered.
CLASS_COLORS = {
    "No_Damage": (46, 180, 95),
    "Minor_Damage": (245, 190, 45),
    "Major_Damage": (245, 125, 45),
    "Destroyed": (220, 55, 70),
}

DAMAGE_WEIGHTS = {
    "No_Damage": 0,
    "Minor_Damage": 1,
    "Major_Damage": 2,
    "Destroyed": 3,
}

CONTEXT_SCALE = 3.0
PREDICTION_BATCH_SIZE = 32
FASTAPI_URL = "http://127.0.0.1:8000"


# MODEL

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

# DAMAGE / PRIORITY


def get_severity(damage_class):
    if damage_class == "No_Damage":
        return "LOW", "🟢", "No significant visible damage detected."
    if damage_class == "Minor_Damage":
        return "MODERATE", "🟡", "Minor damage detected. Further inspection is recommended."
    if damage_class == "Major_Damage":
        return "HIGH", "🟠", "Significant damage detected. Priority assessment is recommended."
    return "CRITICAL", "🔴", "Severe damage or destruction detected. Emergency assessment is recommended."


def get_priority(damage_class):
    if damage_class == "Destroyed":
        return "CRITICAL", 4
    if damage_class == "Major_Damage":
        return "HIGH", 3
    if damage_class == "Minor_Damage":
        return "MODERATE", 2
    return "LOW", 1


# MODEL PREDICTION

def predict_damage(image):
    """Predict damage for one building crop."""
    results = predict_damage_batch([image])
    return results[0]


def predict_damage_batch(images):
    """Predict damage for multiple building crops in batches."""
    all_results = []

    for start in range(0, len(images), PREDICTION_BATCH_SIZE):
        batch_images = images[start:start + PREDICTION_BATCH_SIZE]

        arrays = []
        for image in batch_images:
            image = image.convert("RGB").resize(IMAGE_SIZE)
            arr = np.asarray(image, dtype=np.float32)
            arrays.append(arr)

        batch = np.stack(arrays, axis=0)
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)

        predictions_batch = model.predict(batch, verbose=0)

        for predictions in predictions_batch:
            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(predictions[predicted_index] * 100.0)

            probabilities = {
                CLASS_NAMES[i]: float(predictions[i] * 100.0)
                for i in range(len(CLASS_NAMES))
            }

            all_results.append(
                (predicted_class, confidence, probabilities)
            )

    return all_results


# FASTAPI BACKEND INTEGRATION

def analyze_via_fastapi(uploaded_file):
    """Send the original xBD image to the FastAPI backend."""
    try:
        uploaded_file.seek(0)
        response = requests.post(
            f"{FASTAPI_URL}/api/analyze",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            },
            timeout=600,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not connect to FastAPI backend. "
            "Start it with: python -m uvicorn backend.main:app --reload"
        ) from exc

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text

        raise RuntimeError(
            f"FastAPI analysis failed (HTTP {response.status_code}): {detail}"
        )

    return response.json()


def backend_result_to_frontend_result(item):
    """Convert FastAPI result fields to the existing dashboard schema."""
    damage = item["damage"]
    severity, icon, assessment = get_severity(damage)

    return {
        "building_id": item["building_id"],
        "polygon": item["polygon"],
        "bbox": tuple(item["bbox"]),
        "prediction": damage,
        "confidence": float(item["confidence"]),
        "probabilities": item["probabilities"],
        "severity": severity,
        "priority": item["priority"],
        "priority_rank": int(item["priority_rank"]),
        "assessment": assessment,
        "icon": icon,
        "xbd_subtype": item.get("xbd_subtype", ""),
    }


def convert_fastapi_response(data):
    """Convert the complete FastAPI response for the existing UI."""
    return [
        backend_result_to_frontend_result(item)
        for item in data.get("buildings", [])
    ]


# xBD JSON / POLYGON HELPERS

def find_xbd_json(image_name):
    """Find the exact matching xBD JSON for an original xBD image name.

    xBD pairs use the same filename stem for the image and annotation,
    e.g. ``guatemala-volcano_00000000_post_disaster.jpg`` and
    ``guatemala-volcano_00000000_post_disaster.json``.

    We intentionally do NOT guess another JSON when an exact match is
    unavailable, because using the wrong annotation would place building
    zones on the wrong image.
    """
    if not image_name:
        return None

    stem = os.path.splitext(os.path.basename(image_name))[0]
    target = stem + ".json"

    # Exact filename lookup, recursively, across available xBD label sets.
    for root in XBD_LABEL_DIRS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if target in filenames:
                return os.path.join(dirpath, target)

    return None


def find_xbd_pre_disaster_image(image_name):
    """Find the exact xBD pre-disaster image matching a post-disaster filename."""
    if not image_name or "_post_disaster" not in image_name.lower():
        return None

    pre_name = re.sub(
        r"_post_disaster(?=\.[^.]+$)",
        "_pre_disaster",
        os.path.basename(image_name),
        flags=re.IGNORECASE,
    )

    image_dirs = [
        os.path.join(BASE_DIR, "raw_xbd", "train", "images"),
        os.path.join(BASE_DIR, "raw_xbd", "test", "images"),
        os.path.join(BASE_DIR, "raw_xbd", "tier3", "images"),
    ]

    for image_dir in image_dirs:
        if not os.path.isdir(image_dir):
            continue
        for dirpath, _, filenames in os.walk(image_dir):
            if pre_name in filenames:
                return os.path.join(dirpath, pre_name)

    return None


def is_xbd_post_disaster_filename(image_name):
    """Return True when the uploaded filename follows xBD post-disaster naming."""
    stem = os.path.splitext(os.path.basename(image_name or ""))[0].lower()
    return stem.endswith("_post_disaster")


def parse_wkt_polygon(wkt):
    """
    Parse a simple WKT POLYGON string into [(x,y), ...].
    xBD xy annotations are image-coordinate polygons.
    """
    if not isinstance(wkt, str):
        return []

    match = re.search(r"POLYGON\s*\(\((.*?)\)\)", wkt, flags=re.I)
    if not match:
        return []

    points = []
    for pair in match.group(1).split(","):
        nums = pair.strip().split()
        if len(nums) >= 2:
            try:
                points.append((float(nums[0]), float(nums[1])))
            except ValueError:
                pass
    return points


def extract_xbd_buildings(json_data):
    features = json_data.get("features", {})
    xy_features = features.get("xy", [])

    buildings = []
    for feature in xy_features:
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
            # Keep original xBD subtype only as annotation metadata;
            # predictions below come from the trained model.
            "xbd_subtype": props.get("subtype", "")
        })

    return buildings


def normalize_polygons_to_image(buildings, image_size):
    """
    Normally xBD xy coordinates already match the image pixels.
    If an annotation coordinate range is larger than the actual image,
    scale it into the image bounds rather than producing invalid crops.
    """
    width, height = image_size
    if not buildings:
        return buildings

    max_x = max(max(p[0] for p in b["polygon"]) for b in buildings)
    max_y = max(max(p[1] for p in b["polygon"]) for b in buildings)

    scale_x = width / max_x if max_x > width and max_x > 0 else 1.0
    scale_y = height / max_y if max_y > height and max_y > 0 else 1.0

    if scale_x == 1.0 and scale_y == 1.0:
        return buildings

    normalized = []
    for b in buildings:
        polygon = [
            (x * scale_x, y * scale_y)
            for x, y in b["polygon"]
        ]
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        item = dict(b)
        item["polygon"] = polygon
        item["bbox"] = (min(xs), min(ys), max(xs), max(ys))
        normalized.append(item)

    return normalized


def context_crop(image, bbox, scale=CONTEXT_SCALE):
    """Crop a building bbox with context around it, matching the 3x context idea."""
    width, height = image.size
    x1, y1, x2, y2 = bbox

    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    crop_w = bw * scale
    crop_h = bh * scale

    left = max(0, int(round(cx - crop_w / 2)))
    top = max(0, int(round(cy - crop_h / 2)))
    right = min(width, int(round(cx + crop_w / 2)))
    bottom = min(height, int(round(cy + crop_h / 2)))

    if right <= left:
        right = min(width, left + 2)
    if bottom <= top:
        bottom = min(height, top + 2)

    return image.crop((left, top, right, bottom))


# FULL-IMAGE BUILDING ANALYSIS

def analyze_buildings(image, buildings):
    """Classify every detected xBD building polygon."""
    if not buildings:
        return []

    # Prepare every building crop first, then run MobileNetV2 in batches.
    # This removes the previous 100-building cap and is much faster than
    # calling the model once for every building.
    crops = [
        context_crop(image, building["bbox"])
        for building in buildings
    ]

    predictions = predict_damage_batch(crops)
    results = []

    for index, (building, prediction_data) in enumerate(
        zip(buildings, predictions), start=1
    ):
        prediction, confidence, probabilities = prediction_data
        severity, icon, assessment = get_severity(prediction)
        priority, priority_rank = get_priority(prediction)

        results.append({
            "building_id": f"B{index:03d}",
            "polygon": building["polygon"],
            "bbox": building["bbox"],
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "severity": severity,
            "priority": priority,
            "priority_rank": priority_rank,
            "assessment": assessment,
            "icon": icon,
            "xbd_subtype": building.get("xbd_subtype", ""),
        })

    return results


def annotate_full_image(image, results, label_mode="All Buildings"):
    """
    Draw all building damage polygons and optionally show readable labels.

    label_mode:
      - All Buildings
      - High + Critical
      - Critical Only
      - Destroyed Only
      - No Labels
    """
    base = image.convert("RGB").copy()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype(
            "arial.ttf", max(12, base.width // 90)
        )
    except Exception:
        font = ImageFont.load_default()

    for result in results:
        damage = result["prediction"]
        color = CLASS_COLORS[damage]
        polygon = [
            (int(round(x)), int(round(y)))
            for x, y in result["polygon"]
        ]

        # Always keep every building polygon visible.
        draw.polygon(
            polygon,
            fill=(*color, 55),
            outline=(*color, 255),
        )
        draw.line(
            polygon + [polygon[0]],
            fill=(*color, 255),
            width=max(2, base.width // 500),
            joint="curve",
        )

        show_label = (
            label_mode == "All Buildings"
            or (
                label_mode == "High + Critical"
                and result["priority"] in {"HIGH", "CRITICAL"}
            )
            or (
                label_mode == "Critical Only"
                and result["priority"] == "CRITICAL"
            )
            or (
                label_mode == "Destroyed Only"
                and damage == "Destroyed"
            )
        )

        if not show_label:
            continue

        x1, y1, _, _ = result["bbox"]
        label = (
            f"{result['building_id']} | "
            f"{damage.replace('_', ' ')} | "
            f"{result['confidence']:.0f}% | "
            f"{result['priority']}"
        )

        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label) * 7, 16

        tx = max(0, int(x1))
        ty = max(0, int(y1) - th - 6)

        draw.rounded_rectangle(
            [tx, ty, tx + tw + 10, ty + th + 6],
            radius=4,
            fill=(0, 0, 0, 185),
        )
        draw.text(
            (tx + 5, ty + 3),
            label,
            fill="white",
            font=font,
        )

    return Image.alpha_composite(
        base.convert("RGBA"), overlay
    ).convert("RGB")


# REPORT / EXPORT

def make_csv_bytes(results):
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow([
        "Rank", "Building ID", "Damage", "Confidence",
        "Severity", "Priority", "Assessment",
        "BBox X1", "BBox Y1", "BBox X2", "BBox Y2"
    ])

    ranked = sorted(
        results,
        key=lambda x: (-x["priority_rank"], -x["confidence"])
    )

    for rank, item in enumerate(ranked, start=1):
        x1, y1, x2, y2 = item["bbox"]
        writer.writerow([
            rank,
            item["building_id"],
            item["prediction"].replace("_", " "),
            f"{item['confidence']:.2f}%",
            item["severity"],
            item["priority"],
            item["assessment"],
            round(x1, 2), round(y1, 2),
            round(x2, 2), round(y2, 2),
        ])

    return buffer.getvalue().encode("utf-8")


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_pdf_report(
    original_image,
    annotated_image,
    results,
    disaster_type,
    location,
    reporter,
    notes
):
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=19,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=8,
        spaceAfter=7,
    )
    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
    )

    story = []

    story.append(Paragraph(
        "AI DISASTER RESPONSE ASSISTANT",
        title_style
    ))
    story.append(Paragraph(
        "Building Damage & Rescue Priority Assessment",
        title_style
    ))
    story.append(Spacer(1, 8))

    # Incident information.
    story.append(Paragraph("INCIDENT DETAILS", heading_style))
    incident_data = [
        ["Date & Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Disaster Type", disaster_type],
        ["Location", location or "Not provided"],
        ["Reporter", reporter or "Not provided"],
        ["Notes", notes or "None"],
    ]
    incident_table = Table(incident_data, colWidths=[110, 380])
    incident_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(incident_table)
    story.append(Spacer(1, 10))

    # Summary.
    counts = {c: 0 for c in CLASS_NAMES}
    for item in results:
        counts[item["prediction"]] += 1

    story.append(Paragraph("DAMAGE & PRIORITY SUMMARY", heading_style))
    summary_data = [
        ["Total Buildings", str(len(results))],
        ["No Damage", str(counts["No_Damage"])],
        ["Minor Damage", str(counts["Minor_Damage"])],
        ["Major Damage", str(counts["Major_Damage"])],
        ["Destroyed", str(counts["Destroyed"])],
        ["Critical Priority", str(sum(
            1 for x in results if x["priority"] == "CRITICAL"
        ))],
        ["High Priority", str(sum(
            1 for x in results if x["priority"] == "HIGH"
        ))],
        ["Moderate Priority", str(sum(
            1 for x in results if x["priority"] == "MODERATE"
        ))],
        ["Low Priority", str(sum(
            1 for x in results if x["priority"] == "LOW"
        ))],
    ]
    summary_table = Table(summary_data, colWidths=[170, 320])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # Images.
    story.append(Paragraph("ORIGINAL INPUT IMAGE", heading_style))
    original_bytes = image_to_png_bytes(original_image)
    story.append(PDFImage(
        io.BytesIO(original_bytes),
        width=6.9 * inch,
        height=4.4 * inch,
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("AI DAMAGE ZONE OUTPUT", heading_style))
    annotated_bytes = image_to_png_bytes(annotated_image)
    story.append(PDFImage(
        io.BytesIO(annotated_bytes),
        width=6.9 * inch,
        height=4.4 * inch,
    ))
    story.append(PageBreak())

    # Ranked inventory.
    ranked = sorted(
        results,
        key=lambda x: (-x["priority_rank"], -x["confidence"])
    )

    story.append(Paragraph(
        "COMPLETE BUILDING ASSESSMENT INVENTORY",
        heading_style
    ))

    inventory = [[
        "Rank", "ID", "Damage", "Confidence", "Priority"
    ]]

    for rank, item in enumerate(ranked, start=1):
        inventory.append([
            str(rank),
            item["building_id"],
            item["prediction"].replace("_", " "),
            f"{item['confidence']:.1f}%",
            item["priority"],
        ])

    inventory_table = Table(
        inventory,
        colWidths=[38, 48, 145, 85, 90],
        repeatRows=1,
    )
    inventory_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B7A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(inventory_table)
    story.append(Spacer(1, 12))

    # Detailed priority list.
    story.append(Paragraph(
        "RESCUE PRIORITY ASSESSMENT",
        heading_style
    ))

    for rank, item in enumerate(ranked, start=1):
        story.append(Paragraph(
            f"<b>{rank}. {item['building_id']} — "
            f"{item['priority']} — "
            f"{item['prediction'].replace('_', ' ')} "
            f"({item['confidence']:.2f}%)</b>",
            normal_style
        ))
        story.append(Paragraph(
            item["assessment"],
            normal_style
        ))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 10))
    story.append(Paragraph("MODEL INFORMATION", heading_style))
    model_data = [
        ["Model", "MobileNetV2 Transfer Learning"],
        ["Input Size", "224 × 224"],
        ["Test Accuracy", "86.00%"],
        ["Classes", "No Damage / Minor / Major / Destroyed"],
        ["Localization", "xBD building polygons (xy annotations)"],
        ["Priority", "Rule-based decision-support indicator"],
    ]
    model_table = Table(model_data, colWidths=[130, 360])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Important:</b> Rescue priority is a rule-based "
        "decision-support indicator derived from the AI damage "
        "classification. It does not replace professional structural "
        "inspection or official emergency-response procedures.",
        normal_style
    ))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# SESSION STATE

if "building_results" not in st.session_state:
    st.session_state.building_results = []

if "annotated_image_bytes" not in st.session_state:
    st.session_state.annotated_image_bytes = None

if "original_image_bytes" not in st.session_state:
    st.session_state.original_image_bytes = None

if "backend_response" not in st.session_state:
    st.session_state.backend_response = None

if "current_upload_key" not in st.session_state:
    st.session_state.current_upload_key = None


# SIDEBAR

with st.sidebar:
    st.title("ℹ️ About")
    st.write(
        "AI-based building damage assessment using the "
        "xBD building annotations and a trained MobileNetV2 classifier."
    )

    st.subheader("Damage Classes")
    st.write("🟢 No Damage")
    st.write("🟡 Minor Damage")
    st.write("🟠 Major Damage")
    st.write("🔴 Destroyed")

    st.divider()
    st.write("**Model:** MobileNetV2")
    st.write("**Test Accuracy:** 86%")
    st.write("**Classifier Input:** 224 × 224")
    st.write("**Localization:** xBD building polygons")
    st.write("**GPS / Map:** Not used")
    st.write("**Backend:** FastAPI")
    st.write(f"**API:** {FASTAPI_URL}")


# HEADER


st.title("🚨 AI Disaster Response Assistant")
st.subheader(
    "Aerial Image Damage Zoning, Rescue Priority & Assessment Report"
)
st.caption(
    "Upload a full xBD post-disaster image using its original filename. "
    "The app automatically finds the matching xBD JSON, locates buildings, "
    "and sends the image to the FastAPI backend for MobileNetV2 building-level classification."
)
st.divider()


# QUICK NAVIGATION

# This anchor is used by the section navigation links below.
st.markdown('<div id="quick-nav"></div>', unsafe_allow_html=True)

# INCIDENT DETAILS


st.header("📋 Incident Details")

col1, col2 = st.columns(2)

with col1:
    disaster_type = st.selectbox(
        "Disaster Type",
        [
            "Earthquake", "Flood", "Cyclone", "Wildfire",
            "Tsunami", "Landslide", "Volcanic Eruption", "Other"
        ]
    )
    location = st.text_input(
        "Location",
        placeholder="Optional — e.g. Guatemala / Sector A"
    )

with col2:
    reporter = st.text_input(
        "Reporter Name",
        placeholder="Optional"
    )
    incident_notes = st.text_area(
        "Additional Notes",
        placeholder="Optional incident information..."
    )

st.divider()

# QUICK SECTION NAVIGATION

st.markdown(
    """
    <style>
    .quick-nav-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .quick-nav {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-bottom: 1rem;
    }
    .quick-nav a {
        display: inline-block;
        padding: 0.58rem 0.85rem;
        border: 1px solid rgba(128,128,128,0.45);
        border-radius: 0.55rem;
        text-decoration: none;
        font-weight: 600;
        color: inherit;
        background: rgba(128,128,128,0.10);
    }
    .quick-nav a:hover {
        border-color: rgba(100,160,255,0.85);
        background: rgba(100,160,255,0.12);
    }
    .back-top {
        margin: 0.45rem 0 1rem 0;
    }
    .back-top a {
        text-decoration: none;
        font-weight: 600;
    }
    </style>
    <div class="quick-nav-title">⚡ Quick Navigation</div>
    <div class="quick-nav">
        <a href="#impact-analytics">📊 Disaster Impact Analytics</a>
        <a href="#building-inventory">📋 Complete Building Assessment</a>
        <a href="#detailed-assessments">🔎 Detailed Assessments</a>
        <a href="#assessment-reports">📄 Assessment Reports</a>
        <a href="#image-downloads">🖼️ Image Downloads</a>
    </div>
    """,
    unsafe_allow_html=True,
)


# FULL IMAGE + ANNOTATION INPUT

st.header("🛰️ Full Aerial / Satellite Image")

uploaded_file = st.file_uploader(
    "Upload the full post-disaster image",
    type=["jpg", "jpeg", "png"],
    help=(
        "For real multi-building zoning, use a full xBD image, "
        "not a 34×30 / 35×35 building crop."
    )
)

# Clear stale predictions when the selected image changes or is removed.
if uploaded_file is None:
    if st.session_state.current_upload_key is not None:
        st.session_state.building_results = []
        st.session_state.annotated_image_bytes = None
        st.session_state.original_image_bytes = None
        st.session_state.backend_response = None
        st.session_state.current_upload_key = None
else:
    upload_digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    upload_key = (uploaded_file.name, upload_digest)
    if upload_key != st.session_state.current_upload_key:
        st.session_state.building_results = []
        st.session_state.annotated_image_bytes = None
        st.session_state.original_image_bytes = None
        st.session_state.backend_response = None
        st.session_state.current_upload_key = upload_key

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # --------------------------------------------------------
    # Automatic xBD image -> JSON pairing
    # --------------------------------------------------------
    annotation_source = None
    json_data = None

    if not is_xbd_post_disaster_filename(uploaded_file.name):
        st.warning(
            "This filename does not look like an original xBD "
            "post-disaster image. For xBD automatic pairing, keep the "
            "original filename ending in `_post_disaster`. "
            "Renaming the file to something like `check.jpg` removes "
            "the information needed for exact JSON matching."
        )
    else:
        auto_json = find_xbd_json(uploaded_file.name)
        if auto_json:
            try:
                with open(auto_json, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                annotation_source = auto_json
            except Exception as error:
                st.warning(
                    f"The matching JSON was found but could not be read: {error}"
                )
        else:
            st.warning(
                "No exact matching xBD JSON was found for this image. "
                "The app will not guess another annotation file."
            )

    if json_data is not None:
        buildings = extract_xbd_buildings(json_data)
        buildings = normalize_polygons_to_image(buildings, image.size)
    else:
        buildings = []

   
    # Automatic xBD pre-disaster image pairing

    pre_disaster_path = find_xbd_pre_disaster_image(uploaded_file.name)
    pre_disaster_image = None
    if pre_disaster_path:
        try:
            pre_disaster_image = Image.open(pre_disaster_path).convert("RGB")
        except Exception as error:
            st.warning(
                f"The matching pre-disaster image was found but could not be opened: {error}"
            )


    # Before / After comparison
    st.markdown("### 🛰️ Before & After Comparison")
    before_col, after_col = st.columns(2)

    with before_col:
        st.subheader("🟢 Pre-Disaster (Before)")
        if pre_disaster_image is not None:
            st.image(pre_disaster_image, use_container_width=True)
            st.caption(
                f"Matched automatically: {os.path.basename(pre_disaster_path)}"
            )
        else:
            st.warning(
                "Matching pre-disaster image was not found. "
                "The post-disaster analysis can still continue."
            )

    with after_col:
        st.subheader("🔴 Post-Disaster (After)")
        st.image(image, use_container_width=True)
        st.caption(f"Uploaded: {uploaded_file.name}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("🎯 Building Localization")
        if buildings:
            st.success(
                f"{len(buildings)} building polygons loaded automatically."
            )
            st.caption(f"Matched annotation: {annotation_source}")
        else:
            st.warning(
                "No matching xBD building polygons are available for this image. "
                "Use the original xBD post-disaster filename so the app can "
                "automatically locate its JSON annotation."
            )

    with right:
        if buildings:
            st.info(
                f"Click **Analyze All Buildings** to classify all "
                f"{len(buildings)} buildings and create the damage zones."
            )

    analyze_all = st.button(
        "🔍 Analyze All Buildings & Create Damage Zones",
        type="primary",
        use_container_width=True,
        disabled=(len(buildings) == 0),
    )

    if analyze_all:
        with st.spinner(
            f"Sending image to FastAPI and analyzing all {len(buildings)} buildings..."
        ):
            try:
                backend_data = analyze_via_fastapi(uploaded_file)
                results = convert_fastapi_response(backend_data)

                if not results:
                    raise RuntimeError(
                        "FastAPI returned no building predictions."
                    )

                # Build the initial annotated image using the backend results.
                annotated = annotate_full_image(
                    image,
                    results,
                    label_mode="All Buildings",
                )

                st.session_state.building_results = results
                st.session_state.annotated_image_bytes = image_to_png_bytes(annotated)
                st.session_state.original_image_bytes = image_to_png_bytes(image)
                st.session_state.backend_response = backend_data

                st.success(
                    f"FastAPI analysis completed: {len(results)} buildings classified."
                )

                st.caption(
                    f"Backend: {FASTAPI_URL}/api/analyze | "
                    f"Matched JSON: {backend_data.get('matched_json', 'N/A')}"
                )

            except RuntimeError as error:
                st.error(str(error))
                st.stop()



# RESULTS DASHBOARD

results = st.session_state.building_results

# Analytics are derived from saved session results, not from the one-time button click.
# This keeps the section visible after Streamlit reruns caused by UI controls.
if results:
    counts = {c: 0 for c in CLASS_NAMES}
    priority_counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
    for item in results:
        counts[item["prediction"]] += 1
        priority_counts[item["priority"]] += 1

    st.divider()
    st.markdown('<div id="impact-analytics"></div>', unsafe_allow_html=True)
    st.header("📊 Disaster Impact Analytics")
    st.markdown('<div class="back-top"><a href="#quick-nav">⬆️ Back to Quick Navigation</a></div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Inspected Buildings", len(results))
    m2.metric("Destroyed", counts["Destroyed"])
    m3.metric("Major Damage", counts["Major_Damage"])
    m4.metric("Minor Damage", counts["Minor_Damage"])
    affected = counts["Destroyed"] + counts["Major_Damage"] + counts["Minor_Damage"]
    impact_ratio = (affected / len(results) * 100) if results else 0
    m5.metric("Impact Ratio", f"{impact_ratio:.1f}%")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Damage Distribution")
        st.bar_chart({
            "No Damage": counts["No_Damage"],
            "Minor Damage": counts["Minor_Damage"],
            "Major Damage": counts["Major_Damage"],
            "Destroyed": counts["Destroyed"],
        })
    with chart_col2:
        st.subheader("Rescue Priority Counts")
        st.bar_chart({
            "LOW": priority_counts["LOW"],
            "MODERATE": priority_counts["MODERATE"],
            "HIGH": priority_counts["HIGH"],
            "CRITICAL": priority_counts["CRITICAL"],
        })

if results:
    ranked = sorted(
        results,
        key=lambda x: (-x["priority_rank"], -x["confidence"])
    )

    st.divider()
    st.header("🚨 Rescue Priority Dashboard")

    highest = ranked[0]
    st.error(
        f"Highest Priority: {highest['building_id']} — "
        f"{highest['prediction'].replace('_', ' ')} — "
        f"{highest['priority']} — "
        f"{highest['confidence']:.2f}% confidence"
    )

    st.subheader("🖼️ AI Damage Zone Output")

    # Clean visualization controls. All buildings remain analyzed; these
    # controls only change which labels are drawn on the image.
    label_mode = st.selectbox(
        "Building Label Display",
        [
            "All Buildings",
            "High + Critical",
            "Critical Only",
            "Destroyed Only",
            "No Labels",
        ],
        index=0,
        help=(
            "All building polygons remain visible. This setting only "
            "controls the text labels so dense scenes stay readable."
        ),
    )

    original_for_visual = Image.open(
        io.BytesIO(st.session_state.original_image_bytes)
    ).convert("RGB")

    clean_annotated = annotate_full_image(
        original_for_visual,
        results,
        label_mode=label_mode,
    )

    # Keep the currently selected visualization for the PDF/image download.
    st.session_state.annotated_image_bytes = image_to_png_bytes(clean_annotated)

    st.image(
        clean_annotated,
        use_container_width=True,
        caption=(
            "Green = No Damage | Yellow = Minor | "
            "Orange = Major | Red = Destroyed"
        ),
    )

    st.caption(
        f"Showing {len(results)} analyzed buildings. "
        f"Label mode: {label_mode}."
    )

    st.markdown('<div id="building-inventory"></div>', unsafe_allow_html=True)
    st.subheader("📋 Complete Building Assessment Inventory")
    st.markdown('<div class="back-top"><a href="#quick-nav">⬆️ Back to Quick Navigation</a></div>', unsafe_allow_html=True)

    table_rows = []
    for rank, item in enumerate(ranked, start=1):
        table_rows.append({
            "Rank": rank,
            "Building ID": item["building_id"],
            "Damage": item["prediction"].replace("_", " "),
            "AI Confidence": f"{item['confidence']:.2f}%",
            "Severity": item["severity"],
            "Priority": item["priority"],
        })

    st.dataframe(
        table_rows,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown('<div id="detailed-assessments"></div>', unsafe_allow_html=True)
    st.subheader("🔎 Detailed Building Assessments")
    st.markdown('<div class="back-top"><a href="#quick-nav">⬆️ Back to Quick Navigation</a></div>', unsafe_allow_html=True)

    for rank, item in enumerate(ranked, start=1):
        with st.expander(
            f"{rank}. {item['building_id']} — "
            f"{item['prediction'].replace('_', ' ')} — "
            f"{item['priority']} — "
            f"{item['confidence']:.2f}%"
        ):
            d1, d2 = st.columns(2)

            with d1:
                st.write(f"**Building ID:** {item['building_id']}")
                st.write(f"**Damage:** {item['prediction'].replace('_', ' ')}")
                st.write(f"**Confidence:** {item['confidence']:.2f}%")
                st.write(f"**Severity:** {item['severity']}")
                st.write(f"**Rescue Priority:** {item['priority']}")
                st.write(f"**Assessment:** {item['assessment']}")

            with d2:
                probs = {
                    k.replace("_", " "): round(v, 2)
                    for k, v in item["probabilities"].items()
                }
                st.bar_chart(probs)



# REPORTS


if results and st.session_state.original_image_bytes:
    st.divider()
    st.markdown('<div id="assessment-reports"></div>', unsafe_allow_html=True)
    st.header("📄 Assessment Reports")
    st.markdown('<div class="back-top"><a href="#quick-nav">⬆️ Back to Quick Navigation</a></div>', unsafe_allow_html=True)

    original = Image.open(
        io.BytesIO(st.session_state.original_image_bytes)
    ).convert("RGB")

    annotated = Image.open(
        io.BytesIO(st.session_state.annotated_image_bytes)
    ).convert("RGB")

    pdf_data = generate_pdf_report(
        original_image=original,
        annotated_image=annotated,
        results=results,
        disaster_type=disaster_type,
        location=location,
        reporter=reporter,
        notes=incident_notes,
    )

    csv_data = make_csv_bytes(results)

    report_col1, report_col2 = st.columns(2)

    with report_col1:
        st.download_button(
            "⬇️ Download Complete PDF Report",
            data=pdf_data,
            file_name="AI_Disaster_Response_Assessment.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with report_col2:
        st.download_button(
            "⬇️ Download Building Assessment CSV",
            data=csv_data,
            file_name="building_assessment_inventory.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown('<div id="image-downloads"></div>', unsafe_allow_html=True)
    st.subheader("🖼️ Image Downloads")
    st.markdown('<div class="back-top"><a href="#quick-nav">⬆️ Back to Quick Navigation</a></div>', unsafe_allow_html=True)

    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.download_button(
            "⬇️ Download Original Image",
            data=st.session_state.original_image_bytes,
            file_name="original_disaster_image.png",
            mime="image/png",
            use_container_width=True,
        )

    with img_col2:
        st.download_button(
            "⬇️ Download AI Damage Zone Image",
            data=st.session_state.annotated_image_bytes,
            file_name="AI_damage_zones.png",
            mime="image/png",
            use_container_width=True,
        )



# FOOTER

st.divider()
st.caption(
    "AI Disaster Response Assistant | MobileNetV2 Transfer Learning | "
    "xBD Building Localization | Damage Zoning & Rescue Priority Decision Support"
)
