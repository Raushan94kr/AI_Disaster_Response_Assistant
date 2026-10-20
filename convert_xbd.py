import json
import os
import random
import shutil

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

RAW_XBD = r"D:\AI_Disaster_Response_Assistant\raw_xbd\train"
OUTPUT = r"D:\AI_Disaster_Response_Assistant\dataset"

IMAGE_DIR = os.path.join(RAW_XBD, "images")
LABEL_DIR = os.path.join(RAW_XBD, "labels")

CLASS_MAP = {
    "no-damage": "No_Damage",
    "minor-damage": "Minor_Damage",
    "major-damage": "Major_Damage",
    "destroyed": "Destroyed",
}

# Maximum number of samples to create.
# Start smaller so your laptop can process the dataset comfortably.
MAX_SAMPLES = 3000

# Train / validation / test split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# FUNCTIONS
# ============================================================

def create_output_directories():
    """Create train/validation/test/class folders."""

    splits = ["train", "validation", "test"]

    for split in splits:
        for class_name in CLASS_MAP.values():
            path = os.path.join(OUTPUT, split, class_name)
            os.makedirs(path, exist_ok=True)


def parse_wkt_polygon(wkt):
    """
    Convert a WKT POLYGON string into an OpenCV polygon.

    Example:
    POLYGON ((x y, x y, x y, ...))
    """

    if not wkt.startswith("POLYGON"):
        return None

    try:
        content = wkt.split("((")[1].split("))")[0]

        points = []

        for pair in content.split(","):
            x, y = pair.strip().split()
            points.append([float(x), float(y)])

        return np.array(points, dtype=np.float32)

    except Exception:
        return None


def crop_building(image, polygon, padding=8):
    """
    Crop a building from an image using its polygon.
    """

    if polygon is None or len(polygon) < 3:
        return None

    height, width = image.shape[:2]

    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]

    x_min = max(0, int(np.floor(x_coords.min())) - padding)
    y_min = max(0, int(np.floor(y_coords.min())) - padding)

    x_max = min(width, int(np.ceil(x_coords.max())) + padding)
    y_max = min(height, int(np.ceil(y_coords.max())) + padding)

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = image[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return None

    return crop


def get_image_path(json_filename):
    """
    Convert label filename to corresponding post-disaster image.
    """

    base_name = json_filename.replace(".json", ".png")
    return os.path.join(IMAGE_DIR, base_name)


def collect_samples():
    """
    Read xBD JSON annotations and prepare building crops.
    """

    samples = []

    json_files = [
        file_name
        for file_name in os.listdir(LABEL_DIR)
        if file_name.endswith("_post_disaster.json")
    ]

    print(f"Post-disaster JSON files found: {len(json_files)}")

    for index, json_file in enumerate(json_files, start=1):

        if len(samples) >= MAX_SAMPLES:
            break

        json_path = os.path.join(LABEL_DIR, json_file)

        image_path = get_image_path(json_file)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            features = data.get("features", {})
            xy_features = features.get("xy", [])

            if not isinstance(xy_features, list):
                continue

            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            for feature in xy_features:

                properties = feature.get("properties", {})
                subtype = properties.get("subtype")

                if subtype not in CLASS_MAP:
                    continue

                wkt = feature.get("wkt")

                polygon = parse_wkt_polygon(wkt)

                crop = crop_building(image, polygon)

                if crop is None:
                    continue

                samples.append(
                    {
                        "crop": crop,
                        "class_name": CLASS_MAP[subtype],
                        "source": json_file,
                    }
                )

                if len(samples) >= MAX_SAMPLES:
                    break

        except Exception as error:
            print(f"Error processing {json_file}: {error}")

        if index % 100 == 0:
            print(
                f"Processed JSON files: {index} | "
                f"Samples collected: {len(samples)}"
            )

    return samples


def split_samples(samples):
    """Split samples into train, validation and test."""

    random.seed(RANDOM_SEED)

    random.shuffle(samples)

    total = len(samples)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    return train_samples, val_samples, test_samples


def save_samples(samples, split):
    """Save building crops into class folders."""

    class_counts = {
        class_name: 0
        for class_name in CLASS_MAP.values()
    }

    for index, sample in enumerate(samples):

        class_name = sample["class_name"]

        class_dir = os.path.join(
            OUTPUT,
            split,
            class_name
        )

        os.makedirs(class_dir, exist_ok=True)

        file_name = f"{split}_{index:06d}.jpg"

        output_path = os.path.join(
            class_dir,
            file_name
        )

        cv2.imwrite(
            output_path,
            sample["crop"]
        )

        class_counts[class_name] += 1

    return class_counts


def print_summary(train_counts, val_counts, test_counts):

    print("\n" + "=" * 55)
    print("DATASET SUMMARY")
    print("=" * 55)

    classes = list(CLASS_MAP.values())

    print(
        f"{'Class':<20}"
        f"{'Train':<10}"
        f"{'Validation':<12}"
        f"{'Test':<10}"
    )

    print("-" * 55)

    for class_name in classes:

        print(
            f"{class_name:<20}"
            f"{train_counts[class_name]:<10}"
            f"{val_counts[class_name]:<12}"
            f"{test_counts[class_name]:<10}"
        )

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_test = sum(test_counts.values())

    print("-" * 55)

    print(
        f"{'TOTAL':<20}"
        f"{total_train:<10}"
        f"{total_val:<12}"
        f"{total_test:<10}"
    )

    print("=" * 55)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("xBD DATASET CONVERTER")
    print("=" * 60)

    create_output_directories()

    print("\nReading xBD annotations...")

    samples = collect_samples()

    if not samples:

        print("\nNo samples were created.")
        print("Please check the xBD image and label directories.")

        return

    print(f"\nTotal samples collected: {len(samples)}")

    train_samples, val_samples, test_samples = split_samples(samples)

    print("\nSaving training samples...")
    train_counts = save_samples(
        train_samples,
        "train"
    )

    print("Saving validation samples...")
    val_counts = save_samples(
        val_samples,
        "validation"
    )

    print("Saving test samples...")
    test_counts = save_samples(
        test_samples,
        "test"
    )

    print_summary(
        train_counts,
        val_counts,
        test_counts
    )

    print("\n✅ Dataset conversion completed successfully!")

    print("\nDataset location:")
    print(OUTPUT)


if __name__ == "__main__":
    main()