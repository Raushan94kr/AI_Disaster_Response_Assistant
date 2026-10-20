import json
import os
import random

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

RAW_XBD = r"D:\AI_Disaster_Response_Assistant\raw_xbd\train"

IMAGE_DIR = os.path.join(RAW_XBD, "images")
LABEL_DIR = os.path.join(RAW_XBD, "labels")

OUTPUT_DIR = r"D:\AI_Disaster_Response_Assistant\dataset"


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_MAP = {
    "no-damage": "No_Damage",
    "minor-damage": "Minor_Damage",
    "major-damage": "Major_Damage",
    "destroyed": "Destroyed",
}

# Number of building samples per class
SAMPLES_PER_CLASS = 500

# Final image size for CNN
IMAGE_SIZE = 224

# Context multiplier around the building
# Larger value = more surrounding area
CONTEXT_SCALE = 3.0

# Do not create very tiny crops
MIN_CROP_SIZE = 64

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_directories():
    for split in ["train", "validation", "test"]:
        for class_name in CLASS_MAP.values():
            os.makedirs(
                os.path.join(
                    OUTPUT_DIR,
                    split,
                    class_name
                ),
                exist_ok=True
            )


# ============================================================
# PARSE WKT POLYGON
# ============================================================

def parse_wkt_polygon(wkt):
    """
    Convert:
        POLYGON ((x y, x y, ...))

    into a NumPy array of pixel coordinates.
    """

    if not wkt or not wkt.startswith("POLYGON"):
        return None

    try:
        content = wkt.split("((", 1)[1].split("))", 1)[0]

        points = []

        for pair in content.split(","):
            x, y = pair.strip().split()
            points.append([float(x), float(y)])

        if len(points) < 3:
            return None

        return np.array(points, dtype=np.float32)

    except Exception:
        return None


# ============================================================
# LARGE SQUARE BUILDING CROP
# ============================================================

def crop_building_with_context(image, polygon):
    """
    Create a square crop around the building.

    Instead of tightly cropping the polygon, this function
    adds surrounding context so the CNN gets more useful
    visual information.
    """

    if polygon is None or len(polygon) < 3:
        return None

    height, width = image.shape[:2]

    x_min = float(np.min(polygon[:, 0]))
    x_max = float(np.max(polygon[:, 0]))

    y_min = float(np.min(polygon[:, 1]))
    y_max = float(np.max(polygon[:, 1]))

    building_width = x_max - x_min
    building_height = y_max - y_min

    if building_width <= 0 or building_height <= 0:
        return None

    # Use the larger building dimension
    building_size = max(
        building_width,
        building_height
    )

    # Add surrounding context
    crop_size = max(
        int(building_size * CONTEXT_SCALE),
        MIN_CROP_SIZE
    )

    # Limit extremely large crops
    crop_size = min(crop_size, 256)

    # Building center
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    half_size = crop_size // 2

    crop_x1 = int(round(center_x - half_size))
    crop_y1 = int(round(center_y - half_size))

    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    # --------------------------------------------------------
    # Shift crop back inside image boundaries
    # --------------------------------------------------------

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0

    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0

    if crop_x2 > width:
        shift = crop_x2 - width
        crop_x1 -= shift
        crop_x2 = width

    if crop_y2 > height:
        shift = crop_y2 - height
        crop_y1 -= shift
        crop_y2 = height

    # Final safety correction
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(width, crop_x2)
    crop_y2 = min(height, crop_y2)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    if crop.size == 0:
        return None

    crop_height, crop_width = crop.shape[:2]

    if crop_width < MIN_CROP_SIZE or crop_height < MIN_CROP_SIZE:
        return None

    # Resize to CNN input size
    crop = cv2.resize(
        crop,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return crop


# ============================================================
# COLLECT BALANCED SAMPLES
# ============================================================

def collect_samples():

    samples = {
        class_name: []
        for class_name in CLASS_MAP.values()
    }

    json_files = [
        file_name
        for file_name in os.listdir(LABEL_DIR)
        if file_name.endswith("_post_disaster.json")
    ]

    random.shuffle(json_files)

    print(
        f"Post-disaster JSON files found: {len(json_files)}"
    )

    for file_number, json_file in enumerate(
        json_files,
        start=1
    ):

        # Stop once every class has enough samples
        if all(
            len(samples[class_name]) >= SAMPLES_PER_CLASS
            for class_name in samples
        ):
            break

        json_path = os.path.join(
            LABEL_DIR,
            json_file
        )

        image_name = json_file.replace(
            ".json",
            ".png"
        )

        image_path = os.path.join(
            IMAGE_DIR,
            image_name
        )

        if not os.path.exists(image_path):
            continue

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            xy_features = (
                data
                .get("features", {})
                .get("xy", [])
            )

            if not isinstance(xy_features, list):
                continue

            image = cv2.imread(image_path)

            if image is None:
                continue

            for feature in xy_features:

                properties = feature.get(
                    "properties",
                    {}
                )

                subtype = properties.get(
                    "subtype"
                )

                if subtype not in CLASS_MAP:
                    continue

                class_name = CLASS_MAP[subtype]

                # Skip this class if its quota is full
                if len(samples[class_name]) >= SAMPLES_PER_CLASS:
                    continue

                polygon = parse_wkt_polygon(
                    feature.get("wkt")
                )

                crop = crop_building_with_context(
                    image,
                    polygon
                )

                if crop is None:
                    continue

                samples[class_name].append(crop)

        except Exception as error:

            print(
                f"Error processing {json_file}: {error}"
            )

        if file_number % 100 == 0:

            print(
                f"Processed: {file_number} | "
                f"No Damage: {len(samples['No_Damage'])} | "
                f"Minor: {len(samples['Minor_Damage'])} | "
                f"Major: {len(samples['Major_Damage'])} | "
                f"Destroyed: {len(samples['Destroyed'])}"
            )

    return samples


# ============================================================
# SPLIT DATA
# ============================================================

def split_samples(samples):

    train = {}
    validation = {}
    test = {}

    for class_name, class_samples in samples.items():

        random.shuffle(class_samples)

        total = len(class_samples)

        train_end = int(
            total * TRAIN_RATIO
        )

        validation_end = train_end + int(
            total * VAL_RATIO
        )

        train[class_name] = class_samples[
            :train_end
        ]

        validation[class_name] = class_samples[
            train_end:validation_end
        ]

        test[class_name] = class_samples[
            validation_end:
        ]

    return train, validation, test


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(split_data, split_name):

    counts = {}

    for class_name, images in split_data.items():

        output_folder = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        counts[class_name] = 0

        for index, image in enumerate(images):

            filename = (
                f"{class_name.lower()}_"
                f"{split_name}_"
                f"{index:04d}.jpg"
            )

            output_path = os.path.join(
                output_folder,
                filename
            )

            success = cv2.imwrite(
                output_path,
                image
            )

            if success:
                counts[class_name] += 1

    return counts


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    train_counts,
    validation_counts,
    test_counts
):

    print("\n")
    print("=" * 70)
    print("BALANCED DATASET SUMMARY")
    print("=" * 70)

    print(
        f"{'Class':<20}"
        f"{'Train':<10}"
        f"{'Validation':<12}"
        f"{'Test':<10}"
    )

    print("-" * 70)

    for class_name in CLASS_MAP.values():

        print(
            f"{class_name:<20}"
            f"{train_counts[class_name]:<10}"
            f"{validation_counts[class_name]:<12}"
            f"{test_counts[class_name]:<10}"
        )

    print("-" * 70)

    total_train = sum(
        train_counts.values()
    )

    total_validation = sum(
        validation_counts.values()
    )

    total_test = sum(
        test_counts.values()
    )

    print(
        f"{'TOTAL':<20}"
        f"{total_train:<10}"
        f"{total_validation:<12}"
        f"{total_test:<10}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("xBD IMPROVED BALANCED DATASET CONVERTER")
    print("=" * 70)

    random.seed(RANDOM_SEED)

    create_directories()

    print("\nCollecting building crops...")
    print(
        f"Target: {SAMPLES_PER_CLASS} samples per class"
    )
    print(
        f"Context scale: {CONTEXT_SCALE}x"
    )
    print(
        f"CNN image size: {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    samples = collect_samples()

    print("\nCollected samples:")

    for class_name, images in samples.items():

        print(
            f"{class_name:<15}: {len(images)}"
        )

    # --------------------------------------------------------
    # Check for missing classes
    # --------------------------------------------------------

    incomplete = [
        class_name
        for class_name, images in samples.items()
        if len(images) < SAMPLES_PER_CLASS
    ]

    if incomplete:

        print("\nWARNING:")
        print(
            "Some classes have fewer than "
            f"{SAMPLES_PER_CLASS} usable samples."
        )

        for class_name in incomplete:

            print(
                f"{class_name}: "
                f"{len(samples[class_name])}"
            )

        print(
            "\nUsing all available samples "
            "for those classes."
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, validation, test = split_samples(
        samples
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\nSaving training dataset...")

    train_counts = save_dataset(
        train,
        "train"
    )

    print("Saving validation dataset...")

    validation_counts = save_dataset(
        validation,
        "validation"
    )

    print("Saving test dataset...")

    test_counts = save_dataset(
        test,
        "test"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        train_counts,
        validation_counts,
        test_counts
    )

    print(
        "\n✅ Improved dataset creation completed!"
    )

    print(
        f"\nDataset location:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()