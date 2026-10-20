import os
import cv2
import numpy as np

# Image size for CNN
IMAGE_SIZE = (224, 224)

# Dataset path
DATASET_PATH = "dataset"

# Damage classes
CLASSES = [
    "No_Damage",
    "Minor_Damage",
    "Major_Damage",
    "Destroyed"
]

def load_images():
    images = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        class_path = os.path.join(DATASET_PATH, class_name)

        if not os.path.exists(class_path):
            print(f"Folder not found: {class_path}")
            continue

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            image = image / 255.0

            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)


if __name__ == "__main__":
    images, labels = load_images()

    print("Images Loaded :", len(images))
    print("Labels Loaded :", len(labels))