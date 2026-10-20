from PIL import Image, ImageDraw
import glob
import os
import math

DATASET = r"D:\AI_Disaster_Response_Assistant\dataset\train"

CLASSES = [
    "No_Damage",
    "Minor_Damage",
    "Major_Damage",
    "Destroyed"
]

SAMPLES_PER_CLASS = 4
THUMB_SIZE = (180, 180)

samples = []

for class_name in CLASSES:
    files = glob.glob(
        os.path.join(DATASET, class_name, "*.jpg")
    )[:SAMPLES_PER_CLASS]

    for file in files:
        image = Image.open(file).convert("RGB")
        image.thumbnail(THUMB_SIZE)

        canvas = Image.new("RGB", THUMB_SIZE, "white")

        x = (THUMB_SIZE[0] - image.width) // 2
        y = (THUMB_SIZE[1] - image.height) // 2

        canvas.paste(image, (x, y))

        samples.append((class_name, canvas))

cols = 4
rows = math.ceil(len(samples) / cols)

sheet = Image.new(
    "RGB",
    (cols * THUMB_SIZE[0], rows * 220),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, (class_name, image) in enumerate(samples):

    x = (i % cols) * THUMB_SIZE[0]
    y = (i // cols) * 220

    sheet.paste(image, (x, y))

    draw.text(
        (x + 5, y + 185),
        class_name,
        fill="black"
    )

output = r"D:\AI_Disaster_Response_Assistant\dataset_preview.jpg"

sheet.save(output)

print("Preview created successfully!")
print(output)