import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"D:\AI_Disaster_Response_Assistant"

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# Stage 1: frozen pretrained model
INITIAL_EPOCHS = 8

# Stage 2: fine tuning
FINE_TUNE_EPOCHS = 8

CLASS_NAMES = [
    "No_Damage",
    "Minor_Damage",
    "Major_Damage",
    "Destroyed"
]

NUM_CLASSES = len(CLASS_NAMES)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "disaster_damage_mobilenetv2.keras"
)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 75)
print("AI DISASTER RESPONSE ASSISTANT")
print("TRANSFER LEARNING - MOBILENETV2")
print("=" * 75)

print("\nTensorFlow:", tf.__version__)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Initial epochs:", INITIAL_EPOCHS)
print("Fine-tuning epochs:", FINE_TUNE_EPOCHS)


# ============================================================
# DATA GENERATORS
# ============================================================

print("\nLoading dataset...")


# Training augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    shear_range=0.10,
    horizontal_flip=True,
    fill_mode="nearest"
)


# Validation/Test: only preprocessing
val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


validation_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="categorical",
    shuffle=False
)


test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASS_NAMES,
    class_mode="categorical",
    shuffle=False
)


print("\nDataset:")
print("Training   :", train_generator.samples)
print("Validation :", validation_generator.samples)
print("Test       :", test_generator.samples)

print("\nClass mapping:")
print(train_generator.class_indices)


# ============================================================
# LOAD PRETRAINED MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2 pretrained model...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.40)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# STAGE 1 COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


print("\nModel summary:")
model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# STAGE 1 - FEATURE EXTRACTION
# ============================================================

print("\n")
print("=" * 75)
print("STAGE 1 - TRAINING WITH FROZEN MOBILENETV2")
print("=" * 75)


history_stage1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# STAGE 2 - FINE TUNING
# ============================================================

print("\n")
print("=" * 75)
print("STAGE 2 - FINE TUNING MOBILENETV2")
print("=" * 75)


# Unfreeze MobileNetV2
base_model.trainable = True


# Keep BatchNorm layers frozen for stability
for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False


# Freeze most early layers
fine_tune_from = len(base_model.layers) - 30

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False


print(
    "\nFine-tuning last",
    len(base_model.layers) - fine_tune_from,
    "MobileNetV2 layers."
)


# Recompile with very small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


history_stage2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

print("\n✅ Model saved:")
print(MODEL_PATH)


# ============================================================
# COMBINE TRAINING HISTORY
# ============================================================

history = {}

for key in history_stage1.history:

    history[key] = (
        history_stage1.history[key]
        + history_stage2.history.get(key, [])
    )


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.axvline(
    x=len(history_stage1.history["accuracy"]) - 1,
    linestyle="--",
    label="Fine-tuning starts"
)

plt.title(
    "MobileNetV2 Training and Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

accuracy_path = os.path.join(
    REPORT_DIR,
    "mobilenetv2_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.axvline(
    x=len(history_stage1.history["loss"]) - 1,
    linestyle="--",
    label="Fine-tuning starts"
)

plt.title(
    "MobileNetV2 Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

loss_path = os.path.join(
    REPORT_DIR,
    "mobilenetv2_loss.png"
)

plt.savefig(
    loss_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n")
print("=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)


test_generator.reset()

test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)


print("\nTest Loss:")
print(round(test_loss, 4))

print("\nTest Accuracy:")
print(
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

test_generator.reset()

predictions = model.predict(
    test_generator,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_generator.classes


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\n")
print("=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(report)


report_path = os.path.join(
    REPORT_DIR,
    "mobilenetv2_classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AI Disaster Response Assistant\n"
    )

    file.write(
        "MobileNetV2 Transfer Learning\n\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)


plt.figure(figsize=(8, 7))

plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)

plt.title(
    "MobileNetV2 Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    NUM_CLASSES
)

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

threshold = cm.max() / 2.0

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            color="white"
            if cm[i, j] > threshold
            else "black"
        )


plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")

plt.tight_layout()


cm_path = os.path.join(
    REPORT_DIR,
    "mobilenetv2_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = os.path.join(
    REPORT_DIR,
    "mobilenetv2_training_history.json"
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            key: [
                float(v)
                for v in values
            ]
            for key, values in history.items()
        },
        file,
        indent=4
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

with open(
    class_names_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        CLASS_NAMES,
        file,
        indent=4
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 75)
print("TRANSFER LEARNING COMPLETED")
print("=" * 75)

print("\nModel:")
print(MODEL_PATH)

print("\nReports:")
print(REPORT_DIR)

print("\nFinal Test Accuracy:")
print(
    f"{test_accuracy * 100:.2f}%"
)

print("\n✅ MobileNetV2 training pipeline completed successfully!")