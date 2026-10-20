import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\AI_Disaster_Response_Assistant"

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

CLASS_NAMES = [
    "No_Damage",
    "Minor_Damage",
    "Major_Damage",
    "Destroyed"
]

NUM_CLASSES = len(CLASS_NAMES)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "disaster_damage_model.keras"
)


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

print("=" * 70)
print("AI DISASTER RESPONSE ASSISTANT")
print("CNN BUILDING DAMAGE CLASSIFICATION")
print("=" * 70)

print("\nTensorFlow version:", tf.__version__)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)


# ============================================================
# DATA GENERATORS
# ============================================================

print("\nLoading dataset...")


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)


val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
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


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nDataset information")
print("-" * 50)

print("Training images   :", train_generator.samples)
print("Validation images :", validation_generator.samples)
print("Test images       :", test_generator.samples)

print("\nClass mapping:")
print(train_generator.class_indices)


# ============================================================
# BUILD CNN MODEL
# ============================================================

print("\nBuilding CNN model...")


model = models.Sequential([
    
    layers.Input(
        shape=(224, 224, 3)
    ),

    # Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.BatchNormalization(),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.20),


    # Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.BatchNormalization(),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.25),


    # Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.BatchNormalization(),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.30),


    # Classification head
    layers.Flatten(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Dropout(0.50),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
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


print("\nModel Summary:")
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
    patience=4,
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
# TRAIN MODEL
# ============================================================

print("\n")
print("=" * 70)
print("STARTING CNN TRAINING")
print("=" * 70)


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
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
# TRAINING GRAPHS
# ============================================================

print("\nGenerating training graphs...")


plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training and Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

accuracy_graph = os.path.join(
    REPORT_DIR,
    "accuracy_graph.png"
)

plt.savefig(
    accuracy_graph,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# Loss graph

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid(True)

loss_graph = os.path.join(
    REPORT_DIR,
    "loss_graph.png"
)

plt.savefig(
    loss_graph,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TEST MODEL
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATING MODEL ON TEST DATA")
print("=" * 70)


test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)


print("\nTest Loss     :", round(test_loss, 4))
print("Test Accuracy :", round(test_accuracy * 100, 2), "%")


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
    digits=4
)

print("\nClassification Report")
print("=" * 70)
print(report)


report_path = os.path.join(
    REPORT_DIR,
    "classification_report.txt"
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
        "Building Damage Classification\n\n"
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


plt.figure(
    figsize=(8, 7)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title(
    "Confusion Matrix - Building Damage"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()


cm_path = os.path.join(
    REPORT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    REPORT_DIR,
    "training_history.json"
)

with open(
    history_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            key: [
                float(value)
                for value in values
            ]
            for key, values
            in history.history.items()
        },
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nReports:")
print(REPORT_DIR)

print("\nTest Accuracy:")
print(
    f"{test_accuracy * 100:.2f}%"
)

print("\n✅ CNN training pipeline completed!")