import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Define dataset path
DATA_DIR = "test"  # Change this to your dataset location
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Apply data augmentation
data_gen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0,1] range
    rotation_range=20,  # Randomly rotate images by up to 20 degrees
    width_shift_range=0.1,  # Shift width by up to 10%
    height_shift_range=0.1,  # Shift height by up to 10%
    zoom_range=0.2,  # Randomly zoom by up to 20%
    horizontal_flip=True,  # Flip images horizontally
    validation_split=0.2  # 20% of data for validation
)

# Load training data
train_data = data_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

# Load validation data
val_data = data_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Save class indices for later use
class_indices = train_data.class_indices
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f)

# Load MobileNetV2 (pre-trained model)
base_model = keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base model layers

# Build CNN Model
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(len(train_data.class_indices), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
EPOCHS = 30
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save model
model.save("skin_disease_model.keras")
print("✅ Model saved successfully!")
