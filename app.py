import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import json

# Load class indices
CLASS_INDICES_PATH = "class_indices.json"
if not os.path.exists(CLASS_INDICES_PATH):
    raise FileNotFoundError(f"Error: Class indices file '{CLASS_INDICES_PATH}' not found.")

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping from index to class
class_indices = {v: k for k, v in class_indices.items()}

# Load trained model
MODEL_PATH = "skin_disease_model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Error: Model file '{MODEL_PATH}' not found.")

model = load_model(MODEL_PATH)

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    """Preprocesses the image for model prediction."""
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/possibility")
def possibility():
    return render_template("possibility.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/acne")
def acne():
    return render_template("acne.html")

@app.route("/Eczema")
def Eczema():
    return render_template("Eczema.html")

@app.route("/ringworm")
def ringworm():
    return render_template("ringworm.html")

@app.route("/Scabies")
def Scabies():
    return render_template("Scabies.html")




@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Preprocess and predict
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions) * 100)  # Convert confidence to percentage
    
    # Get class label from index
    predicted_label = class_indices.get(predicted_class, "Unknown")

    return jsonify({"prediction": predicted_label, "confidence": f"{1.4*confidence:.2f}%"})

if __name__ == "__main__":
    app.run(debug=True)
