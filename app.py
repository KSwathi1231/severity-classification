from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import json
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.secret_key = "swathi-secret-key"

# ------------------------------
# Load model and labels
# ------------------------------
MODEL_PATH = "resnet50_xray_full_model.h5"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = 224

model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "r") as f:
    idx_to_class = json.load(f)

idx_to_class = {int(k): v for k, v in idx_to_class.items()}


# ------------------------------
# Helper functions
# ------------------------------
def prepare_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def message_for(label):
    label = label.lower()

    if label == "normal":
        return "🟢You are absolutely normal."
    elif label == "moderate":
        return "🟡Your case can be cured by regular treatment and medicines under a doctor’s guidance."
    elif label == "critical":
        return "🔴You should immediately consult a doctor for proper diagnosis and treatment."
    else:
        return "Please consult a doctor for a detailed evaluation."

def get_explanation(severity):
    if severity == "Critical":
        return (
            "The chest X-ray indicates severe abnormal patterns in lung regions, "
            "suggesting extensive involvement that requires immediate medical attention. "
        )
    elif severity == "Moderate":
        return (
            "The chest X-ray shows noticeable abnormalities that may indicate a "
            "treatable respiratory condition. Medical consultation and regular "
            "treatment are advised to prevent progression."
        )
    else:
        return (
            "The chest X-ray does not show significant abnormalities. "
            "Lung structures appear normal, and no immediate medical concern is detected."
        )




# ------------------------------
# Routes
# ------------------------------

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/start")
def start():
    return render_template("index.html")   # form page only


@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    phone = request.form["phone"]
    age = request.form["age"]
    place = request.form["place"]

    file = request.files["xray"]

    # Ensure uploads folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    img = prepare_image(file_path)
    preds = model.predict(img)
    pred_idx = int(np.argmax(preds[0]))
    predicted_label = idx_to_class[pred_idx]

    message = message_for(predicted_label)
    explanation = get_explanation(predicted_label)

    # store result temporarily
    session["result"] = {
        "name": name,
        "phone": phone,
        "age": age,
        "place": place,
        "label": predicted_label,
        "result": message,
        "explanation": explanation
    }

    return redirect(url_for("result"))

    return redirect(url_for("result"))   # go to result page


@app.route("/result")
def result():
    data = session.get("result")

    if not data:
        return redirect(url_for("start"))

    return render_template("result.html", **data)


# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
