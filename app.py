from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
from  collections import defaultdict
import json
import os
import time
import datetime

current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
now = datetime.datetime.now()
date_str = now.strftime("%d-%m-%Y")
time_str = now.strftime("%H:%M:%S.%f")[:-3]   # milliseconds
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
# Priority mapping
# ------------------------------
priority_map = {
    "Critical": 3,
    "Moderate": 2,
    "Normal": 1
}

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
        return "🟢 You are absolutely normal."
    elif label == "moderate":
        return "🟡 Treatable with proper medication."
    elif label == "critical":
        return "🔴 Immediate doctor consultation required."
    else:
        return "Consult doctor."


def get_explanation(severity):
    if severity == "Critical":
        return "Severe lung abnormalities detected."
    elif severity == "Moderate":
        return "Noticeable abnormalities, needs treatment."
    else:
        return "No major abnormalities detected."

# ------------------------------
# Routes
# ------------------------------

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


# 🔥 MAIN PREDICTION LOGIC
@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    phone = request.form["phone"]
    age = request.form["age"]
    place = request.form["place"]

    file = request.files["xray"]

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # ---------------- MODEL PREDICTION ----------------
    img = prepare_image(file_path)

    preds = model.predict(img)[0]   # flatten
    pred_idx = int(np.argmax(preds))
    predicted_label = idx_to_class[pred_idx]

    confidence = float(preds[pred_idx])   # ⭐ IMPORTANT

    message = message_for(predicted_label)
    explanation = get_explanation(predicted_label)

    # ---------------- STORE PATIENT ----------------
    patient = {
        "date": date_str,      # ⭐ NEW
        "time": time_str ,
        "id": int(time.time() * 1000),  # unique id
        "name": name,
        "phone": phone,
        "age": age,
        "place": place,
        "label": predicted_label,
        "confidence": confidence,
        "result": message,
        "explanation": explanation,
        "status": "Pending",
        
    }

    # store list of patients
    if "patients" not in session:
        session["patients"] = []

    patients = session["patients"]
    patients.append(patient)
    session["patients"] = patients

    # store last result for result page
    session["result"] = patient

    return redirect(url_for("result"))


@app.route("/result")
def result():
    data = session.get("result")

    if not data:
        return redirect(url_for("upload"))

    # ✅ clear after use
    session.pop("result", None)

    return render_template("result.html", **data)


# 🔥 DASHBOARD (PRIORITY SORTING)

@app.route("/dashboard")
def dashboard():
    patients = session.get("patients", [])

    today = datetime.datetime.now()

    filtered = []

    for p in patients:
        # ✅ FIX: handle missing date
        if "date" not in p:
            continue   # skip old records

        p_date = datetime.datetime.strptime(p["date"], "%d-%m-%Y")

        if (today - p_date).days <= 7:
            if p["status"] == "Pending":
                filtered.append(p)

    # ✅ Sort
    sorted_patients = sorted(
        filtered,
        key=lambda x: (
            priority_map[x["label"]],
            x["confidence"]
        ),
        reverse=True
    )

    # ✅ Group by date
    grouped = defaultdict(list)
    for p in sorted_patients:
        grouped[p["date"]].append(p)

    return render_template("dashboard.html", grouped=grouped)

# 🔥 DELETE PATIENT
@app.route("/delete/<int:pid>")
def delete(pid):
    patients = session.get("patients", [])
    patients = [p for p in patients if p["id"] != pid]
    session["patients"] = patients

    return redirect(url_for("dashboard"))


# 🔥 UPDATE STATUS
@app.route("/update_status/<int:pid>")
def update_status(pid):
    patients = session.get("patients", [])

    for p in patients:
        if p["id"] == pid:
            if p["status"] == "Pending":
                p["status"] = "Completed"
            else:
                p["status"] = "Pending"

    session["patients"] = patients

    return redirect(url_for("dashboard"))


# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)