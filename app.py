from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
from  collections import defaultdict
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from flask import send_file
import json
import os
import time
import datetime
import json

DATA_FILE = "patients.json"

def load_patients():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_patients(patients):
    with open(DATA_FILE, "w") as f:
        json.dump(patients, f)

current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
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
    now = datetime.datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S.%f")[:-3]   # milliseconds
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
        "image_path": file_path 
        
    }

    # store list of patients
    if "patients" not in session:
        session["patients"] = []

    patients = load_patients()
    patients.append(patient)
    save_patients(patients)

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
    patients = load_patients()

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

@app.route('/download_report')
def download_report():

    patients = load_patients()   # ✅ FIX
    if not patients:
        return "No data available"

    patient = patients[-1]

    pdf_path = os.path.join(os.getcwd(), "report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)

    styles = getSampleStyleSheet()
    content = []

    from reportlab.lib import colors

    # Title
    content.append(Paragraph("<b><font size=16>AI Medical Diagnostic Report</font></b>", styles['Title']))
    content.append(Spacer(1, 10))

    # Report Info
    content.append(Paragraph(f"<b>Report ID:</b> {patient['id']}", styles['Normal']))
    content.append(Paragraph(f"<b>Date:</b> {patient['date']} {patient['time']}", styles['Normal']))
    content.append(Spacer(1, 10))

    # Patient Section
    content.append(Paragraph("<b>Patient Information</b>", styles['Heading2']))
    content.append(Paragraph(f"Name: {patient['name']}", styles['Normal']))
    content.append(Paragraph(f"Age: {patient['age']}", styles['Normal']))
    content.append(Paragraph(f"Phone: {patient['phone']}", styles['Normal']))
    content.append(Paragraph(f"Location: {patient['place']}", styles['Normal']))
    content.append(Spacer(1, 10))

    # Severity Color Logic
    severity = patient['label']
    if severity == "Critical":
        sev_color = "red"
    elif severity == "Moderate":
        sev_color = "orange"
    else:
        sev_color = "green"

    # Prediction Section
    content.append(Paragraph("<b>Prediction Summary</b>", styles['Heading2']))
    content.append(Paragraph(
        f"<b>Severity:</b> <font color='{sev_color}'><b>{severity}</b></font>",
        styles['Normal']
    ))
    content.append(Paragraph(
        f"<b>Confidence:</b> {round(patient['confidence']*100,2)}%",
        styles['Normal']
    ))
    content.append(Paragraph(
        f"<b>Clinical Note:</b> {patient['explanation']}",
        styles['Normal']
    ))
    content.append(Spacer(1, 15))

    # Image Section
    content.append(Paragraph("<b>Chest X-ray Image</b>", styles['Heading2']))
    try:
        img = RLImage(patient['image_path'], width=250, height=250)
        content.append(img)
    except:
        content.append(Paragraph("Image not available", styles['Normal']))

    content.append(Spacer(1, 20))

    # Recommendation Section
    content.append(Paragraph("<b>Recommendation</b>", styles['Heading2']))
    content.append(Paragraph(patient['result'], styles['Normal']))
    content.append(Spacer(1, 15))

    doc.build(content)

    return send_file(pdf_path, as_attachment=True, download_name="Medical_Report.pdf")

@app.route('/download_report/<int:id>')
def download_report_by_id(id):

    patients = load_patients()

    # 🔍 Find selected patient
    patient = None
    for p in patients:
        if p["id"] == id:
            patient = p
            break

    if not patient:
        return "Patient not found"

    # 📄 Create PDF path
    pdf_path = os.path.join(os.getcwd(), f"report_{id}.pdf")

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    from reportlab.lib import colors

    # 🏥 Title
    content.append(Paragraph("<b><font size=16>AI Medical Diagnostic Report</font></b>", styles['Title']))
    content.append(Spacer(1, 10))

    # 📌 Report Info
    content.append(Paragraph(f"<b>Report ID:</b> {patient['id']}", styles['Normal']))
    content.append(Paragraph(f"<b>Date:</b> {patient['date']} {patient['time']}", styles['Normal']))
    content.append(Spacer(1, 10))

    # 👤 Patient Info
    content.append(Paragraph("<b>Patient Information</b>", styles['Heading2']))
    content.append(Paragraph(f"Name: {patient['name']}", styles['Normal']))
    content.append(Paragraph(f"Age: {patient['age']}", styles['Normal']))
    content.append(Paragraph(f"Phone: {patient['phone']}", styles['Normal']))
    content.append(Paragraph(f"Place: {patient['place']}", styles['Normal']))
    content.append(Spacer(1, 10))

    # 🎯 Severity color logic
    severity = patient['label']
    if severity == "Critical":
        sev_color = "red"
    elif severity == "Moderate":
        sev_color = "orange"
    else:
        sev_color = "green"

    # 🧠 Prediction Section
    content.append(Paragraph("<b>Prediction Summary</b>", styles['Heading2']))
    content.append(Paragraph(
        f"<b>Severity:</b> <font color='{sev_color}'><b>{severity}</b></font>",
        styles['Normal']
    ))
    content.append(Paragraph(
        f"<b>Confidence:</b> {round(patient['confidence']*100,2)}%",
        styles['Normal']
    ))
    content.append(Paragraph(
        f"<b>Explanation:</b> {patient['explanation']}",
        styles['Normal']
    ))
    content.append(Spacer(1, 15))

    # 🖼️ X-ray Image
    content.append(Paragraph("<b>Chest X-ray</b>", styles['Heading2']))
    try:
        img = RLImage(patient['image_path'], width=250, height=250)
        content.append(img)
    except:
        content.append(Paragraph("Image not available", styles['Normal']))

    content.append(Spacer(1, 15))

    # 💊 Recommendation
    content.append(Paragraph("<b>Recommendation</b>", styles['Heading2']))
    content.append(Paragraph(patient['result'], styles['Normal']))
    content.append(Spacer(1, 15))


    # 🏗️ Build PDF
    doc.build(content)

    # 📥 Send file
    return send_file(pdf_path, as_attachment=True, download_name=f"Report_{id}.pdf")




# 🔥 DELETE PATIENT
@app.route("/delete/<int:id>")
def delete(id):
    patients = load_patients()

    patients = [p for p in patients if p["id"] != id]

    save_patients(patients)

    return redirect(url_for("dashboard"))

# 🔥 UPDATE STATUS
@app.route("/update_status/<int:id>")
def update_status(id):
    patients = load_patients()

    for p in patients:
        if p["id"] == id:
            p["status"] = "Completed" if p["status"] == "Pending" else "Pending"

    save_patients(patients)

    return redirect(url_for("dashboard"))


# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)

