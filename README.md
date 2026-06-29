# MediScan — Chest X-ray Severity Classification

![MediScan Healthcare](static/health.jpeg)

## Overview

MediScan is a lightweight Flask web application for chest X-ray diagnosis using a pre-trained ResNet50 model. It classifies X-ray scans into three categories: `Critical`, `Moderate`, and `Normal`, then generates a patient summary report and stores patient history in a local JSON database.

This app is designed for rapid prototyping of AI-assisted medical diagnosis workflows and includes an upload form, prediction result page, dashboard, and PDF report generation.

## Key Features

- Upload patient details and chest X-ray image
- AI-powered severity prediction using a Keras/TensorFlow model
- Confidence score displayed with every prediction
- Patient history dashboard sorted by severity and confidence
- Downloadable PDF medical report for each patient
- Status management for patient records (`Pending` / `Completed`)

## Project Structure

```
Chest X-ray app/
├── app.py
├── class_indices.json
├── patients.json
├── requirements.txt
├── resnet50_xray_full_model.h5
├── runtime.txt
├── static/
│   └── health.jpeg
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── home.html
│   ├── index.html
│   ├── result.html
│   └── upload.html
└── uploads/
```

## How It Works

1. User opens the app and navigates to the upload form.
2. Patient details and an X-ray image are submitted.
3. The app preprocesses the image and passes it to the model.
4. The model returns a predicted label and confidence score.
5. The result is saved in `patients.json` and displayed to the user.
6. Admin dashboard shows recent records and allows report downloads.

## Prediction Classes

The classification model returns one of the following labels:

- `Critical` — immediate doctor consultation required
- `Moderate` — treatable with proper medication
- `Normal` — no major abnormalities detected

## Dependency Summary

The application depends on:

- Flask
- gunicorn
- tensorflow-cpu
- numpy
- Pillow
- reportlab

## Installation

### 1. Create a Python environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

### 4. Open in browser

Visit `http://127.0.0.1:5000/`

## Usage

- Open the home page and click **Get Started**.
- Fill in patient details and upload the chest X-ray image.
- Submit the form to view the prediction result.
- Use the dashboard to review recent patients and download PDF reports.

## Routes

- `/` — Home page
- `/upload` — Upload patient details and X-ray image
- `/predict` — Model prediction endpoint
- `/result` — Display prediction result
- `/dashboard` — Patient history dashboard
- `/download_report` — Download the last report as PDF
- `/download_report/<id>` — Download a report for a specific patient
- `/delete/<id>` — Delete a patient record
- `/update_status/<id>` — Toggle patient record status

## Data Storage

- Patient records are stored in `patients.json`.
- Uploaded images are saved in the `uploads/` folder.
- Reports are generated as PDF files in the app root.

## Example Workflow

```text
[User] -> Upload form -> submit patient + X-ray
      -> app.py preprocesses image
      -> model predicts label + confidence
      -> Save record to patients.json
      -> Show result page + dashboard entry
      -> Download PDF report if needed
```

## Notes

- The model file `resnet50_xray_full_model.h5` must remain in the app root.
- This project is intended for demonstration and educational purposes, not clinical decision-making.
- Ensure the `uploads/` folder exists and is writable.

## Customization

To adapt the app for a different X-ray dataset or model:

- Replace `resnet50_xray_full_model.h5` with a compatible Keras model.
- Update `class_indices.json` if the label mapping changes.
- Adjust `IMG_SIZE` in `app.py` if the model expects a different input size.

## License

This repository does not include a formal license file. Use and modify the project at your own discretion.
