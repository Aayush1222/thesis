from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Initialize Flask app
app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dataset folders
CUT_DIR = 'cricket shot/cut'
SWEEP_DIR = 'cricket shot/sweep'
DRIVE_DIR = 'cricket shot/drive'

# Load DensePose model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = keypointrcnn_resnet50_fpn(pretrained=True)
model.eval().to(device)

# Function to extract keypoints using DensePose
def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)

    # Run DensePose model
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract keypoints from predictions
    if len(predictions[0]['keypoints']) == 0:
        return None

    keypoints = predictions[0]['keypoints'][0].cpu().numpy()
    keypoints_flattened = keypoints[:, :2].flatten().tolist()  # Use x, y only
    return keypoints_flattened

# Train or load the model
def train_model():
    def load_data(directory, label):
        files = os.listdir(directory)
        data = []
        for file in files:
            file_path = os.path.join(directory, file)
            keypoints = extract_keypoints(file_path)
            if keypoints:
                data.append(keypoints + [label])  # Append label at the end
        return pd.DataFrame(data)

    print("Training the model...")

    # Load datasets
    cut_data = load_data(CUT_DIR, label=1)
    sweep_data = load_data(SWEEP_DIR, label=0)
    drive_data = load_data(DRIVE_DIR, label=2)

    # Combine datasets
    data = pd.concat([cut_data, sweep_data, drive_data], ignore_index=True)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Train RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    # Save the model
    joblib.dump(clf, 'cricket_shot_classifier.pkl')
    print("Model training complete.")
    return clf

# Load model or train if not available
if os.path.exists('cricket_shot_classifier.pkl'):
    clf = joblib.load('cricket_shot_classifier.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded.")
else:
    clf = train_model()
    scaler = joblib.load('scaler.pkl')

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'})

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Extract keypoints and predict
    keypoints = extract_keypoints(file_path)
    if not keypoints:
        return jsonify({
            'error': 'No pose detected in the uploaded image.',
            'image_url': f'/uploads/{filename}'
        })

    keypoints_scaled = scaler.transform([keypoints])
    prediction = clf.predict(keypoints_scaled)
    prediction_label = ['Sweep', 'Cut', 'Drive'][prediction[0]]

    return jsonify({
        'prediction': prediction_label,
        'image_url': f'/uploads/{filename}'
    })
if __name__ == '__main__':
    app.run(debug=True)