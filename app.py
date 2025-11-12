# -----------------------------------------------------------
# AI-Based Traffic Sign Recognition and Driver Alert System
# Author: M. Monika
# -----------------------------------------------------------

from flask import Flask, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)

MODEL_PATH = 'traffic_model.h5'
model = load_model(MODEL_PATH)

CLASS_LABELS = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at intersection',
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Dangerous curve left', 'Dangerous curve right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
    'Keep left', 'Roundabout mandatory', 'End of no passing', 'End no passing 3.5 tons'
]

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return "<h2>AI-Based Traffic Sign Recognition System</h2><p>Use /predict_image to upload an image.</p>"

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    sign = CLASS_LABELS[class_index]
    confidence = round(np.max(predictions) * 100, 2)

    return f"Predicted Sign: {sign} (Confidence: {confidence}%)"

if __name__ == '__main__':
    app.run(debug=True)
