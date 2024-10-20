from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Create an uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = tf.keras.models.load_model('trained_model.keras')

# Allowed extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Reduce noise
    img = cv2.resize(img, (128, 128))  # Resize image to (128,128)
    img = img / 255.0  # Normalize to [0, 1]
    return img

@app.route('/')
def upload_form():
    return render_template('upload.html')  # Render the upload form

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the image and make prediction
        img = cv2.imread(file_path)
        img = preprocess_image(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)

        # Here you should map predicted_class to actual class names
        # Example: class_names = ['Healthy', 'Disease1', 'Disease2', ...]
        # predicted_label = class_names[predicted_class[0]]
        
        return jsonify({
            'message': 'File uploaded successfully',
            'predicted_class': int(predicted_class[0])  # Replace with predicted_label if using class names
        })
    return 'File type not allowed'

if __name__ == '__main__':
    app.run(debug=True)
