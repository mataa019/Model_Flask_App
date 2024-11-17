from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import subprocess

app = Flask(__name__)

# Kaggle dataset details
dataset = 'mataajohn/cassava_disease_detection'
model_file = 'Cassava_Disease_Model.h5'
model_path = 'Cassava_Disease_Model.h5'

# Download the model if it does not exist
if not os.path.exists(model_path):
    subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset, '-p', '.', '--unzip'])

# Load your .h5 model
model = load_model(model_path)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 224))  # Resize the image to the input size of your model
        img = np.array(img) / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)

        # Perform prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Return the prediction
        return jsonify({'predicted_class': int(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)