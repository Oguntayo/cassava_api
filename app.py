import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)
# Load the model
model_url = 'https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2'
try:
    model = hub.KerasLayer(model_url)
except Exception as e:
    print(f"Failed to load the model: {e}")
    exit()

# Class mapping
class_names = ['Bacterial Blight', 'Brown Streak Disease', 'Green Mite', 'Mosaic Disease', 'Healthy', 'Unknown']

@app.route('/')
def home():
    return "Cassava Disease Detection API"

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "API is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {e}"}), 400

    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model(image)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    confidence_scores = tf.nn.softmax(predictions).numpy()[0]

    result = {
        "predicted_class": class_names[predicted_class],
        "confidence_scores": {class_names[i]: float(score) for i, score in enumerate(confidence_scores)}
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
