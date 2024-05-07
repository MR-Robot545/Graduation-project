from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
from PIL import Image
import io
from skimage.color import rgb2gray
import os
import shutil
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

def load_model(filepath):
    # Load the model architecture
    with open(filepath + '_architecture.json', 'r') as f:
        loaded_model = model_from_json(f.read())
    # Load the model weights
    loaded_model.load_weights(filepath + '_weights.h5')
    return loaded_model
# Load the fingerprint recognition model
model = load_model('good_model') # Update with your model file path

# Define finger mapping
finger_mapping = {0: 'thumb', 1: 'index', 2: 'middle', 3: 'ring', 4: 'little'}

# Store image data and corresponding labels
stored_images = []
stored_labels = []
#
# @app.route('/add_child', methods=['POST'])
# def add_child():
#     # Receive image data and label from the desktop application
#     image_data = request.files['image'].read()
#     label = request.form['label']
#     # Preprocess the image
#     img_np_array = np.array(Image.open(io.BytesIO(image_data)))
#     gray_image = rgb2gray(img_np_array)
#     gray_image = np.expand_dims(gray_image, axis=-1)
#     real_photo = gray_image.astype(np.float32)
#     # Add image data and label to the storage
#     stored_images.append(real_photo)
#     stored_labels.append(label)
#     return jsonify({'message': 'Child added successfully!'})

@app.route('/predict_similarity', methods=['POST'])
def predict_similarity():
    # Receive image data from the desktop application
    image_data = request.files['image'].read()
    # Preprocess the image
    img_np_array = np.array(Image.open(io.BytesIO(image_data)))
    gray_image = rgb2gray(img_np_array)
    gray_image = np.expand_dims(gray_image, axis=-1)
    real_photo = gray_image.astype(np.float32)
    # Make predictions for similarity
    predictions = []
    for stored_image in stored_images:
        similarity_score = model.predict(np.array([real_photo, stored_image]))[0][0]
        predictions.append(similarity_score)
    return jsonify({'predictions': predictions})
#
# @app.route('/delete_child/<string:label>', methods=['DELETE'])
# def delete_child(label):
#     try:
#         index = stored_labels.index(label)
#         del stored_images[index]
#         del stored_labels[index]
#         return jsonify({'message': 'Child deleted successfully!'})
#     except ValueError:
#         return jsonify({'error': 'Child not found!'})

if __name__ == '__main__':
    app.run(debug=True)