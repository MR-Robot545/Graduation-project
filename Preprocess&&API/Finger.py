import json
from keras.models import model_from_json
import tensorflow as tf
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import os
import numpy as np

def save_model(model, filepath):
    # Save the model weights and architecture
    model.save_weights(filepath + '_weights.h5')
    with open(filepath + '_architecture.json', 'w') as f:
        f.write(model.to_json())
def load_model(filepath):
    # Load the model architecture
    with open(filepath + '_architecture.json', 'r') as f:
        loaded_model = model_from_json(f.read())
    # Load the model weights
    loaded_model.load_weights(filepath + '_weights.h5')
    return loaded_model

model = load_model('good_model')

app = Flask(__name__)

photos_directory = "D:/STUDY/Fouth Year/GP/API_Model/Graduation-project/Preprocess&&API/data/"
def preprocess():
    image_data = []
    target_labels = []
    for filename in os.listdir(photos_directory):
        if filename.endswith(".BMP"):
            subject_id, gender, lr, finger = filename.split('_')
            gender = 0 if gender == 'M' else 1
            lr = 0 if lr == 'Left' else 1

            if finger == 'thumb':
                finger = 0
            elif finger == 'index':
                finger = 1
            elif finger == 'middle':
                finger = 2
            elif finger == 'ring':
                finger = 3
            elif finger == 'little':
                finger = 4
            else:
                finger = 5
            target_labels.append([subject_id, gender, lr, finger])
            path = os.path.join(photos_directory, filename)
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            resized_photo = img.resize((90, 144))
            img_np_array = np.array(resized_photo)
            gray_image = tf.image.rgb_to_grayscale(img_np_array)  # Convert to grayscale using OpenCV
            gray_image = np.expand_dims(gray_image, axis=-1)
            real_photo = gray_image.astype(np.float32)
            image_data.append(real_photo)

    image_data = np.array(image_data)
    target_labels = np.array(target_labels)

    np.save("image_data.npy", image_data)
    np.save("target_labels.npy", target_labels)
def generate_filename(file):
    user_id = request.form.get('user_id')  # Example: '100'
    gender = request.form.get('gender')  # Example: 'M'
    finger_type = request.form.get('finger_type')  # Example: 'Index'
    lr = request.form.get('lr')
    filename = f"{user_id}_{gender}_{lr}_{finger_type}.BMP"
    return filename
def exe(image):
    img = Image.open(image)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    resized_photo = img.resize((90, 144))
    img_np_array = np.array(resized_photo)
    gray_image = tf.image.rgb_to_grayscale(img_np_array)  # Convert to grayscale using OpenCV
    gray_image = np.expand_dims(gray_image, axis=-1)
    real_photo = gray_image.astype(np.float32)
    # preprocess()
    image_data = np.load("image_data.npy")
    target = np.load("target_labels.npy")
    ans = 0
    ansy = -1
    print(len(image_data))
    for i in range(len(image_data)):
        pre = model.predict([real_photo.reshape((1, 144, 90, 1)).astype(np.float32),
                             image_data[i].reshape((1, 144, 90, 1)).astype(np.float32)])
        if ans < pre:
            ansy = target[i][0]
            ans = pre

    return ansy,ans

# Endpoint to receive and process search images
@app.route('/search', methods=['POST'])
def search():

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']

    returned_img,predict= exe(image_file)

    search_result = {
        'index': returned_img,
        'accuracy': float(predict)

    }
    return jsonify(search_result)
@app.route('/Add', methods=['POST'])
def Add_Image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return 'No selected file'
    filename = generate_filename(file)
    if not os.path.exists(photos_directory):
        os.makedirs(photos_directory)

    file.save(os.path.join(photos_directory, filename))
    preprocess()
    return jsonify({'message': 'File uploaded successfully', 'filename': filename})

if __name__ == '__main__':
    app.run(debug=True)
