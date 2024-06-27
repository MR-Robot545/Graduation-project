
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

photos_directory = "data"
@app.route('/pre', methods=['GET'])
def preprocess():
    image_data = []
    target_labels = []
    for filename in os.listdir(photos_directory):
        if filename.endswith(".BMP"):
            subject_id, gender, lr, finger = filename.split('_')
            finger ,_ = finger.split('.')
            gender = 0 if gender == 'M' else 1
            lr = 0 if lr == 'Left' else 1
            # print(finger)
            if finger == 'thumb':
                finger = 0
            elif finger == 'index':
                finger = 1
            elif finger == 'middle':
                finger = 2
            elif finger == 'ring':  #ring
                finger = 3
            elif finger == 'little':
                finger = 4
            else:
                finger = 5
            # print(subject_id," : ",finger)
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
    # return image_data,target_labels
    np.save("image_data.npy", image_data)
    np.save("target_labels.npy", target_labels)
    return "success"
def generate_filename(file):
    user_id = request.form.get('user_id')  # Example: '100'
    gender = request.form.get('gender')  # Example: 'M'
    finger_type = request.form.get('finger_type')  # Example: 'Index'
    lr = request.form.get('lr')
    filename = f"{user_id}_{gender}_{lr}_{finger_type}.BMP"
    return filename

#this function for adding an image for childerns (right and foot)
def generate_newname(file):
    user_id = request.form.get('user_id')  # Example: '100'
    gender = request.form.get('gender')  # Example: 'M'
    finger_type = 'foot'  # Example: 'Index'
    lr = 'Right'
    filename = f"{user_id}_{gender}_{lr}_{finger_type}.BMP"
    return filename

def replace_name(existing_img):
    print(existing_img)
    gender = 'M' if existing_img[1] == '0' else 'F'
    lr = 'Left' if existing_img[2] == '0' else 'Right'

    if existing_img[3] == '0':
        finger = 'thumb'
    elif existing_img[3] == '1':
        finger = 'index'
    elif existing_img[3] == '2':
        finger = 'middle'
    elif existing_img[3] == '3':
        finger = 'ring'
    elif existing_img[3] == '4':
        finger = 'little'
    else:
        finger = 'foot'
    # Rename the new image to match the existing one
    old_filename = f"{existing_img[0]}_{gender}_{lr}_{finger}.BMP"
    gender_new = gender
    lr_new = 'Right'
    finger_new = 'thumb'
    new_filename = f"{existing_img[0]}_{gender_new}_{lr_new}_{finger_new}.BMP"
    print(old_filename, new_filename)
    return old_filename, new_filename

def Add(file):
    if file.filename == '':
        return 'No selected file'
    filename = generate_newname(file)
    if not os.path.exists(photos_directory):
        os.makedirs(photos_directory)

    file.save(os.path.join(photos_directory, filename))
    print(filename)
    preprocess()
    return filename

def check(user_id):
    target = np.load("target_labels.npy")
    for i in range(len(target)):
        if target[i][0] == user_id:
            return target[i]

    return None
def update(user_id, file2):
    if file2.filename == '':
        return 'No selected file'

    # Execute the search to find the existing image
    existing_img = check(user_id)
    # print(existing_img)
    if existing_img is None:
        return 'The provided image does not match any existing image.'

    old_filename, new_filename = replace_name(existing_img)
    print(old_filename)
    # Remove the existing image
    os.remove(os.path.join(photos_directory, old_filename))

    # Save the new image with the existing image's name
    file2.save(os.path.join(photos_directory, new_filename))

    preprocess()

    return jsonify({'message': 'File updated successfully' ,'filename': new_filename})

def exe(image):
    img = Image.open(image)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    resized_photo = img.resize((90, 144))
    img_np_array = np.array(resized_photo)
    gray_image = tf.image.rgb_to_grayscale(img_np_array)  # Convert to grayscale using OpenCV
    gray_image = np.expand_dims(gray_image, axis=-1)
    real_photo = gray_image.astype(np.float32)
    # image_data,target=preprocess()
    image_data = np.load("image_data.npy")
    target = np.load("target_labels.npy")
    ans = 0
    any = None
    print(len(image_data))
    for i in range(len(image_data)):
        pre = model.predict([real_photo.reshape((1, 144, 90, 1)).astype(np.float32 ) /255.,
                             image_data[i].reshape((1, 144, 90, 1)).astype(np.float32 ) /255.])
        if ans < pre:
            any = target[i]
            ans = pre

    return any, ans

# Endpoint to receive and process search images
@app.route('/search', methods=['POST'])
def search():

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']

    returned_img, predict = exe(image_file)

    search_result = {
        'index': returned_img[0],
        'accuracy': float(predict)

    }
    print(55)
    return jsonify(search_result)
@app.route('/Add', methods=['POST'])
def Add_Image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filename = Add(file)
    return jsonify({'message': 'File uploaded successfully', 'filename': filename})
@app.route('/Update', methods=['POST'])
def Update_Image():
    # if 'image' not in request.files:
    #     return jsonify({'error': 'No image provided'}), 400
    if 'new_image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file1 = request.files['new_image']
    user_id = request.form.get('user_id')

    return update(user_id, file1)

if __name__ == '__main__':
    app.run(debug=True)
