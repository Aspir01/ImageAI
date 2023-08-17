import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, request
 
model = tf.keras.models.load_model('model_ok.h5')
 
# Определяем классы объектов
classes = ['clean_photo_good_background', 'good_infographics',
                    'clean_photo_other_background', 'clean_photo_image_background',
                    'bad_infographics', 'other_infographics',
                    'clean_photo_bad_background']
 
def predict(image):
    img = cv2.resize(image, (224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return classes[class_index]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def process_image():
    image = request.files['image']
    result = predict(image)
    return result
