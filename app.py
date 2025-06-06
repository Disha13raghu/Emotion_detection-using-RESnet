import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, url_for # url_for added here
import os
import uuid

app = Flask(__name__)

MODEL_PATH = 'model/emotion_detection_resnet_model.h5'
IMAGE_SIZE = (48, 48)
UPLOAD_FOLDER = 'static/uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    pass

def predict_emotion_from_image_path(image_filepath):
    if model is None:
        return "Model not loaded"

    img = Image.open(image_filepath).convert('L')
    img = img.resize(IMAGE_SIZE)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_emotion_number = np.argmax(predictions, axis=1)[0]
    return int(predicted_emotion_number)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None, image_url=None, error_message=None)

@app.route('/predict', methods=['POST'])
def image_prediction():
    predicted_no = None
    uploaded_image_url = None
    error_message = None
    file_path = None

    if model is None:
        error_message = "Prediction failed: Model could not be loaded on server."
    elif 'image' not in request.files:
        error_message = "No image file provided in the request."
    else:
        image_file = request.files['image']

        if image_file.filename == '':
            error_message = "No selected image file."
        elif image_file:
            try:
                unique_filename = str(uuid.uuid4()) + os.path.splitext(image_file.filename)[1]
                file_path = os.path.join(app.root_path, UPLOAD_FOLDER, unique_filename)
                image_file.save(file_path)

                uploaded_image_url = url_for('static', filename=f'uploads/{unique_filename}')

                predicted_no = predict_emotion_from_image_path(file_path)

            except Exception as e:
                error_message = f"An error occurred during prediction: {e}"
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
    
            vals={0 : "Anger" ,
            1 : "Disgust" ,
            2 : "Fear" ,
            3 : "Happiness" ,
            4 : "Sadness" ,
            5 : "Surprise" ,
            6 : "Neutral" ,
            7:"Contempt" }
            
    return render_template('index.html',
                           prediction=vals[predicted_no],
                           image_url=uploaded_image_url,
                           error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)