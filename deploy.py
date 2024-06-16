from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

model = load_model('clothing_classification.h5')

def preprocess_image(image):
    img = Image.open(image).convert('L').resize((28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img

@app.route('/')
def home():
    return "Clothing Classification Model API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    img = preprocess_image(file)
    prediction = model.predict(img)
    prediction = np.argmax(prediction, axis=1)
    predicted_class = prediction[0]
    
    class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    predicted_class_name = class_labels[predicted_class]

    return jsonify({'predicted_class': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)
