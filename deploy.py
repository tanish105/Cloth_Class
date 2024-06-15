from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = pickle.load(open('clothing_classification.pkl', 'rb'))

def preprocess_image(image):
    # Convert PIL image to numpy array
    image = image.resize((28, 28))  # Adjust size as needed
    image = image.convert('L')  # Convert to grayscale if needed
    image = np.array(image)
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    image = image.flatten()  # Flatten to match the model input
    return image

@app.route('/')
def home():
    return "Clothing Classification Model API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))
        processed_image = preprocess_image(image)

        prediction = model.predict([processed_image])
        return jsonify({'prediction': list(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
