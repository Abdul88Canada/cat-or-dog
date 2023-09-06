from flask import Flask, request, jsonify
import requests
import base64
import numpy as np
import json
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing import image as keras_image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the image from the request
        img_data = request.json.get('image')
        image_data = Image.open(BytesIO(base64.b64decode(img_data.split(",")[1])))
        image_data = image_data.resize((64, 64), Image.LANCZOS)
        test_image = keras_image.img_to_array(image_data)
        #test_image = test_image * (1. / 255)  # Normalize to [0,1]
        test_image = np.expand_dims(test_image, axis=0)
        
        # Convert to list to send as JSON payload
        test_image_list = test_image.tolist()

        # Create payload
        payload = json.dumps({
            "signature_name": "serving_default",
            "instances": test_image_list
        })
        
        # Server URL
        url = 'http://localhost:8501/v1/models/img_classifier:predict'

        # Send request
        headers = {"content-type": "application/json"}
        response = requests.post(url, data=payload, headers=headers)

        # Decode the response JSON and get the result
        result = json.loads(response.content.decode('utf-8'))

        return jsonify({'prediction': result['predictions']})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
