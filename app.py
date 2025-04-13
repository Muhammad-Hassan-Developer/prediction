from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import pickle

app = Flask(__name__)

# Enable CORS to allow cross-origin requests from all domains
CORS(app)

# Load your uploaded model (make sure 'Heart Disease Predictor.pkl' is in the same directory)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "✅ Heart Disease Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the POST request (should be JSON)
        data = request.get_json(force=True)

        # Input features expected (make sure these match the features the model was trained on)
        features = [
            data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
            data['fbs'], data['restecg'], data['thalach'], data['exang'],
            data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]

        # Convert the input data into a format suitable for the model (1D array to 2D)
        input_data = np.array(features).reshape(1, -1)

        # Make the prediction using the loaded model
        prediction = model.predict(input_data)

        # Return the prediction in JSON format
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        # Return an error message if something goes wrong
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the app on the default port (5000) and listen on all IPs (0.0.0.0)
    app.run(debug=True, host='0.0.0.0', port=5000)
