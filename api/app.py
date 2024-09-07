from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-Cors
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to your model file
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({'error': 'Invalid data format'}), 400
        
        features = [d.get('features') for d in data]
        prediction = model.predict(features)
        return jsonify(prediction.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
