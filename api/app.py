from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Path to your model file
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
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
    app.run(host='0.0.0.0', port=5000, debug=True)
