from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
print(f"Model path: {model_path}")  # Debug print
try:
    model = joblib.load(model_path)
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    try:
        df = pd.DataFrame(data)
        prediction = model.predict(df)
        return jsonify(prediction.tolist())
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
