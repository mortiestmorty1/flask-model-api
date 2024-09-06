from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = 'model.pkl'
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    
    # Assuming you pass a single record in the request
    prediction = model.predict(df)
    
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
