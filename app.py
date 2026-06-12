from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from src.prediction import ChurnPredictor

app = Flask(__name__)
CORS(app)

# Initialize predictor
predictor = ChurnPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        result = predictor.predict_single(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)