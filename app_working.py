from flask import Flask, render_template_string, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# HTML Template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 { color: #333; text-align: center; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background: #667eea;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
            font-size: 16px;
        }
        button:hover { background: #764ba2; }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            display: none;
        }
        .result.show { display: block; }
        .risk-high { background: #fee; border-left: 4px solid #e74c3c; }
        .risk-medium { background: #ffeaa7; border-left: 4px solid #f39c12; }
        .risk-low { background: #d5f4e6; border-left: 4px solid #27ae60; }
        .probability { font-size: 24px; font-weight: bold; text-align: center; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Customer Churn Prediction</h1>
        <form id="predictionForm">
            <div class="form-group">
                <label>Tenure (months)</label>
                <input type="number" id="tenure" required value="12">
            </div>
            <div class="form-group">
                <label>Monthly Charges ($)</label>
                <input type="number" step="0.01" id="monthly_charges" required value="70">
            </div>
            <div class="form-group">
                <label>Contract Type</label>
                <select id="contract">
                    <option value="Month-to-month">Month-to-month</option>
                    <option value="One year">One year</option>
                    <option value="Two year">Two year</option>
                </select>
            </div>
            <div class="form-group">
                <label>Internet Service</label>
                <select id="internet_service">
                    <option value="DSL">DSL</option>
                    <option value="Fiber optic">Fiber optic</option>
                    <option value="No">No</option>
                </select>
            </div>
            <button type="submit">Predict Churn Risk</button>
        </form>
        <div id="result" class="result"></div>
    </div>
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = {
                tenure: parseInt(document.getElementById('tenure').value),
                monthly_charges: parseFloat(document.getElementById('monthly_charges').value),
                contract: document.getElementById('contract').value,
                internet_service: document.getElementById('internet_service').value,
                gender: 'Male', partner: 'No', dependents: 'No', phone_service: 'Yes',
                multiple_lines: 'No', online_security: 'No', online_backup: 'No',
                device_protection: 'No', tech_support: 'No', streaming_tv: 'No',
                streaming_movies: 'No', paperless_billing: 'Yes', payment_method: 'Electronic check',
                senior_citizen: 0, total_charges: document.getElementById('tenure').value * document.getElementById('monthly_charges').value
            };
            const response = await fetch('/predict', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(formData)
            });
            const result = await response.json();
            const resultDiv = document.getElementById('result');
            resultDiv.classList.add('show');
            const riskClass = result.risk_level === 'High' ? 'risk-high' : result.risk_level === 'Medium' ? 'risk-medium' : 'risk-low';
            resultDiv.className = `result ${riskClass}`;
            resultDiv.innerHTML = `
                <h3>Prediction Results</h3>
                <div class="probability">Churn Probability: ${(result.churn_probability * 100).toFixed(1)}%</div>
                <div>Risk Level: <strong>${result.risk_level}</strong></div>
                <div>Retention Probability: ${(result.retention_probability * 100).toFixed(1)}%</div>
                <br>
                <div>${result.risk_level === 'High' ? '⚠️ Immediate action recommended!' : result.risk_level === 'Medium' ? '📊 Monitor closely' : '✅ Keep up the good work!'}</div>
            `;
        });
    </script>
</body>
</html>
'''

# Load the trained model and preprocessors
model = None
label_encoders = None
scaler = None

try:
    if os.path.exists('models/best_churn_model.pkl'):
        model = joblib.load('models/best_churn_model.pkl')
        print("✅ Loaded trained model")
    
    if os.path.exists('models/label_encoders.pkl'):
        label_encoders = joblib.load('models/label_encoders.pkl')
        print("✅ Loaded label encoders")
    
    if os.path.exists('models/scaler.pkl'):
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Loaded scaler")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")

def predict_churn(customer_data):
    """Make prediction using the trained model"""
    if model is None:
        # Fallback rule-based prediction
        risk = 0.2
        if customer_data.get('tenure', 12) < 12:
            risk += 0.3
        if customer_data.get('contract') == 'Month-to-month':
            risk += 0.3
        if customer_data.get('internet_service') == 'Fiber optic':
            risk += 0.15
        if customer_data.get('monthly_charges', 70) > 80:
            risk += 0.1
        risk = min(risk, 0.95)
        
        return {
            'churn_prediction': bool(risk > 0.5),
            'churn_probability': float(risk),
            'retention_probability': float(1 - risk),
            'risk_level': 'High' if risk > 0.7 else 'Medium' if risk > 0.3 else 'Low'
        }
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        
        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0]
        
        return {
            'churn_prediction': bool(prediction[0]),
            'churn_probability': float(probability[1]),
            'retention_probability': float(probability[0]),
            'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.3 else 'Low'
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'churn_prediction': False,
            'churn_probability': 0.2,
            'retention_probability': 0.8,
            'risk_level': 'Low'
        }

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        result = predict_churn(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 Customer Churn Prediction System")
    print("=" * 50)
    print("\n✅ Model Status: " + ("Trained ML Model Loaded" if model else "Rule-based Predictor Active"))
    print("\n📱 Open your browser and navigate to:")
    print("   http://localhost:5000")
    print("\n" + "=" * 50)
    print("✅ Server is running on http://localhost:5000")
    print("Press CTRL+C to stop the server\n")
    
    app.run(debug=True, port=5000)