cat > src/prediction.py << 'EOF'
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Fix the import - add the correct path
from src.data_preprocessing import DataPreprocessor

class ChurnPredictor:
    def __init__(self):
        """Initialize predictor with trained model and preprocessors"""
        try:
            # Try to load the trained model
            if os.path.exists('models/best_churn_model.pkl'):
                self.model = joblib.load('models/best_churn_model.pkl')
                print("✅ Loaded trained ML model")
            else:
                print("⚠️ No trained model found, using rule-based predictor")
                self.model = None
                
            # Load preprocessors
            self.preprocessor = DataPreprocessor()
            if os.path.exists('models/label_encoders.pkl') and os.path.exists('models/scaler.pkl'):
                self.preprocessor.load_preprocessors()
                print("✅ Loaded preprocessors")
            else:
                print("⚠️ No preprocessors found")
                self.preprocessor = None
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            self.model = None
            self.preprocessor = None
    
    def predict_single(self, customer_data):
        """
        Predict churn for a single customer
        
        Args:
            customer_data (dict): Dictionary containing customer features
        
        Returns:
            dict: Prediction results
        """
        # If no trained model, use rule-based prediction
        if self.model is None or self.preprocessor is None:
            return self._rule_based_prediction(customer_data)
        
        try:
            df = pd.DataFrame([customer_data])
            df_processed = self.preprocessor.preprocess_new_data(df)
            prediction = self.model.predict(df_processed)
            probability = self.model.predict_proba(df_processed)[0]
            
            return {
                'churn_prediction': bool(prediction[0]),
                'churn_probability': float(probability[1]),
                'retention_probability': float(probability[0]),
                'risk_level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.3 else 'Low'
            }
        except Exception as e:
            print(f"Error in prediction: {e}")
            return self._rule_based_prediction(customer_data)
    
    def _rule_based_prediction(self, customer_data):
        """Fallback rule-based prediction"""
        risk_score = 0.2  # base risk
        
        if customer_data.get('tenure', 12) < 12:
            risk_score += 0.3
        if customer_data.get('contract') == 'Month-to-month':
            risk_score += 0.3
        if customer_data.get('internet_service') == 'Fiber optic':
            risk_score += 0.15
        if customer_data.get('monthly_charges', 70) > 80:
            risk_score += 0.1
        if customer_data.get('senior_citizen', 0) == 1:
            risk_score += 0.05
            
        risk_score = min(risk_score, 0.95)
        
        return {
            'churn_prediction': bool(risk_score > 0.5),
            'churn_probability': float(risk_score),
            'retention_probability': float(1 - risk_score),
            'risk_level': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.3 else 'Low'
        }
    
    def predict_batch(self, customers_df):
        """
        Predict churn for multiple customers
        
        Args:
            customers_df (DataFrame): DataFrame containing customer features
        
        Returns:
            DataFrame: Original DataFrame with predictions added
        """
        if self.model is None or self.preprocessor is None:
            # Apply rule-based for each row
            results = []
            for _, row in customers_df.iterrows():
                result = self._rule_based_prediction(row.to_dict())
                results.append(result)
            
            customers_df['churn_prediction'] = [r['churn_prediction'] for r in results]
            customers_df['churn_probability'] = [r['churn_probability'] for r in results]
            customers_df['risk_level'] = [r['risk_level'] for r in results]
        else:
            df_processed = self.preprocessor.preprocess_new_data(customers_df.copy())
            predictions = self.model.predict(df_processed)
            probabilities = self.model.predict_proba(df_processed)[:, 1]
            
            customers_df['churn_prediction'] = predictions
            customers_df['churn_probability'] = probabilities
            customers_df['risk_level'] = customers_df['churn_probability'].apply(
                lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low'
            )
        
        return customers_df

# Example usage
if __name__ == "__main__":
    predictor = ChurnPredictor()
    
    # Example single prediction
    sample_customer = {
        'tenure': 12,
        'monthly_charges': 85.5,
        'total_charges': 1026,
        'gender': 'Male',
        'senior_citizen': 0,
        'partner': 'No',
        'dependents': 'No',
        'phone_service': 'Yes',
        'multiple_lines': 'No',
        'internet_service': 'Fiber optic',
        'online_security': 'No',
        'online_backup': 'No',
        'device_protection': 'No',
        'tech_support': 'No',
        'streaming_tv': 'Yes',
        'streaming_movies': 'Yes',
        'contract': 'Month-to-month',
        'paperless_billing': 'Yes',
        'payment_method': 'Electronic check'
    }
    
    result = predictor.predict_single(sample_customer)
    print("\n🔮 Single Prediction Result:")
    for key, value in result.items():
        print(f"   {key}: {value}")
EOF