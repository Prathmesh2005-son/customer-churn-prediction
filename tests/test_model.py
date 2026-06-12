import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction import ChurnPredictor

class TestChurnPredictor(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.predictor = ChurnPredictor()
        
    def test_single_prediction(self):
        """Test single customer prediction"""
        customer = {
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
        
        result = self.predictor.predict_single(customer)
        
        self.assertIn('churn_prediction', result)
        self.assertIn('churn_probability', result)
        self.assertIn('risk_level', result)
        self.assertIsInstance(result['churn_probability'], float)
        self.assertGreaterEqual(result['churn_probability'], 0)
        self.assertLessEqual(result['churn_probability'], 1)
        
    def test_prediction_output_types(self):
        """Test prediction output data types"""
        customer = {
            'tenure': 24,
            'monthly_charges': 50.0,
            'total_charges': 1200,
            'gender': 'Female',
            'senior_citizen': 0,
            'partner': 'Yes',
            'dependents': 'Yes',
            'phone_service': 'Yes',
            'multiple_lines': 'Yes',
            'internet_service': 'DSL',
            'online_security': 'Yes',
            'online_backup': 'Yes',
            'device_protection': 'Yes',
            'tech_support': 'Yes',
            'streaming_tv': 'No',
            'streaming_movies': 'No',
            'contract': 'Two year',
            'paperless_billing': 'No',
            'payment_method': 'Bank transfer'
        }
        
        result = self.predictor.predict_single(customer)
        
        self.assertIsInstance(result['churn_prediction'], bool)
        self.assertIsInstance(result['churn_probability'], float)
        self.assertIsInstance(result['risk_level'], str)

if __name__ == '__main__':
    unittest.main()