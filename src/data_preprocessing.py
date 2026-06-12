import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

class DataPreprocessor:
    def __init__(self, data_path='data/churn_data.csv'):
        self.data_path = data_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and initial explore data"""
        df = pd.read_csv(self.data_path)
        print(f"📊 Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"📈 Churn rate: {df['churn'].mean():.2%}")
        return df
    
    def clean_data(self, df):
        """Handle missing values and outliers"""
        if 'customer_id' in df.columns:
            df = df.drop('customer_id', axis=1)
        
        # Fill missing values
        df = df.fillna(df.mode().iloc[0])
        
        # Remove outliers
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
            
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        X = df.drop('churn', axis=1)
        y = df['churn']
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        return X, y
    
    def handle_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print(f"⚖️ Balanced dataset: {X_balanced.shape[0]} samples")
        print(f"📊 New churn rate: {y_balanced.mean():.2%}")
        return X_balanced, y_balanced
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"📚 Train set: {X_train.shape[0]} samples")
        print(f"🔍 Validation set: {X_val.shape[0]} samples")
        print(f"🧪 Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessors(self):
        """Save label encoders and scaler"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("💾 Preprocessors saved to 'models/'")
        
    def load_preprocessors(self):
        """Load label encoders and scaler"""
        self.label_encoders = joblib.load('models/label_encoders.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        
    def preprocess_new_data(self, df):
        """Preprocess new data for prediction"""
        df = self.clean_data(df)
        
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df

if __name__ == "__main__":
    print("=" * 50)
    print("🔄 DATA PREPROCESSING PIPELINE")
    print("=" * 50)
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    df = preprocessor.clean_data(df)
    df = preprocessor.encode_categorical(df)
    X, y = preprocessor.prepare_features(df)
    X_balanced, y_balanced = preprocessor.handle_imbalance(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X_balanced, y_balanced)
    preprocessor.save_preprocessors()
    
    print("\n✅ Preprocessing completed successfully!")