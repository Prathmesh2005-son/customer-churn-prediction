import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models and evaluate on validation set"""
        for name, model in self.models.items():
            print(f"\n🚀 Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1_score': f1_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_pred_proba)
            }
            
            self.results[name] = metrics
            print(f"📊 Results for {name}:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")
            
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.pkl')
        
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        self.best_model = self.models[self.best_model_name]
        print(f"\n🏆 Best model: {self.best_model_name} (ROC-AUC: {self.results[self.best_model_name]['roc_auc']:.4f})")
        
        return self.best_model
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """Perform hyperparameter tuning for the best model"""
        print("\n🔧 Performing hyperparameter tuning...")
        
        if self.best_model_name == 'Random Forest':
            from sklearn.model_selection import GridSearchCV
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.best_model = grid_search.best_estimator_
            print(f"✅ Best parameters: {grid_search.best_params_}")
            
        elif self.best_model_name == 'XGBoost':
            from sklearn.model_selection import RandomizedSearchCV
            param_dist = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0]
            }
            random_search = RandomizedSearchCV(
                XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                param_dist,
                n_iter=10,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42
            )
            random_search.fit(X_train, y_train)
            self.best_model = random_search.best_estimator_
            print(f"✅ Best parameters: {random_search.best_params_}")
        
        y_pred = self.best_model.predict(X_val)
        y_pred_proba = self.best_model.predict_proba(X_val)[:, 1]
        
        print("\n📈 Tuned Model Performance:")
        print(f"   Accuracy: {accuracy_score(y_val, y_pred):.4f}")
        print(f"   ROC-AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")
        
        return self.best_model
    
    def evaluate_on_test(self, X_test, y_test):
        """Final evaluation on test set"""
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        final_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("\n" + "=" * 50)
        print("🎯 FINAL MODEL EVALUATION ON TEST SET")
        print("=" * 50)
        for metric, value in final_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('models/confusion_matrix.png')
        plt.close()
        print("📊 Confusion matrix saved to 'models/confusion_matrix.png'")
        
        # Feature Importance
        if hasattr(self.best_model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'feature': range(len(self.best_model.feature_importances_)),
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance')
            plt.savefig('models/feature_importance.png')
            plt.close()
            print("📊 Feature importance saved to 'models/feature_importance.png'")
        
        joblib.dump(self.best_model, 'models/best_churn_model.pkl')
        print("💾 Best model saved to 'models/best_churn_model.pkl'")
        
        return final_metrics
    
    def save_results(self):
        """Save results to CSV"""
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv('models/model_comparison.csv')
        print("\n📁 Results saved to 'models/model_comparison.csv'")

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 MODEL TRAINING PIPELINE")
    print("=" * 50)
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    df = preprocessor.clean_data(df)
    df = preprocessor.encode_categorical(df)
    X, y = preprocessor.prepare_features(df)
    X_balanced, y_balanced = preprocessor.handle_imbalance(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X_balanced, y_balanced)
    
    trainer = ModelTrainer()
    trainer.train_models(X_train, y_train, X_val, y_val)
    trainer.hyperparameter_tuning(X_train, y_train, X_val, y_val)
    trainer.evaluate_on_test(X_test, y_test)
    trainer.save_results()
    
    print("\n✅ Model training completed successfully!")