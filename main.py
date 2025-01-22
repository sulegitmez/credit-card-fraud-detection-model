import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
   accuracy_score, precision_score, recall_score, f1_score,
   classification_report, roc_auc_score, confusion_matrix,
   roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler, PowerTransformer
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectFromModel
import warnings
import time
from datetime import datetime
import lightgbm as lgb

warnings.filterwarnings('ignore')



class FraudDetection:
   def __init__(self, random_state=42):
       self.random_state = random_state
       # Enhanced model configurations
       self.models = {
           'Random Forest': RandomForestClassifier(
               n_estimators=200,
               max_depth=15,
               min_samples_leaf=4,
               min_samples_split=10,
               class_weight={0: 1, 1: 100},  # Increased weight for fraud class
               n_jobs=-1,
               random_state=self.random_state
           ),
           'XGBoost': XGBClassifier(
               n_estimators=200,
               max_depth=8,
               learning_rate=0.05,
               scale_pos_weight=100,  # Increased weight for fraud class
               min_child_weight=5,
               subsample=0.8,
               colsample_bytree=0.8,
               gamma=1,
               eval_metric='aucpr',  # Focus on PR-AUC
               random_state=self.random_state,
               tree_method='hist'  # Faster training
           ),
           'LightGBM': lgb.LGBMClassifier(
               n_estimators=200,
               max_depth=8,
               learning_rate=0.05,
               num_leaves=31,
               scale_pos_weight=100,  # Increased weight for fraud class
               subsample=0.8,
               colsample_bytree=0.8,
               min_child_samples=5,
               random_state=self.random_state,
               metric='auc_pr'  # Focus on PR-AUC
           ),
           'Logistic Regression': LogisticRegression(
               C=0.1,  # Increased regularization strength
               class_weight={0: 1, 1: 100},  # Increased weight for fraud class
               max_iter=1000,  # Increased max iterations for convergence
               random_state=self.random_state,
               n_jobs=-1,
               solver='saga',  # Efficient solver for large datasets
               penalty='l2'  # L2 regularization to prevent overfitting
           )
       }
       self.scaler = StandardScaler()
       self.power_transformer = PowerTransformer()
       self.results = {}

   # Rest of the class implementation remains the same...
   def load_and_preprocess_data(self, file_path):
       """Enhanced data preprocessing with feature engineering"""
       print("Loading and preprocessing data...")
       start_time = time.time()

       # Load dataset
       self.df = pd.read_csv(file_path)

       # Add feature engineering
       self.engineer_features()

       # Split features and target
       self.X = self.df.drop(columns=['Class'])
       self.y = self.df['Class']

       # Identify numerical columns for SMOTE-NC
       self.numerical_features = self.X.select_dtypes(include=[np.number]).columns
       self.categorical_features = self.X.select_dtypes(exclude=[np.number]).columns

       # Train-test split with stratification
       self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
           self.X, self.y, test_size=0.2, random_state=self.random_state,
           stratify=self.y
       )

       # Advanced preprocessing
       self.preprocess_features()

       # Feature selection
       self.select_features()

       # Advanced resampling with SMOTE-Tomek
       print("\nApplying SMOTE-Tomek for balanced dataset...")
       smote_tomek = SMOTETomek(
           random_state=self.random_state,
           sampling_strategy={1: int(sum(self.y_train == 0) * 0.5)}  # Less aggressive oversampling
       )
       self.X_train_resampled, self.y_train_resampled = smote_tomek.fit_resample(
           self.X_train_scaled, self.y_train
       )

       preprocessing_time = time.time() - start_time
       print(f"Data preprocessing completed in {preprocessing_time:.2f} seconds.")

   def engineer_features(self):
       """Add engineered features"""
       # Amount-based features
       self.df['Amount_Log'] = np.log1p(self.df['Amount'])
       self.df['Amount_Squared'] = self.df['Amount'] ** 2

       # Time-based features
       self.df['Time_Hour'] = self.df['Time'] % 24
       self.df['Time_Day'] = (self.df['Time'] // 24) % 7

       # Interaction features
       for i in range(1, 29):
           self.df[f'V{i}_Amount'] = self.df[f'V{i}'] * self.df['Amount']

       # Drop original Time column
       self.df.drop('Time', axis=1, inplace=True)

   def preprocess_features(self):
       """Advanced feature preprocessing"""
       # Scale features
       self.X_train_scaled = self.scaler.fit_transform(self.X_train)
       self.X_test_scaled = self.scaler.transform(self.X_test)

       # Apply power transformation for normalization
       self.X_train_scaled = self.power_transformer.fit_transform(self.X_train_scaled)
       self.X_test_scaled = self.power_transformer.transform(self.X_test_scaled)

   def select_features(self):
       """Feature selection using Random Forest importance"""
       selector = SelectFromModel(
           RandomForestClassifier(n_estimators=100, random_state=self.random_state),
           prefit=False,
           threshold='median'
       )
       selector.fit(self.X_train_scaled, self.y_train)

       # Apply feature selection
       self.X_train_scaled = selector.transform(self.X_train_scaled)
       self.X_test_scaled = selector.transform(self.X_test_scaled)

       print(f"Selected {self.X_train_scaled.shape[1]} features out of {self.X_train.shape[1]}")

   def custom_threshold_prediction(self, model, X, threshold=0.5):
       """Predict with custom threshold for better precision-recall balance"""
       y_prob = model.predict_proba(X)[:, 1]
       return (y_prob >= threshold).astype(int)

   def train_and_evaluate_models(self):
       """Enhanced training and evaluation with focus on fraud detection"""
       print("\nTraining and evaluating models...")

       # Find optimal threshold using validation set
       val_size = 0.2
       X_train_final, X_val, y_train_final, y_val = train_test_split(
           self.X_train_resampled, self.y_train_resampled,
           test_size=val_size, random_state=self.random_state,
           stratify=self.y_train_resampled
       )

       for model_name, model in self.models.items():
           print(f"\nTraining {model_name}...")
           start_time = time.time()

           # Train model
           if model_name == 'LightGBM':
               eval_set = [(X_train_final, y_train_final), (X_val, y_val)]
               eval_names = ['train', 'valid']
               model.fit(
                   X_train_final,
                   y_train_final,
                   eval_set=eval_set,
                   eval_names=eval_names,
                   eval_metric='auc',
                   callbacks=[
                       lgb.early_stopping(stopping_rounds=10),
                       lgb.log_evaluation(period=10)
                   ]
               )
           else:
               model.fit(X_train_final, y_train_final)

           # Find optimal threshold on validation set
           y_prob_val = model.predict_proba(X_val)[:, 1]
           precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob_val)
           f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
           optimal_idx = np.argmax(f1_scores[:-1])
           optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5

           # Predict with optimal threshold
           y_pred = self.custom_threshold_prediction(
               model, self.X_test_scaled, threshold=optimal_threshold
           )
           y_prob = model.predict_proba(self.X_test_scaled)[:, 1]

           # Calculate and store metrics
           self.calculate_and_store_metrics(
               model_name, model, y_pred, y_prob,
               start_time, optimal_threshold
           )

   def calculate_and_store_metrics(self, model_name, model, y_pred, y_prob, start_time, threshold):
       """Calculate and store comprehensive metrics"""
       training_time = time.time() - start_time

       metrics = {
           'model': model,
           'accuracy': accuracy_score(self.y_test, y_pred),
           'precision': precision_score(self.y_test, y_pred),
           'recall': recall_score(self.y_test, y_pred),
           'f1': f1_score(self.y_test, y_pred),
           'roc_auc': roc_auc_score(self.y_test, y_prob),
           'avg_precision': average_precision_score(self.y_test, y_prob),
           'confusion_matrix': confusion_matrix(self.y_test, y_pred),
           'optimal_threshold': threshold,
           'training_time': training_time,
           'y_prob': y_prob  # Store probabilities for plotting
       }

       self.results[model_name] = metrics

       # Print detailed metrics
       print(f"\n{model_name} Results:")
       print(f"Optimal Threshold: {threshold:.3f}")
       print(f"Training Time: {training_time:.2f} seconds")
       print(f"Test Accuracy: {metrics['accuracy']:.4f}")
       print(f"Test Precision (Fraud): {metrics['precision']:.4f}")
       print(f"Test Recall (Fraud): {metrics['recall']:.4f}")
       print(f"Test F1 (Fraud): {metrics['f1']:.4f}")
       print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
       print(f"Test Average Precision: {metrics['avg_precision']:.4f}")

       print("\nClassification Report:")
       print(classification_report(self.y_test, y_pred))

   def plot_results(self, save_path='plots'):
       """Generate and save visualization plots for model comparison including Logistic Regression"""
       import os
       os.makedirs(save_path, exist_ok=True)
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

       # 1. ROC Curves with all models including Logistic Regression
       plt.figure(figsize=(10, 6))
       for model_name, result in self.results.items():
           if 'y_prob' in result:
               y_prob = result['y_prob']
               fpr, tpr, _ = roc_curve(self.y_test, y_prob)
               plt.plot(fpr, tpr, label=f'{model_name} (AUC = {result["roc_auc"]:.3f})')

       plt.plot([0, 1], [0, 1], 'k--', label='Random')
       plt.xlabel('False Positive Rate')
       plt.ylabel('True Positive Rate')
       plt.title('ROC Curves Comparison')
       plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig(f'{save_path}/roc_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
       plt.close()

       # 2. Precision-Recall Curves with all models including Logistic Regression
       plt.figure(figsize=(10, 6))
       for model_name, result in self.results.items():
           if 'y_prob' in result:
               y_prob = result['y_prob']
               precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
               plt.plot(recall, precision,
                       label=f'{model_name} (AP = {result["avg_precision"]:.3f})')

       plt.xlabel('Recall')
       plt.ylabel('Precision')
       plt.title('Precision-Recall Curves Comparison')
       plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig(f'{save_path}/precision_recall_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
       plt.close()

       # 3. Confusion Matrices for each model
       for model_name, result in self.results.items():
           plt.figure(figsize=(8, 6))
           cm = result['confusion_matrix']
           sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['Normal', 'Fraud'],
                      yticklabels=['Normal', 'Fraud'])
           plt.title(f'Confusion Matrix - {model_name}')
           plt.ylabel('True Label')
           plt.xlabel('Predicted Label')
           plt.tight_layout()
           plt.savefig(f'{save_path}/confusion_matrix_{model_name.lower().replace(" ", "_")}_{timestamp}.png',
                      dpi=300, bbox_inches='tight')
           plt.close()

       # 4. Performance Metrics Comparison
       metrics_df = pd.DataFrame({
           'Model': list(self.results.keys()),
           'Accuracy': [result['accuracy'] for result in self.results.values()],
           'Precision': [result['precision'] for result in self.results.values()],
           'Recall': [result['recall'] for result in self.results.values()],
           'F1 Score': [result['f1'] for result in self.results.values()],
           'ROC AUC': [result['roc_auc'] for result in self.results.values()],
           'Avg Precision': [result['avg_precision'] for result in self.results.values()],
           'Training Time': [result['training_time'] for result in self.results.values()]
       })

       # 5. Bar plot for key metrics
       plt.figure(figsize=(12, 6))
       metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'ROC AUC']
       ax = metrics_df.set_index('Model')[metrics_to_plot].plot(kind='bar', width=0.8)
       plt.title('Model Performance Metrics Comparison')
       plt.ylabel('Score')
       plt.xlabel('Model')
       plt.xticks(rotation=45)
       plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig(f'{save_path}/performance_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
       plt.close()

       # 6. Training Time Comparison
       plt.figure(figsize=(10, 6))
       training_times = [result['training_time'] for result in self.results.values()]
       plt.bar(list(self.results.keys()), training_times)
       plt.title('Training Time Comparison')
       plt.ylabel('Time (seconds)')
       plt.xticks(rotation=45)
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig(f'{save_path}/training_times_{timestamp}.png', dpi=300, bbox_inches='tight')
       plt.close()

       # Save metrics to CSV
       metrics_df.to_csv(f'{save_path}/model_metrics_{timestamp}.csv', index=False)

       # Print summary table
       print(f"\nPlots and metrics have been saved to the '{save_path}' directory.")
       print("\nMetrics Summary:")
       print(metrics_df.to_string(float_format=lambda x: '{:.4f}'.format(x) if isinstance(x, float) else str(x)))


def main():
   # Initialize the improved fraud detection system
   fraud_detection = FraudDetection()

   # Load and preprocess data
   fraud_detection.load_and_preprocess_data('creditcard.csv')

   # Train and evaluate models
   fraud_detection.train_and_evaluate_models()

   # Generate and save plots
   fraud_detection.plot_results()


if __name__ == "__main__":
   main()

