import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             matthews_corrcoef, roc_curve, auc, precision_recall_curve,
                             average_precision_score, f1_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter

# Plot settings
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Load and explore the data
print("Loading and analyzing data...")
data = pd.read_csv('Data/CreditCardData.csv')

# Check class distribution
print("\nClass Distribution:")
print(data['Class'].value_counts(normalize=True))

# 2. Data preprocessing
scaler = StandardScaler()
data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)

X = data.drop('Class', axis=1)
y = data['Class']

# Check and remove NaN values
print("\nChecking and removing NaN values...")
X = X.dropna()
y = y[X.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Apply SMOTE (only to the training set)
print("\nSMOTE before training set distribution:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("SMOTE after training set distribution:", Counter(y_train_resampled))

# 3. Model definitions
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        class_weight='balanced',
        max_depth=10,
        n_estimators=100,
        random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
    scale_pos_weight=len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1]),
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    enable_categorical=True
)
}

# 4. Cross-validation and model evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    print(f"\n{name} training...")

    if name == 'XGBoost':
        # Special handling for XGBoost
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Manual cross-validation for XGBoost
        roc_auc_scores = []
        auprc_scores = []

        for train_idx, val_idx in cv.split(X_train_resampled, y_train_resampled):
            X_train_cv, X_val_cv = X_train_resampled.iloc[train_idx], X_train_resampled.iloc[val_idx]
            y_train_cv, y_val_cv = y_train_resampled.iloc[train_idx], y_train_resampled.iloc[val_idx]

            model_cv = xgb.XGBClassifier(
                scale_pos_weight=len(y_train_cv[y_train_cv == 0]) / len(y_train_cv[y_train_cv == 1]),
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )

            model_cv.fit(X_train_cv, y_train_cv)
            y_val_pred_proba = model_cv.predict_proba(X_val_cv)[:, 1]

            roc_auc_scores.append(roc_auc_score(y_val_cv, y_val_pred_proba))
            auprc_scores.append(average_precision_score(y_val_cv, y_val_pred_proba))

        print(f"Cross-validation ROC-AUC scores: {np.mean(roc_auc_scores):.3f} (+/- {np.std(roc_auc_scores) * 2:.3f})")
        print(f"Cross-validation AUPRC scores: {np.mean(auprc_scores):.3f} (+/- {np.std(auprc_scores) * 2:.3f})")

    else:
        # Original code for other models
        cv_results = cross_validate(model, X_train_resampled, y_train_resampled,
                                    cv=cv, scoring=['roc_auc', 'average_precision'],
                                    return_train_score=True)
        roc_auc_scores = cv_results['test_roc_auc']
        auprc_scores = cv_results['test_average_precision']
        print(f"Cross-validation ROC-AUC scores: {roc_auc_scores.mean():.3f} (+/- {roc_auc_scores.std() * 2:.3f})")
        print(f"Cross-validation AUPRC scores: {auprc_scores.mean():.3f} (+/- {auprc_scores.std() * 2:.3f})")

        # Train the model
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    predictions[name] = y_pred
    probabilities[name] = y_proba

    # Calculate metrics
    results[name] = {
        'accuracy': (y_test == y_pred).mean(),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'auprc': average_precision_score(y_test, y_proba)
    }

    # Detailed classification report
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))

# 5. Visualizations
# Confusion Matrices
plt.figure(figsize=(20, 5))
for idx, (name, y_pred) in enumerate(predictions.items(), 1):
    plt.subplot(1, 3, idx)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name}\nConfusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# ROC Curves
plt.figure(figsize=(10, 8))
for name, y_scores in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curves.png')
plt.close()

# Precision-Recall Curves
plt.figure(figsize=(10, 8))
for name, y_scores in probabilities.items():
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    auprc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name} (AUPRC = {auprc:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves Comparison')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('pr_curves.png')
plt.close()

# Metric Comparison
plt.figure(figsize=(12, 6))
metrics = ['accuracy', 'roc_auc', 'f1', 'mcc', 'auprc']
metric_data = {metric: [results[model][metric] for model in models.keys()]
               for metric in metrics}

x = np.arange(len(models))
width = 0.15

for i, metric in enumerate(metrics):
    plt.bar(x + i * width, metric_data[metric], width, label=metric)

plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width * 2.5, models.keys(), rotation=45)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('metric_comparison.png')
plt.close()

# Feature Importance (Random Forest)
rf_model = models['Random Forest']
importances = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(12, 8))
importances.plot(kind='barh')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Final results
print("\nDetailed Results:")
print("-" * 50)
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

print("\nVisualizations saved:")
print("1. confusion_matrices.png")
print("2. roc_curves.png")
print("3. pr_curves.png")
print("4. metric_comparison.png")
print("5. feature_importance.png")
