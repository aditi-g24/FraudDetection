# Fraud Detection - Machine Learning Project

"""
Fraud Detection System using Machine Learning
Author: Your Name
Date: 2024
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================

# Load the dataset
df = pd.read_csv("Downloads/Fraud.csv")

# Display first few rows
print("Dataset Preview:")
print(df.head())
print("\n" + "="*80 + "\n")

# Dataset shape
print(f"Dataset Shape: {df.shape}")
print(f"Total Records: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")
print("\n" + "="*80 + "\n")

# Dataset size in memory
print(f"Dataset Size: {df.size}")
print("\n" + "="*80 + "\n")

# Dataset information
print("Dataset Information:")
print(df.info())
print("\n" + "="*80 + "\n")

# Statistical summary
print("Statistical Summary:")
print(df.describe())
print("\n" + "="*80 + "\n")

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())
print("\n" + "="*80 + "\n")

# Check for duplicates
print(f"Duplicate Rows: {df.duplicated().sum()}")
print("\n" + "="*80 + "\n")

# Value counts for categorical columns
print("Transaction Types Distribution:")
print(df['type'].value_counts())
print("\n" + "="*80 + "\n")

# Check class imbalance
if 'isFraud' in df.columns:
    print("Fraud Cases Distribution:")
    print(df['isFraud'].value_counts())
    print("\nPercentage Distribution:")
    print(df['isFraud'].value_counts(normalize=True) * 100)
    print("\n" + "="*80 + "\n")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

# Handle missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['type'], drop_first=True)

# Feature engineering (if needed)
# Example: Create new features
if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
    df_encoded['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    
if 'oldbalanceDest' in df.columns and 'newbalanceDest' in df.columns:
    df_encoded['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

# Correlation heatmap
plt.figure(figsize=(14, 10))
correlation_matrix = df_encoded.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Distribution of transaction amounts
if 'amount' in df.columns:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['amount'], bins=50, edgecolor='black')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')
    plt.title('Distribution of Transaction Amounts')
    
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(df['amount']), bins=50, edgecolor='black', color='orange')
    plt.xlabel('Log(Transaction Amount)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Log Transaction Amounts')
    plt.tight_layout()
    plt.savefig('amount_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# Fraud vs Non-Fraud comparison
if 'isFraud' in df.columns and 'amount' in df.columns:
    plt.figure(figsize=(10, 6))
    df.boxplot(column='amount', by='isFraud')
    plt.title('Transaction Amount by Fraud Status')
    plt.suptitle('')
    plt.xlabel('Is Fraud')
    plt.ylabel('Amount')
    plt.savefig('fraud_amount_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 4. MODEL BUILDING
# ============================================================================

# Separate features and target
X = df_encoded.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1, errors='ignore')
y = df_encoded['isFraud']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training Set Size: {X_train.shape[0]}")
print(f"Test Set Size: {X_test.shape[0]}")
print("\n" + "="*80 + "\n")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 5. LOGISTIC REGRESSION MODEL
# ============================================================================

print("Training Logistic Regression Model...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predictions
lr_predictions = lr_model.predict(X_test_scaled)
lr_probabilities = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("\nLogistic Regression Results:")
print("="*80)
print(f"Accuracy: {accuracy_score(y_test, lr_predictions):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, lr_probabilities):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))
print("\n" + "="*80 + "\n")

# ============================================================================
# 6. RANDOM FOREST MODEL
# ============================================================================

print("Training Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nRandom Forest Results:")
print("="*80)
print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_probabilities):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print("\n" + "="*80 + "\n")

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================

# Random Forest Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))
print("\n" + "="*80 + "\n")

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. ROC CURVE COMPARISON
# ============================================================================

# Calculate ROC curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probabilities)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probabilities)

# Plot ROC curves
plt.figure(figsize=(10, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, lr_probabilities):.3f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_probabilities):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. CONFUSION MATRIX VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Logistic Regression
sns.heatmap(confusion_matrix(y_test, lr_predictions), annot=True, fmt='d', 
            cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Logistic Regression - Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Random Forest
sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt='d', 
            cmap='Greens', ax=axes[1], cbar=False)
axes[1].set_title('Random Forest - Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 10. SAVE MODELS
# ============================================================================

import pickle

# Save models
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models saved successfully!")
print("\n" + "="*80 + "\n")

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print("FINAL MODEL COMPARISON")
print("="*80)
print(f"{'Model':<25} {'Accuracy':<15} {'ROC-AUC':<15}")
print("-"*80)
print(f"{'Logistic Regression':<25} {accuracy_score(y_test, lr_predictions):<15.4f} {roc_auc_score(y_test, lr_probabilities):<15.4f}")
print(f"{'Random Forest':<25} {accuracy_score(y_test, rf_predictions):<15.4f} {roc_auc_score(y_test, rf_probabilities):<15.4f}")
print("="*80)

print("\n✓ Analysis Complete!")
