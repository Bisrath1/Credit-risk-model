# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load processed data (assumes processed data from Task 3 and Task 4)
data = pd.read_csv(r"C:\10x AIMastery\Credit-risk-model\data\processed\processed_data.csv")  # Adjust path as needed
X = data.drop(columns=["is_high_risk"])  # Features
y = data["is_high_risk"]  # Target (from Task 4)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save train/test split for reproducibility
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)