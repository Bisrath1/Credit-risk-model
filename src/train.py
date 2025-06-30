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
X = data.drop(columns=["is_high_risk", "CustomerId"])

y = data["is_high_risk"]  # Target (from Task 4)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save train/test split for reproducibility
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)


# src/train.py (continued)
def train_model(model, model_name, X_train, y_train, X_test, y_test):
    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for ROC-AUC

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} Metrics:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        return model

# Define models
models = {
    "LogisticRegression": LogisticRegression(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    trained_model = train_model(model, model_name, X_train, y_train, X_test, y_test)


# src/train.py (continued)
from sklearn.model_selection import GridSearchCV

def tune_model():
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Log best parameters and model
    with mlflow.start_run(run_name="RandomForest_Tuned"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_roc_auc", grid_search.best_score_)
        mlflow.sklearn.log_model(grid_search.best_estimator_, "RandomForest_Tuned")

        # Evaluate on test set
        y_pred = grid_search.predict(X_test)
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric("test_roc_auc", roc_auc)

        print(f"Tuned RandomForest Best Params: {grid_search.best_params_}")
        print(f"Tuned RandomForest Test ROC-AUC: {roc_auc:.4f}")

    return grid_search.best_estimator_

# Run hyperparameter tuning
best_rf = tune_model()


# src/train.py (continued)
import mlflow
from mlflow.models import Model

# Register the best model (example: assuming RandomForest_Tuned is the best)
model_uri = "runs:/<run_id>/RandomForest_Tuned"  # Replace <run_id> with actual run ID
mlflow.register_model(model_uri, "CreditRiskModel")