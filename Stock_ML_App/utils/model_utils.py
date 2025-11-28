# =====================================================
# model_utils.py
# Handles model loading, caching, and prediction
# =====================================================

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score

# -----------------------------------------------------
# Paths
# -----------------------------------------------------
project_root = Path("C:/JupyterProjects/Stock_ML_Project")
models_dir = project_root / "Models"
data_dir = project_root / "Data" / "Processed" / "enhanced"

# -----------------------------------------------------
# Model Loading
# -----------------------------------------------------
def load_model(ticker: str, model_name: str, task_type="regression"):
    """
    Load a trained model file dynamically based on stock, model, and task.
    """
    suffix = "regression" if task_type == "regression" else "classification"
    model_path = models_dir / f"{ticker}_{model_name}_{suffix}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model

# -----------------------------------------------------
# Data Loading
# -----------------------------------------------------
def load_data(ticker: str):
    """
    Load enhanced dataset for a given ticker.
    """
    file_path = data_dir / f"{ticker.lower()}_final_model_ready.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data not found for {ticker}: {file_path}")
    df = pd.read_csv(file_path)
    return df

# -----------------------------------------------------
# Regression Prediction
# -----------------------------------------------------
def predict_regression(ticker: str, model_name: str, horizon=1):
    """
    Predict next N days of stock price using a regression model.
    horizon = number of days ahead (max 5)
    """
    df = load_data(ticker)
    model = load_model(ticker, model_name, "regression")

    # Select numeric features
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    target_col = [c for c in numeric_df.columns if "Target_Reg" in c]
    if target_col:
        X = numeric_df.drop(columns=target_col, errors="ignore")
    else:
        X = numeric_df

    # Take last N rows for prediction
    recent_data = X.tail(horizon)
    preds = model.predict(recent_data)

    return preds[-1] if horizon == 1 else preds

# -----------------------------------------------------
# Classification Prediction
# -----------------------------------------------------
def predict_classification(ticker: str, model_name: str):
    """
    Predict next-day class (Up/Down) using classification model.
    """
    df = load_data(ticker)
    model = load_model(ticker, model_name, "classification")

    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    target_col = [c for c in numeric_df.columns if "Target_Cls" in c]
    if target_col:
        X = numeric_df.drop(columns=target_col, errors="ignore")
    else:
        X = numeric_df

    recent_data = X.tail(1)
    pred_class = model.predict(recent_data)[0]
    pred_proba = model.predict_proba(recent_data)[0][1] if hasattr(model, "predict_proba") else None

    return pred_class, pred_proba

# -----------------------------------------------------
# Metrics Helper
# -----------------------------------------------------
def evaluate_regression(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def evaluate_classification(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }
