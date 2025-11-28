# =====================================================
# utils/cache_utils.py
# =====================================================

import os
import joblib
import pandas as pd
from pathlib import Path
import streamlit as st

from utils.model_utils import load_data
from utils.ensemble_utils import run_ensemble
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path("C:/JupyterProjects/Stock_ML_Project")
MODELS_DIR = PROJECT_ROOT / "Models"
DATA_DIR = PROJECT_ROOT / "Data" / "Processed" / "enhanced"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def clear_all_streamlit_cache():
    """Clear Streamlit’s data/resource caches."""
    st.cache_data.clear()
    st.cache_resource.clear()
    return "✅ Streamlit cache cleared successfully."


def retrain_all_models(tickers=["RELIANCE", "TCS", "HDFCBANK"]):
    """Retrain and resave regression models for all tickers."""
    MODELS_DIR.mkdir(exist_ok=True)
    summary = []

    for ticker in tickers:
        try:
            df = load_data(ticker)
            df = df.replace([float("inf"), -float("inf")], None).dropna(how="all", axis=1)
            target_col = next((c for c in ["Target_Reg", "Target_Reg_y"] if c in df.columns), None)
            if not target_col:
                summary.append((ticker, "❌ No target column"))
                continue

            numeric = df.select_dtypes("number").copy()
            imputer = SimpleImputer(strategy="mean")
            numeric[numeric.columns] = imputer.fit_transform(numeric)
            y = numeric[target_col]
            X = numeric.drop(columns=[target_col])
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(random_state=42),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                "SVR": SVR(kernel="rbf"),
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                save_path = MODELS_DIR / f"{ticker}_{name}_retrained.pkl"
                joblib.dump(model, save_path)
            summary.append((ticker, "✅ Retrained Successfully"))
        except Exception as e:
            summary.append((ticker, f"❌ Error: {e}"))

    return pd.DataFrame(summary, columns=["Ticker", "Status"])


def retrain_ensemble(tickers=["RELIANCE", "TCS", "HDFCBANK"]):
    """Recalculate ensemble metrics and optionally resave blended results."""
    results = []
    for ticker in tickers:
        try:
            df = load_data(ticker)
            target_col = next((c for c in ["Target_Reg", "Target_Reg_y"] if c in df.columns), None)
            if not target_col:
                continue
            numeric = df.select_dtypes("number").copy()
            imputer = SimpleImputer(strategy="mean")
            numeric[numeric.columns] = imputer.fit_transform(numeric)
            y = numeric[target_col]
            X = numeric.drop(columns=[target_col])
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

            result = run_ensemble(X_train, X_test, y_train, y_test)
            result["Ticker"] = ticker
            results.append(result)
        except Exception as e:
            results.append({"Ticker": ticker, "Error": str(e)})

    return results
