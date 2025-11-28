# =====================================================
# utils/ensemble_utils.py
# =====================================================

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from datetime import timedelta

# --------------------------
# Evaluation Helper
# --------------------------
def evaluate(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# --------------------------
# Blend Predictions
# --------------------------
def blend_predictions(preds_dict, weights=None):
    model_names = list(preds_dict.keys())
    preds_matrix = np.column_stack(list(preds_dict.values()))

    if weights is None:
        weights = np.ones(len(model_names)) / len(model_names)
    weights = np.array(weights) / np.sum(weights)
    blended = np.dot(preds_matrix, weights)
    return blended


# --------------------------
# Offline ensemble trainer   (metrics-only; used in notebooks)
# --------------------------
def run_ensemble(X_train, X_test, y_train, y_test):
    """
    Blend Linear, Ridge, Lasso regressors and return evaluation scores.
    This is for offline experimentation / Step 9, not used directly by the dashboard.
    """
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),
    }

    preds, scores = {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        scores[name] = evaluate(y_test, y_pred)

    # Simple average blend
    blended_simple = blend_predictions(preds)
    metrics_simple = evaluate(y_test, blended_simple)

    # Weighted blend (inverse RMSE)
    inv_rmse = np.array([1 / scores[m]["RMSE"] for m in models])
    blended_weighted = blend_predictions(preds, inv_rmse)
    metrics_weighted = evaluate(y_test, blended_weighted)

    return {
        "individual": scores,
        "blend_simple": metrics_simple,
        "blend_weighted": metrics_weighted,
    }


# =====================================================
# Dashboard Wrapper – simple deterministic forecast
# =====================================================

def run_ensemble_dashboard(df_recent, horizon: int = 30):
    """
    Lightweight, deterministic forecasting for the Streamlit dashboard.

    For now we don't load your heavy trained models here.
    Instead we fit a simple LinearRegression on the last N days (time vs Close)
    and extrapolate 'horizon' days into the future.

    IMPORTANT: For a given company + selected_date, this is deterministic:
    no np.random.uniform, no self-imports, no hidden randomness.
    """
    df_recent = df_recent.copy()
    df_recent = df_recent.sort_values("Date")
    closes = df_recent["Close"].values

    if len(closes) < 5:
        # If we don't have enough points, just keep the last price flat.
        last_close = float(closes[-1])
        forecast = np.full(horizon, last_close)
        return {
            "next_day": last_close,
            "next_30": last_close,
            "forecast_series": forecast,
        }

    # Use up to the last 120 days for trend fitting
    window = min(120, len(closes))
    y = closes[-window:]
    X = np.arange(window).reshape(-1, 1)

    # Fit simple linear trend
    trend_model = LinearRegression()
    trend_model.fit(X, y)

    # Forecast next 'horizon' points
    X_future = np.arange(window, window + horizon).reshape(-1, 1)
    future_pred = trend_model.predict(X_future)

    next_day = float(future_pred[0])
    next_30 = float(future_pred[-1])

    return {
        "next_day": next_day,
        "next_30": next_30,
        "forecast_series": future_pred,  # length = horizon
    }

# =====================================================
# True Ensemble Forecast Generator
# =====================================================
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
import numpy as np
import pandas as pd

# =====================================================
# True Ensemble Forecast Generator (Continuity Fixed)
# =====================================================
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
import numpy as np
import pandas as pd


def predict_with_ensemble(df, horizon=30):
    """
    True predictive ensemble:
    - Uses recent 'Close' data to fit models
    - Predicts next N days recursively
    - Ensures continuity with the last observed close
    """
    closes = df["Close"].values
    if len(closes) < 30:
        raise ValueError("Not enough data to make predictions")

    # ---- Create supervised features ----
    window = 5  # number of past days to look back
    X, y = [], []
    for i in range(window, len(closes)):
        X.append(closes[i - window:i])
        y.append(closes[i])
    X, y = np.array(X), np.array(y)

    # ---- Train models ----
    models = {
        "ridge": Ridge(alpha=0.5),
        "lasso": Lasso(alpha=0.001),
        "rf": RandomForestRegressor(n_estimators=100, random_state=42),
        "gb": GradientBoostingRegressor(n_estimators=150, random_state=42),
    }

    for m in models.values():
        m.fit(X, y)

    # ---- Recursive prediction starting from last window (continuity fix) ----
    current_window = closes[-window:].tolist()
    forecast = []

    for _ in range(horizon):
        X_input = np.array(current_window[-window:]).reshape(1, -1)
        pred_each = [m.predict(X_input)[0] for m in models.values()]
        blended_pred = np.mean(pred_each)
        forecast.append(blended_pred)
        current_window.append(blended_pred)

    next_day = float(forecast[0])
    next_30 = float(forecast[-1])

    return {
        "next_day": next_day,
        "next_30": next_30,
        "forecast_series": np.array(forecast),
    }

def run_weighted_blend(df, horizon=1):
    """
    True weighted ensemble using Ridge, RandomForest, GradientBoosting, and LinearRegression.
    Returns predictions, weights, and forecast series.
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.metrics import mean_squared_error
    import numpy as np

    closes = df["Close"].values
    if len(closes) < 40:
        raise ValueError("Not enough data to compute weighted blend")

    # Prepare supervised data
    window = 5
    X, y = [], []
    for i in range(window, len(closes)):
        X.append(closes[i - window:i])
        y.append(closes[i])
    X, y = np.array(X), np.array(y)

    models = {
        "Ridge": Ridge(alpha=0.5),
        "Linear": LinearRegression(),
        "RF": RandomForestRegressor(n_estimators=100, random_state=42),
        "GB": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    preds, rmses = {}, {}

    # Train each and compute RMSE
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        preds[name] = model
        rmses[name] = rmse

    # Compute weights (inverse RMSE)
    inv_rmse = np.array([1 / rmses[m] for m in models])
    weights = inv_rmse / inv_rmse.sum()

    # Generate weighted prediction
    last_window = closes[-window:]
    X_input = last_window.reshape(1, -1)
    pred_each = [m.predict(X_input)[0] for m in preds.values()]
    blended_pred = float(np.dot(pred_each, weights))

    return {
        "models": list(models.keys()),
        "weights": weights.round(3).tolist(),
        "pred_each": np.round(pred_each, 2).tolist(),
        "blended_next_day": blended_pred,
    }
