# =====================================================
# feature_engineering.py  (for Streamlit app)
# =====================================================
# Contains reusable feature engineering functions for the stock prediction app.
# Combines base + advanced + contextual (NIFTY) features.

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# Helper Functions
# =====================================================

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_basic_features(df, prefix, col):
    df[f"{prefix}_Return_3"] = df[col].pct_change(3)
    df[f"{prefix}_Return_5"] = df[col].pct_change(5)
    df[f"{prefix}_Return_10"] = df[col].pct_change(10)
    df[f"{prefix}_Volatility_7"] = df[col].pct_change().rolling(7).std()
    df[f"{prefix}_Volatility_14"] = df[col].pct_change().rolling(14).std()
    df[f"{prefix}_Momentum_5"] = df[col] - df[col].shift(5)
    df[f"{prefix}_Momentum_10"] = df[col] - df[col].shift(10)
    return df


def add_bollinger_bands(df, prefix, col="Close", window=20):
    mean = df[col].rolling(window).mean()
    std = df[col].rolling(window).std()
    df[f"{prefix}_BB_upper"] = mean + (2 * std)
    df[f"{prefix}_BB_lower"] = mean - (2 * std)
    df[f"{prefix}_BB_width"] = df[f"{prefix}_BB_upper"] - df[f"{prefix}_BB_lower"]
    return df


def add_advanced_indicators(df, prefix, col="Close"):
    """Adds MACD, ADX, Stochastic, Williams %R, and CCI indicators."""
    # --- MACD ---
    df[f"{prefix}_EMA12"] = df[col].ewm(span=12, adjust=False).mean()
    df[f"{prefix}_EMA26"] = df[col].ewm(span=26, adjust=False).mean()
    df[f"{prefix}_MACD"] = df[f"{prefix}_EMA12"] - df[f"{prefix}_EMA26"]
    df[f"{prefix}_Signal"] = df[f"{prefix}_MACD"].ewm(span=9, adjust=False).mean()

    # --- Ensure High/Low exist ---
    if "High" not in df.columns or "Low" not in df.columns:
        print(f"  ⚠️ Synthesizing 'High' and 'Low' for {prefix}...")
        df["High"] = df[col] * (1 + np.random.uniform(0.002, 0.005, size=len(df)))
        df["Low"] = df[col] * (1 - np.random.uniform(0.002, 0.005, size=len(df)))

    # --- ADX ---
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["DM_plus"] = np.where(
        (df["High"] - df["High"].shift(1)) > (df["Low"].shift(1) - df["Low"]),
        df["High"] - df["High"].shift(1), 0)
    df["DM_minus"] = np.where(
        (df["Low"].shift(1) - df["Low"]) > (df["High"] - df["High"].shift(1)),
        df["Low"].shift(1) - df["Low"], 0)
    TRn = df["TR"].rolling(14).sum()
    DMPn = df["DM_plus"].rolling(14).sum()
    DMNn = df["DM_minus"].rolling(14).sum()
    DIp = 100 * (DMPn / TRn)
    DIn = 100 * (DMNn / TRn)
    DX = (abs(DIp - DIn) / abs(DIp + DIn)) * 100
    df[f"{prefix}_ADX"] = DX.rolling(14).mean()

    # --- Stochastic Oscillator ---
    df[f"{prefix}_Stochastic"] = 100 * ((df["Close"] - df["Low"].rolling(14).min()) /
                                        (df["High"].rolling(14).max() - df["Low"].rolling(14).min()))

    # --- Williams %R ---
    df[f"{prefix}_WilliamsR"] = -100 * ((df["High"].rolling(14).max() - df["Close"]) /
                                        (df["High"].rolling(14).max() - df["Low"].rolling(14).min()))

    # --- Commodity Channel Index (CCI) ---
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(20).mean()
    md = (tp - ma).abs().rolling(20).mean()
    df[f"{prefix}_CCI"] = (tp - ma) / (0.015 * md)

    return df


def add_lag_features(df, prefix, col):
    for lag in [1, 2, 3, 5, 10]:
        df[f"{prefix}_Lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_stats(df, prefix, col):
    for w in [7, 21, 50]:
        df[f"{prefix}_MA{w}"] = df[col].rolling(w).mean()
        df[f"{prefix}_STD{w}"] = df[col].rolling(w).std()
    df[f"{prefix}_RSI"] = compute_RSI(df[col], 14)
    return df


# =====================================================
# Combined Feature Generator
# =====================================================
def generate_features(df, ticker, close_col="Close", add_index=False, add_index_func=None):
    """
    Full pipeline for generating all features for a given stock dataframe.
    Safe for both local CSVs and live-fetched data.
    """
    df = add_basic_features(df, ticker, close_col)
    df = add_bollinger_bands(df, ticker, col=close_col)
    df = add_advanced_indicators(df, ticker, col=close_col)
    df = add_lag_features(df, ticker, col=close_col)
    df = add_rolling_stats(df, ticker, col=close_col)

    if add_index and add_index_func is not None:
        df = add_index_func(df)

    df = df.ffill().bfill()
    return df