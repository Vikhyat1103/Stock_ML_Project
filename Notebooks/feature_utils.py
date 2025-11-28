import pandas as pd
import numpy as np
import yfinance as yf

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def add_index_features(df, index_symbol="^NSEI"):
    """Adds NIFTY-based contextual features safely."""
    nifty = yf.download(index_symbol, start="2018-12-31", end="2025-01-01", auto_adjust=False)
    nifty.index = pd.to_datetime(nifty.index).tz_localize(None)
    nifty = nifty[["Close"]].copy()
    nifty.columns = ["NIFTY_Close"]

    if "Date" not in df.columns:
        df["Date"] = pd.date_range(start="2018-01-01", periods=len(df), freq="D")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    nifty_reset = nifty.reset_index().rename(columns={"Date": "NIFTY_Date"})
    merged = pd.merge_asof(df.sort_values("Date"), nifty_reset.sort_values("NIFTY_Date"),
                           left_on="Date", right_on="NIFTY_Date", direction="backward")

    merged["NIFTY_Return"] = merged["NIFTY_Close"].pct_change()
    merged["NIFTY_MA7"] = merged["NIFTY_Close"].rolling(7).mean()
    merged["NIFTY_MA21"] = merged["NIFTY_Close"].rolling(21).mean()
    merged["NIFTY_RSI"] = compute_RSI(merged["NIFTY_Close"], 14)
    merged = merged.drop(columns=["NIFTY_Date"])
    return merged
