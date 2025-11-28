# utils/data_fetcher.py
# ----------------------------------------------------
# Utilities to fetch & cache stock price data via yfinance
# ----------------------------------------------------

from pathlib import Path
import pandas as pd
import yfinance as yf


# Project paths
PROJECT_ROOT = Path("C:/JupyterProjects/Stock_ML_Project")
RAW_DIR = PROJECT_ROOT / "Data" / "Raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _download_from_yf(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """
    Internal helper: download OHLCV data from yfinance and
    return a clean dataframe with flat column names.
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    # Handle MultiIndex columns (e.g., when downloading multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]  # Flatten to single level

    # yfinance returns Date as index
    df = df.reset_index()

    # Ensure Date is datetime and timezone-naive
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    return df


def fetch_stock_data(
    ticker: str,
    start: str = "2015-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a given ticker between [start, end].

    - Uses local CSV cache in Data/Raw/<ticker>.csv
    - If cache exists and is recent enough, reuses it.
    - If force_refresh=True, always re-download from yfinance.

    Returns a dataframe with at least:
    ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cache_file = RAW_DIR / f"{ticker}_raw.csv"

    # If cache exists and we are not forcing a refresh, try to reuse it
    if cache_file.exists() and not force_refresh:
        df = pd.read_csv(cache_file, parse_dates=["Date"])
        df["Date"] = df["Date"].dt.tz_localize(None)

        # Simple staleness check: if last date is within 5 days of today, reuse
        last_date = df["Date"].max()
        if (pd.Timestamp.today().normalize() - last_date.normalize()).days <= 5:
            return df

        # Otherwise, fall through to redownload (will overwrite cache)

    # Download from yfinance
    df = _download_from_yf(ticker, start=start, end=end)

    # Save to cache
    df.to_csv(cache_file, index=False)

    return df
