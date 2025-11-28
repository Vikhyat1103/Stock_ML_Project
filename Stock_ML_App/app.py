# =====================================================
# app.py – Stock Price Prediction App (Prototype + Enhanced UI)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from utils.about_page import about_page

# our helper functions
from utils.model_utils import (
    load_model,     # currently unused but kept for future
    load_data,
    predict_regression,
    predict_classification,
)

# =====================================================
# Page Config
# =====================================================
st.set_page_config(
    page_title="Stock ML Prediction App",
    page_icon="📊",
    layout="wide",
)

# =====================================================
# Styling and Theme
# =====================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-card h2 {
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# Paths / Config
# =====================================================
project_root = Path("C:/JupyterProjects/Stock_ML_Project")
data_dir = project_root / "Data" / "Processed" / "enhanced"
results_dir = project_root / "Results"
models_dir = project_root / "Models"

TICKERS = ["RELIANCE", "TCS", "HDFCBANK"]

# =====================================================
# Available Data Files
# =====================================================
DATA_FILES = {
    "RELIANCE": data_dir / "reliance_final_model_ready.csv",
    "TCS": data_dir / "tcs_final_model_ready.csv",
    "HDFCBANK": data_dir / "hdfcbank_final_model_ready.csv",
}

# Wrapper so we can change later if needed
@st.cache_data
def load_dataset(ticker: str) -> pd.DataFrame:
    """Use the same loader you already use in model_utils."""
    return load_data(ticker)


# =====================================================
# Sidebar Navigation
# =====================================================
with st.sidebar:
    st.title("🔍 Navigation")
    choice = st.radio(
        "Choose Section",
        ["🏠 Dashboard", "ℹ️ About", "⚙️ Settings"]
    )


# =====================================================
# Dashboard Page – Ensemble Integrated Forecast Version
# =====================================================
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import timedelta

from utils.company_map import COMPANY_TICKER_MAP
from utils.data_fetcher import fetch_stock_data
from utils.ensemble_utils import run_ensemble_dashboard
from utils.ensemble_utils import predict_with_ensemble as run_ensemble
from utils.ensemble_utils import run_weighted_blend

def render_stock_chart(df_full, company_name, selected_date, next_30_forecast):
    df_full = df_full.copy()
    df_full["Date"] = pd.to_datetime(df_full["Date"])
    df_full = df_full.sort_values("Date")

    df_sel = df_full[df_full["Date"] <= selected_date]
    if df_sel.empty:
        st.warning("⚠️ No data to plot.")
        return None

    ranges = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "4Y": 1460, "MAX": None}
    st.markdown("### 📈 Price Trend (2015 → Selected Date)")
    st.caption("Historical prices with dynamic range selection and model-based forecast region")

    # Range buttons
    cols = st.columns(len(ranges))
    selected_range = st.session_state.get("selected_range", "1Y")
    for i, (label, _) in enumerate(ranges.items()):
        if cols[i].button(label, key=f"range_btn_{label}"):
            selected_range = label
            st.session_state["selected_range"] = label

    # Filter range data
    if ranges[selected_range] is None:
        df_range = df_sel
    else:
        start_date = selected_date - timedelta(days=ranges[selected_range])
        df_range = df_sel[(df_sel["Date"] >= start_date) & (df_sel["Date"] <= selected_date)]
        if df_range.empty:
            df_range = df_sel

    # Y-axis padding (increase top space to avoid cutoff)
    y_min, y_max = df_range["Close"].min(), df_range["Close"].max()
    pad = (y_max - y_min) * 0.15  # increased from 0.05 to 0.15
    y_range = [y_min - pad, y_max + pad]

    last_close = df_sel["Close"].iloc[-1]
    prev_close = df_sel["Close"].iloc[-2] if len(df_sel) > 1 else last_close

    # ------------- Plot Figure -------------
    fig = go.Figure()

    # Historical section
    fig.add_trace(go.Scatter(
        x=df_range["Date"], y=df_range["Close"],
        mode="lines", line=dict(color="green", width=2),
        fill="tozeroy", fillcolor="rgba(0,200,0,0.15)",
        name="Actual Price",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: ₹%{y:.2f}<extra></extra>"
    ))

    # =====================================================
    # Forecast region (blue) — continuity alignment fixed
    # =====================================================

    # 1️⃣ Generate forecast date range correctly (continuous from last point)
    forecast_dates = pd.date_range(
        start=df_sel["Date"].iloc[-1],  # start from same day as last actual
        periods=len(next_30_forecast) + 1  # include extra for continuity
    )[1:]  # skip first to maintain correct forecast day labeling

    # 2️⃣ Optional micro bridge (for pixel-perfect continuity)
    fig.add_trace(go.Scatter(
        x=[df_sel["Date"].iloc[-1], forecast_dates[0]],
        y=[df_sel["Close"].iloc[-1], next_30_forecast[0]],
        mode="lines",
        line=dict(color="royalblue", width=1, dash="dot"),
        showlegend=False
    ))

    # 3️⃣ Main forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=next_30_forecast,
        mode="lines",
        line=dict(color="royalblue", width=2, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(65,105,225,0.2)",
        name="Next 30 Days (Predicted)",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Predicted: ₹%{y:.2f}<extra></extra>"
    ))


    # Previous close marker
    fig.add_hline(
        y=prev_close, line_dash="dot", line_color="gray",
        annotation_text="Prev Close", annotation_position="top left"
    )

    fig.update_layout(
        title=f"{company_name} – Closing Price Over Time",
        xaxis_title="Date", yaxis_title="Price (₹)",
        yaxis_range=y_range, hovermode="x unified",
        template="plotly_white", margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False, font=dict(size=13)
    )

    st.plotly_chart(fig, use_container_width=True)
    progress = len(df_range) / len(df_sel)
    st.progress(progress, text=f"Viewing {selected_range} range ({len(df_range)} / {len(df_sel)} records)")

    # Debug expander to verify forecast values
    with st.expander("🔍 Model Forecast Data"):
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted_Close": next_30_forecast
        })
        st.write(forecast_df.head(10))

    return last_close

def dashboard_page():
    st.title("📊 Stock Price Prediction Dashboard (AI-Powered)")
    st.caption("Live data • Ensemble-style forecasts • Interactive chart")

    # 1️⃣ Select company
    company_name = st.selectbox("🏢 Select Company:", list(COMPANY_TICKER_MAP.keys()), index=0)
    ticker = COMPANY_TICKER_MAP[company_name]

    # 2️⃣ Fetch live data
    with st.spinner(f"Fetching live data for {company_name} ({ticker})..."):
        df = fetch_stock_data(ticker)

    if df is None or df.empty:
        st.error("❌ Data fetch failed.")
        return

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
    selected_date = st.date_input(
        "📅 Select a date:",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )
    selected_ts = pd.Timestamp(selected_date)

    df_upto = df[df["Date"] <= selected_ts]
    if df_upto.empty:
        st.warning("⚠️ No data found before this date.")
        return

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("📊 Records", f"{len(df_upto):,}")
    with c2:
        st.metric("🕒 Start", f"{df_upto['Date'].min().date()}")
    with c3:
        st.metric("📆 Selected", f"{selected_date}")
    st.markdown("---")

    # 3️⃣ Run dashboard ensemble wrapper (deterministic)
    with st.spinner("🧠 Generating model-style forecast..."):
        try:
            preds = run_ensemble(df_upto)
            predicted_next_day = preds["next_day"]
            predicted_next_30 = preds["next_30"]
            next_30_forecast = preds["forecast_series"]
        except Exception as e:
            st.error(f"⚠️ Model prediction failed: {e}")
            return

    # 4️⃣ Render chart
    last_close = render_stock_chart(df, company_name, selected_ts, next_30_forecast)

        # 5️⃣ Show forecast metrics
    st.markdown("### 🔮 Forecasts")
    delta_next_day = f"{((predicted_next_day - last_close) / last_close) * 100:+.2f}%"
    delta_next_30 = f"{((predicted_next_30 - last_close) / last_close) * 100:+.2f}%"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📅 Predicted Price (Next Day)", f"₹ {predicted_next_day:,.2f}", delta=delta_next_day)
    with col2:
        st.metric("📆 Predicted Price (Next 30 Days)", f"₹ {predicted_next_30:,.2f}", delta=delta_next_30)

    if predicted_next_day > last_close:
        st.success("📈 Model indicates upward movement (▲)")
    else:
        st.error("📉 Model indicates possible decline (▼)")

    # ----------------------------------------------------
    # 6️⃣ Accuracy Evaluation (for *next* trading day)
    #     – reuse the SAME next_day prediction
    # ----------------------------------------------------
    st.markdown("### 🎯 Model Accuracy (for Selected Date)")

    # Last trading day used for training
    train_last_date = df_upto["Date"].max()
    full_last_date = df["Date"].max()

    if train_last_date < full_last_date:
        # Find the next trading date after train_last_date
        future_dates = df.loc[df["Date"] > train_last_date, "Date"].sort_values()
        if not future_dates.empty:
            target_date = future_dates.iloc[0]        # this is the date our next_day prediction refers to
            actual_price = df.loc[df["Date"] == target_date, "Close"].iloc[0]

            predicted_for_target = predicted_next_day  # 🔑 reuse main forecast
            error_pct = abs(predicted_for_target - actual_price) / actual_price * 100.0
            accuracy = 100.0 - error_pct

            st.caption(f"Predicted vs Actual on {target_date.date()}")
            st.metric("Accuracy", f"{accuracy:.2f} %")

            # Nice explanatory chip
            delta_label = (
                f"Predicted ₹{predicted_for_target:,.2f} vs Actual ₹{actual_price:,.2f}"
            )
            if accuracy >= 90:
                st.success(f"✅ {delta_label}")
            else:
                st.warning(f"⚠️ {delta_label}")

        else:
            st.info("Selected date is the most recent trading day — no future actual price available to score the model yet.")
    else:
        st.info("Selected date is the most recent trading day — no future actual price available to score the model yet.")

    
    # =====================================================
    # 7️⃣ Weighted Ensemble Evaluation (2x2 Organized Layout)
    # =====================================================
    st.markdown("---")
    st.markdown("### 🧠 Weighted Ensemble Evaluation")

    try:
        blend = run_weighted_blend(df_upto)

        # First row: Ensemble Models + Weights
        colA, colB = st.columns(2)
        with colA:
            st.metric("⚖️ Ensemble Models", ", ".join(blend["models"]))
        with colB:
            st.metric("🧩 Weights", ", ".join([str(w) for w in blend["weights"]]))

        # Second row: Prediction + Accuracy
        colC, colD = st.columns(2)
        with colC:
            st.metric(
                "🧮 Weighted Blend Next Day Prediction",
                f"₹ {blend['blended_next_day']:,.2f}"
            )

        with colD:
            # Evaluate on the SAME next-trading-day as the main accuracy block
            if selected_ts < df["Date"].max():
                future_dates = df.loc[df["Date"] > selected_ts, "Date"].sort_values()
                if not future_dates.empty:
                    target_date = future_dates.iloc[0]          # next trading day
                    actual_price = df.loc[df["Date"] == target_date, "Close"].iloc[0]

                    error = abs(blend["blended_next_day"] - actual_price)
                    accuracy = max(0.0, 100.0 - (error / actual_price * 100.0))

                    st.metric(
                        "🎯 Weighted Ensemble Accuracy",
                        f"{accuracy:.2f}%",
                        help=f"Evaluated on {target_date.date()} (next trading day)."
                    )
                else:
                    st.caption("📅 No future trading day available to compute accuracy.")
            else:
                st.caption("📅 Accuracy available only for past dates.")


        # Expandable details
        with st.expander("🔍 Model Details"):
            details_df = pd.DataFrame({
                "Model": blend["models"],
                "Weight": blend["weights"],
                "Predicted (Next Day)": blend["pred_each"]
            })
            st.dataframe(details_df)

    except Exception as e:
        st.info(f"⚙️ Weighted blend not available: {e}")

    st.info("""
    🚀 **Now Using Deterministic Trend-Based Forecasts (Placeholder)**
    - Stable for a given date & company (no randomness)
    - Blue region is always fully visible
    - Shape will improve once we plug in your full Step-9 ensemble models
    """)


# =====================================================
# SETTINGS PAGE
# =====================================================
def settings_page():
    st.title("🎨 App Settings & Customization")
    st.write("Toggle theme, clear cache, or reload models (future work).")

    dark_mode = st.toggle("🌙 Dark Mode (very simple demo)")
    if dark_mode:
        st.markdown(
            "<style>body{background-color:#1e1e1e;color:white;}</style>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("🧹 Maintenance & Cache Control")

    from utils.cache_utils import clear_all_streamlit_cache, retrain_all_models, retrain_ensemble

    if st.button("🧠 Retrain All Models"):
        with st.spinner("Retraining all models..."):
            result_df = retrain_all_models()
        st.success("Retraining completed.")
        st.dataframe(result_df)

    if st.button("🔄 Recalculate Ensemble Metrics"):
        with st.spinner("Running ensemble blending again..."):
            ensemble_results = retrain_ensemble()
        st.success("Ensemble recalculated successfully.")
        st.write(ensemble_results)

    if st.button("🧽 Clear Streamlit Cache"):
        msg = clear_all_streamlit_cache()
        st.info(msg)


# =====================================================
# ROUTER
# =====================================================
if choice == "🏠 Dashboard":
    dashboard_page()
elif choice == "ℹ️ About":
    about_page()
elif choice == "⚙️ Settings":
    settings_page()

