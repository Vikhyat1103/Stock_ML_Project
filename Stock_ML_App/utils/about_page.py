import streamlit as st

def about_page():
    # --- Header ---
    st.markdown("<h1 style='font-size:42px;'>ℹ️ About This Dashboard</h1>", unsafe_allow_html=True)
    st.caption("Understand how the AI-powered stock prediction system works and how to interpret its outputs.")
    st.markdown("---")

    # --- Overview Section ---
    with st.container():
        st.markdown("### 📘 Overview")
        st.markdown("""
        This interactive dashboard leverages **machine learning models** to forecast stock prices for over **100+ companies**, 
        including major Indian and global corporations.  
        
        It combines **traditional regression algorithms** with **modern ensemble learning techniques** 
        to deliver accurate, real-time short-term forecasts and performance analytics.
        """)

        st.success("💡 The system automatically adjusts for different time windows, ensuring the most recent data drives predictions.")

    # --- How It Works ---
    st.markdown("---")
    with st.expander("⚙️ How It Works — Click to Expand", expanded=True):
        st.markdown("""
        - **📊 Data Source:** Historical stock data fetched live from **Yahoo Finance API**
        - **🧠 Models Used:** Ridge Regression, Linear Regression, Random Forest, and Gradient Boosting
        - **⚖️ Ensemble Method:** Weighted blending of multiple models to improve robustness
        - **🎯 Accuracy Metric:** Comparison of predicted vs actual price for the next trading day
        """)
        st.info("Our ensemble achieves accuracy close to **99% on stable stocks** under normal conditions.")

    # --- Dashboard Insights ---
    st.markdown("---")
    st.markdown("### 📈 What You’ll Find in the Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - 🟢 **Next-Day Forecast:** Predicts price movement (rise/fall)
        - 🔵 **30-Day Forecast:** Projects broader trendline
        - 🧩 **Model Accuracy:** Evaluates precision using real data  
        """)
    with col2:
        st.markdown("""
        - ⚗️ **Weighted Ensemble:** Combines multiple models using adaptive weights  
        - 📏 **Trend Indicators:** Visual cues for bullish/bearish movement  
        - 📑 **Transparent Results:** Forecasts are fully reproducible  
        """)

    st.success("✅ The forecasts shown on the Dashboard are deterministic — the same date and company will always produce identical results.")

    # --- Example Section ---
    st.markdown("---")
    st.markdown("### 🧩 Example Interpretation")
    st.markdown("""
    Suppose you select **Reliance Industries** and pick **12th September 2022**:
    - The model predicts **₹1,182.85** for the next trading day.  
    - The actual price on **13th September 2022** was **₹1,195.91**.  
    - That yields an **accuracy of 98.9%**, indicating excellent model performance.  
    - The weighted ensemble confirms consistency across models.
    """)
    st.info("📊 A close match between Predicted and Actual prices means the ensemble captured recent market behavior accurately.")

    # Optional demo chart/image
    with st.expander("🖼️ View Example Forecast Chart"):
        st.image("assets/demo_forecast_example.png", caption="Example: Predicted vs Actual Price", use_container_width=True)

    # --- Future Enhancements ---
    st.markdown("---")
    st.markdown("### 🚀 Future Enhancements")
    st.markdown("""
    - 🧭 Model comparison visualizations  
    - 🧩 Feature importance breakdowns  
    - 📰 Integration with market sentiment & news data  
    - 📈 Real-time adaptive retraining pipeline  
    """)

    # --- Author Note ---
    st.markdown("---")
    st.markdown("### 👨‍💻 Author Notes")
    st.markdown("""
    This project was built by **Vikhyat Pandey** — combining **Data Science**, **Machine Learning**, and **Finance**  
    to build an accessible, educational, and powerful predictive analytics tool.
    """)
    st.caption("Version 2.0 • Updated November 2025 • Streamlit-powered")

    st.success("💬 Tip: Try switching between companies and dates on the Dashboard — the AI adapts dynamically to every dataset.")
