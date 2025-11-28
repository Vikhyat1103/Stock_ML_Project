<h1 align="center">📊 Comparative Study of Machine Learning Algorithms for Stock Market Prediction and Automated Trend Forecasting</h1>

<p align="center">
<b>Course:</b> Machine Learning for Electronics <br>
<b>Team 18:</b> Vikhyat Pandey · Akash Ranjan · Subhrajit Kalita <br>
<b>Instructor:</b> Prof. [Professor’s Name]  
</p>

---

## 🧠 Abstract

An end-to-end machine learning system for short-term stock price forecasting and directional trend prediction. The project combines feature engineering, multiple regression/classification models, and a weighted ensemble to generate stable, interpretable forecasts exposed through an interactive dashboard.

---

## 🚀 Overview (concise)

Markets are volatile and complex; this project combines rich feature engineering and a mix of linear and non-linear models with a weighted ensemble to produce both numeric price forecasts and directional (up/down) signals. Results are surfaced through a Streamlit dashboard for easy interpretation.

### 🎯 Project Goals
- Forecast next-day closing prices for selected companies.  
- Predict next-day market direction (up/down).  
- Compare regression and classification algorithms and build a weighted ensemble.  
- Provide a user-facing dashboard for visualization and interpretation.

### 🧠 Highlights
- 10+ years of historical data (yfinance)  
- 20+ technical indicators (RSI, MACD, SMA/EMA, Bollinger Bands, momentum)  
- Regression & classification pipelines with ensemble blending  
- Streamlit dashboard for interactive visualization

---

## 🧱 Project Structure

This reflects the current repository layout.

```text
Stock_ML_Project/
│
├── Data/                         # data storage
│   └── Processed/                # cleaned & feature-engineered datasets (processed CSVs)
│
├── Models/                       # trained model artifacts (if any)
│
├── Notebooks/                    # Jupyter notebooks for each phase
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 03A_EDA.ipynb
│   └── 04_target_creation.ipynb
│
├── Stock_ML_App/                 # Streamlit dashboard source (app.py, pages, utils)
│
├── figures/                      # visualization assets and saved EDA plots
│
├── results/                      # model evaluation outputs / CSVs
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧩 Key Components

| Component | Description |
|------------|-------------|
| 📥 **Data Source** | Yahoo Finance API (`yfinance`) for historical stock data |
| ⚙️ **Feature Engineering** | RSI, MACD, Bollinger Bands, SMA/EMA, Momentum, Lag Features |
| 🔍 **Regression Models** | Linear Regression, Ridge, Random Forest, Gradient Boosting, LightGBM |
| 📈 **Classification Models** | Logistic Regression, Random Forest, Gradient Boosting |
| ⚖️ **Blending Strategy** | Weighted Ensemble of Ridge + Linear + RF + GB |
| 📊 **Visualization** | Streamlit dashboard with accuracy evaluation |
| 📉 **Metrics** | R², RMSE, MAE, Accuracy, F1-score |

---


## ⚙️ Methodology

### 1️⃣ Data Collection & Preprocessing
- Extracted 10+ years of data via **Yahoo Finance API (`yfinance`)**  
- Cleaned missing values and adjusted for splits/dividends  
- Integrated **NIFTY 50 index** as a contextual feature  

### 2️⃣ Feature Engineering
Generated 25+ features, including:
- **Technical Indicators:** SMA, EMA, RSI, MACD, Bollinger Bands  
- **Lag Features:** 3D, 5D, 10D rolling returns  
- **Volatility & Momentum:** Rolling standard deviation and momentum metrics  
- **Seasonality Features:** Day-of-week, month, quarter  

### 3️⃣ Model Building
We created **two parallel pipelines**:
- **Regression Models:** Linear, Ridge, Random Forest, Gradient Boosting, LightGBM  
- **Classification Models:** Logistic Regression, RF, Gradient Boosting  

### 4️⃣ Weighted Ensemble Blending
A **meta-model layer** combined predictions:

$$
P_{final} = w_1 P_{ridge} + w_2 P_{linear} + w_3 P_{rf} + w_4 P_{gb}
$$

Weights were optimized to minimize validation RMSE.

### 5️⃣ Evaluation Metrics
| Model Type | Metrics |
|-------------|----------|
| Regression | R², RMSE, MAE, MAPE |
| Classification | Accuracy, F1, Precision, Recall, ROC-AUC |

---

## 📊 Streamlit Dashboard

The **Streamlit App** provides a fully interactive dashboard.

### ✨ Features:
✅ Predict **Next-Day & 30-Day Prices**  
✅ Compare **Predicted vs Actual** values  
✅ Display **Model Accuracy** dynamically  
✅ Evaluate **Weighted Ensemble Predictions**  
✅ Visualize **Trends with Charts**  
✅ Learn from **About Page Interpretation**

---

## 🧮 Sample Forecast Output

| Metric | Example Output |
|---------|----------------|
| Predicted Price (Next Day) | ₹1,182.85 |
| Predicted Price (Next 30 Days) | ₹1,176.83 |
| Model Accuracy | 98.91% |
| Weighted Ensemble Accuracy | 99.73% |
| Models Used | Ridge, Linear, RF, GB |
| Weights | 0.173, 0.173, 0.427, 0.226 |

---

## 💡 Example Interpretation

Suppose you select **Reliance Industries** and pick **12th September 2022**:

- Predicted next-day price → ₹1,182.85  
- Actual market close → ₹1,195.91  
- Model accuracy → **98.9%**  
- Weighted ensemble confirms consistent prediction ✅  

> 🧠 *This indicates that the ensemble captured short-term behavior accurately, aligning with market movement.*

---

## 💻 Run Locally

```bash
# Clone the repository
git clone https://github.com/Vikhyat1103/Stock_ML_Project.git
cd Stock_ML_Project

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run Stock_ML_App/app.py
```

---

## 🧭 Insights & Learnings

- Weighted blending increased prediction stability.  
- Ridge and Random Forest offered a good bias–variance tradeoff.  
- Accuracy remained high for historical validation data in experiments.  
- Ensemble forecasts were deterministic and repeatable.  
- Streamlit made insights easily interpretable for non-technical users.

---

## 🏁 Conclusion

This project demonstrates how ensemble ML techniques, combined with intelligent feature engineering and visual dashboards, can produce highly accurate and interpretable forecasts.

It bridges data science, finance, and UI design into one cohesive predictive system. With its modular architecture, the system can easily scale to new companies or longer forecasting horizons.

---

## 🧾 References

- Yahoo Finance API Documentation  
- Scikit-learn ML Library  
- Streamlit Official Documentation  
- Kaggle Stock Forecasting Datasets  
- Ensemble Learning Research Papers
