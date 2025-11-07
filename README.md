<h1 align="center">📊 Comparative Study of Machine Learning Algorithms for Stock Market Prediction and Automated Trend Forecasting</h1>

<p align="center">
<b>Course:</b> ML for Electronics <br>
<b>Team 18:</b> Vikhyat Pandey · Akash Ranjan · Subhrajit Kalita <br>
<b>Instructor:</b> Prof. [Professor’s Name]  
</p>

---

## 🧠 Abstract
This project aims to design and evaluate an **automated machine learning pipeline** for short-term **stock market prediction and trend forecasting**.  
We compare the performance of multiple **classical ML and deep learning algorithms** for predicting **next-day closing prices** and **directional movement (up/down)** of selected Indian equities.  

Data is collected using the **Yahoo Finance API (`yfinance`)** for:
- **Reliance Industries (RELIANCE.NS)**
- **Tata Consultancy Services (TCS.NS)**
- **HDFC Bank (HDFCBANK.NS)**
- **NIFTY 50 Index (^NSEI)** – used as a contextual feature  

The pipeline includes:
- Data Collection 📥  
- Preprocessing & Cleaning 🧹  
- Feature Engineering ⚙️  
- Exploratory Data Analysis (EDA) 📈  
- Target Creation 🎯  
- Model Training & Evaluation 🤖  

---

## 🧱 Project Structure

```bash
Stock_ML_Project/
│
├── data/
│   ├── raw/                  # Raw data downloaded using yfinance
│   └── processed/            # Cleaned and feature-engineered datasets
│       ├── stocks_clean.csv
│       ├── stocks_features.csv
│       ├── reliance_model_ready.csv
│       ├── tcs_model_ready.csv
│       └── hdfcbank_model_ready.csv
│
├── notebooks/                # Jupyter notebooks (each stage of the pipeline)
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 03A_EDA.ipynb
│   └── 04_target_creation.ipynb
│
├── figures/                  # Visualizations & EDA plots
│   └── EDA/
│       ├── price_trends.png
│       ├── returns_distribution.png
│       ├── moving_averages.png
│       ├── RSI.png
│       └── correlation_heatmap.png
│
├── src/                      # Scripts for reusable code (pipeline ready)
│   ├── data_download.py
│   ├── preprocessing.py
│   └── feature_engineering.py
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignore raw/large data and environment files
└── README.md                 # Project documentation
```

## ⚙️ **Technologies Used**

- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `yfinance`  
- **Environment:** Jupyter Notebook / VS Code  
- **Version Control:** Git + GitHub  

---

## 🚀 **Current Progress (Mid-Sem)**

✅ **Data Collection**  
✅ **Data Cleaning & Preprocessing**  
✅ **Feature Engineering:** MA, EMA, RSI, Returns  
✅ **EDA & Visualization**  
✅ **Target Creation & Train-Test Split**  

---

## 🔮 **Future Work (Post-Midsem)**

- 🔹 Train regression models — *Linear, Random Forest, SVR, XGBoost*  
- 🔹 Train classification models — *Logistic, KNN, Random Forest, SVM*  
- 🔹 Add deep learning models — *MLP, CNN, LSTM*  
- 🔹 Evaluate & compare models on **RMSE, MAE, Accuracy, F1, Directional Accuracy**  
- 🔹 Build automated prediction pipeline and simple demo app (*Streamlit*)  
- 🔹 Write final report, video presentation, and documentation  

---

## 🏗️ **How to Reproduce**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vikhyat1103/Stock_ML_Project.git
   cd Stock_ML_Project
