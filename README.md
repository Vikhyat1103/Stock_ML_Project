\# 🧠 Comparative Study of Machine Learning Algorithms for Stock Market Prediction and Automated Trend Forecasting



\### 📚 Course: ML for Electronics  

\### 👥 Team 18 — Vikhyat Pandey, Akash Ranjan, Subhrajit Kalita  



---



\## 🎯 Abstract

This project focuses on building an automated machine learning pipeline that compares various ML and DL algorithms for short-term stock market prediction.  

The aim is to analyze and forecast next-day stock prices and movement trends (up/down) for selected Indian equities using both classical and deep learning approaches.



We utilize historical OHLCV data for \*\*Reliance\*\*, \*\*TCS\*\*, and \*\*HDFC Bank\*\*, with the \*\*NIFTY Index\*\* as a contextual feature.  

The pipeline includes data collection, preprocessing, feature engineering (technical indicators), exploratory analysis, target creation, model training, evaluation, and visualization.



---



\## 🧱 Project Structure



Stock\_ML\_Project/

│

├── data/

│ ├── raw/ # Raw downloaded CSVs (ignored in .gitignore)

│ └── processed/ # Cleaned \& feature-engineered data

│ ├── stocks\_clean.csv

│ ├── stocks\_features.csv

│ ├── reliance\_model\_ready.csv

│ ├── tcs\_model\_ready.csv

│ └── hdfcbank\_model\_ready.csv

│

├── notebooks/ # Jupyter notebooks for each phase

│ ├── 01\_data\_collection.ipynb

│ ├── 02\_preprocessing.ipynb

│ ├── 03\_feature\_engineering.ipynb

│ ├── 03A\_EDA.ipynb

│ └── 04\_target\_creation.ipynb

│

├── figures/ # Visualization and EDA plots

│ └── EDA/

│ ├── price\_trends.png

│ ├── returns\_distribution.png

│ ├── moving\_averages.png

│ ├── RSI.png

│ └── correlation\_heatmap.png

│

├── src/ # Future pipeline code modules

│ ├── data\_download.py

│ ├── preprocessing.py

│ └── feature\_engineering.py

│

├── requirements.txt # Python dependencies

├── .gitignore # Ignore large/raw data and system files

└── README.md # Project overview and documentation





---



\## ⚙️ Technologies Used

\- \*\*Language:\*\* Python  

\- \*\*Libraries:\*\* `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `yfinance`  

\- \*\*Environment:\*\* Jupyter Notebook / VS Code  

\- \*\*Version Control:\*\* Git + GitHub  



---



\## 🚀 Current Progress (Mid-Sem)

✅ Data Collection  

✅ Data Cleaning \& Preprocessing  

✅ Feature Engineering (MA, EMA, RSI, Returns)  

✅ EDA \& Visualization  

✅ Target Creation \& Train-Test Split  



---



\## 🔮 Future Work (Post-Midsem)

\- Train regression models (Linear, Random Forest, SVR, XGBoost)

\- Train classification models (Logistic, KNN, Random Forest, SVM)

\- Add deep learning (MLP, CNN, LSTM)

\- Evaluate \& compare models on RMSE, MAE, Accuracy, F1, Directional Accuracy

\- Build automated prediction pipeline and simple demo app (Streamlit)

\- Write final report, video presentation, and project documentation



---



\## 🏗️ How to Reproduce

1\. Clone the repository:

&nbsp;  ```bash

&nbsp;  git clone https://github.com/Vikhyat1103/Stock\_ML\_Project.git

&nbsp;  cd Stock\_ML\_Project



