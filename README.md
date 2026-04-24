# 📈 Pump and Dump Detection using ML Algorithms

A machine learning project that detects **pump-and-dump schemes** in financial markets — a form of market manipulation where the price of an asset is artificially inflated ("pumped") through misleading activity, then sold off ("dumped"), leaving other investors at a loss.

---

## 🧠 Project Overview

This project applies supervised machine learning algorithms to identify pump-and-dump patterns in historical market data. The pipeline covers data collection from financial APIs, feature engineering on price/volume signals, model training, and evaluation.

---

## 📁 Repository Structure

```
Pump-and-dump-detection-using-ML-algorithms/
│
├── data/
│   └── raw/                    # Raw collected market data (CSV files)
│
├── data_collection.ipynb       # Data collection pipeline using financial APIs
├── Model.ipynb                 # Feature engineering, model training & evaluation
│
└── README.md
```

---

## 🔄 Workflow

### 1. Data Collection (`data_collection.ipynb`)
- Fetches historical OHLCV (Open, High, Low, Close, Volume) data for selected stocks/crypto assets
- Stores raw data under `data/raw/`
- Labels data with pump-and-dump event windows

### 2. Modelling (`Model.ipynb`)
- **Feature Engineering** — derives indicators such as:
  - Price return over rolling windows
  - Volume surge ratio
  - Volatility metrics
  - Moving averages and their deviations
- **Model Training** — experiments with multiple ML classifiers:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting / XGBoost
- **Evaluation** — compares models using accuracy, precision, recall, F1-score, and ROC-AUC; addresses class imbalance via appropriate metrics and sampling strategies

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Notebooks | Jupyter Notebook |
| Data | pandas, NumPy |
| ML | scikit-learn, XGBoost |
| Visualization | matplotlib, seaborn |
| Data Source | PyCoinGecko AI |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn yfinance jupyter
```

### Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Harishkumar4771/Pump-and-dump-detection-using-ML-algorithms.git
   cd Pump-and-dump-detection-using-ML-algorithms
   ```

2. Start with data collection:
   ```bash
   jupyter notebook data_collection.ipynb
   ```

3. Then run the modelling pipeline:
   ```bash
   jupyter notebook Model.ipynb
   ```

---

## 📊 Key Concepts

**Pump and Dump** is a market manipulation scheme where:
- Coordinated buying activity artificially inflates an asset's price
- Media hype or fake news amplifies the effect
- Manipulators sell at the peak, crashing the price and causing losses for retail investors

ML models can detect these patterns by learning from abnormal spikes in price and volume that historically coincide with manipulation events.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not intended to be used as financial advice or a trading signal system.

---
## Future scopes
   -This project can be further developed into a NLP project,it checks the opinions and hypes about the cryptocurrency coins on social media.
   -This can be also extended as a Deep Learning project for better detection and integrate with real time alert system.

## 👤 Author

**Harishkumar4771**  
[GitHub Profile](https://github.com/Harishkumar4771)
