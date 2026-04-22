import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from matplotlib.patches import Patch

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Pump & Dump Detector",
    page_icon="🚨",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    # XGBoost
    if os.path.exists("models/xgb_model.json"):
        xgb = XGBClassifier()
        xgb.load_model("models/xgb_model.json")
        models["XGBoost"] = xgb
    
    # Random Forest
    if os.path.exists("models/rf_model.joblib"):
        models["Random Forest"] = joblib.load("models/rf_model.joblib")
    
    # LightGBM
    if os.path.exists("models/lgbm_model.joblib"):
        models["LightGBM"] = joblib.load("models/lgbm_model.joblib")
    
    # SVM
    if os.path.exists("models/svm_model.joblib"):
        models["SVM"] = joblib.load("models/svm_model.joblib")
    
    return models

# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    files = glob.glob("data/raw/*_labeled.csv")
    dfs = {}
    for f in files:
        # Extract coin name from filename in a cross-platform way
        coin = os.path.basename(f).replace("_labeled.csv", "")
        df = pd.read_csv(f, index_col="timestamp", parse_dates=True)
        # Calculate derived features used for training
        df["high_low_range"]  = (df["high"] - df["low"]) / df["low"]
        df["close_open_diff"] = (df["close"] - df["open"]) / df["open"]
        dfs[coin] = df
    return dfs

# Initialize model and data
try:
    available_models = load_models()
    all_data = load_data()
except Exception as e:
    st.error(f"❌ Error loading model or data: {e}")
    st.stop()

if not all_data:
    st.error("❌ No data found in data/raw/*_labeled.csv")
    st.stop()

if not available_models:
    st.error("❌ No models found in models/ folder.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
coin     = st.sidebar.selectbox("Select Coin", list(all_data.keys()))
model_name = st.sidebar.selectbox("Select Model", list(available_models.keys()))
window   = st.sidebar.slider("Lookback Window (days)", 50, 500, 200)

features = ["open", "high", "low", "close", "volume", "trades",
            "high_low_range", "close_open_diff"]

# ── Data for selected coin ───────────────────────────────────
df = all_data[coin].copy().tail(window)
X  = df[features].dropna()
df = df.loc[X.index]

# ── Predictions ──────────────────────────────────────────────
current_model = available_models[model_name]
df["prediction"]  = current_model.predict(X)
df["risk_score"]  = current_model.predict_proba(X)[:, 1]

# ── Header ───────────────────────────────────────────────────
st.title("🚨 Crypto Pump & Dump Detector")
st.markdown(f"Analyzing **{coin}** with **{model_name}** — last **{window}** days")

# ── Metrics row ──────────────────────────────────────────────
total     = len(df)
detected  = int(df["prediction"].sum())
avg_risk  = df["risk_score"].mean()
max_risk  = df["risk_score"].max()

c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Total Candles",    total)
c2.metric("🚨 P&D Events",       detected)
c3.metric("⚠️ Avg Risk Score",   f"{avg_risk:.2%}")
c4.metric("🔴 Max Risk Score",   f"{max_risk:.2%}")

st.divider()

# ── Price chart with P&D highlights ──────────────────────────
st.subheader("📈 Price Chart with Detected Events")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df.index, df["close"], color="steelblue", linewidth=1, label="Close Price")

pump_points = df[df["prediction"] == 1]
ax.scatter(pump_points.index, pump_points["close"],
           color="red", zorder=5, s=60, label=f"🚨 {model_name} Detected")

ax.set_ylabel("Price (USDT)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# ── Volume chart ─────────────────────────────────────────────
st.subheader("📊 Volume Chart")

fig2, ax2 = plt.subplots(figsize=(14, 3))
colors = ["red" if p == 1 else "steelblue" for p in df["prediction"]]
ax2.bar(df.index, df["volume"], color=colors, width=1.5)
ax2.set_ylabel("Volume")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)

ax2.legend(handles=[
    Patch(color="steelblue", label="Normal"),
    Patch(color="red",       label="P&D Detected")
])
plt.tight_layout()
st.pyplot(fig2)

# ── Risk score chart ─────────────────────────────────────────
st.subheader("🎯 Risk Score Over Time")

fig3, ax3 = plt.subplots(figsize=(14, 3))
ax3.fill_between(df.index, df["risk_score"], alpha=0.4, color="orange")
ax3.plot(df.index, df["risk_score"], color="orange", linewidth=1)
ax3.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Threshold (0.5)")
ax3.set_ylabel("Risk Score")
ax3.set_xlabel("Date")
ax3.set_ylim(0, 1)
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)

st.divider()

# ── Recent detections table ───────────────────────────────────
st.subheader("🗂️ Recent P&D Detections")

recent = df[df["prediction"] == 1][
    ["close", "volume", "risk_score"]
].sort_index(ascending=False).head(10)

if len(recent) == 0:
    st.success("✅ No P&D events detected in this window.")
else:
    recent.index = recent.index.strftime("%Y-%m-%d")
    recent.columns = ["Close Price", "Volume", "Risk Score"]
    recent["Risk Score"] = recent["Risk Score"].map("{:.2%}".format)
    st.dataframe(recent, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("Built with Scikit-Learn, XGBoost & LightGBM | Data: pycoingecko | Deployed on Streamlit Cloud")
