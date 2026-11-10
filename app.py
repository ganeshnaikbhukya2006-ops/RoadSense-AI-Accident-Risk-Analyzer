import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="RoadSense AI", page_icon="🚦", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(135deg,#07111f 0%,#0b1930 55%,#101d35 100%); color:#edf4ff;}
section[data-testid="stSidebar"] {background:#0a1628;border-right:1px solid #203652;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;color:#f2f7ff;}
.hero {padding:26px 30px;border:1px solid #24415f;border-radius:22px;background:linear-gradient(110deg,#102943,#102039 70%,#142c45);margin-bottom:22px;}
.eyebrow {color:#54d6cf;font-weight:700;letter-spacing:2px;font-size:12px;}
.hero h1 {font-size:36px;margin:6px 0;}
.hero p {color:#a9bdd4;margin:0;font-size:15px;}
.metric {background:#10233a;border:1px solid #23415f;border-radius:16px;padding:17px 19px;}
.metric-label {color:#9db2ca;font-size:12px;text-transform:uppercase;letter-spacing:1px;}
.metric-value {color:#f3f8ff;font-family:'Space Grotesk';font-size:27px;font-weight:700;margin-top:5px;}
.panel {background:#0e2035;border:1px solid #203b58;border-radius:18px;padding:18px 20px;}
div[data-testid="stMetric"] {background:#10233a;padding:14px;border:1px solid #23415f;border-radius:14px;}
.stButton>button {background:linear-gradient(90deg,#13b8b0,#2589dc);color:white;border:0;border-radius:10px;font-weight:700;padding:10px 18px;}
.small-note {color:#91a8c2;font-size:12px;}
</style>
""", unsafe_allow_html=True)


def generate_demo_data(n=5000, seed=42):
    """Create synthetic sample records when no external CSV dataset is provided."""
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    day = rng.integers(0, 7, n)
    weather = rng.choice(["Clear", "Rain", "Fog", "Storm"], n, p=[0.55, 0.25, 0.12, 0.08])
    road = rng.choice(["Highway", "Urban", "Rural", "Intersection"], n, p=[0.28, 0.38, 0.18, 0.16])
    traffic = rng.choice(["Low", "Medium", "High"], n, p=[0.25, 0.45, 0.30])
    speed = rng.integers(20, 121, n)
    lighting = np.where((hour >= 19) | (hour <= 5), "Dark", "Daylight")
    visibility = np.select(
        [weather == "Fog", weather == "Storm", weather == "Rain"],
        [rng.integers(100, 500, n), rng.integers(300, 1200, n), rng.integers(500, 2500, n)],
        default=rng.integers(2000, 10000, n)
    )
    risk = (
        (speed > 80) * 1.0 + (weather == "Rain") * 0.7 +
        (weather == "Fog") * 1.2 + (weather == "Storm") * 1.4 +
        (road == "Highway") * 0.3 + (road == "Intersection") * 0.6 +
        (traffic == "High") * 0.5 + (lighting == "Dark") * 0.6 +
        (hour >= 22) * 0.4 + (visibility < 500) * 0.8 +
        rng.normal(0, 0.55, n)
    )
    threshold = np.quantile(risk, 0.68)
    labels = np.where(risk >= threshold, "High",
                      np.where(risk >= threshold - 0.75, "Medium", "Low"))
    return pd.DataFrame({
        "hour": hour, "day_of_week": day, "weather": weather,
        "road_type": road, "traffic_level": traffic, "speed_kmh": speed,
        "lighting": lighting, "visibility_m": visibility, "risk_level": labels
    })

@st.cache_data
def load_data():
    # Use a supplied CSV when available; otherwise create demo records automatically.
    csv_path = "data/road_accident_demo.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return generate_demo_data()

@st.cache_resource
def train_model(data):
    features=["hour","day_of_week","weather","road_type","traffic_level","speed_kmh","lighting","visibility_m"]
    X=data[features]; y=data["risk_level"]
    cat=["weather","road_type","traffic_level","lighting"]
    num=[c for c in features if c not in cat]
    prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",StandardScaler(),num)])
    model=Pipeline([("prep",prep),("knn",KNeighborsClassifier(n_neighbors=5,weights="distance",metric="minkowski",p=2))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model.fit(Xtr,ytr)
    return model, accuracy_score(yte,model.predict(Xte))

st.markdown("""<div class="hero"><div class="eyebrow">INTELLIGENT TRANSPORT • MACHINE LEARNING</div>
<h1>🚦 RoadSense AI</h1><p>Intelligent Road Accident Risk Prediction & Decision Support</p></div>""",unsafe_allow_html=True)

st.markdown("### 📁 Upload your accident-risk dataset")
st.caption("Upload a CSV with the required columns below. If you do not have a dataset yet, the app uses generated demo data.")
template = generate_demo_data(30)
st.download_button(
    "⬇️ Download sample CSV template",
    data=template.to_csv(index=False).encode("utf-8"),
