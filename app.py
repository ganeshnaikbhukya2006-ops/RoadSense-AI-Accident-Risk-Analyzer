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

@st.cache_data
def load_data():
    return pd.read_csv("data/road_accident_demo.csv")

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

data=load_data()
model, accuracy=train_model(data)
st.markdown("""<div class="hero"><div class="eyebrow">INTELLIGENT TRANSPORT • MACHINE LEARNING</div>
<h1>🚦 RoadSense AI</h1><p>Intelligent Road Accident Risk Prediction & Decision Support</p></div>""",unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Risk scenario")
    st.caption("Enter road and environment conditions")
    hour_in=st.slider("Time of day",0,23,18,format="%d:00")
    day_in=st.selectbox("Day of week",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    weather_in=st.selectbox("Weather",["Clear","Rain","Fog","Storm"])
    road_in=st.selectbox("Road type",["Highway","Urban","Rural","Intersection"])
    traffic_in=st.select_slider("Traffic density",["Low","Medium","High"],value="Medium")
    speed_in=st.slider("Vehicle speed (km/h)",20,120,65)
    light_in=st.selectbox("Lighting",["Daylight","Dark"],index=1 if hour_in>=19 or hour_in<=5 else 0)
    visibility_in=st.slider("Visibility (metres)",100,10000,2500,step=100)
    predict=st.button("PREDICT ACCIDENT RISK",use_container_width=True)

st.markdown("### Live risk assessment")
a,b,c=st.columns(3)
a.metric("Model validation accuracy",f"{accuracy*100:.1f}%")
b.metric("Training records",f"{len(data):,}")
c.metric("Risk categories","3 levels")
st.caption("Model trained on a generated demonstration dataset. Accuracy is a classroom-demo metric, not a real-world safety guarantee.")

features=["hour","day_of_week","weather","road_type","traffic_level","speed_kmh","lighting","visibility_m"]
day_num=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day_in)
sample=pd.DataFrame([{"hour":hour_in,"day_of_week":day_num,"weather":weather_in,"road_type":road_in,"traffic_level":traffic_in,"speed_kmh":speed_in,"lighting":light_in,"visibility_m":visibility_in}])
probs=model.predict_proba(sample)[0]
classes=model.named_steps["rf"].classes_
pred=classes[int(np.argmax(probs))]
confidence=float(np.max(probs))*100
colors={"Low":"#21c58e","Medium":"#f4b942","High":"#ff6577"}
left,right=st.columns([1.05,1])
with left:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.markdown("#### Predicted risk level")
    st.markdown(f'<h1 style="color:{colors[pred]};font-size:42px;margin:0">{pred.upper()} RISK</h1>',unsafe_allow_html=True)
    st.progress(int(confidence))
    st.write(f"Model confidence: **{confidence:.1f}%**")
    tips={"Low":"Continue following speed limits and normal defensive-driving practices.","Medium":"Reduce speed, increase following distance and stay alert for changing road conditions.","High":"Avoid unnecessary travel if possible. Reduce speed substantially and use a safer route or wait for conditions to improve."}
    st.info("**Suggested action:** "+tips[pred])
    st.markdown('</div>',unsafe_allow_html=True)
with right:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.markdown("#### Class probability")
    p=pd.DataFrame({"Risk level":classes,"Probability":probs})
    fig=px.bar(p,x="Risk level",y="Probability",color="Risk level",color_discrete_map=colors, range_y=[0,1])
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=270,margin=dict(l=5,r=5,t=10,b=5),showlegend=False)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('</div>',unsafe_allow_html=True)

st.markdown("### Dataset intelligence")
x,y=st.columns(2)
with x:
    counts=data["risk_level"].value_counts().rename_axis("Risk level").reset_index(name="Records")
    fig=px.pie(counts,names="Risk level",values="Records",hole=.58,color="Risk level",color_discrete_map=colors)
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=310,margin=dict(t=15,b=5))
    st.plotly_chart(fig,use_container_width=True)
with y:
    fig=px.histogram(data,x="speed_kmh",color="risk_level",nbins=24,barmode="overlay",opacity=.78,color_discrete_map=colors)
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=310,margin=dict(t=15,b=5),legend_title_text="Risk")
    st.plotly_chart(fig,use_container_width=True)

with st.expander("🔎 Explore sample records"):
    st.dataframe(data.sample(min(12,len(data)),random_state=4),use_container_width=True,hide_index=True)
with st.expander("🧠 About the machine-learning model"):
    st.write("A K-Nearest Neighbors (KNN) classifier compares a new road scenario with the five most similar training records. Categorical inputs are one-hot encoded and numerical inputs are scaled in a preprocessing pipeline. Distance-weighted voting gives closer records more influence.")
    st.write("The data included in this project is synthetically generated for academic demonstration. Replace it with a properly licensed, verified accident dataset and retrain before making any operational decisions.")

st.markdown("---")
st.markdown('<p class="small-note">RoadSense AI • Academic project prototype • Not an emergency service or a substitute for official road-safety systems.</p>',unsafe_allow_html=True)
