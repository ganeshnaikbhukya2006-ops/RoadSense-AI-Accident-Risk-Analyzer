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
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="RoadSense AI",
    page_icon="🚦",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg,#07111f,#0b1930,#101d35);
    color: #edf4ff;
}

section[data-testid="stSidebar"] {
    background: #0a1628;
    border-right: 1px solid #203652;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: #f2f7ff;
}

.hero {
    padding: 26px 30px;
    border: 1px solid #24415f;
    border-radius: 22px;
    background: linear-gradient(110deg,#102943,#102039,#142c45);
    margin-bottom: 22px;
}

.eyebrow {
    color: #54d6cf;
    font-weight: 700;
    letter-spacing: 2px;
    font-size: 12px;
}

.panel {
    background: #0e2035;
    border: 1px solid #203b58;
    border-radius: 18px;
    padding: 18px 20px;
}

div[data-testid="stMetric"] {
    background: #10233a;
    padding: 14px;
    border: 1px solid #23415f;
    border-radius: 14px;
}

.stButton>button {
    background: linear-gradient(90deg,#13b8b0,#2589dc);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 18px;
}

.small-note {
    color: #91a8c2;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- DEMO DATA GENERATOR ----------------

def generate_demo_data(n=5000, seed=42):

    rng = np.random.default_rng(seed)

    hour = rng.integers(0, 24, n)
    day = rng.integers(0, 7, n)

    weather = rng.choice(
        ["Clear", "Rain", "Fog", "Storm"],
        n,
        p=[0.55, 0.25, 0.12, 0.08]
    )

    road = rng.choice(
        ["Highway", "Urban", "Rural", "Intersection"],
        n
    )

    traffic = rng.choice(
        ["Low", "Medium", "High"],
        n
    )

    speed = rng.integers(20, 121, n)

    lighting = np.where(
        (hour >= 19) | (hour <= 5),
        "Dark",
        "Daylight"
    )

    visibility = np.select(
        [
            weather == "Fog",
            weather == "Storm",
            weather == "Rain"
        ],
        [
            rng.integers(100, 500, n),
            rng.integers(300, 1200, n),
            rng.integers(500, 2500, n)
        ],
        default=rng.integers(2000, 10000, n)
    )

    risk = (
        (speed > 80) * 1.0 +
        (weather == "Rain") * 0.7 +
        (weather == "Fog") * 1.2 +
        (weather == "Storm") * 1.4 +
        (road == "Highway") * 0.3 +
        (road == "Intersection") * 0.6 +
        (traffic == "High") * 0.5 +
        (lighting == "Dark") * 0.6 +
        (hour >= 22) * 0.4 +
        (visibility < 500) * 0.8 +
        rng.normal(0, 0.55, n)
    )

    threshold = np.quantile(risk, 0.68)

    labels = np.where(
        risk >= threshold,
        "High",
        np.where(
            risk >= threshold - 0.75,
            "Medium",
            "Low"
        )
    )

    return pd.DataFrame({
        "hour": hour,
        "day_of_week": day,
        "weather": weather,
        "road_type": road,
        "traffic_level": traffic,
        "speed_kmh": speed,
        "lighting": lighting,
        "visibility_m": visibility,
        "risk_level": labels
    })


# ---------------- DATA LOADING ----------------

def load_data():

    csv_path = "data/road_accident_demo.csv"

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    return generate_demo_data()


# ---------------- MACHINE LEARNING MODEL ----------------

def train_model(data):

    features = [
        "hour",
        "day_of_week",
        "weather",
        "road_type",
        "traffic_level",
        "speed_kmh",
        "lighting",
        "visibility_m"
    ]

    X = data[features]
    y = data["risk_level"]

    categorical = [
        "weather",
        "road_type",
        "traffic_level",
        "lighting"
    ]

    numerical = [
        c for c in features if c not in categorical
    ]

    preprocessing = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ),
        (
            "numerical",
            StandardScaler(),
            numerical
        )
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="minkowski",
                p=2
            )
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">
        INTELLIGENT TRANSPORT • MACHINE LEARNING
    </div>

    <h1>🚦 RoadSense AI</h1>

    <p>
        Intelligent Road Accident Risk Prediction
        & Decision Support
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- CSV UPLOAD ----------------

st.markdown("### 📁 Upload your accident-risk dataset")

st.caption(
    "Upload your own CSV dataset to train the KNN model."
)

template = generate_demo_data(30)

st.download_button(
    "⬇️ Download sample CSV template",
    template.to_csv(index=False).encode("utf-8"),
    "road_accident_dataset_template.csv",
    "text/csv"
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

required_columns = [
    "hour",
    "day_of_week",
    "weather",
    "road_type",
    "traffic_level",
    "speed_kmh",
    "lighting",
    "visibility_m",
    "risk_level"
]

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        missing = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing:
            st.error(
                "Missing columns: " + ", ".join(missing)
            )
            st.stop()

        data = data[required_columns].dropna().copy()

        data["risk_level"] = (
            data["risk_level"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        invalid = set(data["risk_level"]) - {
            "Low", "Medium", "High"
        }

        if invalid:
            st.error(
                "Risk labels must be Low, Medium or High."
            )
            st.stop()

        numeric_columns = [
            "hour",
            "day_of_week",
            "speed_kmh",
            "visibility_m"
        ]

        for col in numeric_columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

        data = data.dropna()

        if len(data) < 15:
            st.error("Upload at least 15 valid records.")
            st.stop()

        if data["risk_level"].nunique() != 3:
            st.error(
                "Dataset must contain Low, Medium and High classes."
            )
            st.stop()

        if data["risk_level"].value_counts().min() < 2:
            st.error(
                "Each risk category needs at least 2 records."
            )
            st.stop()

        st.success(
            f"Successfully uploaded {len(data):,} records!"
        )

    except Exception as e:

        st.error(f"CSV processing error: {e}")
        st.stop()

else:

    data = load_data()

    st.info(
        "Using generated demonstration data. "
        "Upload your own CSV to train on your dataset."
    )


# ---------------- TRAIN MODEL ----------------

model, accuracy = train_model(data)


# ---------------- SIDEBAR INPUTS ----------------

with st.sidebar:

    st.markdown("## ⚙️ Risk Scenario")

    st.caption("Enter road and environmental conditions.")

    hour_in = st.slider(
        "Time of day",
        0, 23, 18
    )

    day_in = st.selectbox(
        "Day of week",
        [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ]
    )

    weather_in = st.selectbox(
        "Weather",
        ["Clear", "Rain", "Fog", "Storm"]
    )

    road_in = st.selectbox(
        "Road type",
        ["Highway", "Urban", "Rural", "Intersection"]
    )

    traffic_in = st.select_slider(
        "Traffic density",
        ["Low", "Medium", "High"],
        value="Medium"
    )

    speed_in = st.slider(
        "Vehicle speed (km/h)",
        20, 120, 65
    )

    light_in = st.selectbox(
        "Lighting",
        ["Daylight", "Dark"]
    )

    visibility_in = st.slider(
        "Visibility (metres)",
        100, 10000, 2500, step=100
    )

    predict = st.button(
        "🚦 PREDICT ACCIDENT RISK",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------

st.markdown("### Live Risk Assessment")

a, b, c = st.columns(3)

a.metric(
    "Model Validation Accuracy",
    f"{accuracy * 100:.1f}%"
)

b.metric(
    "Training Records",
    f"{len(data):,}"
)

c.metric(
    "Risk Categories",
    "3 Levels"
)

st.caption(
    "Results are based on the selected dataset. "
    "Synthetic demo results are not real-world safety guarantees."
)

features = [
    "hour",
    "day_of_week",
    "weather",
    "road_type",
    "traffic_level",
    "speed_kmh",
    "lighting",
    "visibility_m"
]

days = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

day_num = days.index(day_in)

sample = pd.DataFrame([{
    "hour": hour_in,
    "day_of_week": day_num,
    "weather": weather_in,
    "road_type": road_in,
    "traffic_level": traffic_in,
    "speed_kmh": speed_in,
    "lighting": light_in,
    "visibility_m": visibility_in
}])

# Predict when button is clicked
if predict:

    probabilities = model.predict_proba(sample)[0]

    classes = model.named_steps["knn"].classes_

    predicted = classes[np.argmax(probabilities)]

    confidence = float(np.max(probabilities)) * 100

    colors = {
        "Low": "#21c58e",
        "Medium": "#f4b942",
        "High": "#ff6577"
    }

    left, right = st.columns([1.05, 1])

    with left:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown("#### Predicted Risk Level")

        st.markdown(
            f'<h1 style="color:{colors[predicted]};'
            f'font-size:42px;margin:0">'
            f'{predicted.upper()} RISK</h1>',
            unsafe_allow_html=True
        )

        st.progress(int(confidence))

        st.write(
            f"Model confidence: **{confidence:.1f}%**"
        )

        tips = {
            "Low": "Continue following speed limits and normal defensive-driving practices.",
            "Medium": "Reduce speed and increase following distance.",
            "High": "Avoid unnecessary travel if possible and wait for safer conditions."
        }

        st.info(
            "**Suggested action:** " + tips[predicted]
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown("#### Class Probability")

        probability_data = pd.DataFrame({
            "Risk level": classes,
            "Probability": probabilities
        })

        fig = px.bar(
            probability_data,
            x="Risk level",
            y="Probability",
            color="Risk level",
            color_discrete_map=colors,
            range_y=[0, 1]
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=270,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

else:

    st.info(
        "👈 Select road conditions in the sidebar "
        "and click PREDICT ACCIDENT RISK to see the result."
    )


# ---------------- DATASET ANALYTICS ----------------

st.markdown("### 📊 Dataset Intelligence")

x, y = st.columns(2)

colors = {
    "Low": "#21c58e",
    "Medium": "#f4b942",
    "High": "#ff6577"
}

with x:

    counts = (
        data["risk_level"]
        .value_counts()
        .rename_axis("Risk level")
        .reset_index(name="Records")
    )

    fig = px.pie(
        counts,
        names="Risk level",
        values="Records",
        hole=0.58,
        color="Risk level",
        color_discrete_map=colors
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=310
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with y:

    fig = px.histogram(
        data,
        x="speed_kmh",
        color="risk_level",
        nbins=24,
        barmode="overlay",
        opacity=0.78,
        color_discrete_map=colors
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=310
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ---------------- SAMPLE RECORDS ----------------

with st.expander("🔎 Explore Sample Records"):

    st.dataframe(
        data.sample(
            min(12, len(data)),
            random_state=4
        ),
        use_container_width=True,
        hide_index=True
    )


# ---------------- ABOUT MODEL ----------------

with st.expander("🧠 About the Machine Learning Model"):

    st.write("""
    The application uses the K-Nearest Neighbors (KNN)
    classification algorithm.

    KNN compares a new road scenario with the five
    closest training records and predicts the risk
    category using distance-weighted voting.

    Numerical features are standardized, and
    categorical features are one-hot encoded.
    """)

    st.write("""
    The built-in dataset is synthetically generated
    for academic demonstration. Replace it with
    verified accident records for research use.
    """)


# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    '<p class="small-note">'
    'RoadSense AI • Academic Project Prototype • '
    'Not an emergency service or a substitute '
    'for official road-safety systems.'
    '</p>',
    unsafe_allow_html=True
)
