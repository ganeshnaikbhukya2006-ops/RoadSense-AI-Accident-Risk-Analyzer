
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

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="RoadSense AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. CUSTOM CSS DESIGN
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #07111f, #0b1930, #101d35);
    color: #edf4ff;
    font-family: 'DM Sans', sans-serif;
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
    padding: 28px;
    border: 1px solid #24415f;
    border-radius: 20px;
    background: linear-gradient(110deg, #102943, #102039, #142c45);
    margin-bottom: 25px;
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
    padding: 20px;
}

div[data-testid="stMetric"] {
    background: #10233a;
    padding: 15px;
    border: 1px solid #23415f;
    border-radius: 14px;
}

.stButton > button {
    background: linear-gradient(90deg, #13b8b0, #2589dc);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}

.small-note {
    color: #91a8c2;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. DEMO DATA GENERATOR
# =====================================================

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
        n,
        p=[0.28, 0.38, 0.18, 0.16]
    )

    traffic = rng.choice(
        ["Low", "Medium", "High"],
        n,
        p=[0.25, 0.45, 0.30]
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
        (speed > 80) * 1.0
        + (weather == "Rain") * 0.7
        + (weather == "Fog") * 1.2
        + (weather == "Storm") * 1.4
        + (road == "Highway") * 0.3
        + (road == "Intersection") * 0.6
        + (traffic == "High") * 0.5
        + (lighting == "Dark") * 0.6
        + (hour >= 22) * 0.4
        + (visibility < 500) * 0.8
        + rng.normal(0, 0.55, n)
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


# =====================================================
# 4. MACHINE LEARNING MODEL - KNN
# =====================================================

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

    categorical = [
        "weather",
        "road_type",
        "traffic_level",
        "lighting"
    ]

    numerical = [
        col for col in features
        if col not in categorical
    ]

    X = data[features]
    y = data["risk_level"]

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


# =====================================================
# 5. DASHBOARD HEADER
# =====================================================

st.markdown(
    '<div class="eyebrow">INTELLIGENT TRANSPORT • MACHINE LEARNING</div>',
    unsafe_allow_html=True
)

st.title("🚦 RoadSense AI")

st.subheader(
    "Intelligent Road Accident Risk Prediction & Decision Support"
)

st.markdown("---")


# =====================================================
# 6. CSV DATASET UPLOAD
# =====================================================

st.markdown("### 📁 Upload Your Accident-Risk Dataset")

st.caption(
    "Upload your CSV dataset to train the KNN model."
)

template = generate_demo_data(30)

st.download_button(
    label="⬇️ Download Sample CSV Template",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="road_accident_dataset_template.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader(
    "Choose your CSV file",
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

        missing_columns = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing_columns:
            st.error(
                "Missing columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        data = data[required_columns].copy()

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

        data["risk_level"] = (
            data["risk_level"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        data = data.dropna()

        if len(data) < 15:
            st.error(
                "Please upload at least 15 valid records."
            )
            st.stop()

        if data["risk_level"].nunique() != 3:
            st.error(
                "Dataset must contain Low, Medium and High risk classes."
            )
            st.stop()

        if data["risk_level"].value_counts().min() < 2:
            st.error(
                "Each risk category must contain at least 2 records."
            )
            st.stop()

        if not data["hour"].between(0, 23).all():
            st.error("Hour must be between 0 and 23.")
            st.stop()

        if not data["day_of_week"].between(0, 6).all():
            st.error("Day of week must be between 0 and 6.")
            st.stop()

        if not data["speed_kmh"].between(0, 300).all():
            st.error("Please check the vehicle speed values.")
            st.stop()

        if not data["visibility_m"].between(0, 100000).all():
            st.error("Please check the visibility values.")
            st.stop()

        st.success(
            f"Successfully uploaded {len(data):,} records!"
        )

    except Exception as error:

        st.error(f"Dataset error: {error}")
        st.stop()

else:

    data = generate_demo_data()

    st.info(
        "Using automatically generated demonstration data. "
        "Upload your own CSV to train on your dataset."
    )


# =====================================================
# 7. MODEL TRAINING
# =====================================================

try:

    with st.spinner("Training KNN Machine Learning model..."):

        model, accuracy = train_model(data)

    st.success("KNN model trained successfully!")

except Exception as error:

    st.error(f"Model training failed: {error}")
    st.stop()


# =====================================================
# 8. SIDEBAR INPUTS
# =====================================================

with st.sidebar:

    st.markdown("## ⚙️ Risk Scenario")

    st.caption(
        "Enter road and environmental conditions."
    )

    hour_in = st.slider(
        "Time of Day",
        0, 23, 18
    )

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    day_in = st.selectbox(
        "Day of Week",
        days
    )

    weather_in = st.selectbox(
        "Weather",
        ["Clear", "Rain", "Fog", "Storm"]
    )

    road_in = st.selectbox(
        "Road Type",
        ["Highway", "Urban", "Rural", "Intersection"]
    )

    traffic_in = st.select_slider(
        "Traffic Density",
        ["Low", "Medium", "High"],
        value="Medium"
    )

    speed_in = st.slider(
        "Vehicle Speed (km/h)",
        0, 200, 65
    )

    light_in = st.selectbox(
        "Lighting",
        ["Daylight", "Dark"]
    )

    visibility_in = st.slider(
        "Visibility (metres)",
        100, 10000, 2500,
        step=100
    )

    predict = st.button(
        "🚦 PREDICT ACCIDENT RISK",
        use_container_width=True
    )


# =====================================================
# 9. MODEL PERFORMANCE
# =====================================================

st.markdown("### 📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Validation Accuracy",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "Training Dataset",
    f"{len(data):,} Records"
)

col3.metric(
    "Algorithm",
    "KNN (K=5)"
)

st.caption(
    "Accuracy is measured on a held-out test split. "
    "Synthetic dataset performance is illustrative only."
)


# =====================================================
# 10. ACCIDENT RISK PREDICTION
# =====================================================

st.markdown("### 🔍 Live Accident Risk Prediction")

day_number = days.index(day_in)

sample = pd.DataFrame([{
    "hour": hour_in,
    "day_of_week": day_number,
    "weather": weather_in,
    "road_type": road_in,
    "traffic_level": traffic_in,
    "speed_kmh": speed_in,
    "lighting": light_in,
    "visibility_m": visibility_in
}])

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

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown("#### Predicted Risk Level")

        st.markdown(
            f'<h1 style="color:{colors[predicted]}">'
            f'{predicted.upper()} RISK</h1>',
            unsafe_allow_html=True
        )

        st.progress(int(confidence))

        st.write(
            f"Prediction confidence: **{confidence:.2f}%**"
        )

        recommendations = {
            "Low": (
                "Continue following traffic rules and maintain "
                "safe driving practices."
            ),
            "Medium": (
                "Reduce speed, increase following distance "
                "and stay alert."
            ),
            "High": (
                "Avoid unnecessary travel if possible. "
                "Wait for safer road conditions."
            )
        }

        st.info(
            "**Safety Recommendation:** "
            + recommendations[predicted]
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True
        )

        st.markdown("#### Risk Probability")

        probability_data = pd.DataFrame({
            "Risk Level": classes,
            "Probability": probabilities
        })

        fig = px.bar(
            probability_data,
            x="Risk Level",
            y="Probability",
            color="Risk Level",
            color_discrete_map=colors,
            range_y=[0, 1]
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

else:

    st.info(
        "Select road conditions in the sidebar and click "
        "'PREDICT ACCIDENT RISK' to view the prediction."
    )


# =====================================================
# 11. DATASET ANALYTICS
# =====================================================

st.markdown("### 📈 Dataset Intelligence")

left, right = st.columns(2)

risk_colors = {
    "Low": "#21c58e",
    "Medium": "#f4b942",
    "High": "#ff6577"
}

with left:

    counts = (
        data["risk_level"]
        .value_counts()
        .rename_axis("Risk Level")
        .reset_index(name="Records")
    )

    fig = px.pie(
        counts,
        names="Risk Level",
        values="Records",
        hole=0.58,
        color="Risk Level",
        color_discrete_map=risk_colors
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    fig = px.histogram(
        data,
        x="speed_kmh",
        color="risk_level",
        nbins=25,
        barmode="overlay",
        opacity=0.8,
        color_discrete_map=risk_colors
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# 12. DATASET PREVIEW
# =====================================================

with st.expander("🔎 Explore Dataset Records"):

    st.dataframe(
        data.head(20),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 13. ABOUT THE MODEL
# =====================================================

with st.expander("🧠 About KNN Algorithm"):

    st.write("""
    K-Nearest Neighbors is a supervised Machine Learning
    algorithm used for classification.

    This project uses K=5 and distance-weighted voting.

    It identifies the five nearest training records
    based on their features and predicts the accident
    risk category.

    Numerical features are standardized, while
    categorical features are one-hot encoded.
    """)

    st.write("""
    The built-in demonstration dataset is synthetic.
    Use verified real-world accident records for
    research and further model evaluation.
    """)


# =====================================================
# 14. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    '<p class="small-note">'
    'RoadSense AI | Intelligent Road Accident Prediction '
    '| Academic Project Prototype'
    '</p>',
    unsafe_allow_html=True
)

st.caption(
    "This application is for academic demonstration "
    "and road-safety awareness, not actual accident forecasting."
)
