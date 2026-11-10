"""Train and evaluate the K-Nearest Neighbors (KNN) risk classifier."""
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("data/road_accident_demo.csv")
features = ["hour","day_of_week","weather","road_type","traffic_level","speed_kmh","lighting","visibility_m"]
categorical = ["weather","road_type","traffic_level","lighting"]
numerical = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numerical", StandardScaler(), numerical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2))
])

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["risk_level"], test_size=0.2,
    random_state=42, stratify=df["risk_level"]
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("KNN validation accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print(classification_report(y_test, predictions))
joblib.dump(model, "road_risk_knn_model.joblib")
print("Saved road_risk_knn_model.joblib")
