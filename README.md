# RoadSense AI — Intelligent Road Accident Risk Prediction

A polished Streamlit + Machine Learning academic project prototype.

## Features
- Interactive road-condition inputs (time, weather, road type, traffic, speed, lighting and visibility)
- K-Nearest Neighbors (KNN) classification into Low / Medium / High risk
- Class probability chart and dataset analytics
- Responsive dark dashboard UI
- Included 5,000-row synthetic demonstration dataset

## Run on Windows
1. Install Python 3.10–3.12.
2. Extract this ZIP.
3. Open Command Prompt in the extracted `Intelligent_Road_Accident_Prediction` folder.
4. Run:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The browser opens the dashboard at the local Streamlit address.

## Run on Ubuntu / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
- `app.py` — Streamlit dashboard and prediction interface
- `data/road_accident_demo.csv` — generated sample dataset
- `train_model.py` — optional standalone model training and evaluation
- `requirements.txt` — Python dependencies

## Important academic note
The bundled dataset is synthetically generated for demonstration, not collected from actual police, hospital, or traffic records. The model predicts a risk category from the supplied scenario; it does not predict the exact occurrence, location, or time of a future crash. Do not use it for real-world driving or emergency decisions. For a research-grade project, replace the demo dataset with a verified, documented accident dataset and re-evaluate the model.


## KNN configuration
The application uses `KNeighborsClassifier` with `n_neighbors=5`, distance-weighted voting, and Euclidean distance (Minkowski distance with p=2). Numerical features are standardized because KNN is distance-based. The value K=5 is a starting configuration and should be tuned using validation data for a research-grade result.
