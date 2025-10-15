import pandas as pd
import numpy as np
import joblib
import pickle
import pyarrow as pa
from datetime import datetime

def build_features_for_prediction(station_name: str, future_datetime: datetime):
    # Load precomputed data
    avg_flow = joblib.load("/home/rishika/Intro_to_DS/MiniProject/DS_mini_project/avg_flow.pkl")
    recent_state = joblib.load("/home/rishika/Intro_to_DS/MiniProject/DS_mini_project/recent_state.pkl")
    station_map = joblib.load("/home/rishika/Intro_to_DS/MiniProject/DS_mini_project/station_map.pkl")

    # Get station details
    row_station = station_map[station_map["name"].str.lower() == station_name.lower()]
    if row_station.empty:
        raise ValueError(f"Station '{station_name}' not found!")
    station_id = row_station.iloc[0]["station_id"]
    capacity = row_station.iloc[0]["capacity"]

    # Get recent state (for lag features)
    recent_row = recent_state[recent_state["station_id"] == station_id]
    if not recent_row.empty:
        prev_available = float(recent_row["bikesAvailable"].iloc[0])
        prev_arrivals = float(recent_row["arrivals"].iloc[0])
        prev_departures = float(recent_row["departures"].iloc[0])
    else:
        prev_available, prev_arrivals, prev_departures = 0, 0, 0

    # Lookup typical arrivals/departures
    hour = future_datetime.hour
    day_of_week = future_datetime.weekday()
    flow = avg_flow[
        (avg_flow["station_id"] == station_id)
        & (avg_flow["hour_of_day"] == hour)
        & (avg_flow["day_of_week"] == day_of_week)
    ]

    if not flow.empty:
        arrivals = float(flow["arrivals"].iloc[0])
        departures = float(flow["departures"].iloc[0])
    else:
        arrivals, departures = prev_arrivals, prev_departures

    # Build feature row
    df_ready = pd.DataFrame([{
        "station_id": station_id,
        "hour_of_day": hour,
        "day_of_week": day_of_week,
        "month": future_datetime.month,
        "year": future_datetime.year,
        "is_weekend": int(day_of_week >= 5),
        "departures": departures,
        "arrivals": arrivals,
        "capacity": capacity,
        "prev_available": prev_available,
        "prev_departures": prev_departures,
        "prev_arrivals": prev_arrivals
    }])

    return df_ready



def main():
    #example how to use -- the date time input should come from the user --> datetime(year, month, day, hour)
    features = build_features_for_prediction("Töölönlahdenkatu", datetime(2025, 10, 3, 12))
    # print(features) #gives input for the model

    MODEL_PATH = "/home/rishika/Intro_to_DS/MiniProject/SGDmodel_stable_full_tuned.pkl" #"/home/rishika/Intro_to_DS/MiniProject/DS_mini_project/SGDmodel_stable_full_tuned.pkl"

    bundle = joblib.load(MODEL_PATH)
    model = bundle['model']
    preprocessor = bundle['preprocessor']
    X_transformed = preprocessor.transform(features)
    predictions = model.predict(X_transformed)

    predictions = np.clip(predictions, 0, features['capacity'].to_numpy())
    print("Predicted available bikes:", predictions)
    return predictions

main()