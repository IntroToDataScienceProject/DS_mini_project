import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

# Function 1: Create dataframe

def create_station_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[['ID', 'Nimi', 'Kapasiteet', 'x', 'y']]
    return df

# Function 2: Predict availability
def predict_availability(model_path: str, capacity: int, station_id: int,
                         hour_of_day: int, day_of_week: int, month: int, is_weekend: int):
    # Loading model
    data = joblib.load(model_path)
    model = data['model']
    preprocessor = data['preprocessor']

    
    # Preparing input for model
    X_input = pd.DataFrame([{
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "station_id": station_id,
        "capacity": capacity,
        "departures": 5, 
        "arrivals": 3,
        "prev_available": 21,
        "prev_departures": 2,
        "prev_arrivals": 13

    }])

    X_processed = preprocessor.transform(X_input)
    
    # Predicting number of available bikes
    num_available = model.predict(X_processed)[0]
    num_available = max(0, min(num_available, capacity))

    return {
        "available": int(num_available),
        "empty": int(capacity - num_available),
        "capacity": int(capacity)
    }

# Function 3: Pie chart
def plot_pie_chart(model_path: str, capacity_csv: str, station_id: int,
                   hour_of_day: int, day_of_week: int, month: int, is_weekend: int):
    pred = predict_availability(model_path, capacity_csv, station_id,
                                hour_of_day, day_of_week, month, is_weekend)
    labels = ['Available', 'Empty']
    sizes = [pred['available'], pred['empty']]
    colors = ['#66b3ff', '#ff9999']  #(i avoided dark red & green)
    
    plt.figure(figsize = (4, 4))
    plt.pie(sizes, labels = labels, autopct = '%1.1f%%', colors = colors, startangle = 90)
    plt.title(f"Station {station_id} — Capacity {pred['capacity']}")
    plt.show()

# Function 4: Nearby stations
def fetch_nearby_stations(station_name: str, capacity_csv: str):
    df = pd.read_csv(capacity_csv)
    df = df[['Nimi', 'x', 'y']]
    
    if station_name not in df['Nimi'].values:
        raise ValueError(f"Station '{station_name}' not found.")
    
    current = df[df['Nimi'] == station_name].iloc[0]
    x1, y1 = current['x'], current['y']
    
    def distance(x2, y2):
        R = 6371  # Earth radius in km
        dlon = radians(x2 - x1)
        dlat = radians(y2 - y1)
        a = sin(dlat/2)**2 + cos(radians(y1)) * cos(radians(y2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    
    df['distance_km'] = df.apply(lambda row: distance(row['x'], row['y']), axis = 1)
    nearby = df[df['Nimi'] != station_name].nsmallest(5, 'distance_km')
    
    return list(zip(nearby['Nimi'], nearby['distance_km']))

# Example Usage (for testing)

if __name__ == "__main__":
    csv_path = "Helsingin_ja_Espoon_kaupunkipyöräasemat_avoin_7704606743268189464 (1).csv"  
    model_path = "basic_linreg_model_042019.pkl"

    # 1. Create dataframe
    df = create_station_df(csv_path)
    print(df.head())

    # 2. Predict
    result = predict_availability(model_path, csv_path, station_id = 501, hour_of_day = 10,
                                  day_of_week = 2, month = 4, is_weekend = 0)
    print(result)

    # 3. Pie chart
    plot_pie_chart(model_path, csv_path, station_id = 501, hour_of_day = 10,
                   day_of_week = 2, month = 4, is_weekend = 0)

    # 4. Nearby stations
    near = fetch_nearby_stations("Apollonkatu", csv_path)
    print(near)