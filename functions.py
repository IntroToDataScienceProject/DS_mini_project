import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import predict_funct
from datetime import datetime

def create_station_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[['ID', 'Nimi', 'Kapasiteet', 'x', 'y']]
    df.to_csv('Data/station_data.csv')
    
def stations_dict(csv_path):
    df=pd.read_csv(csv_path)
    stations = df.to_dict(orient='records')
    return stations

# Function 1: Create dataframe

def create_station_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[['ID', 'Nimi', 'Kapasiteet', 'x', 'y']]
    return df

# Function 2: Predict availability
def predict_availability(model_path: str, station_name: str, future_datetime: datetime):
    # Loading model
    data = joblib.load(model_path)
    model = data['model']
    preprocessor = data['preprocessor']

    features = predict_funct.build_features_for_prediction(station_name, future_datetime)

    # Preparing input for model
    X_processed = preprocessor.transform(features)
    
    # Predicting number of available bikes
    num_available = model.predict(X_processed)[0]
    capacity = int(features['capacity'].iloc[0])
    num_available = int(np.clip(num_available, 0, capacity))

    return {
        "available": num_available,
        "empty": capacity - num_available,
        "capacity": capacity
    }

# Function 3: Pie chart
def plot_pie_chart(available, empty):

    labels = ['Available', 'Empty']
    sizes = [available, empty]
    colors = ['#66b3ff', '#ff9999']  #(i avoided dark red & green)
    
    plt.figure(figsize = (4, 4))
    plt.pie(sizes, labels = labels, autopct = '%1.1f%%', colors = colors, startangle = 90)

    plt.savefig('static/pie_chart.png', bbox_inches='tight')
    plt.close()

# Function 4: Nearby stations
def fetch_nearby_stations(station_name: str, capacity_csv: str):
    df = pd.read_csv(capacity_csv)
    df = df[['Nimi', 'x', 'y']]
    
    if station_name not in df['Nimi'].values:
        raise ValueError(f"Station '{station_name}' not found.")
    
    current = df[df['Nimi'] == station_name].iloc[0]
    x1, y1 = current['x'], current['y']
    
    def distance(x2, y2):
        R = 6371000  # Earth radius in meters
        dlon = radians(x2 - x1)
        dlat = radians(y2 - y1)
        a = sin(dlat/2)**2 + cos(radians(y1)) * cos(radians(y2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    
    df['distance_m'] = df.apply(lambda row: round(distance(row['x'], row['y'])), axis = 1)
    nearby = df[df['Nimi'] != station_name].nsmallest(5, 'distance_m')
    
    return list(zip(nearby['Nimi'], nearby['distance_m']))

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