import pandas as pd
import numpy as np
import joblib
import pickle
import pyarrow as pa

def preprocess_bike_data(activity_parquet, metadata_parquet, station_csv, days_window=30):
    print("🔧 Loading raw data...")
    df_activity = pd.read_parquet(activity_parquet, engine="pyarrow")
    df_meta = pd.read_parquet(metadata_parquet, engine="pyarrow")
    df_stations = pd.read_csv(station_csv)

    # Filter numeric station IDs
    df_activity = df_activity[df_activity['id'].str.isnumeric()]
    df_meta = df_meta[df_meta['id'].str.isnumeric()]

    # Normalize columns
    df_stations.rename(columns={"Nimi": "name", "Kapasiteet": "capacity", "ID": "station_id"}, inplace=True)
    df_meta.rename(columns={"id": "station_id"}, inplace=True)
    df_activity.rename(columns={"id": "station_id"}, inplace=True)

    # Convert to int
    for df in [df_activity, df_meta, df_stations]:
        df["station_id"] = df["station_id"].astype(int)

    # Keep only recent data
    df_activity["time"] = pd.to_datetime(df_activity["time"])
    cutoff = df_activity["time"].max() - pd.Timedelta(days=days_window)
    df_recent = df_activity[df_activity["time"] >= cutoff].copy()

    print("🕒 Aggregating to hourly data...")
    # Round timestamps to nearest hour
    df_recent["hour_time"] = df_recent["time"].dt.floor("H")

    # Compute lag-based features
    df_recent["prev_available"] = df_recent.groupby("station_id")["bikesAvailable"].shift(1)
    df_recent["departures"] = (df_recent["prev_available"] - df_recent["bikesAvailable"]).clip(lower=0)
    df_recent["arrivals"] = (df_recent["bikesAvailable"] - df_recent["prev_available"]).clip(lower=0)
    df_recent.dropna(subset=["prev_available"], inplace=True)

    # Aggregate to hourly resolution
    df_hourly = (
        df_recent.groupby(["station_id", "hour_time"])
        .agg({
            "bikesAvailable": "mean",   # mean availability within the hour
            "arrivals": "sum",          # total arrivals in the hour
            "departures": "sum",        # total departures in the hour
        })
        .reset_index()
    )

    # Add temporal features
    df_hourly["hour_of_day"] = df_hourly["hour_time"].dt.hour
    df_hourly["day_of_week"] = df_hourly["hour_time"].dt.dayofweek

    # Add station metadata efficiently via maps
    cap_map = dict(zip(df_stations["station_id"], df_stations["capacity"]))
    name_map = dict(zip(df_stations["station_id"], df_stations["name"]))

    df_hourly["capacity"] = df_hourly["station_id"].map(cap_map)
    df_hourly["name"] = df_hourly["station_id"].map(name_map)

    # Add lag columns at hourly level
    df_hourly["prev_available"] = df_hourly.groupby("station_id")["bikesAvailable"].shift(1)
    df_hourly["prev_arrivals"] = df_hourly.groupby("station_id")["arrivals"].shift(1)
    df_hourly["prev_departures"] = df_hourly.groupby("station_id")["departures"].shift(1)
    df_hourly.dropna(inplace=True)

    print("📈 Computing average hourly flow patterns...")
    avg_flow = (
        df_hourly.groupby(["station_id", "hour_of_day", "day_of_week"])[["arrivals", "departures"]]
        .mean()
        .reset_index()
    )

    print("📊 Storing recent state per station...")
    last_state = (
        df_hourly.sort_values("hour_time")
        .groupby("station_id")
        .tail(1)[
            ["station_id", "bikesAvailable", "arrivals", "departures",
             "prev_available", "prev_arrivals", "prev_departures",
             "capacity", "name"]
        ]
    )

    # Save lightweight precomputed data
    joblib.dump(avg_flow, "avg_flow.pkl")
    joblib.dump(last_state, "recent_state.pkl")
    joblib.dump(df_stations[["station_id", "name", "capacity"]], "station_map.pkl")

    print("✅ Preprocessing complete!")
    print("Saved: avg_flow.pkl, recent_state.pkl, station_map.pkl.")


def main():
    preprocess_bike_data(
    activity_parquet="/home/rishika/Intro_to_DS/MiniProject/data/data_2025.parquet",
    metadata_parquet="/home/rishika/Intro_to_DS/MiniProject/data/tellingit_2017-2025.parquet",
    station_csv="/home/rishika/Intro_to_DS/MiniProject/data/Helsingin_ja_Espoon_kaupunkipyöräasemat_avoin_7704606743268189464.csv",
    )