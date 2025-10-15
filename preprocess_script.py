

def preprocess(path: str):
    import pandas as pd
    import numpy as np

    data = pd.read_csv(path)
    df = data.copy()
    df['Departure'] = pd.to_datetime(df['Departure'], errors='coerce', format='mixed')
    df['Return'] = pd.to_datetime(df['Return'], errors='coerce', format='mixed')

    df['dep_hour'] = df['Departure'].dt.floor('h')
    df['ret_hour'] = df['Return'].dt.floor('h')

    # --- Hourly DEPARTURES per station ---
    departures = (
        df.groupby(['Departure station id', 'dep_hour'])
        .size()
        .reset_index(name='departures')
        .rename(columns={'Departure station id': 'station_id', 'dep_hour': 'hour'})
    )
    # --- Hourly ARRIVALS per station ---
    arrivals = (
        df.groupby(['Return station id', 'ret_hour'])
        .size()
        .reset_index(name='arrivals')
        .rename(columns={'Return station id': 'station_id', 'ret_hour': 'hour'})
    )
    # --- Merge them together (outer join to include hours with only dep/arr) ---
    station_hourly = pd.merge(departures, arrivals,on=['station_id', 'hour'], how='outer').fillna(0)

    # Get full range of hours and all station IDs
    # hours = pd.date_range(df['Departure'].min().floor('h'),
    #                     df['Return'].max().ceil('h'),
    #                     freq='H')
    # stations = df['Departure station id'].unique()
    # Create full index
    # multi_index = pd.MultiIndex.from_product([stations, hours], names=['station_id', 'hour'])
    # full = pd.DataFrame(index=multi_index).reset_index()
    # Merge with your counts and fill missing values with 0
    # station_hourly_full = pd.merge(full, station_hourly,on=['station_id', 'hour'], how='left').fillna(0)

    # station_hourly_full['weekday'] = station_hourly_full['hour'].dt.dayofweek
    # station_hourly_full['month']   = station_hourly_full['hour'].dt.month
    # station_hourly_full['hour_of_day'] = station_hourly_full['hour'].dt.hour
    # station_hourly_full['year'] = station_hourly_full['hour'].dt.year

    cap_df = pd.read_csv("data/Helsingin_ja_Espoon_kaupunkipyöräasemat_avoin_7704606743268189464.csv")
    cap_df = cap_df[['ID', 'Kapasiteet']]
    cap_df.rename(columns={'ID': 'station_id', 'Kapasiteet': 'capacity'}, inplace=True)

    stations_binary = station_hourly.copy()
    stations_binary = stations_binary.merge(cap_df, on="station_id", how="left")
    stations_binary.dropna(subset=['capacity'], inplace=True)

    # Ensure datetime format
    stations_binary['hour'] = pd.to_datetime(stations_binary['hour'])
    # Create full hourly timeline for all stations
    full_range = pd.date_range(stations_binary['hour'].min(), stations_binary['hour'].max(), freq='H')
    stations = stations_binary['station_id'].unique()
    # Cartesian product: all (station, hour) pairs
    full_index = pd.MultiIndex.from_product([stations, full_range], names=['station_id', 'hour'])
    stations_binary_full = (
        stations_binary.set_index(['station_id', 'hour'])
        .reindex(full_index)
        .reset_index())
    # Fill missing values
    stations_binary_full['departures'] = stations_binary_full['departures'].fillna(0)
    stations_binary_full['arrivals'] = stations_binary_full['arrivals'].fillna(0)
    stations_binary_full['capacity'] = stations_binary_full.groupby('station_id')['capacity'].transform(lambda x: x.ffill().bfill())
    # Compute net flow
    stations_binary_full['net_flow'] = stations_binary_full['arrivals'] - stations_binary_full['departures']
    # Compute cumulative available bikes per station
    def compute_availability(g):
        start_val = g['capacity'].iloc[0] / 2  # assume half capacity at start
        available = start_val + g['net_flow'].cumsum()
        available = np.clip(available, 0, g['capacity'].iloc[0])  # keep within [0, capacity]
        g['available_bikes'] = available
        return g
    stations_binary_full = stations_binary_full.groupby('station_id', group_keys=False).apply(compute_availability)
    # Add binary column “empty”
    stations_binary_full['empty'] = (stations_binary_full['available_bikes'] <= 0).astype(int)
    # Add time-based covariates
    stations_binary_full['hour_of_day'] = stations_binary_full['hour'].dt.hour
    stations_binary_full['day_of_week'] = stations_binary_full['hour'].dt.dayofweek  # Monday = 0
    stations_binary_full['month'] = stations_binary_full['hour'].dt.month
    stations_binary_full['is_weekend'] = stations_binary_full['day_of_week'].isin([5, 6]).astype(int)
    stations_binary_full['year'] = stations_binary_full['hour'].dt.year # ADDED
    stations_binary_full['day'] = stations_binary_full['hour'].dt.day # ADDED

    df_full = stations_binary_full.copy()

    return df_full

def feature_engineer(df_full):
    import pandas as pd
    import numpy as np
    df_lag = df_full.copy()
    df_lag['hour'] = pd.to_datetime(df_lag['hour'])

    # Sort for lag features
    df_lag = df_lag.sort_values(['station_id', 'hour'])

    # --- ADD LAG FEATURES ---
    df_lag['prev_available'] = df_lag.groupby('station_id')['available_bikes'].shift(1)
    df_lag['prev_departures'] = df_lag.groupby('station_id')['departures'].shift(1)
    df_lag['prev_arrivals'] = df_lag.groupby('station_id')['arrivals'].shift(1)

    # Drop first hour for each station (NaNs)
    df_lag = df_lag.dropna(subset=['prev_available', 'prev_departures', 'prev_arrivals'])

    # # --- FEATURE ENGINEERING ---
    # df_lag['hour_of_day'] = df_lag['hour'].dt.hour
    # df_lag['day_of_week'] = df_lag['hour'].dt.dayofweek
    # df_lag['month'] = df_lag['hour'].dt.month
    # df_lag['is_weekend'] = (df_lag['day_of_week'] >= 5).astype(int)

    return df_lag


def merge_weather(weather, df_lag):
    import pandas as pd
    # weather = pd.read_csv("/home/rishika/Intro_to_DS/MiniProject/data/espoo_rain_042019.csv")
    rain = weather.copy()
    rain = rain.rename(columns={'Precipitation [mm]': "rained"})

    # 1️⃣ Combine the year, month, day, and time columns into one datetime column
    rain['datetime'] = pd.to_datetime(
        rain['Year'].astype(str) + '-' +
        rain['Month'].astype(str) + '-' +
        rain['Day'].astype(str) + ' ' +
        rain['Time [Local time]']
    )

    # 2️⃣ Now extract the useful datetime components
    rain['hour_of_day'] = rain['datetime'].dt.hour
    rain['day'] = rain['datetime'].dt.day
    rain['month'] = rain['datetime'].dt.month
    rain['year'] = rain['datetime'].dt.year

    cols = ['year', 'month', 'day', 'hour_of_day']
    df_lag[cols] = df_lag[cols].astype(int)
    rain[cols] = rain[cols].astype(int)

    merged_df = pd.merge(
        df_lag,
        rain[['year', 'month', 'day', 'hour_of_day', 'rained']],
        how='left',   # keeps all rows from bikes, adds rain info where available
        on=['year', 'month', 'day', 'hour_of_day']
    )

    return merged_df

def main():
    import pandas as pd
    # path  = "/home/rishika/Intro_to_DS/MiniProject/data/all_data_2017-2019.csv"
    path  = "/home/rishika/Intro_to_DS/MiniProject/data/all_data_2018-2019.csv"
    df_processed = preprocess(path)
    df_processed.to_csv("/home/rishika/Intro_to_DS/MiniProject/data/full_data2.csv")
    print("saved!")

    df_lag = feature_engineer(df_processed)
    df_lag.to_csv("full_df_lag2.csv")
    print("df_lag computed and saved as csv file!")

main()



