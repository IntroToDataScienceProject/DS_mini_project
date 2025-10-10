import pandas as pd
import numpy as np

def create_station_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[['ID', 'Nimi', 'Kapasiteet', 'x', 'y']]
    df.to_csv('Data/station_data.csv')

create_station_df("Data/Helsingin_ja_Espoon_kaupunkipyöräasemat_avoin.csv")

def stations_dict(csv_path):
    df=pd.read_csv(csv_path)
    stations = df.to_dict(orient='records')
    return stations