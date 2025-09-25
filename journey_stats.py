import pandas as pd
import glob
import os

data_folder = os.path.expanduser("~/DataS")
all_files = glob.glob(os.path.join(data_folder, "*.csv"))

frames = []
for file in all_files:
    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip().str.replace('\n','').str.lower()
    df.rename(columns={
        "duration (sec.)": "duration",
        "covered distance (m)": "distance"
    }, inplace=True)
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

data["duration"] = pd.to_numeric(data["duration"], errors='coerce')
data["distance"] = pd.to_numeric(data["distance"], errors='coerce')

data = data.dropna(subset=["duration", "distance"])

print("Journey Duration (sec) — mean:", data["duration"].mean(),
      ", median:", data["duration"].median())
print("Journey Distance (m) — mean:", data["distance"].mean(),
      ", median:", data["distance"].median())
