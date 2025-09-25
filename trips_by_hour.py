import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

data_folder = os.path.expanduser("~/DataS")
all_files = glob.glob(os.path.join(data_folder, "*.csv"))

frames = []
for file in all_files:
    df = pd.read_csv(file)
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

data["Departure"] = pd.to_datetime(data["Departure"], errors='coerce')

data["Hour"] = data["Departure"].dt.hour

hour_counts = data["Hour"].value_counts().sort_index()

plt.figure(figsize=(10,5))
plt.bar(hour_counts.index, hour_counts.values, edgecolor='black')
plt.xlabel("Hour of the day")
plt.ylabel("Number of trips")
plt.title("Number of trips by time of day")
plt.xticks(range(0,24))
plt.grid(axis='y', alpha=0.7)
plt.show()
