import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

data_folder = os.path.expanduser("~/DataS")

all_files = glob.glob(os.path.join(data_folder, "*.csv"))

print(f"Found {len(all_files)} files")

frames = []
for file in all_files:
    try:
        df = pd.read_csv(file)
        year = os.path.basename(file).split("-")[0]
        df["Year"] = year
        frames.append(df)
        print(f"Loaded {file} with {len(df)} rows")
    except Exception as e:
        print(f"Error reading {file}: {e}")

data = pd.concat(frames, ignore_index=True)

years = sorted(data["Year"].unique())
n_years = len(years)

fig, axes = plt.subplots(1, n_years, figsize=(6*n_years, 5), sharey=True)

if n_years == 1:  
    axes = [axes]

for ax, year in zip(axes, years):
    year_data = data[data["Year"] == year]
    station_counts = year_data["Departure station name"].value_counts()
    ax.hist(station_counts.values, bins=30, edgecolor="black")
    ax.set_title(f"Year {year}")
    ax.set_xlabel("Trips per station")
    ax.set_ylabel("Number of stations")
    ax.grid(axis='y', alpha=0.7)

plt.suptitle("Distribution of station usage (proxy for capacity) by year")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
