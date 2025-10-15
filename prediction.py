import pandas as pd
import numpy as np
import joblib

# === CONFIG ===
DATA_PATH = "/home/rishika/Intro_to_DS/MiniProject/DS_mini_project/full_df_lag_train.csv"
MODEL_PATH = "/home/rishika/Intro_to_DS/MiniProject/SGDmodel_stable_full_tuned.pkl" #"/home/rishika/Intro_to_DS/MiniProject/DS_mini_project/SGDmodel_stable_full_tuned.pkl"
BATCH_SIZE = 100_000

# === FEATURE COLUMNS ===
numeric_features = [
    'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'year',
    'departures', 'arrivals', 'capacity',
    'prev_available', 'prev_departures', 'prev_arrivals'
]
categorical_features = ['station_id']
target = 'available_bikes'

# === PREDICTION EXAMPLE ===
sample = pd.read_csv("/home/rishika/Intro_to_DS/MiniProject/DS_mini_project/full_df_lag_validation.csv")
X_sample = sample[numeric_features + categorical_features]

bundle = joblib.load(MODEL_PATH)
model = bundle['model']
preprocessor = bundle['preprocessor']

X_transformed = preprocessor.transform(X_sample)
predictions = model.predict(X_transformed)

train_full = pd.read_csv(DATA_PATH)
y_mean = train_full[target].mean()
predictions += y_mean - predictions.mean() #to unshift everything
predictions = np.clip(predictions, 0, sample['capacity'].to_numpy())
np.save("SGD_stable_preds_for_validation.npy", predictions)

print("Predicted available bikes:", predictions)


