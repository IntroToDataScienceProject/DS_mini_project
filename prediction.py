import pandas as pd
import numpy as np
import joblib

# === CONFIG ===
DATA_PATH = "/home/rishika/Intro_to_DS/MiniProject/DS_mini_project/full_df_lag_train.csv"
MODEL_PATH = "/home/rishika/Intro_to_DS/MiniProject/DS_mini_project/SGDmodel_stable_full.pkl"
BATCH_SIZE = 100_000

# === FEATURE COLUMNS ===
numeric_features = [
    'hour_of_day', 'day_of_week', 'month', 'is_weekend',
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
np.save("SGD_stable_preds_for_validation.npy", predictions)

print("Predicted available bikes:", predictions)


# import joblib
# import numpy as np
# from scipy.sparse import hstack

# # === FEATURE COLUMNS ===
# numeric_features = [
#     'hour_of_day', 'day_of_week', 'month', 'is_weekend',
#     'departures', 'arrivals', 'capacity',
#     'prev_available', 'prev_departures', 'prev_arrivals'
# ]
# categorical_features = ['station_id']
# target = 'available_bikes'

# bundle = joblib.load("DS_mini_project/SGDmodel.pkl")
# model = bundle['model']
# scaler = bundle['scaler']
# encoder = bundle['encoder']
# poly = bundle['poly']

# # Example input
# sample = {
#     'station_id': ['1'],
#     'hour_of_day': [8],
#     'day_of_week': [2],
#     'month': [4],
#     'is_weekend': [0],
#     'departures': [5],
#     'arrivals': [3],
#     'capacity': [20],
#     'prev_available': [10],
#     'prev_departures': [4],
#     'prev_arrivals': [2]
# }

# import pandas as pd
# X_sample = pd.DataFrame(sample)

# # Transform features
# X_num = scaler.transform(X_sample[numeric_features])
# X_cat = encoder.transform(X_sample[categorical_features])
# X_combined = hstack([X_num, X_cat]).tocsr()
# X_poly = poly.transform(X_num)
# X_full = np.hstack([X_poly, X_combined.toarray()[:, :10]])

# # Predict
# pred = model.predict(X_full)
# print("Predicted available bikes:", pred)


#     # X_combined = hstack([X_num_scaled, X_cat_encoded]).tocsr()

#     # # --- ADD POLYNOMIAL FEATURES (careful with memory!) ---
#     # # For sparse matrices, PolynomialFeatures can explode in size, so we limit it
#     # # to numeric columns only for tractability.
#     # X_poly = poly.fit_transform(X_num_scaled) if first else poly.transform(X_num_scaled)

#     # X_full = np.hstack([X_poly, X_combined.toarray()[:, :10]])  # use subset to limit memory