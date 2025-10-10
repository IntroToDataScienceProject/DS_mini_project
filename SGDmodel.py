import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import joblib
import gc

# === CONFIG ===
DATA_PATH = "full_df_lag_train.csv"
MODEL_PATH = "SGDmodel_stable_full.pkl"
BATCH_SIZE = 100_000

# === FEATURE COLUMNS ===
numeric_features = [
    'hour_of_day', 'day_of_week', 'month', 'is_weekend',
    'departures', 'arrivals', 'capacity',
    'prev_available', 'prev_departures', 'prev_arrivals'
]
categorical_features = ['station_id']
target = 'available_bikes'

# === PIPELINE ===
numeric_pipe = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

preprocessor = ColumnTransformer([
    ('num', numeric_pipe, numeric_features),
    ('cat', categorical_pipe, categorical_features)
])

# Use stable SGD parameters
model = SGDRegressor(
    loss='squared_error',
    penalty='l2',
    alpha=0.001,             # regularization
    learning_rate='invscaling',
    eta0=0.01,              # initial learning rate
    max_iter=1,
    warm_start=True,
    fit_intercept=True
)

# === TRAIN IN BATCHES ===
iterator = pd.read_csv(DATA_PATH, chunksize=BATCH_SIZE)

print("Training incremental regression model in batches...")

first_batch = True
for i, chunk in enumerate(iterator):
    chunk = chunk.dropna(subset=numeric_features + categorical_features + [target])
    chunk = shuffle(chunk, random_state=42)

    X_chunk = chunk[numeric_features + categorical_features]
    y_chunk = chunk[target].values

    # Fit or transform the preprocessor
    if first_batch:
        X_transformed = preprocessor.fit_transform(X_chunk)
    else:
        X_transformed = preprocessor.transform(X_chunk)

    # Incremental training
    model.partial_fit(X_transformed, y_chunk)

    print(f"✅ Trained on batch {i+1}, rows: {len(chunk)}")

    # Memory cleanup
    del chunk, X_chunk, X_transformed
    gc.collect()

    first_batch = False

# === SAVE MODEL + PREPROCESSOR ===
joblib.dump({
    'model': model,
    'preprocessor': preprocessor
}, MODEL_PATH)
print(f"✅ Model and preprocessor saved to {MODEL_PATH}")

# # === PREDICTION EXAMPLE ===
# sample = pd.read_csv("df_lag_test_042019.csv")
# X_sample = sample[numeric_features + categorical_features]

# bundle = joblib.load(MODEL_PATH)
# model = bundle['model']
# preprocessor = bundle['preprocessor']

# X_transformed = preprocessor.transform(X_sample)
# predictions = model.predict(X_transformed)

# print("Predicted available bikes:", predictions)




# import pandas as pd
# import numpy as np
# from sklearn.linear_model import SGDRegressor
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.utils import shuffle
# import joblib
# import gc

# # === CONFIG ===
# DATA_PATH = "full_df_lag.csv"
# MODEL_PATH = "SGDmodel.pkl"
# BATCH_SIZE = 100_000  # adjust depending on your RAM
# POLY_DEGREE = 2

# # === FEATURE COLUMNS ===
# numeric_features = [
#     'hour_of_day', 'day_of_week', 'month', 'is_weekend',
#     'departures', 'arrivals', 'capacity',
#     'prev_available', 'prev_departures', 'prev_arrivals'
# ]
# categorical_features = ['station_id']
# target = 'available_bikes'

# # === PREPARE TRANSFORMERS ===
# scaler = StandardScaler()
# encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
# poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)

# # === INITIALIZE MODEL ===
# model = SGDRegressor(loss="squared_error", penalty="l2", alpha=0.001, max_iter=1)

# # === TRAIN IN BATCHES ===
# iterator = pd.read_csv(DATA_PATH, chunksize=BATCH_SIZE)

# print("Training incremental polynomial regression model in batches...")

# first = True
# for i, chunk in enumerate(iterator):
#     chunk = chunk.dropna(subset=numeric_features + categorical_features + [target])
#     chunk = shuffle(chunk, random_state=42)

#     X_num = chunk[numeric_features].astype(float)
#     X_cat = chunk[categorical_features].astype(str)
#     y_batch = chunk[target].values

#     # --- SCALE NUMERICAL FEATURES ---
#     if first:
#         X_num_scaled = scaler.fit_transform(X_num)
#     else:
#         X_num_scaled = scaler.transform(X_num)

#     # --- ENCODE CATEGORICAL FEATURES ---
#     if first:
#         X_cat_encoded = encoder.fit_transform(X_cat)
#     else:
#         X_cat_encoded = encoder.transform(X_cat)

#     # --- COMBINE NUMERIC + CATEGORICAL ---
#     from scipy.sparse import hstack
#     X_combined = hstack([X_num_scaled, X_cat_encoded]).tocsr()

#     # --- ADD POLYNOMIAL FEATURES (careful with memory!) ---
#     # For sparse matrices, PolynomialFeatures can explode in size, so we limit it
#     # to numeric columns only for tractability.
#     X_poly = poly.fit_transform(X_num_scaled) if first else poly.transform(X_num_scaled)

#     X_full = np.hstack([X_poly, X_combined.toarray()[:, :10]])  # use subset to limit memory

#     # --- TRAIN INCREMENTALLY ---
#     model.partial_fit(X_full, y_batch)

#     print(f"✅ Trained on batch {i+1}, rows: {len(chunk)}")

#     # --- MEMORY MANAGEMENT ---
#     del chunk, X_num, X_cat, X_combined, X_poly, X_full
#     gc.collect()

#     first = False

# # === SAVE MODEL + TRANSFORMERS ===
# joblib.dump({
#     'model': model,
#     'scaler': scaler,
#     'encoder': encoder,
#     'poly': poly
# }, MODEL_PATH)
# print(f"✅ Model and preprocessors saved to {MODEL_PATH}")



# import pandas as pd
# import numpy as np
# from sklearn.linear_model import SGDRegressor
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
# from sklearn.utils import shuffle
# from sklearn.metrics import mean_squared_error, r2_score
# from scipy.sparse import hstack
# from joblib import Parallel, delayed, dump
# import gc
# import os
# from datetime import datetime

# # ==== CONFIGURATION ====
# DATA_PATH = "full_df_lag.csv"
# MODEL_PATH = "SGDmodel.pkl"
# LOG_PATH = "training_log.csv"

# BATCH_SIZE = 100_000      # rows per sub-batch
# N_JOBS = os.cpu_count()//2 or 2
# POLY_DEGREE = 2
# SUBSET_CATEGORIES = 10     # limit number of encoded category columns for memory

# # ==== FEATURES ====
# numeric_features = [
#     'hour_of_day', 'day_of_week', 'month', 'is_weekend',
#     'departures', 'arrivals', 'capacity',
#     'prev_available', 'prev_departures', 'prev_arrivals'
# ]
# categorical_features = ['station_id']
# target = 'available_bikes'

# # ==== TRANSFORMERS ====
# scaler = StandardScaler()
# encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
# poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)

# # ==== MODEL ====
# model = SGDRegressor(
#     loss="squared_error", 
#     penalty="l2", 
#     alpha=0.001, 
#     max_iter=1,
#     learning_rate="adaptive",
#     eta0=0.01
# )

# # ==== Helper: process one batch ====
# def process_batch(batch_df, scaler, encoder, poly, first=False):
#     batch_df = batch_df.dropna(subset=numeric_features + categorical_features + [target])
#     batch_df = shuffle(batch_df, random_state=42)

#     X_num = batch_df[numeric_features].astype(float)
#     X_cat = batch_df[categorical_features].astype(str)
#     y_batch = batch_df[target].values

#     # Scale numeric
#     X_num_scaled = scaler.fit_transform(X_num) if first else scaler.transform(X_num)

#     # Encode categorical
#     X_cat_encoded = encoder.fit_transform(X_cat) if first else encoder.transform(X_cat)

#     # Combine numeric + limited categorical
#     X_combined = hstack([X_num_scaled, X_cat_encoded[:, :SUBSET_CATEGORIES]]).tocsr()

#     # Polynomial expansion (numeric only)
#     X_poly = poly.fit_transform(X_num_scaled) if first else poly.transform(X_num_scaled)

#     # Combine final feature set
#     X_final = np.hstack([X_poly, X_combined.toarray()])
#     return X_final, y_batch


# # ==== MAIN TRAINING LOOP ====
# def train_in_parallel():
#     iterator = pd.read_csv(DATA_PATH, chunksize=BATCH_SIZE * N_JOBS)
#     first = True
#     batch_idx = 0

#     # Initialize log file
#     log_cols = ["timestamp", "batch_idx", "rows", "rmse", "r2"]
#     with open(LOG_PATH, "w") as f:
#         f.write(",".join(log_cols) + "\n")

#     print(f"Training on {DATA_PATH} using {N_JOBS} parallel workers...")

#     for batch_group in iterator:
#         sub_batches = np.array_split(batch_group, N_JOBS)

#         results = Parallel(n_jobs=N_JOBS)(
#             delayed(process_batch)(sub_batch, scaler, encoder, poly, first=first and j == 0)
#             for j, sub_batch in enumerate(sub_batches)
#         )

#         # Train incrementally + compute metrics on the fly
#         all_y_true, all_y_pred = [], []

#         for X_batch, y_batch in results:
#             model.partial_fit(X_batch, y_batch)

#             # Evaluate this batch
#             y_pred = model.predict(X_batch)
#             all_y_true.extend(y_batch)
#             all_y_pred.extend(y_pred)

#         rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
#         r2 = r2_score(all_y_true, all_y_pred)

#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(LOG_PATH, "a") as f:
#             f.write(f"{timestamp},{batch_idx},{len(batch_group)},{rmse:.4f},{r2:.4f}\n")

#         print(f"✅ Batch {batch_idx+1}: {len(batch_group)} rows | RMSE={rmse:.3f}, R²={r2:.3f}")

#         first = False
#         batch_idx += 1

#         del batch_group, results, all_y_true, all_y_pred
#         gc.collect()

#     # Save trained model + preprocessors
#     dump({
#         'model': model,
#         'scaler': scaler,
#         'encoder': encoder,
#         'poly': poly
#     }, MODEL_PATH)

#     print(f"Training complete — model saved to {MODEL_PATH}")
#     print(f"Training log written to {LOG_PATH}")


# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.linear_model import SGDRegressor
# from joblib import Parallel, delayed
# import pandas as pd
# import numpy as np

# def process_batch(X_batch, y_batch, scaler, poly, model, first=False):
#     # Only fit scaler and poly once, in the first batch
#     if first:
#         X_num_scaled = scaler.fit_transform(X_batch)
#         X_poly = poly.fit_transform(X_num_scaled)
#     else:
#         X_num_scaled = scaler.transform(X_batch)
#         X_poly = poly.transform(X_num_scaled)
    
#     model.partial_fit(X_poly, y_batch)
#     return model, scaler, poly


# def train_in_batches(df, batch_size=100_000):
#     features = ['hour_of_day', 'day_of_week', 'month', 'is_weekend',
#                 'departures', 'arrivals', 'capacity', 'prev_available',
#                 'prev_departures', 'prev_arrivals']
#     target = 'available_bikes'

#     X = df[features].values
#     y = df[target].values

#     scaler = StandardScaler()
#     poly = PolynomialFeatures(degree=2, include_bias=False)
#     model = SGDRegressor(loss="squared_error", penalty="l2", max_iter=5)

#     n = len(X)
#     for i in range(0, n, batch_size):
#         X_batch = X[i:i+batch_size]
#         y_batch = y[i:i+batch_size]
#         first = (i == 0)
#         model, scaler, poly = process_batch(X_batch, y_batch, scaler, poly, model, first)
#         print(f"Batch {i//batch_size + 1} / {n//batch_size + 1} complete")

#     return model, scaler, poly



# if __name__ == "__main__":
#     train_in_parallel()
