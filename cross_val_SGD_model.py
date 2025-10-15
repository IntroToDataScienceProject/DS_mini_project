import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
import joblib
import gc

# === CONFIG ===
DATA_PATH = "/home/rishika/Intro_to_DS/MiniProject/DS_mini_project/full_df_lag_train.csv"
MODEL_PATH = "SGDmodel_stable_full_tuned.pkl"
BATCH_SIZE = 100_000
N_SPLITS = 3   # 3 years → roughly 3 folds (1 per year)
SAMPLE_FRAC = 0.15  # tune on 15% of data for speed

# === FEATURE COLUMNS ===
numeric_features = [
    'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'year',
    'departures', 'arrivals', 'capacity',
    'prev_available', 'prev_departures', 'prev_arrivals'
]
categorical_features = ['station_id']
target = 'available_bikes'

# === PREPROCESSOR ===
numeric_pipe = Pipeline([
    ('scaler', StandardScaler())
])
categorical_pipe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

preprocessor = ColumnTransformer([
    ('num', numeric_pipe, numeric_features),
    ('cat', categorical_pipe, categorical_features)
])

# === STEP 1: Load a subset for hyperparameter tuning ===
print("📂 Loading subset of data for hyperparameter tuning...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=numeric_features + categorical_features + [target])
df = df.sort_values("hour")  # ensure time ordering
df_sample = df.sample(frac=SAMPLE_FRAC, random_state=42).sort_values("hour")

X = df_sample[numeric_features + categorical_features]
y = df_sample[target].values
print(f"✅ Sample size for tuning: {len(df_sample)} rows")

# === STEP 2: Define Time Series CV ===
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# === STEP 3: Define Hyperparameter Grid ===
param_grid = {
    'alpha': [1e-2, 1e-3, 1e-4],
    'eta0': [0.1, 0.01, 0.001],
    'learning_rate': ['invscaling', 'adaptive'],
    'penalty': ['l2', 'elasticnet']
}

grid = list(ParameterGrid(param_grid))
print(f"🔍 Trying {len(grid)} hyperparameter combinations...")

best_rmse = float('inf')
best_params = None

# === STEP 4: Loop through parameter grid ===
for i, params in enumerate(grid):
    print(f"\n⚙️ [{i+1}/{len(grid)}] Testing params: {params}")

    fold_rmses = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Preprocess
        X_train_t = preprocessor.fit_transform(X_train)
        X_val_t = preprocessor.transform(X_val)

        # Model
        model = SGDRegressor(
            loss='squared_error',
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            **params
        )
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_val_t)
        # rmse = mean_squared_error(y_val, y_pred, squared=False)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        fold_rmses.append(rmse)
        print(f"   Fold {fold+1} RMSE: {rmse:.3f}")

    mean_rmse = np.mean(fold_rmses)
    print(f"➡️ Mean CV RMSE: {mean_rmse:.3f}")

    if mean_rmse < best_rmse:
        best_rmse = mean_rmse
        best_params = params

print("\n🏆 Best Parameters:")
print(best_params)
print(f"📉 Best RMSE: {best_rmse:.3f}")

# === STEP 5: Train Final Model Incrementally on Full Data ===
print("\n🚀 Training final model on full dataset with best parameters...")

model = SGDRegressor(
    loss='squared_error',
    max_iter=1,
    warm_start=True,
    fit_intercept=True,
    **best_params
)

iterator = pd.read_csv(DATA_PATH, chunksize=BATCH_SIZE)
first_batch = True

for i, chunk in enumerate(iterator):
    chunk = chunk.dropna(subset=numeric_features + categorical_features + [target])
    X_chunk = chunk[numeric_features + categorical_features]
    y_chunk = chunk[target].values

    # Fit or transform the preprocessor incrementally
    if first_batch:
        X_transformed = preprocessor.fit_transform(X_chunk)
    else:
        X_transformed = preprocessor.transform(X_chunk)

    model.partial_fit(X_transformed, y_chunk)
    print(f"✅ Trained on batch {i+1}, rows: {len(chunk)}")

    del chunk, X_chunk, X_transformed
    gc.collect()
    first_batch = False

# === SAVE MODEL + PREPROCESSOR ===
joblib.dump({
    'model': model,
    'preprocessor': preprocessor,
    'best_params': best_params
}, MODEL_PATH)

print(f"\n✅ Model and preprocessor saved to {MODEL_PATH}")
