import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


print("=" * 70)
print("PROJECT FORESIGHT - MACHINE LEARNING DEMAND FORECASTING")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading forecasting dataset...")

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "forecasting"
    / "daily_forecasting_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. FEATURES
# ============================================================

features = [
    "avg_discount",
    "promotion_flag",

    "year",
    "month",
    "quarter",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "is_weekend",

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",

    "rolling_std_7",
    "rolling_std_28"
]

target = "units_sold"


print("\n" + "=" * 70)
print("FEATURES")
print("=" * 70)

print(features)


# ============================================================
# 3. REMOVE ROWS WITH LAG NA VALUES
# ============================================================

model_df = df.dropna(
    subset=features + [target]
).copy()

print("\nRows after removing missing feature values:")
print(len(model_df))


# ============================================================
# 4. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

TEST_DAYS = 90

train = model_df.iloc[:-TEST_DAYS].copy()
test = model_df.iloc[-TEST_DAYS:].copy()


X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("Training rows:", len(train))
print("Testing rows :", len(test))

print(
    "\nTraining period:",
    train["date"].min(),
    "to",
    train["date"].max()
)

print(
    "Testing period :",
    test["date"].min(),
    "to",
    test["date"].max()
)


# ============================================================
# 5. RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Random Forest training completed!")


# ============================================================
# 6. PREDICTIONS
# ============================================================

predictions = model.predict(X_test)

# Demand cannot be negative
predictions = np.maximum(
    predictions,
    0
)

test["rf_forecast"] = predictions


# ============================================================
# 7. EVALUATION
# ============================================================

actual = y_test.values
predicted = predictions


mae = mean_absolute_error(
    actual,
    predicted
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)

non_zero = actual != 0

mape = np.mean(
    np.abs(
        (actual[non_zero] - predicted[non_zero])
        / actual[non_zero]
    )
) * 100

wape = (
    np.sum(
        np.abs(
            actual - predicted
        )
    )
    /
    np.sum(
        np.abs(actual)
    )
) * 100


print("\n" + "=" * 70)
print("RANDOM FOREST PERFORMANCE")
print("=" * 70)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"WAPE : {wape:.2f}%")


# ============================================================
# 8. COMPARE WITH BASELINE
# ============================================================

baseline_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "forecasting"
    / "baseline_model_results.csv"
)

baseline = pd.read_csv(
    baseline_path
)

rf_result = pd.DataFrame({
    "model": ["Random Forest"],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE": [mape],
    "WAPE": [wape]
})

comparison = pd.concat(
    [
        baseline,
        rf_result
    ],
    ignore_index=True
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 10. SAMPLE FORECASTS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE RANDOM FOREST FORECASTS")
print("=" * 70)

sample = test[
    [
        "date",
        "units_sold",
        "rf_forecast"
    ]
].head(20)

print(
    sample.to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "forecasting"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


comparison_path = (
    OUTPUT_DIR
    / "model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


importance_path = (
    OUTPUT_DIR
    / "feature_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False
)


forecast_path = (
    OUTPUT_DIR
    / "random_forest_forecasts.csv"
)

test[
    [
        "date",
        "units_sold",
        "rf_forecast"
    ]
].to_csv(
    forecast_path,
    index=False
)


print("\n" + "=" * 70)
print("MACHINE LEARNING FORECASTING COMPLETED")
print("=" * 70)

print("\nModel comparison saved to:")
print(comparison_path)

print("\nFeature importance saved to:")
print(importance_path)

print("\nForecasts saved to:")
print(forecast_path)