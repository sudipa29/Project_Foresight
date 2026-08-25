import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("PROJECT FORESIGHT - BASELINE DEMAND FORECASTING")
print("=" * 70)


# ============================================================
# 1. LOAD FORECASTING DATASET
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
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATE RANGE")
print("=" * 70)

print("Start Date:", df["date"].min())
print("End Date  :", df["date"].max())
print("Total Days:", len(df))


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

# Last 90 days are used as test data
TEST_DAYS = 90

train = df.iloc[:-TEST_DAYS].copy()
test = df.iloc[-TEST_DAYS:].copy()

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("Training observations:", len(train))
print("Testing observations :", len(test))

print("\nTraining period:")
print(train["date"].min(), "to", train["date"].max())

print("\nTesting period:")
print(test["date"].min(), "to", test["date"].max())


# ============================================================
# 4. ACTUAL DEMAND
# ============================================================

y_train = train["units_sold"].values
y_test = test["units_sold"].values


# ============================================================
# 5. BASELINE 1 — NAIVE FORECAST
# ============================================================

# Tomorrow's demand = today's demand

test["naive_forecast"] = test["lag_1"]


# ============================================================
# 6. BASELINE 2 — 7-DAY SEASONAL NAIVE
# ============================================================

# Demand forecast = demand from same weekday previous week

test["seasonal_naive_7"] = test["lag_7"]


# ============================================================
# 7. BASELINE 3 — 7-DAY MOVING AVERAGE
# ============================================================

test["moving_average_7"] = test["rolling_mean_7"]


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    # Avoid division by zero
    non_zero = actual != 0

    if non_zero.sum() > 0:

        mape = np.mean(
            np.abs(
                (actual[non_zero] - predicted[non_zero])
                / actual[non_zero]
            )
        ) * 100

    else:

        mape = np.nan

    # Weighted Absolute Percentage Error
    wape = (
        np.sum(np.abs(actual - predicted))
        / np.sum(np.abs(actual))
    ) * 100

    return mae, rmse, mape, wape


# ============================================================
# 9. CALCULATE BASELINE METRICS
# ============================================================

results = []

models = {
    "Naive": test["naive_forecast"],
    "Seasonal Naive (7-Day)": test["seasonal_naive_7"],
    "Moving Average (7-Day)": test["moving_average_7"]
}


for model_name, predictions in models.items():

    valid = (
        predictions.notna()
        & test["units_sold"].notna()
    )

    actual = test.loc[valid, "units_sold"]
    predicted = predictions.loc[valid]

    mae, rmse, mape, wape = calculate_metrics(
        actual,
        predicted
    )

    results.append({
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "WAPE": wape
    })


results_df = pd.DataFrame(results)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BASELINE MODEL PERFORMANCE")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 11. BEST BASELINE
# ============================================================

best_model = (
    results_df
    .sort_values("WAPE")
    .iloc[0]
)

print("\n" + "=" * 70)
print("BEST BASELINE MODEL")
print("=" * 70)

print("Model :", best_model["model"])
print("MAE   :", round(best_model["MAE"], 2))
print("RMSE  :", round(best_model["RMSE"], 2))
print("MAPE  :", round(best_model["MAPE"], 2), "%")
print("WAPE  :", round(best_model["WAPE"], 2), "%")


# ============================================================
# 12. SAMPLE FORECASTS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE FORECAST RESULTS")
print("=" * 70)

sample = test[
    [
        "date",
        "units_sold",
        "naive_forecast",
        "seasonal_naive_7",
        "moving_average_7"
    ]
].head(15)

print(sample.to_string(index=False))


# ============================================================
# 13. SAVE RESULTS
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


RESULT_PATH = (
    OUTPUT_DIR
    / "baseline_model_results.csv"
)

results_df.to_csv(
    RESULT_PATH,
    index=False
)


FORECAST_PATH = (
    OUTPUT_DIR
    / "baseline_forecasts.csv"
)

test[
    [
        "date",
        "units_sold",
        "naive_forecast",
        "seasonal_naive_7",
        "moving_average_7"
    ]
].to_csv(
    FORECAST_PATH,
    index=False
)


print("\n" + "=" * 70)
print("BASELINE FORECASTING COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(RESULT_PATH)

print("\nForecasts saved to:")
print(FORECAST_PATH)