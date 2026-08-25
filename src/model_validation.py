import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

print("=" * 70)
print("PROJECT FORESIGHT - MODEL VALIDATION")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORECAST_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "forecasting"
)

VALIDATION_DIR = os.path.join(
    FORECAST_DIR,
    "validation"
)

os.makedirs(VALIDATION_DIR, exist_ok=True)


# ============================================================
# 2. LOAD ADVANCED FORECAST RESULTS
# ============================================================

print("\nLoading advanced forecast results...")

forecast_file = os.path.join(
    FORECAST_DIR,
    "advanced_model_forecasts.csv"
)

df = pd.read_csv(forecast_file)

df["date"] = pd.to_datetime(df["date"])

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "units_sold",
    "rf_forecast",
    "xgb_forecast",
    "lgb_forecast"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 4. CALCULATE ERRORS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING MODEL ERRORS")
print("=" * 70)


df["rf_error"] = (
    df["units_sold"] - df["rf_forecast"]
)

df["xgb_error"] = (
    df["units_sold"] - df["xgb_forecast"]
)

df["lgb_error"] = (
    df["units_sold"] - df["lgb_forecast"]
)


df["rf_abs_error"] = abs(df["rf_error"])

df["xgb_abs_error"] = abs(df["xgb_error"])

df["lgb_abs_error"] = abs(df["lgb_error"])


df["lgb_pct_error"] = (
    abs(df["lgb_error"])
    / df["units_sold"].replace(0, np.nan)
) * 100


# ============================================================
# 5. VALIDATION METRICS
# ============================================================

def calculate_metrics(actual, predicted):

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
            (
                actual[non_zero]
                - predicted[non_zero]
            )
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

    return mae, rmse, mape, wape


actual = df["units_sold"]


rf_metrics = calculate_metrics(
    actual,
    df["rf_forecast"]
)

xgb_metrics = calculate_metrics(
    actual,
    df["xgb_forecast"]
)

lgb_metrics = calculate_metrics(
    actual,
    df["lgb_forecast"]
)


validation_results = pd.DataFrame({

    "model": [
        "Random Forest",
        "XGBoost",
        "LightGBM"
    ],

    "MAE": [
        rf_metrics[0],
        xgb_metrics[0],
        lgb_metrics[0]
    ],

    "RMSE": [
        rf_metrics[1],
        xgb_metrics[1],
        lgb_metrics[1]
    ],

    "MAPE": [
        rf_metrics[2],
        xgb_metrics[2],
        lgb_metrics[2]
    ],

    "WAPE": [
        rf_metrics[3],
        xgb_metrics[3],
        lgb_metrics[3]
    ]

})


print("\n" + "=" * 70)
print("VALIDATION METRICS")
print("=" * 70)

print(
    validation_results.round(2).to_string(
        index=False
    )
)


# ============================================================
# 6. BEST MODEL
# ============================================================

best_model = validation_results.loc[
    validation_results["MAE"].idxmin()
]

print("\n" + "=" * 70)
print("BEST VALIDATED MODEL")
print("=" * 70)

print(
    "Model:",
    best_model["model"]
)

print(
    "MAE:",
    round(best_model["MAE"], 2)
)

print(
    "RMSE:",
    round(best_model["RMSE"], 2)
)

print(
    "MAPE:",
    round(best_model["MAPE"], 2),
    "%"
)

print(
    "WAPE:",
    round(best_model["WAPE"], 2),
    "%"
)


# ============================================================
# 7. LIGHTGBM ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("LIGHTGBM ERROR ANALYSIS")
print("=" * 70)


print(
    "\nAverage absolute error:",
    round(
        df["lgb_abs_error"].mean(),
        2
    )
)

print(
    "Maximum absolute error:",
    round(
        df["lgb_abs_error"].max(),
        2
    )
)

print(
    "Average percentage error:",
    round(
        df["lgb_pct_error"].mean(),
        2
    ),
    "%"
)


# ============================================================
# 8. WORST FORECAST DAYS
# ============================================================

worst_days = (
    df[
        [
            "date",
            "units_sold",
            "lgb_forecast",
            "lgb_error",
            "lgb_abs_error",
            "lgb_pct_error"
        ]
    ]
    .sort_values(
        "lgb_abs_error",
        ascending=False
    )
    .head(20)
)


print("\n" + "=" * 70)
print("TOP 20 WORST LIGHTGBM FORECAST DAYS")
print("=" * 70)

print(
    worst_days.round(2).to_string(
        index=False
    )
)


# ============================================================
# 9. BEST FORECAST DAYS
# ============================================================

best_days = (
    df[
        [
            "date",
            "units_sold",
            "lgb_forecast",
            "lgb_error",
            "lgb_abs_error",
            "lgb_pct_error"
        ]
    ]
    .sort_values(
        "lgb_abs_error",
        ascending=True
    )
    .head(20)
)


print("\n" + "=" * 70)
print("TOP 20 MOST ACCURATE LIGHTGBM DAYS")
print("=" * 70)

print(
    best_days.round(2).to_string(
        index=False
    )
)


# ============================================================
# 10. ACTUAL VS FORECAST
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["date"],
    df["units_sold"],
    label="Actual Demand"
)

plt.plot(
    df["date"],
    df["lgb_forecast"],
    label="LightGBM Forecast"
)

plt.title(
    "Actual vs LightGBM Demand Forecast"
)

plt.xlabel("Date")

plt.ylabel("Units Sold")

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()


actual_forecast_plot = os.path.join(
    VALIDATION_DIR,
    "actual_vs_lightgbm.png"
)

plt.savefig(
    actual_forecast_plot,
    dpi=150
)

plt.close()


# ============================================================
# 11. FORECAST ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["lgb_error"],
    bins=30
)

plt.axvline(
    0,
    linestyle="--"
)

plt.title(
    "LightGBM Forecast Error Distribution"
)

plt.xlabel(
    "Forecast Error"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()


error_plot = os.path.join(
    VALIDATION_DIR,
    "lightgbm_error_distribution.png"
)

plt.savefig(
    error_plot,
    dpi=150
)

plt.close()


# ============================================================
# 12. ACTUAL VS PREDICTED SCATTER
# ============================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    df["units_sold"],
    df["lgb_forecast"],
    alpha=0.5
)

min_value = min(
    df["units_sold"].min(),
    df["lgb_forecast"].min()
)

max_value = max(
    df["units_sold"].max(),
    df["lgb_forecast"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title(
    "Actual vs Predicted Demand - LightGBM"
)

plt.xlabel(
    "Actual Units Sold"
)

plt.ylabel(
    "Predicted Units Sold"
)

plt.tight_layout()


scatter_plot = os.path.join(
    VALIDATION_DIR,
    "actual_vs_predicted_scatter.png"
)

plt.savefig(
    scatter_plot,
    dpi=150
)

plt.close()


# ============================================================
# 13. SAVE VALIDATION METRICS
# ============================================================

metrics_file = os.path.join(
    VALIDATION_DIR,
    "validation_metrics.csv"
)

validation_results.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# 14. SAVE ERROR ANALYSIS
# ============================================================

error_file = os.path.join(
    VALIDATION_DIR,
    "prediction_errors.csv"
)

df.to_csv(
    error_file,
    index=False
)


# ============================================================
# 15. SAVE WORST DAYS
# ============================================================

worst_file = os.path.join(
    VALIDATION_DIR,
    "worst_forecast_days.csv"
)

worst_days.to_csv(
    worst_file,
    index=False
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL VALIDATION COMPLETED")
print("=" * 70)

print("\nValidation files saved to:")

print(
    VALIDATION_DIR
)

print("\nGenerated files:")

print(
    "1. validation_metrics.csv"
)

print(
    "2. prediction_errors.csv"
)

print(
    "3. worst_forecast_days.csv"
)

print(
    "4. actual_vs_lightgbm.png"
)

print(
    "5. lightgbm_error_distribution.png"
)

print(
    "6. actual_vs_predicted_scatter.png"
)

print("\n" + "=" * 70)