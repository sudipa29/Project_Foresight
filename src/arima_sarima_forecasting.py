import os
import warnings

import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")


print("=" * 70)
print("PROJECT FORESIGHT - ARIMA & SARIMA FORECASTING")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FORECAST_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "forecasting"
)

ARIMA_DIR = os.path.join(
    FORECAST_DIR,
    "arima_sarima"
)

os.makedirs(
    ARIMA_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD FORECASTING DATASET
# ============================================================

print("\nLoading forecasting dataset...")

forecast_file = os.path.join(
    FORECAST_DIR,
    "daily_forecasting_dataset.csv"
)

df = pd.read_csv(
    forecast_file
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.sort_values(
    "date"
)


print("Dataset loaded successfully!")

print(
    "Shape:",
    df.shape
)

print(
    "Historical period:",
    df["date"].min(),
    "to",
    df["date"].max()
)


# ============================================================
# 3. PREPARE TIME SERIES
# ============================================================

print("\n" + "=" * 70)
print("PREPARING TIME SERIES")
print("=" * 70)


time_series = (
    df[
        [
            "date",
            "units_sold"
        ]
    ]
    .groupby(
        "date",
        as_index=False
    )["units_sold"]
    .sum()
)


time_series = time_series.set_index(
    "date"
)


time_series = time_series.asfreq(
    "D"
)


# Fill missing demand dates if any
time_series["units_sold"] = (
    time_series["units_sold"]
    .fillna(0)
)


y = time_series[
    "units_sold"
]


print(
    "Time-series observations:",
    len(y)
)

print(
    "Start:",
    y.index.min()
)

print(
    "End:",
    y.index.max()
)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)


test_size = 90


train = y.iloc[:-test_size]

test = y.iloc[-test_size:]


print(
    "Training observations:",
    len(train)
)

print(
    "Testing observations:",
    len(test)
)

print(
    "Training period:",
    train.index.min(),
    "to",
    train.index.max()
)

print(
    "Testing period:",
    test.index.min(),
    "to",
    test.index.max()
)


# ============================================================
# 5. METRICS FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

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

    if non_zero.sum() > 0:

        mape = (
            np.mean(
                np.abs(
                    (
                        actual[non_zero]
                        -
                        predicted[non_zero]
                    )
                    /
                    actual[non_zero]
                )
            )
            * 100
        )

    else:

        mape = np.nan


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


    return (
        mae,
        rmse,
        mape,
        wape
    )


# ============================================================
# 6. ARIMA
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ARIMA")
print("=" * 70)


arima_order = (
    5,
    1,
    2
)


print(
    "ARIMA order:",
    arima_order
)


arima_model = ARIMA(
    train,
    order=arima_order
)


arima_result = arima_model.fit()


arima_forecast = (
    arima_result
    .forecast(
        steps=test_size
    )
)


arima_forecast.index = test.index


print(
    "ARIMA completed."
)


# ============================================================
# 7. SARIMA
# ============================================================

print("\n" + "=" * 70)
print("TRAINING SARIMA")
print("=" * 70)


sarima_order = (
    1,
    1,
    1
)

seasonal_order = (
    1,
    1,
    1,
    7
)


print(
    "SARIMA order:",
    sarima_order
)

print(
    "Seasonal order:",
    seasonal_order
)


sarima_model = SARIMAX(
    train,
    order=sarima_order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)


sarima_result = sarima_model.fit(
    disp=False
)


sarima_forecast = (
    sarima_result
    .forecast(
        steps=test_size
    )
)


sarima_forecast.index = test.index


print(
    "SARIMA completed."
)


# ============================================================
# 8. MODEL METRICS
# ============================================================

print("\n" + "=" * 70)
print("ARIMA / SARIMA MODEL COMPARISON")
print("=" * 70)


arima_metrics = calculate_metrics(
    test,
    arima_forecast
)


sarima_metrics = calculate_metrics(
    test,
    sarima_forecast
)


comparison = pd.DataFrame({

    "model": [
        "ARIMA",
        "SARIMA"
    ],

    "MAE": [
        arima_metrics[0],
        sarima_metrics[0]
    ],

    "RMSE": [
        arima_metrics[1],
        sarima_metrics[1]
    ],

    "MAPE": [
        arima_metrics[2],
        sarima_metrics[2]
    ],

    "WAPE": [
        arima_metrics[3],
        sarima_metrics[3]
    ]

})


print(
    comparison
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 9. BEST STATISTICAL MODEL
# ============================================================

best_model = comparison.loc[
    comparison["MAE"].idxmin()
]


print("\n" + "=" * 70)
print("BEST STATISTICAL MODEL")
print("=" * 70)


print(
    "Model:",
    best_model["model"]
)

print(
    "MAE:",
    round(
        best_model["MAE"],
        2
    )
)

print(
    "RMSE:",
    round(
        best_model["RMSE"],
        2
    )
)

print(
    "MAPE:",
    round(
        best_model["MAPE"],
        2
    ),
    "%"
)

print(
    "WAPE:",
    round(
        best_model["WAPE"],
        2
    ),
    "%"
)


# ============================================================
# 10. SAVE FORECAST RESULTS
# ============================================================

forecast_results = pd.DataFrame({

    "date": test.index,

    "actual_units": test.values,

    "arima_forecast": (
        arima_forecast.values
    ),

    "sarima_forecast": (
        sarima_forecast.values
    )

})


forecast_results["arima_error"] = (
    forecast_results["actual_units"]
    -
    forecast_results["arima_forecast"]
)


forecast_results["sarima_error"] = (
    forecast_results["actual_units"]
    -
    forecast_results["sarima_forecast"]
)


forecast_file = os.path.join(
    ARIMA_DIR,
    "arima_sarima_forecasts.csv"
)


forecast_results.to_csv(
    forecast_file,
    index=False
)


# ============================================================
# 11. SAVE MODEL COMPARISON
# ============================================================

comparison_file = os.path.join(
    ARIMA_DIR,
    "arima_sarima_comparison.csv"
)


comparison.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# 12. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("ARIMA & SARIMA FORECASTING COMPLETED")
print("=" * 70)


print(
    "\nFiles saved to:"
)

print(
    ARIMA_DIR
)


print(
    "\nGenerated files:"
)

print(
    "1. arima_sarima_forecasts.csv"
)

print(
    "2. arima_sarima_comparison.csv"
)


print("\n" + "=" * 70)