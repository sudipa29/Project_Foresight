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
print("PROJECT FORESIGHT - ARIMA / SARIMA MODEL TUNING")
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


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading forecasting dataset...")

file_path = os.path.join(
    FORECAST_DIR,
    "daily_forecasting_dataset.csv"
)

df = pd.read_csv(
    file_path
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.sort_values(
    "date"
)


print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# 3. PREPARE DAILY TIME SERIES
# ============================================================

daily = (
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


daily = daily.set_index(
    "date"
)


daily = daily.asfreq(
    "D"
)


daily["units_sold"] = (
    daily["units_sold"]
    .fillna(0)
)


y = daily["units_sold"]


print(
    "Time-series observations:",
    len(y)
)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

TEST_SIZE = 90


train = y.iloc[:-TEST_SIZE]

test = y.iloc[-TEST_SIZE:]


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)


print(
    "Training:",
    train.index.min(),
    "to",
    train.index.max()
)

print(
    "Testing:",
    test.index.min(),
    "to",
    test.index.max()
)


# ============================================================
# 5. METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    actual = np.asarray(actual)

    predicted = np.asarray(predicted)


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


    denominator = np.sum(
        np.abs(actual)
    )


    if denominator != 0:

        wape = (
            np.sum(
                np.abs(
                    actual - predicted
                )
            )
            /
            denominator
        ) * 100

    else:

        wape = np.nan


    return (
        mae,
        rmse,
        mape,
        wape
    )


# ============================================================
# 6. ARIMA PARAMETER GRID
# ============================================================

arima_orders = [

    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),

    (2, 0, 0),
    (2, 1, 0),
    (2, 1, 1),

    (3, 0, 0),
    (3, 1, 0),
    (3, 1, 1),

    (5, 1, 0),
    (5, 1, 1),
    (5, 1, 2)

]


# ============================================================
# 7. SARIMA PARAMETER GRID
# ============================================================

sarima_orders = [

    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),

    (2, 0, 0),
    (2, 1, 0),
    (2, 1, 1)

]


seasonal_orders = [

    (1, 0, 0, 7),
    (1, 1, 0, 7),
    (1, 1, 1, 7),

    (0, 1, 1, 7),

    (2, 1, 1, 7)

]


# ============================================================
# 8. RESULTS
# ============================================================

results = []


# ============================================================
# 9. ARIMA TUNING
# ============================================================

print("\n" + "=" * 70)
print("TUNING ARIMA")
print("=" * 70)


for order in arima_orders:

    print(
        f"Testing ARIMA{order}..."
    )

    try:

        model = ARIMA(
            train,
            order=order
        )

        fitted = model.fit()


        forecast = fitted.forecast(
            steps=TEST_SIZE
        )


        metrics = calculate_metrics(
            test,
            forecast
        )


        results.append({

            "model": "ARIMA",

            "order": str(order),

            "seasonal_order": "-",

            "MAE": metrics[0],

            "RMSE": metrics[1],

            "MAPE": metrics[2],

            "WAPE": metrics[3]

        })


        print(
            f"MAE: {metrics[0]:.2f}"
        )


    except Exception as e:

        print(
            f"Failed ARIMA{order}: {e}"
        )


# ============================================================
# 10. SARIMA TUNING
# ============================================================

print("\n" + "=" * 70)
print("TUNING SARIMA")
print("=" * 70)


for order in sarima_orders:

    for seasonal_order in seasonal_orders:

        print(
            f"Testing SARIMA"
            f"{order}"
            f"x"
            f"{seasonal_order}..."
        )


        try:

            model = SARIMAX(

                train,

                order=order,

                seasonal_order=seasonal_order,

                enforce_stationarity=False,

                enforce_invertibility=False

            )


            fitted = model.fit(
                disp=False
            )


            forecast = fitted.forecast(
                steps=TEST_SIZE
            )


            metrics = calculate_metrics(
                test,
                forecast
            )


            results.append({

                "model": "SARIMA",

                "order": str(order),

                "seasonal_order":
                    str(seasonal_order),

                "MAE": metrics[0],

                "RMSE": metrics[1],

                "MAPE": metrics[2],

                "WAPE": metrics[3]

            })


            print(
                f"MAE: {metrics[0]:.2f}"
            )


        except Exception as e:

            print(
                "Failed:",
                e
            )


# ============================================================
# 11. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
    .sort_values(
        "MAE",
        ascending=True
    )
    .reset_index(drop=True)
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TUNING RESULTS")
print("=" * 70)


print(
    results_df.round(2).to_string(
        index=False
    )
)


# ============================================================
# 13. BEST STATISTICAL MODEL
# ============================================================

best = results_df.iloc[0]


print("\n" + "=" * 70)
print("BEST STATISTICAL CONFIGURATION")
print("=" * 70)


print(
    "Model:",
    best["model"]
)

print(
    "Order:",
    best["order"]
)

print(
    "Seasonal Order:",
    best["seasonal_order"]
)

print(
    "MAE:",
    round(
        best["MAE"],
        2
    )
)

print(
    "RMSE:",
    round(
        best["RMSE"],
        2
    )
)

print(
    "MAPE:",
    round(
        best["MAPE"],
        2
    ),
    "%"
)

print(
    "WAPE:",
    round(
        best["WAPE"],
        2
    ),
    "%"
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

output_file = os.path.join(
    FORECAST_DIR,
    "arima_sarima",
    "arima_sarima_tuning_results.csv"
)


results_df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)
print("TUNING COMPLETED")
print("=" * 70)


print(
    "\nSaved:"
)

print(
    output_file
)


print("\n" + "=" * 70)