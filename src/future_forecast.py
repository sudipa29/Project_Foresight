import os
import warnings
import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor


warnings.filterwarnings("ignore")


print("=" * 70)
print("PROJECT FORESIGHT - FUTURE DEMAND FORECASTING")
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

OUTPUT_DIR = os.path.join(
    FORECAST_DIR,
    "future"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD FORECASTING DATASET
# ============================================================

print("\nLoading forecasting dataset...")

input_file = os.path.join(
    FORECAST_DIR,
    "daily_forecasting_dataset.csv"
)

df = pd.read_csv(input_file)

df["date"] = pd.to_datetime(
    df["date"]
)

df = df.sort_values(
    "date"
).reset_index(drop=True)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print(
    "Historical period:",
    df["date"].min(),
    "to",
    df["date"].max()
)


# ============================================================
# 3. CREATE ADVANCED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING ADVANCED FEATURES")
print("=" * 70)


def create_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    data["year"] = data["date"].dt.year

    data["month"] = data["date"].dt.month

    data["quarter"] = data["date"].dt.quarter

    data["day_of_week"] = data["date"].dt.dayofweek

    data["day_of_month"] = data["date"].dt.day

    data["week_of_year"] = (
        data["date"].dt.isocalendar().week.astype(int)
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)


    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    data["trend"] = (
        np.arange(len(data))
    )


    # --------------------------------------------------------
    # Cyclical features
    # --------------------------------------------------------

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    data["day_of_week_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["day_of_week_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )


    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    data["lag_1"] = (
        data["units_sold"].shift(1)
    )

    data["lag_2"] = (
        data["units_sold"].shift(2)
    )

    data["lag_3"] = (
        data["units_sold"].shift(3)
    )

    data["lag_7"] = (
        data["units_sold"].shift(7)
    )

    data["lag_14"] = (
        data["units_sold"].shift(14)
    )

    data["lag_21"] = (
        data["units_sold"].shift(21)
    )

    data["lag_28"] = (
        data["units_sold"].shift(28)
    )

    data["lag_56"] = (
        data["units_sold"].shift(56)
    )


    # --------------------------------------------------------
    # Rolling demand features
    # --------------------------------------------------------

    data["rolling_mean_7"] = (
        data["units_sold"]
        .rolling(7)
        .mean()
    )

    data["rolling_mean_14"] = (
        data["units_sold"]
        .rolling(14)
        .mean()
    )

    data["rolling_mean_28"] = (
        data["units_sold"]
        .rolling(28)
        .mean()
    )

    data["rolling_mean_56"] = (
        data["units_sold"]
        .rolling(56)
        .mean()
    )


    data["rolling_std_7"] = (
        data["units_sold"]
        .rolling(7)
        .std()
    )

    data["rolling_std_14"] = (
        data["units_sold"]
        .rolling(14)
        .std()
    )

    data["rolling_std_28"] = (
        data["units_sold"]
        .rolling(28)
        .std()
    )


    return data


feature_df = create_features(df)

print("Advanced features created.")


# ============================================================
# 4. MODEL FEATURES
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

    "trend",

    "month_sin",
    "month_cos",

    "day_of_week_sin",
    "day_of_week_cos",

    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "lag_56",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_mean_56",

    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28"
]


print("\nNumber of features:", len(features))


# ============================================================
# 5. REMOVE INITIAL MISSING ROWS
# ============================================================

model_data = feature_df.dropna(
    subset=features + ["units_sold"]
).copy()

print(
    "Training rows:",
    len(model_data)
)


# ============================================================
# 6. TRAIN FINAL LIGHTGBM MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL LIGHTGBM MODEL")
print("=" * 70)


X_train = model_data[
    features
]

y_train = model_data[
    "units_sold"
]


final_model = LGBMRegressor(

    objective="regression",

    n_estimators=500,

    learning_rate=0.03,

    num_leaves=31,

    max_depth=-1,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    verbosity=-1
)


final_model.fit(
    X_train,
    y_train
)


print(
    "Final LightGBM model trained successfully!"
)


# ============================================================
# 7. FORECAST HORIZON
# ============================================================

FORECAST_DAYS = 30

last_date = df["date"].max()

future_dates = pd.date_range(

    start=last_date + pd.Timedelta(days=1),

    periods=FORECAST_DAYS,

    freq="D"
)


print("\n" + "=" * 70)
print("FUTURE FORECAST PERIOD")
print("=" * 70)

print(
    "Start:",
    future_dates.min()
)

print(
    "End:",
    future_dates.max()
)

print(
    "Forecast days:",
    len(future_dates)
)


# ============================================================
# 8. PREPARE RECURSIVE FORECAST DATA
# ============================================================

history = df[
    [
        "date",
        "units_sold",
        "avg_discount",
        "promotion_flag"
    ]
].copy()


# ------------------------------------------------------------
# Use recent historical discount/promotion patterns
# ------------------------------------------------------------

recent_discount = (
    history["avg_discount"]
    .tail(28)
    .mean()
)

recent_promotion_rate = (
    history["promotion_flag"]
    .tail(28)
    .mean()
)


print("\nRecent average discount:",
      round(recent_discount, 2))

print(
    "Recent promotion rate:",
    round(recent_promotion_rate, 2)
)


# ============================================================
# 9. RECURSIVE FUTURE FORECAST
# ============================================================

print("\n" + "=" * 70)
print("GENERATING FUTURE FORECAST")
print("=" * 70)


future_results = []


for future_date in future_dates:

    # --------------------------------------------------------
    # Assume future discount based on recent average
    # --------------------------------------------------------

    future_discount = recent_discount

    # --------------------------------------------------------
    # Assume future promotion based on recent pattern
    # --------------------------------------------------------

    future_promotion = int(
        round(recent_promotion_rate)
    )


    # --------------------------------------------------------
    # Add temporary future row
    # --------------------------------------------------------

    new_row = pd.DataFrame({

        "date": [
            future_date
        ],

        "units_sold": [
            np.nan
        ],

        "avg_discount": [
            future_discount
        ],

        "promotion_flag": [
            future_promotion
        ]
    })


    temp = pd.concat(
        [
            history,
            new_row
        ],
        ignore_index=True
    )


    # --------------------------------------------------------
    # Create features
    # --------------------------------------------------------

    temp_features = create_features(
        temp
    )


    current_features = (
        temp_features
        .iloc[-1:]
        [features]
    )


    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    prediction = final_model.predict(
        current_features
    )[0]


    # Prevent negative demand
    prediction = max(
        0,
        prediction
    )


    # --------------------------------------------------------
    # Store prediction
    # --------------------------------------------------------

    future_results.append({

        "date": future_date,

        "forecast_units": prediction,

        "avg_discount": future_discount,

        "promotion_flag": future_promotion
    })


    # --------------------------------------------------------
    # Add prediction to history
    # --------------------------------------------------------

    history = pd.concat(

        [
            history,

            pd.DataFrame({

                "date": [
                    future_date
                ],

                "units_sold": [
                    prediction
                ],

                "avg_discount": [
                    future_discount
                ],

                "promotion_flag": [
                    future_promotion
                ]
            })
        ],

        ignore_index=True
    )


# ============================================================
# 10. CREATE FORECAST DATAFRAME
# ============================================================

future_forecast = pd.DataFrame(
    future_results
)


# ============================================================
# 11. DEMAND LEVEL
# ============================================================

historical_mean = (
    df["units_sold"]
    .mean()
)

historical_std = (
    df["units_sold"]
    .std()
)


high_threshold = (
    historical_mean
    + 0.5 * historical_std
)

low_threshold = (
    historical_mean
    - 0.5 * historical_std
)


def demand_level(value):

    if value >= high_threshold:
        return "High Demand"

    elif value <= low_threshold:
        return "Low Demand"

    else:
        return "Normal Demand"


future_forecast[
    "demand_level"
] = future_forecast[
    "forecast_units"
].apply(
    demand_level
)


# ============================================================
# 12. DEMAND RISK
# ============================================================

def demand_risk(level):

    if level == "High Demand":
        return "High"

    elif level == "Low Demand":
        return "Low"

    else:
        return "Medium"


future_forecast[
    "demand_risk"
] = future_forecast[
    "demand_level"
].apply(
    demand_risk
)


# ============================================================
# 13. ROUND FORECAST
# ============================================================

future_forecast[
    "forecast_units"
] = future_forecast[
    "forecast_units"
].round(0)


future_forecast[
    "avg_discount"
] = future_forecast[
    "avg_discount"
].round(2)


# ============================================================
# 14. PRINT FORECAST
# ============================================================

print("\n" + "=" * 70)
print("30-DAY FUTURE DEMAND FORECAST")
print("=" * 70)

print(
    future_forecast.to_string(
        index=False
    )
)


# ============================================================
# 15. FORECAST SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FORECAST SUMMARY")
print("=" * 70)


print(
    "Average forecast demand:",
    round(
        future_forecast[
            "forecast_units"
        ].mean(),
        2
    )
)


print(
    "Maximum forecast demand:",
    round(
        future_forecast[
            "forecast_units"
        ].max(),
        2
    )
)


print(
    "Minimum forecast demand:",
    round(
        future_forecast[
            "forecast_units"
        ].min(),
        2
    )
)


print("\nDemand level distribution:")

print(
    future_forecast[
        "demand_level"
    ].value_counts()
)


# ============================================================
# 16. SAVE FUTURE FORECAST
# ============================================================

forecast_file = os.path.join(
    OUTPUT_DIR,
    "future_30_day_forecast.csv"
)


future_forecast.to_csv(
    forecast_file,
    index=False
)


# ============================================================
# 17. HIGH DEMAND DAYS
# ============================================================

high_demand_days = (
    future_forecast[
        future_forecast[
            "demand_level"
        ] == "High Demand"
    ]
)


high_demand_file = os.path.join(
    OUTPUT_DIR,
    "high_demand_days.csv"
)


high_demand_days.to_csv(
    high_demand_file,
    index=False
)


# ============================================================
# 18. LOW DEMAND DAYS
# ============================================================

low_demand_days = (
    future_forecast[
        future_forecast[
            "demand_level"
        ] == "Low Demand"
    ]
)


low_demand_file = os.path.join(
    OUTPUT_DIR,
    "low_demand_days.csv"
)


low_demand_days.to_csv(
    low_demand_file,
    index=False
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FUTURE FORECASTING COMPLETED")
print("=" * 70)

print("\nFiles saved to:")

print(
    OUTPUT_DIR
)

print("\nGenerated files:")

print(
    "1. future_30_day_forecast.csv"
)

print(
    "2. high_demand_days.csv"
)

print(
    "3. low_demand_days.csv"
)

print("\n" + "=" * 70)