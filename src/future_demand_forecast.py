# ============================================================
# PROJECT FORESIGHT
# Phase 6.0 - Future Demand Forecast
# Final Production Model: LightGBM
#
# Optimized for large dataset:
# 17M+ rows
# 50 stores
# 200 SKUs
# 10,000 Store-SKU combinations
#
# Forecast Horizons:
# 30 Days
# 60 Days
# 90 Days
# ============================================================

import os

# Limit unnecessary thread oversubscription
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from pathlib import Path
import gc
import warnings

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(
    r"E:\Zidio_Development_Internship\Project_Foresight"
)

PROCESSED_PATH = BASE_PATH / "data" / "processed"

FORECASTING_PATH = (
    PROCESSED_PATH / "forecasting"
)

OUTPUT_PATH = (
    FORECASTING_PATH / "future"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILE
# ============================================================

INPUT_FILE = (
    FORECASTING_PATH /
    "forecast_demand_daily.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

FORECAST_30_FILE = (
    OUTPUT_PATH /
    "future_30_day_forecast.csv"
)

FORECAST_60_FILE = (
    OUTPUT_PATH /
    "future_60_day_forecast.csv"
)

FORECAST_90_FILE = (
    OUTPUT_PATH /
    "future_90_day_forecast.csv"
)

ALL_FORECAST_FILE = (
    OUTPUT_PATH /
    "future_demand_forecast_30_60_90.csv"
)

STORE_SUMMARY_FILE = (
    OUTPUT_PATH /
    "store_future_demand_summary.csv"
)

SKU_SUMMARY_FILE = (
    OUTPUT_PATH /
    "sku_future_demand_summary.csv"
)

HIGH_DEMAND_FILE = (
    OUTPUT_PATH /
    "high_demand_forecast_items.csv"
)

LOW_DEMAND_FILE = (
    OUTPUT_PATH /
    "low_demand_forecast_items.csv"
)

MODEL_INFO_FILE = (
    OUTPUT_PATH /
    "future_forecast_model_info.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_DAYS = 365

FORECAST_HORIZONS = [30, 60, 90]

RANDOM_STATE = 42

# LightGBM parameters deliberately kept moderate
# so the 3.3M+ training rows remain manageable.
N_ESTIMATORS = 300

LEARNING_RATE = 0.05

NUM_LEAVES = 31

MAX_DEPTH = -1

N_JOBS = 4


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def reduce_memory(df):
    """
    Reduce numerical memory usage.
    """

    for col in df.columns:

        if df[col].dtype == "int64":

            df[col] = pd.to_numeric(
                df[col],
                downcast="integer"
            )

        elif df[col].dtype == "float64":

            df[col] = pd.to_numeric(
                df[col],
                downcast="float"
            )

    return df


def create_features(df):

    """
    Create forecasting features.

    IMPORTANT:
    Features are created separately by Store-SKU.
    """

    df = df.sort_values(
        ["store_id", "sku_id", "date"]
    ).copy()

    grouped = df.groupby(
        ["store_id", "sku_id"],
        sort=False
    )

    print("Creating lag features...")

    df["lag_1"] = grouped["units_sold"].shift(1)

    df["lag_7"] = grouped["units_sold"].shift(7)

    df["lag_14"] = grouped["units_sold"].shift(14)

    df["lag_30"] = grouped["units_sold"].shift(30)


    print("Creating rolling features...")

    # Shift first to prevent data leakage.
    shifted = grouped["units_sold"].shift(1)

    df["rolling_mean_7"] = (
        shifted
        .groupby(
            [
                df["store_id"],
                df["sku_id"]
            ]
        )
        .transform(
            lambda x:
            x.rolling(
                7,
                min_periods=1
            ).mean()
        )
    )

    df["rolling_mean_14"] = (
        shifted
        .groupby(
            [
                df["store_id"],
                df["sku_id"]
            ]
        )
        .transform(
            lambda x:
            x.rolling(
                14,
                min_periods=1
            ).mean()
        )
    )

    df["rolling_mean_30"] = (
        shifted
        .groupby(
            [
                df["store_id"],
                df["sku_id"]
            ]
        )
        .transform(
            lambda x:
            x.rolling(
                30,
                min_periods=1
            ).mean()
        )
    )


    print("Creating intermittency features...")

    df["active_days_7"] = (
        shifted
        .groupby(
            [
                df["store_id"],
                df["sku_id"]
            ]
        )
        .transform(
            lambda x:
            x.rolling(
                7,
                min_periods=1
            )
            .apply(
                lambda z:
                np.sum(z > 0),
                raw=True
            )
        )
    )

    df["active_days_30"] = (
        shifted
        .groupby(
            [
                df["store_id"],
                df["sku_id"]
            ]
        )
        .transform(
            lambda x:
            x.rolling(
                30,
                min_periods=1
            )
            .apply(
                lambda z:
                np.sum(z > 0),
                raw=True
            )
        )
    )


    print("Creating days-since-demand feature...")

    occurrence = (
        shifted
        .groupby(
            [
                df["store_id"],
                df["sku_id"]
            ]
        )
        .transform(
            lambda x:
            x.gt(0)
        )
    )

    # Calculate days since previous positive demand.
    temp_dates = df["date"].where(
        occurrence
    )

    previous_demand_date = (
        temp_dates
        .groupby(
            [
                df["store_id"],
                df["sku_id"]
            ]
        )
        .ffill()
    )

    df["days_since_demand"] = (
        df["date"] -
        previous_demand_date
    ).dt.days

    df["days_since_demand"] = (
        df["days_since_demand"]
        .fillna(999)
    )


    print("Creating trend features...")

    df["trend_7_vs_30"] = (
        df["rolling_mean_7"] /
        (
            df["rolling_mean_30"] +
            1e-6
        )
    )


    print("Creating calendar features...")

    df["day_of_week"] = (
        df["date"]
        .dt.dayofweek
        .astype("int8")
    )

    df["month"] = (
        df["date"]
        .dt.month
        .astype("int8")
    )

    df["quarter"] = (
        df["date"]
        .dt.quarter
        .astype("int8")
    )

    df["day_of_month"] = (
        df["date"]
        .dt.day
        .astype("int8")
    )

    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype("int8")
    )

    df["is_weekend"] = (
        df["day_of_week"]
        >= 5
    ).astype("int8")


    # Occurrence rate
    df["demand_occurrence"] = (
        df["units_sold"] > 0
    ).astype("int8")


    # Clean numerical features
    feature_cols = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
        "active_days_7",
        "active_days_30",
        "days_since_demand",
        "trend_7_vs_30",
    ]

    for col in feature_cols:

        df[col] = (
            df[col]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
            .astype("float32")
        )


    return df


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT - FUTURE DEMAND FORECAST")
print("=" * 70)


# ============================================================
# CHECK INPUT
# ============================================================

print_section("CHECKING INPUT FILE")

if not INPUT_FILE.exists():

    print("ERROR: Input file not found.")

    print()
    print("Expected:")
    print(INPUT_FILE)

    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

print("Input file found:")
print(INPUT_FILE)


# ============================================================
# LOAD ONLY REQUIRED COLUMNS
# ============================================================

print_section(
    "LOADING FORECAST DEMAND DATASET"
)

print("Loading only required columns...")

print(
    "IMPORTANT: Date will initially remain "
    "as STRING to avoid expensive conversion "
    "of 17M+ rows."
)

required_columns = [
    "store_id",
    "sku_id",
    "date",
    "units_sold"
]


demand = pd.read_csv(
    INPUT_FILE,
    usecols=required_columns,
    dtype={
        "store_id": "int16",
        "sku_id": "int16",
        "date": "string",
        "units_sold": "float32"
    },
    low_memory=False
)

print(
    f"Loaded shape: {demand.shape}"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print_section(
    "BASIC DATA VALIDATION"
)

print("Checking missing values...")

print(
    "Missing dates:",
    demand["date"].isna().sum()
)

print(
    "Missing demand:",
    demand["units_sold"].isna().sum()
)

print(
    "Negative demand:",
    (
        demand["units_sold"] < 0
    ).sum()
)


# ============================================================
# DATE RANGE USING STRING
# ============================================================

print_section(
    "DETERMINING HISTORICAL PERIOD"
)

# Dataset uses YYYY-MM-DD format.
# String comparison is much faster than
# converting all 17M dates.

min_date_string = (
    demand["date"]
    .min()
)

max_date_string = (
    demand["date"]
    .max()
)

print(
    "Minimum date:",
    min_date_string
)

print(
    "Maximum date:",
    max_date_string
)


# ============================================================
# FILTER LAST 365 DAYS
# ============================================================

latest_date = pd.Timestamp(
    max_date_string
)

training_start = (
    latest_date -
    pd.Timedelta(
        days=TRAINING_DAYS - 1
    )
)

training_start_string = (
    training_start.strftime(
        "%Y-%m-%d"
    )
)

latest_date_string = (
    latest_date.strftime(
        "%Y-%m-%d"
    )
)

print()
print(
    "Training start:",
    training_start_string
)

print(
    "Training end:",
    latest_date_string
)


# ============================================================
# FILTER BEFORE DATE CONVERSION
# ============================================================

print_section(
    "FILTERING TRAINING PERIOD"
)

demand = demand.loc[
    (
        demand["date"] >=
        training_start_string
    )
    &
    (
        demand["date"] <=
        latest_date_string
    )
].copy()

print(
    "Filtered shape:",
    demand.shape
)


# ============================================================
# NOW CONVERT ONLY FILTERED DATES
# ============================================================

print_section(
    "CONVERTING FILTERED DATES"
)

demand["date"] = pd.to_datetime(
    demand["date"],
    format="%Y-%m-%d"
)

print(
    "Date conversion completed."
)


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

demand = demand.loc[
    demand["units_sold"].notna()
].copy()

demand = demand.loc[
    demand["units_sold"] >= 0
].copy()


# ============================================================
# SORT
# ============================================================

print_section(
    "PREPARING HISTORICAL DATA"
)

demand = demand.sort_values(
    [
        "store_id",
        "sku_id",
        "date"
    ]
).reset_index(
    drop=True
)

print(
    "Historical rows:",
    len(demand)
)

print(
    "Stores:",
    demand["store_id"].nunique()
)

print(
    "SKUs:",
    demand["sku_id"].nunique()
)

print(
    "Store-SKU combinations:",
    demand[
        [
            "store_id",
            "sku_id"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)

print(
    "Latest historical date:",
    demand["date"].max()
)


# ============================================================
# CREATE FEATURES
# ============================================================

print_section(
    "CREATING MODEL FEATURES"
)

featured = create_features(
    demand
)

del demand

gc.collect()


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "store_id",
    "sku_id",

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",

    "active_days_7",
    "active_days_30",

    "days_since_demand",
    "trend_7_vs_30",

    "day_of_week",
    "month",
    "quarter",
    "day_of_month",
    "week_of_year",
    "is_weekend",
]


TARGET = "units_sold"


# ============================================================
# REMOVE EARLY LAG ROWS
# ============================================================

print_section(
    "PREPARING TRAINING DATA"
)

featured = featured.loc[
    featured["lag_30"].notna()
].copy()

print(
    "Training feature shape:",
    featured.shape
)


# ============================================================
# MEMORY OPTIMIZATION
# ============================================================

featured = reduce_memory(
    featured
)


# ============================================================
# MODEL DATA
# ============================================================

X = featured[
    FEATURES
].copy()

y = featured[
    TARGET
].astype(
    "float32"
).copy()


print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# ============================================================
# LIGHTGBM
# ============================================================

print_section(
    "TRAINING FINAL LIGHTGBM MODEL"
)

print(
    "Production model: LightGBM"
)

print(
    "Training rows:",
    len(X)
)

print(
    "Features:",
    len(FEATURES)
)

print(
    "Estimators:",
    N_ESTIMATORS
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Number of leaves:",
    NUM_LEAVES
)


model = LGBMRegressor(

    objective="regression",

    n_estimators=N_ESTIMATORS,

    learning_rate=LEARNING_RATE,

    num_leaves=NUM_LEAVES,

    max_depth=MAX_DEPTH,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.1,

    reg_lambda=0.1,

    random_state=RANDOM_STATE,

    n_jobs=N_JOBS,

    verbosity=-1
)


model.fit(
    X,
    y
)

print()
print(
    "LightGBM training completed."
)


# ============================================================
# FREE TRAINING DATA
# ============================================================

del X
del y

gc.collect()


# ============================================================
# PREPARE LAST HISTORICAL ROWS
# ============================================================

print_section(
    "PREPARING FORECAST STARTING POINT"
)

# Keep only the latest 30 days.
# These rows contain enough information
# to construct recursive forecasts.

history = featured[
    [
        "store_id",
        "sku_id",
        "date",
        "units_sold"
    ]
].copy()

history = history.sort_values(
    [
        "store_id",
        "sku_id",
        "date"
    ]
)

print(
    "Historical records available:",
    len(history)
)

print(
    "Latest historical date:",
    history["date"].max()
)


# ============================================================
# STORE-SKU LIST
# ============================================================

series = (
    history[
        [
            "store_id",
            "sku_id"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "store_id",
            "sku_id"
        ]
    )
    .reset_index(
        drop=True
    )
)

print(
    "Store-SKU series:",
    len(series)
)


# ============================================================
# FORECAST FUNCTION
# ============================================================

def generate_future_forecast(
    model,
    history_df,
    series_df,
    horizon
):

    print()
    print(
        "-" * 70
    )

    print(
        f"GENERATING {horizon}-DAY FORECAST"
    )

    print(
        "-" * 70
    )

    latest_date = (
        history_df["date"].max()
    )

    future_dates = pd.date_range(
        start=latest_date +
        pd.Timedelta(days=1),

        periods=horizon,

        freq="D"
    )


    # --------------------------------------------------------
    # Keep only last 30 observations per series
    # --------------------------------------------------------

    working = history_df.copy()

    working = (
        working
        .sort_values(
            [
                "store_id",
                "sku_id",
                "date"
            ]
        )
        .groupby(
            [
                "store_id",
                "sku_id"
            ],
            group_keys=False
        )
        .tail(30)
        .copy()
    )


    forecasts = []


    # --------------------------------------------------------
    # Recursive forecasting
    # --------------------------------------------------------

    for step, forecast_date in enumerate(
        future_dates,
        start=1
    ):

        if step % 10 == 0 or step == 1:

            print(
                f"Forecast day {step}/{horizon}: "
                f"{forecast_date.date()}"
            )


        rows = []

        # ----------------------------------------------------
        # Construct features for each Store-SKU
        # ----------------------------------------------------

        for row in series_df.itertuples(
            index=False
        ):

            store_id = row.store_id
            sku_id = row.sku_id

            series_data = working.loc[
                (
                    working["store_id"] ==
                    store_id
                )
                &
                (
                    working["sku_id"] ==
                    sku_id
                )
            ].sort_values(
                "date"
            )


            values = (
                series_data[
                    "units_sold"
                ]
                .astype(float)
                .values
            )


            if len(values) == 0:

                lag_1 = 0.0
                lag_7 = 0.0
                lag_14 = 0.0
                lag_30 = 0.0

                mean_7 = 0.0
                mean_14 = 0.0
                mean_30 = 0.0

                active_7 = 0.0
                active_30 = 0.0

                days_since = 999.0

            else:

                lag_1 = (
                    values[-1]
                    if len(values) >= 1
                    else 0.0
                )

                lag_7 = (
                    values[-7]
                    if len(values) >= 7
                    else 0.0
                )

                lag_14 = (
                    values[-14]
                    if len(values) >= 14
                    else 0.0
                )

                lag_30 = (
                    values[-30]
                    if len(values) >= 30
                    else 0.0
                )


                mean_7 = np.mean(
                    values[-7:]
                )

                mean_14 = np.mean(
                    values[-14:]
                )

                mean_30 = np.mean(
                    values[-30:]
                )


                active_7 = np.sum(
                    values[-7:] > 0
                )

                active_30 = np.sum(
                    values[-30:] > 0
                )


                positive_positions = np.where(
                    values > 0
                )[0]

                if len(
                    positive_positions
                ) == 0:

                    days_since = 999.0

                else:

                    days_since = (
                        len(values) -
                        1 -
                        positive_positions[-1]
                    )


            trend = (
                mean_7 /
                (
                    mean_30 +
                    1e-6
                )
            )


            day_of_week = (
                forecast_date.dayofweek
            )

            month = (
                forecast_date.month
            )

            quarter = (
                forecast_date.quarter
            )

            day_of_month = (
                forecast_date.day
            )

            week_of_year = (
                forecast_date.isocalendar().week
            )

            is_weekend = int(
                day_of_week >= 5
            )


            rows.append(
                {
                    "store_id": store_id,
                    "sku_id": sku_id,

                    "lag_1": lag_1,
                    "lag_7": lag_7,
                    "lag_14": lag_14,
                    "lag_30": lag_30,

                    "rolling_mean_7": mean_7,
                    "rolling_mean_14": mean_14,
                    "rolling_mean_30": mean_30,

                    "active_days_7": active_7,
                    "active_days_30": active_30,

                    "days_since_demand":
                        days_since,

                    "trend_7_vs_30":
                        trend,

                    "day_of_week":
                        day_of_week,

                    "month":
                        month,

                    "quarter":
                        quarter,

                    "day_of_month":
                        day_of_month,

                    "week_of_year":
                        week_of_year,

                    "is_weekend":
                        is_weekend
                }
            )


        feature_frame = pd.DataFrame(
            rows
        )


        feature_frame = (
            feature_frame[
                FEATURES
            ]
        )


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predictions = model.predict(
            feature_frame
        )


        # Demand cannot be negative.
        predictions = np.maximum(
            predictions,
            0
        )


        # ----------------------------------------------------
        # Save daily prediction
        # ----------------------------------------------------

        daily_prediction = pd.DataFrame(
            {
                "store_id":
                    series_df[
                        "store_id"
                    ].values,

                "sku_id":
                    series_df[
                        "sku_id"
                    ].values,

                "date":
                    forecast_date,

                "forecast_units":
                    predictions
            }
        )


        forecasts.append(
            daily_prediction
        )


        # ----------------------------------------------------
        # Add prediction to working history
        # ----------------------------------------------------

        new_history = pd.DataFrame(
            {
                "store_id":
                    series_df[
                        "store_id"
                    ].values,

                "sku_id":
                    series_df[
                        "sku_id"
                    ].values,

                "date":
                    forecast_date,

                "units_sold":
                    predictions
            }
        )


        working = pd.concat(
            [
                working,
                new_history
            ],
            ignore_index=True
        )


        # Keep only last 30 rows per series
        working = (
            working
            .sort_values(
                [
                    "store_id",
                    "sku_id",
                    "date"
                ]
            )
            .groupby(
                [
                    "store_id",
                    "sku_id"
                ],
                group_keys=False
            )
            .tail(30)
            .copy()
        )


        del feature_frame
        del new_history

        gc.collect()


    result = pd.concat(
        forecasts,
        ignore_index=True
    )


    result["forecast_units"] = (
        result["forecast_units"]
        .astype("float32")
    )


    return result


# ============================================================
# GENERATE 30 DAY
# ============================================================

forecast_30 = generate_future_forecast(
    model,
    history,
    series,
    30
)

forecast_30.to_csv(
    FORECAST_30_FILE,
    index=False
)

print()
print(
    "30-day forecast saved:"
)

print(
    FORECAST_30_FILE
)


# ============================================================
# GENERATE 60 DAY
# ============================================================

forecast_60 = generate_future_forecast(
    model,
    history,
    series,
    60
)

forecast_60.to_csv(
    FORECAST_60_FILE,
    index=False
)

print()
print(
    "60-day forecast saved:"
)

print(
    FORECAST_60_FILE
)


# ============================================================
# GENERATE 90 DAY
# ============================================================

forecast_90 = generate_future_forecast(
    model,
    history,
    series,
    90
)

forecast_90.to_csv(
    FORECAST_90_FILE,
    index=False
)

print()
print(
    "90-day forecast saved:"
)

print(
    FORECAST_90_FILE
)


# ============================================================
# COMBINE 30/60/90
# ============================================================

print_section(
    "COMBINING FUTURE FORECASTS"
)

all_forecasts = pd.concat(
    [
        forecast_30.assign(
            horizon_days=30
        ),

        forecast_60.assign(
            horizon_days=60
        ),

        forecast_90.assign(
            horizon_days=90
        )
    ],
    ignore_index=True
)


all_forecasts.to_csv(
    ALL_FORECAST_FILE,
    index=False
)

print(
    "Combined forecast saved:"
)

print(
    ALL_FORECAST_FILE
)


# ============================================================
# STORE SUMMARY
# ============================================================

print_section(
    "CREATING STORE-LEVEL FORECAST SUMMARY"
)

store_summary = (
    forecast_90
    .groupby(
        "store_id",
        as_index=False
    )
    .agg(
        forecast_90d_units=(
            "forecast_units",
            "sum"
        ),

        avg_daily_forecast=(
            "forecast_units",
            "mean"
        ),

        max_daily_forecast=(
            "forecast_units",
            "max"
        )
    )
    .sort_values(
        "forecast_90d_units",
        ascending=False
    )
)


store_summary.to_csv(
    STORE_SUMMARY_FILE,
    index=False
)

print(
    "Store summary saved:"
)

print(
    STORE_SUMMARY_FILE
)


# ============================================================
# SKU SUMMARY
# ============================================================

print_section(
    "CREATING SKU-LEVEL FORECAST SUMMARY"
)

sku_summary = (
    forecast_90
    .groupby(
        "sku_id",
        as_index=False
    )
    .agg(
        forecast_90d_units=(
            "forecast_units",
            "sum"
        ),

        avg_daily_forecast=(
            "forecast_units",
            "mean"
        ),

        max_daily_forecast=(
            "forecast_units",
            "max"
        )
    )
    .sort_values(
        "forecast_90d_units",
        ascending=False
    )
)


sku_summary.to_csv(
    SKU_SUMMARY_FILE,
    index=False
)

print(
    "SKU summary saved:"
)

print(
    SKU_SUMMARY_FILE
)


# ============================================================
# STORE-SKU SUMMARY
# ============================================================

print_section(
    "CREATING STORE-SKU DEMAND RANKING"
)

item_summary = (
    forecast_90
    .groupby(
        [
            "store_id",
            "sku_id"
        ],
        as_index=False
    )
    .agg(
        forecast_90d_units=(
            "forecast_units",
            "sum"
        ),

        avg_daily_forecast=(
            "forecast_units",
            "mean"
        ),

        max_daily_forecast=(
            "forecast_units",
            "max"
        )
    )
)


# ============================================================
# HIGH DEMAND
# ============================================================

high_demand = (
    item_summary
    .sort_values(
        "forecast_90d_units",
        ascending=False
    )
    .head(100)
    .copy()
)

high_demand[
    "demand_priority"
] = "HIGH"


high_demand.to_csv(
    HIGH_DEMAND_FILE,
    index=False
)


# ============================================================
# LOW DEMAND
# ============================================================

low_demand = (
    item_summary
    .sort_values(
        "forecast_90d_units",
        ascending=True
    )
    .head(100)
    .copy()
)

low_demand[
    "demand_priority"
] = "LOW"


low_demand.to_csv(
    LOW_DEMAND_FILE,
    index=False
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print_section(
    "SAVING MODEL INFORMATION"
)

model_info = pd.DataFrame(
    [
        {
            "production_model":
                "LightGBM",

            "training_days":
                TRAINING_DAYS,

            "training_start":
                training_start.strftime(
                    "%Y-%m-%d"
                ),

            "training_end":
                latest_date.strftime(
                    "%Y-%m-%d"
                ),

            "forecast_horizons":
                "30,60,90",

            "n_estimators":
                N_ESTIMATORS,

            "learning_rate":
                LEARNING_RATE,

            "num_leaves":
                NUM_LEAVES,

            "training_rows":
                len(featured),

            "store_count":
                series[
                    "store_id"
                ].nunique(),

            "sku_count":
                series[
                    "sku_id"
                ].nunique(),

            "store_sku_count":
                len(series)
        }
    ]
)


model_info.to_csv(
    MODEL_INFO_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print_section(
    "FUTURE FORECAST SUMMARY"
)

print(
    "Historical end:",
    latest_date.date()
)

print(
    "30-day forecast:",
    forecast_30["date"].min().date(),
    "to",
    forecast_30["date"].max().date()
)

print(
    "60-day forecast:",
    forecast_60["date"].min().date(),
    "to",
    forecast_60["date"].max().date()
)

print(
    "90-day forecast:",
    forecast_90["date"].min().date(),
    "to",
    forecast_90["date"].max().date()
)

print()

print(
    "30-day total forecast:",
    round(
        forecast_30[
            "forecast_units"
        ].sum(),
        2
    )
)

print(
    "60-day total forecast:",
    round(
        forecast_60[
            "forecast_units"
        ].sum(),
        2
    )
)

print(
    "90-day total forecast:",
    round(
        forecast_90[
            "forecast_units"
        ].sum(),
        2
    )
)


# ============================================================
# COMPLETED
# ============================================================

print()
print("=" * 70)
print("PHASE 6.0 COMPLETED")
print("=" * 70)

print()
print(
    "Final production model: LightGBM"
)

print()
print(
    "Outputs saved to:"
)

print(
    OUTPUT_PATH
)

print()
print(
    "30-day forecast:"
)

print(
    FORECAST_30_FILE
)

print()
print(
    "60-day forecast:"
)

print(
    FORECAST_60_FILE
)

print()
print(
    "90-day forecast:"
)

print(
    FORECAST_90_FILE
)

print()
print(
    "Combined forecast:"
)

print(
    ALL_FORECAST_FILE
)

print()
print(
    "Store summary:"
)

print(
    STORE_SUMMARY_FILE
)

print()
print(
    "SKU summary:"
)

print(
    SKU_SUMMARY_FILE
)

print()
print(
    "High-demand items:"
)

print(
    HIGH_DEMAND_FILE
)

print()
print(
    "Low-demand items:"
)

print(
    LOW_DEMAND_FILE
)

print()
print("=" * 70)
print(
    "NEXT PHASE: FORECAST EVALUATION & BUSINESS INSIGHTS"
)
print("=" * 70)