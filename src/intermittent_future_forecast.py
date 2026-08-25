# ============================================================
# PROJECT FORESIGHT
# Phase 6.1 - Intermittent Future Demand Forecast
#
# Two-Stage LightGBM:
#   Stage 1 -> Demand occurrence probability
#   Stage 2 -> Positive demand quantity
#
# Final:
#   Expected Demand =
#   P(Demand > 0) * E(Demand | Demand > 0)
#
# Forecast Horizons:
#   30 Days
#   60 Days
#   90 Days
# ============================================================

import os

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

from lightgbm import LGBMClassifier, LGBMRegressor

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
    FORECASTING_PATH / "future" / "intermittent_corrected"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


INPUT_FILE = (
    FORECASTING_PATH /
    "forecast_demand_daily.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

FORECAST_30_FILE = (
    OUTPUT_PATH /
    "intermittent_future_30_day_forecast.csv"
)

FORECAST_60_FILE = (
    OUTPUT_PATH /
    "intermittent_future_60_day_forecast.csv"
)

FORECAST_90_FILE = (
    OUTPUT_PATH /
    "intermittent_future_90_day_forecast.csv"
)

ALL_FORECAST_FILE = (
    OUTPUT_PATH /
    "intermittent_future_forecast_30_60_90.csv"
)

MODEL_INFO_FILE = (
    OUTPUT_PATH /
    "intermittent_model_info.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_DAYS = 365

FORECAST_HORIZONS = [
    30,
    60,
    90
]

RANDOM_STATE = 42

N_ESTIMATORS = 300

LEARNING_RATE = 0.05

NUM_LEAVES = 31

N_JOBS = 4


# ============================================================
# HELPER
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(df):

    df = (
        df
        .sort_values(
            [
                "store_id",
                "sku_id",
                "date"
            ]
        )
        .copy()
    )

    grouped = df.groupby(
        [
            "store_id",
            "sku_id"
        ],
        sort=False
    )

    # --------------------------------------------------------
    # LAGS
    # --------------------------------------------------------

    print("Creating lag features...")

    df["lag_1"] = (
        grouped["units_sold"]
        .shift(1)
    )

    df["lag_7"] = (
        grouped["units_sold"]
        .shift(7)
    )

    df["lag_14"] = (
        grouped["units_sold"]
        .shift(14)
    )

    df["lag_30"] = (
        grouped["units_sold"]
        .shift(30)
    )


    # --------------------------------------------------------
    # SHIFTED SERIES
    # --------------------------------------------------------

    shifted = (
        grouped["units_sold"]
        .shift(1)
    )


    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------

    print("Creating rolling features...")

    shifted_grouped = shifted.groupby(
        [
            df["store_id"],
            df["sku_id"]
        ]
    )

    df["rolling_mean_7"] = (
        shifted_grouped
        .transform(
            lambda x:
            x.rolling(
                7,
                min_periods=1
            ).mean()
        )
    )

    df["rolling_mean_14"] = (
        shifted_grouped
        .transform(
            lambda x:
            x.rolling(
                14,
                min_periods=1
            ).mean()
        )
    )

    df["rolling_mean_30"] = (
        shifted_grouped
        .transform(
            lambda x:
            x.rolling(
                30,
                min_periods=1
            ).mean()
        )
    )


    # --------------------------------------------------------
    # ACTIVE DAYS
    # --------------------------------------------------------

    print("Creating intermittency features...")

    df["active_days_7"] = (
        shifted_groupby := shifted.groupby(
            [
                df["store_id"],
                df["sku_id"]
            ]
        )
    ).transform(
        lambda x:
        x.rolling(
            7,
            min_periods=1
        ).apply(
            lambda z:
            np.sum(z > 0),
            raw=True
        )
    )

    df["active_days_30"] = (
        shifted.groupby(
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
            ).apply(
                lambda z:
                np.sum(z > 0),
                raw=True
            )
        )
    )


    # --------------------------------------------------------
    # DAYS SINCE DEMAND
    # --------------------------------------------------------

    print("Creating days-since-demand...")

    occurrence = (
        shifted > 0
    )

    temp_dates = (
        df["date"]
        .where(occurrence)
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


    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    df["trend_7_vs_30"] = (
        df["rolling_mean_7"] /
        (
            df["rolling_mean_30"] +
            1e-6
        )
    )


    # --------------------------------------------------------
    # CALENDAR
    # --------------------------------------------------------

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
        df["day_of_week"] >= 5
    ).astype("int8")


    # --------------------------------------------------------
    # CLEAN FEATURES
    # --------------------------------------------------------

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
        "trend_7_vs_30"
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT")
print("PHASE 6.1 - INTERMITTENT FUTURE DEMAND FORECAST")
print("=" * 70)


print_section(
    "CHECKING INPUT"
)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

print(
    "Input:",
    INPUT_FILE
)


# ============================================================
# LOAD
# ============================================================

print_section(
    "LOADING HISTORICAL DEMAND"
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
    "Original shape:",
    demand.shape
)


# ============================================================
# DATE RANGE
# ============================================================

latest_date_string = (
    demand["date"].max()
)

latest_date = pd.Timestamp(
    latest_date_string
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

print(
    "Training start:",
    training_start_string
)

print(
    "Training end:",
    latest_date_string
)


# ============================================================
# FILTER
# ============================================================

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


# ============================================================
# DATE CONVERSION
# ============================================================

demand["date"] = pd.to_datetime(
    demand["date"],
    format="%Y-%m-%d"
)


# ============================================================
# CLEAN
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

demand = (
    demand
    .sort_values(
        [
            "store_id",
            "sku_id",
            "date"
        ]
    )
    .reset_index(drop=True)
)


print(
    "Filtered shape:",
    demand.shape
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
    "Store-SKU:",
    demand[
        [
            "store_id",
            "sku_id"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


# ============================================================
# CREATE FEATURES
# ============================================================

print_section(
    "CREATING FEATURES"
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
    "is_weekend"
]


# ============================================================
# REMOVE EARLY LAG ROWS
# ============================================================

featured = featured.loc[
    featured["lag_30"].notna()
].copy()


# ============================================================
# TARGETS
# ============================================================

featured["demand_occurs"] = (
    featured["units_sold"] > 0
).astype("int8")


# ============================================================
# TRAINING MATRICES
# ============================================================

X = featured[
    FEATURES
].copy()


y_occurrence = (
    featured[
        "demand_occurs"
    ]
    .astype("int8")
)


# ============================================================
# POSITIVE DEMAND DATA
# ============================================================

positive_mask = (
    featured["units_sold"] > 0
)

X_positive = (
    featured.loc[
        positive_mask,
        FEATURES
    ]
    .copy()
)

y_positive = (
    featured.loc[
        positive_mask,
        "units_sold"
    ]
    .astype("float32")
    .copy()
)


print_section(
    "TRAINING DATA SUMMARY"
)

print(
    "Total training rows:",
    len(X)
)

print(
    "Positive demand rows:",
    len(X_positive)
)

print(
    "Zero demand rows:",
    len(X) - len(X_positive)
)

print(
    "Positive demand percentage:",
    round(
        len(X_positive) /
        len(X) *
        100,
        2
    )
)


# ============================================================
# OCCURRENCE MODEL
# ============================================================

print_section(
    "TRAINING DEMAND OCCURRENCE MODEL"
)

occurrence_model = LGBMClassifier(

    objective="binary",

    n_estimators=N_ESTIMATORS,

    learning_rate=LEARNING_RATE,

    num_leaves=NUM_LEAVES,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.1,

    reg_lambda=0.1,

    random_state=RANDOM_STATE,

    n_jobs=N_JOBS,

    verbosity=-1
)


occurrence_model.fit(
    X,
    y_occurrence
)


print(
    "Occurrence model completed."
)


# ============================================================
# POSITIVE DEMAND MODEL
# ============================================================

print_section(
    "TRAINING POSITIVE DEMAND MODEL"
)

quantity_model = LGBMRegressor(

    objective="regression",

    n_estimators=N_ESTIMATORS,

    learning_rate=LEARNING_RATE,

    num_leaves=NUM_LEAVES,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.1,

    reg_lambda=0.1,

    random_state=RANDOM_STATE,

    n_jobs=N_JOBS,

    verbosity=-1
)


quantity_model.fit(
    X_positive,
    y_positive
)


print(
    "Positive quantity model completed."
)


del X
del X_positive
del y_occurrence
del y_positive

gc.collect()


# ============================================================
# HISTORICAL STARTING POINT
# ============================================================

history = featured[
    [
        "store_id",
        "sku_id",
        "date",
        "units_sold"
    ]
].copy()


history = (
    history
    .sort_values(
        [
            "store_id",
            "sku_id",
            "date"
        ]
    )
)


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
    .reset_index(drop=True)
)


# ============================================================
# FUTURE FORECAST FUNCTION
# ============================================================

def generate_future_forecast(
    occurrence_model,
    quantity_model,
    history_df,
    series_df,
    horizon
):

    print()
    print(
        "-" * 70
    )

    print(
        f"GENERATING CORRECTED {horizon}-DAY FORECAST"
    )

    print(
        "-" * 70
    )


    latest_date = (
        history_df["date"].max()
    )


    future_dates = pd.date_range(

        start=
        latest_date +
        pd.Timedelta(days=1),

        periods=horizon,

        freq="D"
    )


    working = (
        history_df
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


    for step, forecast_date in enumerate(
        future_dates,
        start=1
    ):

        if step == 1 or step % 10 == 0:

            print(
                f"Forecast day {step}/{horizon}: "
                f"{forecast_date.date()}"
            )


        rows = []


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


                positive_positions = (
                    np.where(values > 0)[0]
                )


                if len(
                    positive_positions
                ) == 0:

                    days_since = 999.0

                else:

                    days_since = (
                        len(values)
                        - 1
                        - positive_positions[-1]
                    )


            trend = (
                mean_7 /
                (
                    mean_30 +
                    1e-6
                )
            )


            rows.append(
                {

                    "store_id":
                        store_id,

                    "sku_id":
                        sku_id,

                    "lag_1":
                        lag_1,

                    "lag_7":
                        lag_7,

                    "lag_14":
                        lag_14,

                    "lag_30":
                        lag_30,

                    "rolling_mean_7":
                        mean_7,

                    "rolling_mean_14":
                        mean_14,

                    "rolling_mean_30":
                        mean_30,

                    "active_days_7":
                        active_7,

                    "active_days_30":
                        active_30,

                    "days_since_demand":
                        days_since,

                    "trend_7_vs_30":
                        trend,

                    "day_of_week":
                        forecast_date.dayofweek,

                    "month":
                        forecast_date.month,

                    "quarter":
                        forecast_date.quarter,

                    "day_of_month":
                        forecast_date.day,

                    "week_of_year":
                        forecast_date.isocalendar().week,

                    "is_weekend":
                        int(
                            forecast_date.dayofweek >= 5
                        )
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
        # OCCURRENCE PROBABILITY
        # ----------------------------------------------------

        occurrence_probability = (
            occurrence_model
            .predict_proba(
                feature_frame
            )[:, 1]
        )


        # ----------------------------------------------------
        # POSITIVE DEMAND QUANTITY
        # ----------------------------------------------------

        positive_quantity = (
            quantity_model
            .predict(
                feature_frame
            )
        )


        positive_quantity = (
            np.maximum(
                positive_quantity,
                0
            )
        )


        # ----------------------------------------------------
        # EXPECTED DEMAND
        # ----------------------------------------------------

        predictions = (
            occurrence_probability *
            positive_quantity
        )


        predictions = (
            np.maximum(
                predictions,
                0
            )
        )


        # ----------------------------------------------------
        # SAVE FORECAST
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

                "occurrence_probability":
                    occurrence_probability,

                "positive_demand_quantity":
                    positive_quantity,

                "forecast_units":
                    predictions
            }
        )


        forecasts.append(
            daily_prediction
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT treat every fractional expected demand
        # as an actual demand event.
        #
        # For recursive history we use occurrence-aware
        # simulated demand.
        # ----------------------------------------------------

        simulated_demand = (
            np.where(
                occurrence_probability >= 0.50,
                positive_quantity,
                0.0
            )
        )


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
                    simulated_demand
            }
        )


        working = pd.concat(
            [
                working,
                new_history
            ],
            ignore_index=True
        )


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


    return pd.concat(
        forecasts,
        ignore_index=True
    )


# ============================================================
# GENERATE 30 DAYS
# ============================================================

forecast_30 = generate_future_forecast(
    occurrence_model,
    quantity_model,
    history,
    series,
    30
)

forecast_30.to_csv(
    FORECAST_30_FILE,
    index=False
)


print(
    "30-day corrected forecast saved:"
)

print(
    FORECAST_30_FILE
)


# ============================================================
# GENERATE 60 DAYS
# ============================================================

forecast_60 = generate_future_forecast(
    occurrence_model,
    quantity_model,
    history,
    series,
    60
)

forecast_60.to_csv(
    FORECAST_60_FILE,
    index=False
)


print(
    "60-day corrected forecast saved:"
)

print(
    FORECAST_60_FILE
)


# ============================================================
# GENERATE 90 DAYS
# ============================================================

forecast_90 = generate_future_forecast(
    occurrence_model,
    quantity_model,
    history,
    series,
    90
)

forecast_90.to_csv(
    FORECAST_90_FILE,
    index=False
)


print(
    "90-day corrected forecast saved:"
)

print(
    FORECAST_90_FILE
)


# ============================================================
# COMBINE
# ============================================================

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


# ============================================================
# MODEL INFORMATION
# ============================================================

model_info = pd.DataFrame(
    [
        {

            "production_model":
                "Two-Stage LightGBM",

            "occurrence_model":
                "LightGBMClassifier",

            "quantity_model":
                "LightGBMRegressor",

            "training_days":
                TRAINING_DAYS,

            "training_start":
                training_start_string,

            "training_end":
                latest_date_string,

            "forecast_horizons":
                "30,60,90",

            "n_estimators":
                N_ESTIMATORS,

            "learning_rate":
                LEARNING_RATE,

            "num_leaves":
                NUM_LEAVES,

            "store_count":
                series["store_id"].nunique(),

            "sku_count":
                series["sku_id"].nunique(),

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
# FINAL SUMMARY
# ============================================================

print_section(
    "CORRECTED FORECAST SUMMARY"
)

print(
    "30-day total:",
    round(
        forecast_30[
            "forecast_units"
        ].sum(),
        2
    )
)

print(
    "60-day total:",
    round(
        forecast_60[
            "forecast_units"
        ].sum(),
        2
    )
)

print(
    "90-day total:",
    round(
        forecast_90[
            "forecast_units"
        ].sum(),
        2
    )
)

print()

print(
    "Average occurrence probability:",
    round(
        all_forecasts[
            "occurrence_probability"
        ].mean(),
        4
    )
)

print(
    "Average positive-demand quantity:",
    round(
        all_forecasts[
            "positive_demand_quantity"
        ].mean(),
        4
    )
)

print()

print(
    "Corrected forecasts saved to:"
)

print(
    OUTPUT_PATH
)

print()

print(
    "=" * 70
)

print(
    "PHASE 6.1 COMPLETED"
)

print(
    "=" * 70
)