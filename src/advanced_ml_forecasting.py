# ============================================================
# PROJECT FORESIGHT
# Phase 5.7 - Advanced ML Forecasting
# ============================================================

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# OPTIONAL ML LIBRARIES
# ============================================================

try:

    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True

except ImportError:

    XGBOOST_AVAILABLE = False


try:

    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True

except ImportError:

    LIGHTGBM_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(
    r"E:\Zidio_Development_Internship\Project_Foresight"
)

PROCESSED_PATH = (
    BASE_PATH
    / "data"
    / "processed"
)

FORECASTING_PATH = (
    PROCESSED_PATH
    / "forecasting"
)

INPUT_PATH = (
    FORECASTING_PATH
    / "forecast_demand_daily.csv"
)

OUTPUT_PATH = (
    FORECASTING_PATH
    / "advanced_ml"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_END = pd.Timestamp(
    "2025-10-01"
)

VALIDATION_START = pd.Timestamp(
    "2025-10-02"
)

VALIDATION_END = pd.Timestamp(
    "2025-10-31"
)

TRAIN_LOOKBACK_DAYS = 365

RANDOM_STATE = 42


# ============================================================
# METRICS
# ============================================================

def mae(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return mean_absolute_error(
        actual,
        predicted
    )


def rmse(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )


def bias(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return np.mean(
        predicted - actual
    )


def wape(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:

        return 0.0

    return (
        np.sum(
            np.abs(
                actual - predicted
            )
        )
        /
        denominator
        *
        100
    )


def active_wape(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    mask = actual > 0

    if mask.sum() == 0:

        return 0.0

    return wape(
        actual[mask],
        predicted[mask]
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "PROJECT FORESIGHT - ADVANCED ML FORECASTING"
)

print("=" * 70)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_PATH}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\nLoading forecast demand dataset..."
)

demand = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

print(
    "Original shape:",
    demand.shape
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "store_id",
    "sku_id",
    "units_sold"
]

missing_columns = [
    col
    for col in required_columns
    if col not in demand.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# DATE
# ============================================================

demand["date"] = pd.to_datetime(
    demand["date"],
    errors="coerce"
)

invalid_dates = (
    demand["date"]
    .isna()
    .sum()
)

print(
    "Invalid dates:",
    invalid_dates
)


if invalid_dates > 0:

    demand = demand[
        demand["date"].notna()
    ]


# ============================================================
# DEMAND
# ============================================================

demand["units_sold"] = pd.to_numeric(
    demand["units_sold"],
    errors="coerce"
)

missing_demand = (
    demand["units_sold"]
    .isna()
    .sum()
)

print(
    "Missing demand:",
    missing_demand
)


demand["units_sold"] = (
    demand["units_sold"]
    .fillna(0)
    .clip(lower=0)
)


# ============================================================
# SORT
# ============================================================

demand = demand.sort_values(
    [
        "store_id",
        "sku_id",
        "date"
    ]
).reset_index(
    drop=True
)


# ============================================================
# TRAINING WINDOW
# ============================================================

training_start = (
    TRAIN_END
    -
    pd.Timedelta(
        days=TRAIN_LOOKBACK_DAYS
    )
)


print("\n" + "=" * 70)

print(
    "TRAINING WINDOW"
)

print("=" * 70)

print(
    "Training start:",
    training_start.date()
)

print(
    "Training end:",
    TRAIN_END.date()
)

print(
    "Validation start:",
    VALIDATION_START.date()
)

print(
    "Validation end:",
    VALIDATION_END.date()
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)

print(
    "CREATING ML FEATURES"
)

print("=" * 70)


group_cols = [
    "store_id",
    "sku_id"
]


# ============================================================
# LAGS
# ============================================================

print(
    "\nCreating lag features..."
)

grouped = (
    demand
    .groupby(
        group_cols,
        sort=False
    )["units_sold"]
)


demand["lag_1"] = (
    grouped.shift(1)
)

demand["lag_7"] = (
    grouped.shift(7)
)

demand["lag_14"] = (
    grouped.shift(14)
)

demand["lag_30"] = (
    grouped.shift(30)
)

demand["lag_60"] = (
    grouped.shift(60)
)


# ============================================================
# ROLLING FEATURES
# ============================================================

print(
    "Creating rolling features..."
)


demand["rolling_mean_7"] = (
    demand
    .groupby(
        group_cols,
        sort=False
    )["units_sold"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            7,
            min_periods=7
        )
        .mean()
    )
)


demand["rolling_mean_14"] = (
    demand
    .groupby(
        group_cols,
        sort=False
    )["units_sold"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            14,
            min_periods=14
        )
        .mean()
    )
)


demand["rolling_mean_30"] = (
    demand
    .groupby(
        group_cols,
        sort=False
    )["units_sold"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            30,
            min_periods=30
        )
        .mean()
    )
)


demand["rolling_std_7"] = (
    demand
    .groupby(
        group_cols,
        sort=False
    )["units_sold"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            7,
            min_periods=7
        )
        .std()
    )
)


demand["rolling_std_30"] = (
    demand
    .groupby(
        group_cols,
        sort=False
    )["units_sold"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            30,
            min_periods=30
        )
        .std()
    )
)


# ============================================================
# INTERMITTENCY
# ============================================================

print(
    "Creating intermittency features..."
)


demand["positive_demand"] = (
    demand["units_sold"] > 0
).astype(
    np.int8
)


demand["positive_days_30"] = (
    demand
    .groupby(
        group_cols,
        sort=False
    )["positive_demand"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            30,
            min_periods=30
        )
        .sum()
    )
)


demand["demand_occurrence_rate_30"] = (
    demand["positive_days_30"]
    /
    30.0
)


# ============================================================
# DAYS SINCE DEMAND
# ============================================================

print(
    "Creating days-since-demand feature..."
)


demand["last_positive_date"] = (
    demand["date"]
    .where(
        demand["units_sold"] > 0
    )
)


demand["last_positive_date"] = (
    demand
    .groupby(
        group_cols,
        sort=False
    )["last_positive_date"]
    .ffill()
)


demand["days_since_demand"] = (
    demand["date"]
    -
    demand["last_positive_date"]
).dt.days


demand["days_since_demand"] = (
    demand["days_since_demand"]
    .fillna(999)
)


# ============================================================
# CALENDAR FEATURES
# ============================================================

print(
    "Creating calendar features..."
)


demand["day_of_week"] = (
    demand["date"].dt.dayofweek
)

demand["day_of_month"] = (
    demand["date"].dt.day
)

demand["month"] = (
    demand["date"].dt.month
)

demand["week_of_year"] = (
    demand["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

demand["is_weekend"] = (
    demand["day_of_week"] >= 5
).astype(
    np.int8
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    "store_id",
    "sku_id",

    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "lag_60",

    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",

    "rolling_std_7",
    "rolling_std_30",

    "positive_days_30",
    "demand_occurrence_rate_30",

    "days_since_demand",

    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend"
]


# ============================================================
# MODELING DATA
# ============================================================

print("\n" + "=" * 70)

print(
    "PREPARING MODELING DATA"
)

print("=" * 70)


model_data = demand[
    (
        demand["date"]
        >= training_start
    )
    &
    (
        demand["date"]
        <= VALIDATION_END
    )
]


model_data = model_data.dropna(
    subset=[
        "lag_60",
        "rolling_mean_30",
        "rolling_std_30",
        "positive_days_30"
    ]
)


model_data = (
    model_data
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


model_data = model_data.dropna(
    subset=FEATURES + ["units_sold"]
)


print(
    "Modeling dataset shape:",
    model_data.shape
)


# ============================================================
# TRAIN / VALIDATION
# ============================================================

train = model_data[
    model_data["date"]
    <= TRAIN_END
]


validation = model_data[
    (
        model_data["date"]
        >= VALIDATION_START
    )
    &
    (
        model_data["date"]
        <= VALIDATION_END
    )
]


print(
    "\nTraining rows:",
    len(train)
)

print(
    "Validation rows:",
    len(validation)
)


print(
    "Training Store-SKU:",
    train[
        group_cols
    ]
    .drop_duplicates()
    .shape[0]
)


print(
    "Validation Store-SKU:",
    validation[
        group_cols
    ]
    .drop_duplicates()
    .shape[0]
)


# ============================================================
# X / Y
# ============================================================

X_train = train[
    FEATURES
].copy()

y_train = train[
    "units_sold"
].copy()


X_valid = validation[
    FEATURES
].copy()

y_valid = validation[
    "units_sold"
].copy()


# ============================================================
# ENCODING
# ============================================================

print(
    "\nEncoding Store and SKU identifiers..."
)


store_values = (
    demand["store_id"]
    .drop_duplicates()
    .sort_values()
)


sku_values = (
    demand["sku_id"]
    .drop_duplicates()
    .sort_values()
)


store_map = {
    value: i
    for i, value
    in enumerate(store_values)
}


sku_map = {
    value: i
    for i, value
    in enumerate(sku_values)
}


X_train["store_id"] = (
    X_train["store_id"]
    .map(store_map)
)


X_valid["store_id"] = (
    X_valid["store_id"]
    .map(store_map)
)


X_train["sku_id"] = (
    X_train["sku_id"]
    .map(sku_map)
)


X_valid["sku_id"] = (
    X_valid["sku_id"]
    .map(sku_map)
)


# ============================================================
# FINAL VALIDATION
# ============================================================

if X_train.isna().any().any():

    raise ValueError(
        "Missing values found in X_train."
    )


if X_valid.isna().any().any():

    raise ValueError(
        "Missing values found in X_valid."
    )


# ============================================================
# MODELS
# ============================================================

models = {}


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)

print(
    "TRAINING RANDOM FOREST"
)

print("=" * 70)


rf_model = RandomForestRegressor(

    n_estimators=100,

    max_depth=18,

    min_samples_leaf=5,

    n_jobs=-1,

    random_state=RANDOM_STATE

)


rf_model.fit(
    X_train,
    y_train
)


models[
    "RandomForest"
] = rf_model


print(
    "Random Forest training completed."
)


# ============================================================
# XGBOOST
# ============================================================

if XGBOOST_AVAILABLE:

    print("\n" + "=" * 70)

    print(
        "TRAINING XGBOOST"
    )

    print("=" * 70)


    xgb_model = XGBRegressor(

        n_estimators=300,

        max_depth=8,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        eval_metric="mae",

        n_jobs=-1,

        random_state=RANDOM_STATE

    )


    xgb_model.fit(
        X_train,
        y_train,
        verbose=False
    )


    models[
        "XGBoost"
    ] = xgb_model


    print(
        "XGBoost training completed."
    )

else:

    print(
        "\nXGBoost is not installed."
    )


# ============================================================
# LIGHTGBM
# ============================================================

if LIGHTGBM_AVAILABLE:

    print("\n" + "=" * 70)

    print(
        "TRAINING LIGHTGBM"
    )

    print("=" * 70)


    lgb_model = LGBMRegressor(

        n_estimators=300,

        learning_rate=0.05,

        num_leaves=64,

        max_depth=-1,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="regression",

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1

    )


    lgb_model.fit(
        X_train,
        y_train
    )


    models[
        "LightGBM"
    ] = lgb_model


    print(
        "LightGBM training completed."
    )

else:

    print(
        "\nLightGBM is not installed."
    )


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 70)

print(
    "GENERATING VALIDATION PREDICTIONS"
)

print("=" * 70)


prediction_results = validation[
    [
        "date",
        "store_id",
        "sku_id",
        "units_sold"
    ]
].copy()


prediction_results = (
    prediction_results
    .rename(
        columns={
            "units_sold":
                "actual_units"
        }
    )
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

for model_name, model in models.items():

    print(
        f"\nPredicting with {model_name}..."
    )


    predictions = model.predict(
        X_valid
    )


    predictions = np.maximum(
        predictions,
        0
    )


    prediction_results[
        f"{model_name}_forecast"
    ] = predictions


# ============================================================
# SAVE CACHE
# ============================================================

prediction_cache_path = (
    OUTPUT_PATH
    /
    "ml_validation_predictions_cache.csv"
)


prediction_results.to_csv(
    prediction_cache_path,
    index=False
)


print(
    "\nML prediction cache saved to:"
)

print(
    prediction_cache_path
)


# ============================================================
# DAILY MODEL METRICS
# ============================================================

metrics = []


actual_daily = (
    y_valid
    .to_numpy()
)


for model_name in models.keys():

    predictions = (
        prediction_results[
            f"{model_name}_forecast"
        ]
        .to_numpy()
    )


    metrics.append(
        {
            "model":
                model_name,

            "validation_rows":
                len(validation),

            "MAE":
                mae(
                    actual_daily,
                    predictions
                ),

            "RMSE":
                rmse(
                    actual_daily,
                    predictions
                ),

            "Bias":
                bias(
                    actual_daily,
                    predictions
                ),

            "WAPE_pct":
                wape(
                    actual_daily,
                    predictions
                ),

            "Active_WAPE_pct":
                active_wape(
                    actual_daily,
                    predictions
                ),

            "Mean_actual":
                actual_daily.mean(),

            "Mean_forecast":
                predictions.mean(),

            "Min_forecast":
                predictions.min(),

            "Max_forecast":
                predictions.max()
        }
    )


# ============================================================
# ROLLING 30D BENCHMARK
# ============================================================

print("\n" + "=" * 70)

print(
    "CALCULATING ROLLING 30D BENCHMARK"
)

print("=" * 70)


# ------------------------------------------------------------
# IMPORTANT MEMORY FIX
#
# DO NOT create:
#
# train_before_validation = demand[
#     demand["date"] <= TRAIN_END
# ].copy()
#
# That attempts to copy millions of rows and all feature
# columns.
#
# Instead, only take the final 30 days and only the four
# columns required for the benchmark.
# ------------------------------------------------------------

benchmark_start = (
    TRAIN_END
    -
    pd.Timedelta(
        days=29
    )
)


print(
    "Benchmark period:",
    benchmark_start.date(),
    "to",
    TRAIN_END.date()
)


benchmark_mask = (
    demand["date"]
    >= benchmark_start
) & (
    demand["date"]
    <= TRAIN_END
)


benchmark_data = demand.loc[
    benchmark_mask,
    [
        "store_id",
        "sku_id",
        "units_sold"
    ]
]


print(
    "Benchmark rows:",
    len(benchmark_data)
)


benchmark = (
    benchmark_data
    .groupby(
        [
            "store_id",
            "sku_id"
        ],
        as_index=False
    )["units_sold"]
    .mean()
    .rename(
        columns={
            "units_sold":
                "Rolling30D_forecast"
        }
    )
)


# ============================================================
# MERGE BENCHMARK
# ============================================================

prediction_results = (
    prediction_results
    .merge(
        benchmark,
        on=[
            "store_id",
            "sku_id"
        ],
        how="left"
    )
)


prediction_results[
    "Rolling30D_forecast"
] = (
    prediction_results[
        "Rolling30D_forecast"
    ]
    .fillna(0)
)


benchmark_predictions = (
    prediction_results[
        "Rolling30D_forecast"
    ]
    .to_numpy()
)


# ============================================================
# BENCHMARK METRICS
# ============================================================

metrics.append(
    {
        "model":
            "Rolling30D",

        "validation_rows":
            len(validation),

        "MAE":
            mae(
                actual_daily,
                benchmark_predictions
            ),

        "RMSE":
            rmse(
                actual_daily,
                benchmark_predictions
            ),

        "Bias":
            bias(
                actual_daily,
                benchmark_predictions
            ),

        "WAPE_pct":
            wape(
                actual_daily,
                benchmark_predictions
            ),

        "Active_WAPE_pct":
            active_wape(
                actual_daily,
                benchmark_predictions
            ),

        "Mean_actual":
            actual_daily.mean(),

        "Mean_forecast":
            benchmark_predictions.mean(),

        "Min_forecast":
            benchmark_predictions.min(),

        "Max_forecast":
            benchmark_predictions.max()
    }
)


# ============================================================
# MODEL COMPARISON
# ============================================================

metrics_df = pd.DataFrame(
    metrics
)


metrics_df = (
    metrics_df
    .sort_values(
        [
            "MAE",
            "RMSE",
            "WAPE_pct"
        ]
    )
    .reset_index(
        drop=True
    )
)


metrics_df["rank"] = (
    np.arange(
        1,
        len(metrics_df) + 1
    )
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)

print(
    "ADVANCED ML MODEL COMPARISON"
)

print("=" * 70)


print(
    metrics_df.to_string(
        index=False
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)

print(
    "CALCULATING FEATURE IMPORTANCE"
)

print("=" * 70)


feature_importance_records = []


for model_name, model in models.items():

    if hasattr(
        model,
        "feature_importances_"
    ):

        importance = (
            model
            .feature_importances_
        )


        for feature, value in zip(
            FEATURES,
            importance
        ):

            feature_importance_records.append(
                {
                    "model":
                        model_name,

                    "feature":
                        feature,

                    "importance":
                        float(value)
                }
            )


feature_importance_df = (
    pd.DataFrame(
        feature_importance_records
    )
)


if not feature_importance_df.empty:

    feature_importance_df = (
        feature_importance_df
        .sort_values(
            [
                "model",
                "importance"
            ],
            ascending=[
                True,
                False
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PRELIMINARY MODEL SELECTION
# ============================================================

best_model_name = (
    metrics_df
    .iloc[0]["model"]
)


best_model_mae = (
    metrics_df
    .iloc[0]["MAE"]
)


best_model_rmse = (
    metrics_df
    .iloc[0]["RMSE"]
)


best_model_wape = (
    metrics_df
    .iloc[0]["WAPE_pct"]
)


print("\n" + "=" * 70)

print(
    "PRELIMINARY MODEL RESULT"
)

print("=" * 70)


print(
    "Best validation model:",
    best_model_name
)


print(
    "MAE:",
    round(
        best_model_mae,
        6
    )
)


print(
    "RMSE:",
    round(
        best_model_rmse,
        6
    )
)


print(
    "WAPE:",
    round(
        best_model_wape,
        6
    ),
    "%"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = (
    OUTPUT_PATH
    /
    "advanced_ml_model_metrics.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_path = (
    OUTPUT_PATH
    /
    "advanced_ml_validation_predictions.csv"
)


prediction_results.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = (
    OUTPUT_PATH
    /
    "advanced_ml_feature_importance.csv"
)


feature_importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model_summary_path = (
    OUTPUT_PATH
    /
    "advanced_ml_model_summary.csv"
)


metrics_df[
    [
        "model",
        "MAE",
        "RMSE",
        "Bias",
        "WAPE_pct",
        "Active_WAPE_pct",
        "rank"
    ]
].to_csv(
    model_summary_path,
    index=False
)


# ============================================================
# VALIDATION SUMMARY
# ============================================================

validation_summary_path = (
    OUTPUT_PATH
    /
    "advanced_ml_validation_summary.csv"
)


validation_summary = pd.DataFrame(
    {
        "training_start": [
            training_start.date()
        ],

        "training_end": [
            TRAIN_END.date()
        ],

        "validation_start": [
            VALIDATION_START.date()
        ],

        "validation_end": [
            VALIDATION_END.date()
        ],

        "training_rows": [
            len(train)
        ],

        "validation_rows": [
            len(validation)
        ],

        "store_count": [
            demand[
                "store_id"
            ].nunique()
        ],

        "sku_count": [
            demand[
                "sku_id"
            ].nunique()
        ],

        "store_sku_count": [
            validation[
                [
                    "store_id",
                    "sku_id"
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ]
    }
)


validation_summary.to_csv(
    validation_summary_path,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)

print(
    "PHASE 5.7 COMPLETED"
)

print("=" * 70)


print(
    "\nMetrics saved to:"
)

print(
    metrics_path
)


print(
    "\nValidation predictions saved to:"
)

print(
    predictions_path
)


print(
    "\nPrediction cache saved to:"
)

print(
    prediction_cache_path
)


print(
    "\nFeature importance saved to:"
)

print(
    importance_path
)


print(
    "\nModel summary saved to:"
)

print(
    model_summary_path
)


print(
    "\nValidation summary saved to:"
)

print(
    validation_summary_path
)


print("\n" + "=" * 70)

print(
    "NEXT PHASE: ARIMA / SARIMA COMPARISON"
)

print("=" * 70)