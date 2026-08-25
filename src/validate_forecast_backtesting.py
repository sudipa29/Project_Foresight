# ============================================================
# PROJECT FORESIGHT
# Phase 5.6 - Forecast Backtesting Validation
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(
    r"E:\Zidio_Development_Internship\Project_Foresight"
)

PROCESSED_PATH = (
    BASE_PATH /
    "data" /
    "processed"
)

FORECASTING_PATH = (
    PROCESSED_PATH /
    "forecasting"
)

INPUT_PATH = (
    FORECASTING_PATH /
    "forecast_demand_daily.csv"
)

OUTPUT_PATH = (
    FORECASTING_PATH /
    "validation"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

# Forecast horizon
HORIZON = 30

# Rolling validation windows
# Each cutoff generates a 30-day forecast
BACKTEST_CUTOFFS = [
    "2025-07-03",
    "2025-08-01",
    "2025-09-01",
    "2025-10-01",
]

# Minimum historical observations required
MIN_HISTORY = 60

# Rolling mean windows
WINDOWS = [7, 14, 30]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_mae(actual, forecast):
    """
    Mean Absolute Error
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    return np.mean(
        np.abs(actual - forecast)
    )


def calculate_rmse(actual, forecast):
    """
    Root Mean Squared Error
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    return np.sqrt(
        np.mean(
            (actual - forecast) ** 2
        )
    )


def calculate_bias(actual, forecast):
    """
    Forecast bias.

    Positive value:
        Under-forecasting

    Negative value:
        Over-forecasting
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    return np.mean(
        forecast - actual
    )


def calculate_wape(actual, forecast):
    """
    Weighted Absolute Percentage Error
    """

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:
        return 0.0

    return (
        np.sum(
            np.abs(actual - forecast)
        )
        /
        denominator
        *
        100
    )


def calculate_mape_active(actual, forecast):
    """
    MAPE calculated only where actual demand > 0.
    """

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    mask = actual > 0

    if mask.sum() == 0:
        return 0.0

    return np.mean(
        np.abs(
            (actual[mask] - forecast[mask])
            /
            actual[mask]
        )
    ) * 100


def evaluate_predictions(
    actual,
    forecast
):

    return {
        "MAE": calculate_mae(
            actual,
            forecast
        ),
        "RMSE": calculate_rmse(
            actual,
            forecast
        ),
        "Bias": calculate_bias(
            actual,
            forecast
        ),
        "WAPE_pct": calculate_wape(
            actual,
            forecast
        ),
        "MAPE_active_pct": calculate_mape_active(
            actual,
            forecast
        )
    }


def safe_round(value):
    return round(
        float(value),
        6
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print(
    "PROJECT FORESIGHT - FORECAST BACKTESTING VALIDATION"
)
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading forecast demand dataset...")

demand = pd.read_csv(
    INPUT_PATH,
    low_memory=False
)

print(
    "Dataset shape:",
    demand.shape
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATA VALIDATION")
print("=" * 70)

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
        f"Missing required columns: {missing_columns}"
    )

print(
    "\nRequired columns:",
    required_columns
)

print(
    "Required columns found successfully."
)


# ============================================================
# DATE CONVERSION
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
    "\nInvalid dates:",
    invalid_dates
)

if invalid_dates > 0:

    demand = demand[
        demand["date"].notna()
    ].copy()


# ============================================================
# NUMERIC CONVERSION
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
    "Missing demand values:",
    missing_demand
)

demand["units_sold"] = (
    demand["units_sold"]
    .fillna(0)
)


# ============================================================
# NEGATIVE DEMAND CHECK
# ============================================================

negative_rows = (
    demand["units_sold"] < 0
).sum()

print(
    "Negative demand rows:",
    negative_rows
)

if negative_rows > 0:

    print(
        "WARNING: negative demand detected."
    )

    demand["units_sold"] = (
        demand["units_sold"]
        .clip(lower=0)
    )


# ============================================================
# SORT DATA
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
# DUPLICATE CHECK
# ============================================================

duplicate_count = (
    demand
    .duplicated(
        subset=[
            "date",
            "store_id",
            "sku_id"
        ]
    )
    .sum()
)

print(
    "\nDuplicate Store-SKU-Date rows:",
    duplicate_count
)

if duplicate_count > 0:

    print(
        "WARNING: duplicate observations detected."
    )

    demand = (
        demand
        .groupby(
            [
                "date",
                "store_id",
                "sku_id"
            ],
            as_index=False
        )[
            "units_sold"
        ]
        .sum()
    )

    demand = demand.sort_values(
        [
            "store_id",
            "sku_id",
            "date"
        ]
    )


# ============================================================
# GLOBAL DATA PROFILE
# ============================================================

print("\n" + "=" * 70)
print("DATA PROFILE")
print("=" * 70)

print(
    "\nDate range:",
    demand["date"].min(),
    "to",
    demand["date"].max()
)

print(
    "Stores:",
    demand["store_id"].nunique()
)

print(
    "SKUs:",
    demand["sku_id"].nunique()
)

store_sku_count = (
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
    "Store-SKU combinations:",
    store_sku_count
)

print(
    "Rows:",
    len(demand)
)

zero_rows = (
    demand["units_sold"] == 0
).sum()

positive_rows = (
    demand["units_sold"] > 0
).sum()

print(
    "Zero-demand observations:",
    zero_rows
)

print(
    "Positive-demand observations:",
    positive_rows
)

print(
    "Zero-demand percentage:",
    round(
        zero_rows /
        len(demand) *
        100,
        2
    )
)


# ============================================================
# BACKTEST FUNCTION
# ============================================================

def run_backtest(
    data,
    cutoff_date,
    horizon=30
):

    cutoff_date = pd.Timestamp(
        cutoff_date
    )

    validation_start = (
        cutoff_date +
        pd.Timedelta(days=1)
    )

    validation_end = (
        cutoff_date +
        pd.Timedelta(days=horizon)
    )

    print("\n" + "-" * 70)
    print(
        "BACKTEST WINDOW"
    )
    print("-" * 70)

    print(
        "Training end:",
        cutoff_date.date()
    )

    print(
        "Validation start:",
        validation_start.date()
    )

    print(
        "Validation end:",
        validation_end.date()
    )

    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    train = data[
        data["date"] <= cutoff_date
    ].copy()

    # --------------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------------

    validation = data[
        (
            data["date"] >= validation_start
        )
        &
        (
            data["date"] <= validation_end
        )
    ].copy()

    print(
        "Training rows:",
        len(train)
    )

    print(
        "Validation rows:",
        len(validation)
    )

    # --------------------------------------------------------
    # IMPORTANT LEAKAGE CHECK
    # --------------------------------------------------------

    if len(train) > 0 and len(validation) > 0:

        if train["date"].max() >= validation["date"].min():

            raise ValueError(
                "DATA LEAKAGE DETECTED: "
                "training data overlaps validation data."
            )

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    train_history = (
        train
        .groupby(
            [
                "store_id",
                "sku_id"
            ]
        )["units_sold"]
        .agg(
            [
                "count",
                "sum"
            ]
        )
        .reset_index()
    )

    train_history = train_history.rename(
        columns={
            "count": "history_days",
            "sum": "training_units"
        }
    )

    # --------------------------------------------------------
    # FORECAST BASELINES
    # --------------------------------------------------------

    train_grouped = (
        train
        .sort_values(
            [
                "store_id",
                "sku_id",
                "date"
            ]
        )
    )

    # Last 7-day average
    forecast_7 = (
        train_grouped
        .groupby(
            [
                "store_id",
                "sku_id"
            ]
        )["units_sold"]
        .apply(
            lambda x:
            x.tail(7).mean()
        )
        .reset_index(
            name="forecast_7d_daily"
        )
    )

    # Last 14-day average
    forecast_14 = (
        train_grouped
        .groupby(
            [
                "store_id",
                "sku_id"
            ]
        )["units_sold"]
        .apply(
            lambda x:
            x.tail(14).mean()
        )
        .reset_index(
            name="forecast_14d_daily"
        )
    )

    # Last 30-day average
    forecast_30 = (
        train_grouped
        .groupby(
            [
                "store_id",
                "sku_id"
            ]
        )["units_sold"]
        .apply(
            lambda x:
            x.tail(30).mean()
        )
        .reset_index(
            name="forecast_30d_daily"
        )
    )

    # --------------------------------------------------------
    # MERGE FORECASTS
    # --------------------------------------------------------

    forecast = train_history.merge(
        forecast_7,
        on=[
            "store_id",
            "sku_id"
        ],
        how="left"
    )

    forecast = forecast.merge(
        forecast_14,
        on=[
            "store_id",
            "sku_id"
        ],
        how="left"
    )

    forecast = forecast.merge(
        forecast_30,
        on=[
            "store_id",
            "sku_id"
        ],
        how="left"
    )

    # --------------------------------------------------------
    # VALIDATION ACTUALS
    # --------------------------------------------------------

    actual = (
        validation
        .groupby(
            [
                "store_id",
                "sku_id"
            ]
        )["units_sold"]
        .sum()
        .reset_index(
            name="actual_units_30d"
        )
    )

    # --------------------------------------------------------
    # COMPLETE STORE-SKU EVALUATION
    # --------------------------------------------------------

    evaluation = forecast.merge(
        actual,
        on=[
            "store_id",
            "sku_id"
        ],
        how="left"
    )

    evaluation[
        "actual_units_30d"
    ] = evaluation[
        "actual_units_30d"
    ].fillna(0)

    # Convert daily forecast to 30-day forecast
    for window in WINDOWS:

        col = (
            f"forecast_{window}d_daily"
        )

        forecast_col = (
            f"forecast_{window}d_total"
        )

        evaluation[
            forecast_col
        ] = (
            evaluation[col] *
            horizon
        )

    # --------------------------------------------------------
    # FILTER SUFFICIENT HISTORY
    # --------------------------------------------------------

    evaluation["sufficient_history"] = (
        evaluation["history_days"]
        >= MIN_HISTORY
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results = []

    model_columns = {
        "Rolling_7D": "forecast_7d_total",
        "Rolling_14D": "forecast_14d_total",
        "Rolling_30D": "forecast_30d_total"
    }

    for model_name, forecast_col in model_columns.items():

        metrics_all = evaluate_predictions(
            evaluation["actual_units_30d"],
            evaluation[forecast_col]
        )

        active_mask = (
            evaluation["actual_units_30d"]
            > 0
        )

        if active_mask.sum() > 0:

            active_metrics = evaluate_predictions(
                evaluation.loc[
                    active_mask,
                    "actual_units_30d"
                ],
                evaluation.loc[
                    active_mask,
                    forecast_col
                ]
            )

        else:

            active_metrics = {
                "MAE": 0,
                "RMSE": 0,
                "Bias": 0,
                "WAPE_pct": 0,
                "MAPE_active_pct": 0
            }

        sufficient = evaluation[
            evaluation["sufficient_history"]
        ].copy()

        if len(sufficient) > 0:

            sufficient_metrics = evaluate_predictions(
                sufficient[
                    "actual_units_30d"
                ],
                sufficient[
                    forecast_col
                ]
            )

        else:

            sufficient_metrics = {
                "MAE": np.nan,
                "RMSE": np.nan,
                "Bias": np.nan,
                "WAPE_pct": np.nan,
                "MAPE_active_pct": np.nan
            }

        results.append(
            {
                "cutoff_date":
                    cutoff_date.date(),

                "validation_start":
                    validation_start.date(),

                "validation_end":
                    validation_end.date(),

                "model":
                    model_name,

                "store_sku_count":
                    len(evaluation),

                "sufficient_history_count":
                    len(sufficient),

                "MAE":
                    metrics_all["MAE"],

                "RMSE":
                    metrics_all["RMSE"],

                "Bias":
                    metrics_all["Bias"],

                "WAPE_pct":
                    metrics_all["WAPE_pct"],

                "MAPE_active_pct":
                    metrics_all[
                        "MAPE_active_pct"
                    ],

                "Active_MAE":
                    active_metrics["MAE"],

                "Active_RMSE":
                    active_metrics["RMSE"],

                "Active_Bias":
                    active_metrics["Bias"],

                "Active_WAPE_pct":
                    active_metrics[
                        "WAPE_pct"
                    ],

                "SufficientHistory_MAE":
                    sufficient_metrics["MAE"],

                "SufficientHistory_RMSE":
                    sufficient_metrics["RMSE"],

                "SufficientHistory_Bias":
                    sufficient_metrics["Bias"],

                "SufficientHistory_WAPE_pct":
                    sufficient_metrics[
                        "WAPE_pct"
                    ]
            }
        )

    # --------------------------------------------------------
    # ITEM LEVEL RESULTS
    # --------------------------------------------------------

    evaluation["cutoff_date"] = (
        cutoff_date.date()
    )

    evaluation["validation_start"] = (
        validation_start.date()
    )

    evaluation["validation_end"] = (
        validation_end.date()
    )

    return (
        pd.DataFrame(results),
        evaluation
    )


# ============================================================
# RUN ROLLING BACKTESTS
# ============================================================

all_results = []

all_item_results = []

for cutoff in BACKTEST_CUTOFFS:

    metrics_df, item_df = run_backtest(
        demand,
        cutoff,
        HORIZON
    )

    all_results.append(
        metrics_df
    )

    all_item_results.append(
        item_df
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

backtest_results = pd.concat(
    all_results,
    ignore_index=True
)

backtest_item_results = pd.concat(
    all_item_results,
    ignore_index=True
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ROLLING BACKTEST RESULTS")
print("=" * 70)

display_columns = [
    "cutoff_date",
    "model",
    "store_sku_count",
    "sufficient_history_count",
    "MAE",
    "RMSE",
    "Bias",
    "WAPE_pct",
    "Active_MAE",
    "Active_WAPE_pct"
]

print(
    backtest_results[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# AGGREGATED MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("AGGREGATED MODEL PERFORMANCE")
print("=" * 70)

aggregated = (
    backtest_results
    .groupby(
        "model",
        as_index=False
    )
    .agg(
        {
            "MAE": "mean",
            "RMSE": "mean",
            "Bias": "mean",
            "WAPE_pct": "mean",
            "MAPE_active_pct": "mean",
            "Active_MAE": "mean",
            "Active_RMSE": "mean",
            "Active_Bias": "mean",
            "Active_WAPE_pct": "mean",
            "SufficientHistory_MAE": "mean",
            "SufficientHistory_RMSE": "mean",
            "SufficientHistory_Bias": "mean",
            "SufficientHistory_WAPE_pct": "mean"
        }
    )
)

print(
    aggregated.to_string(
        index=False
    )
)


# ============================================================
# MODEL RANKING
# ============================================================

ranking = aggregated.copy()

ranking["MAE_rank"] = (
    ranking["MAE"]
    .rank(
        method="min",
        ascending=True
    )
)

ranking["RMSE_rank"] = (
    ranking["RMSE"]
    .rank(
        method="min",
        ascending=True
    )
)

ranking["WAPE_rank"] = (
    ranking["WAPE_pct"]
    .rank(
        method="min",
        ascending=True
    )
)

ranking["overall_rank_score"] = (
    ranking["MAE_rank"]
    +
    ranking["RMSE_rank"]
    +
    ranking["WAPE_rank"]
)

ranking = ranking.sort_values(
    "overall_rank_score"
)


# ============================================================
# WIN COUNT
# ============================================================

print("\n" + "=" * 70)
print("ROLLING WINDOW MODEL WIN COUNT")
print("=" * 70)

win_records = []

for cutoff in BACKTEST_CUTOFFS:

    current = backtest_results[
        backtest_results["cutoff_date"]
        == pd.Timestamp(cutoff).date()
    ].copy()

    if current.empty:
        continue

    best_mae = current.loc[
        current["MAE"].idxmin(),
        "model"
    ]

    best_rmse = current.loc[
        current["RMSE"].idxmin(),
        "model"
    ]

    best_wape = current.loc[
        current["WAPE_pct"].idxmin(),
        "model"
    ]

    win_records.append(
        {
            "cutoff_date":
                pd.Timestamp(cutoff).date(),

            "best_MAE_model":
                best_mae,

            "best_RMSE_model":
                best_rmse,

            "best_WAPE_model":
                best_wape
        }
    )

win_summary = pd.DataFrame(
    win_records
)

print(
    win_summary.to_string(
        index=False
    )
)


# ============================================================
# DATA LEAKAGE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATA LEAKAGE VALIDATION")
print("=" * 70)

leakage_errors = []

for cutoff in BACKTEST_CUTOFFS:

    cutoff_date = pd.Timestamp(
        cutoff
    )

    train_dates = demand.loc[
        demand["date"] <= cutoff_date,
        "date"
    ]

    validation_dates = demand.loc[
        (
            demand["date"]
            > cutoff_date
        )
        &
        (
            demand["date"]
            <=
            cutoff_date
            +
            pd.Timedelta(
                days=HORIZON
            )
        ),
        "date"
    ]

    if len(train_dates) > 0 and len(validation_dates) > 0:

        if train_dates.max() >= validation_dates.min():

            leakage_errors.append(
                cutoff
            )

if len(leakage_errors) == 0:

    print(
        "PASS: No temporal leakage detected."
    )

else:

    print(
        "FAIL: Leakage detected for:",
        leakage_errors
    )


# ============================================================
# VALIDATION OF FORECAST HORIZON
# ============================================================

print("\n" + "=" * 70)
print("FORECAST HORIZON VALIDATION")
print("=" * 70)

horizon_checks = []

for cutoff in BACKTEST_CUTOFFS:

    cutoff_date = pd.Timestamp(
        cutoff
    )

    validation_dates = demand.loc[
        (
            demand["date"]
            > cutoff_date
        )
        &
        (
            demand["date"]
            <=
            cutoff_date
            +
            pd.Timedelta(
                days=HORIZON
            )
        ),
        "date"
    ].drop_duplicates()

    horizon_checks.append(
        {
            "cutoff_date":
                cutoff_date.date(),

            "expected_days":
                HORIZON,

            "actual_validation_days":
                len(validation_dates),

            "valid":
                len(validation_dates)
                == HORIZON
        }
    )

horizon_validation = pd.DataFrame(
    horizon_checks
)

print(
    horizon_validation.to_string(
        index=False
    )
)


# ============================================================
# FORECAST SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("FORECAST SANITY CHECK")
print("=" * 70)

forecast_columns = [
    "forecast_7d_total",
    "forecast_14d_total",
    "forecast_30d_total"
]

for column in forecast_columns:

    negative_count = (
        backtest_item_results[column]
        < 0
    ).sum()

    missing_count = (
        backtest_item_results[column]
        .isna()
        .sum()
    )

    print(
        f"\n{column}"
    )

    print(
        "Missing:",
        missing_count
    )

    print(
        "Negative:",
        negative_count
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_output = (
    OUTPUT_PATH /
    "rolling_backtest_results.csv"
)

summary_output = (
    OUTPUT_PATH /
    "rolling_backtest_summary.csv"
)

item_output = (
    OUTPUT_PATH /
    "rolling_backtest_item_results.csv"
)

ranking_output = (
    OUTPUT_PATH /
    "rolling_backtest_model_ranking.csv"
)

win_output = (
    OUTPUT_PATH /
    "rolling_backtest_model_wins.csv"
)

horizon_output = (
    OUTPUT_PATH /
    "backtest_horizon_validation.csv"
)


backtest_results.to_csv(
    results_output,
    index=False
)

aggregated.to_csv(
    summary_output,
    index=False
)

backtest_item_results.to_csv(
    item_output,
    index=False
)

ranking.to_csv(
    ranking_output,
    index=False
)

win_summary.to_csv(
    win_output,
    index=False
)

horizon_validation.to_csv(
    horizon_output,
    index=False
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL BACKTESTING VALIDATION")
print("=" * 70)

print(
    "\nBacktest windows:",
    len(BACKTEST_CUTOFFS)
)

print(
    "Forecast horizon:",
    HORIZON,
    "days"
)

print(
    "Models tested:",
    len(aggregated)
)

print(
    "Leakage errors:",
    len(leakage_errors)
)

print(
    "Horizon validation failures:",
    (
        ~horizon_validation["valid"]
    ).sum()
)


# ============================================================
# FINAL MODEL
# ============================================================

if (
    len(leakage_errors) == 0
    and
    (
        ~horizon_validation["valid"]
    ).sum() == 0
):

    print(
        "\nBACKTESTING VALIDATION PASSED"
    )

else:

    print(
        "\nBACKTESTING VALIDATION REQUIRES REVIEW"
    )


print("\n" + "=" * 70)
print(
    "PHASE 5.6 COMPLETED"
)
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    results_output
)

print(
    summary_output
)

print(
    item_output
)

print(
    ranking_output
)

print(
    win_output
)

print(
    horizon_output
)

print("\n" + "=" * 70)
print(
    "NEXT PHASE: ADVANCED ML FORECASTING"
)
print("=" * 70)