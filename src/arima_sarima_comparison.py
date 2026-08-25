# ============================================================
# PROJECT FORESIGHT
# Phase 5.8 - ARIMA / SARIMA Comparison
# CORRECTED REPRESENTATIVE SAMPLING VERSION
# ============================================================

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
    / "arima_sarima"
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

TRAIN_START = pd.Timestamp(
    "2024-10-01"
)

VALIDATION_DAYS = 30

RANDOM_STATE = 42

# Target number of Store-SKU combinations
SAMPLE_COMBINATIONS = 30

# Minimum positive demand days
MIN_POSITIVE_DAYS = 5


# ============================================================
# METRICS
# ============================================================

def calculate_mae(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return np.mean(
        np.abs(
            actual - predicted
        )
    )


def calculate_rmse(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return np.sqrt(
        np.mean(
            (
                actual - predicted
            ) ** 2
        )
    )


def calculate_bias(actual, predicted):

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


def calculate_wape(actual, predicted):

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


def calculate_active_wape(
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

    return calculate_wape(
        actual[mask],
        predicted[mask]
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "PROJECT FORESIGHT - ARIMA / SARIMA COMPARISON"
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
    usecols=[
        "date",
        "store_id",
        "sku_id",
        "units_sold"
    ],
    low_memory=False
)

print(
    "Loaded shape:",
    demand.shape
)


# ============================================================
# DATA TYPES
# ============================================================

demand["date"] = pd.to_datetime(
    demand["date"],
    errors="coerce"
)

demand["units_sold"] = pd.to_numeric(
    demand["units_sold"],
    errors="coerce"
)

demand["units_sold"] = (
    demand["units_sold"]
    .fillna(0)
    .clip(lower=0)
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)

print(
    "BASIC DATA VALIDATION"
)

print("=" * 70)

print(
    "Invalid dates:",
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
# FILTER PERIOD
# ============================================================

print("\n" + "=" * 70)

print(
    "FILTERING ARIMA/SARIMA PERIOD"
)

print("=" * 70)

arima_data = demand[
    (
        demand["date"]
        >= TRAIN_START
    )
    &
    (
        demand["date"]
        <= VALIDATION_END
    )
].copy()

print(
    "Filtered rows:",
    len(arima_data)
)


# ============================================================
# STORE-SKU SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "ANALYZING STORE-SKU DEMAND ACTIVITY"
)

print("=" * 70)

summary = (
    arima_data
    .groupby(
        [
            "store_id",
            "sku_id"
        ]
    )
    .agg(
        total_demand=(
            "units_sold",
            "sum"
        ),

        positive_days=(
            "units_sold",
            lambda x:
            (x > 0).sum()
        ),

        observations=(
            "units_sold",
            "size"
        )
    )
    .reset_index()
)

summary["positive_rate"] = (
    summary["positive_days"]
    /
    summary["observations"]
)


print(
    "Total Store-SKU combinations:",
    len(summary)
)


print(
    "Combinations with at least",
    MIN_POSITIVE_DAYS,
    "positive days:",
    (
        summary["positive_days"]
        >= MIN_POSITIVE_DAYS
    ).sum()
)


# ============================================================
# REPRESENTATIVE SAMPLING
# ============================================================

print("\n" + "=" * 70)

print(
    "SELECTING REPRESENTATIVE STORE-SKU SERIES"
)

print("=" * 70)


eligible = summary[
    summary["positive_days"]
    >= MIN_POSITIVE_DAYS
].copy()


if len(eligible) == 0:

    raise RuntimeError(
        "No eligible Store-SKU combinations found."
    )


# ------------------------------------------------------------
# Divide into activity groups
# ------------------------------------------------------------

eligible["activity_group"] = pd.qcut(
    eligible["positive_rate"],
    q=3,
    labels=[
        "Low_Activity",
        "Medium_Activity",
        "High_Activity"
    ],
    duplicates="drop"
)


# ------------------------------------------------------------
# Calculate number per group
# ------------------------------------------------------------

target_n = min(
    SAMPLE_COMBINATIONS,
    len(eligible)
)


group_counts = (
    eligible["activity_group"]
    .value_counts()
)


groups = list(
    group_counts.index
)


# Start with equal allocation
base_per_group = (
    target_n
    //
    len(groups)
)


remaining = (
    target_n
    -
    base_per_group
    * len(groups)
)


selected_parts = []


# ------------------------------------------------------------
# Select from each activity group
# ------------------------------------------------------------

for i, group in enumerate(groups):

    group_data = eligible[
        eligible["activity_group"]
        == group
    ].copy()

    n_select = base_per_group

    if i < remaining:

        n_select += 1

    n_select = min(
        n_select,
        len(group_data)
    )

    if n_select > 0:

        # Sort by demand activity but
        # include randomization for diversity.

        group_data = (
            group_data
            .sort_values(
                [
                    "positive_rate",
                    "total_demand"
                ],
                ascending=[
                    True,
                    False
                ]
            )
        )

        if len(group_data) > n_select:

            rng = np.random.default_rng(
                RANDOM_STATE + i
            )

            # Select evenly across the group
            positions = np.linspace(
                0,
                len(group_data) - 1,
                n_select,
                dtype=int
            )

            selected_group = (
                group_data
                .iloc[positions]
            )

        else:

            selected_group = group_data

        selected_parts.append(
            selected_group
        )


# ------------------------------------------------------------
# Combine selections
# ------------------------------------------------------------

sample_series = pd.concat(
    selected_parts,
    ignore_index=True
)


# ------------------------------------------------------------
# Safety fill
# ------------------------------------------------------------

if len(sample_series) < target_n:

    remaining_candidates = (
        eligible[
            ~eligible.set_index(
                [
                    "store_id",
                    "sku_id"
                ]
            ).index.isin(
                sample_series.set_index(
                    [
                        "store_id",
                        "sku_id"
                    ]
                ).index
            )
        ]
        .sort_values(
            [
                "positive_rate",
                "total_demand"
            ],
            ascending=[
                True,
                False
            ]
        )
    )

    needed = (
        target_n
        -
        len(sample_series)
    )

    sample_series = pd.concat(
        [
            sample_series,
            remaining_candidates.head(
                needed
            )
        ],
        ignore_index=True
    )


# ------------------------------------------------------------
# Final maximum
# ------------------------------------------------------------

sample_series = (
    sample_series
    .drop_duplicates(
        subset=[
            "store_id",
            "sku_id"
        ]
    )
    .head(
        target_n
    )
    .copy()
)


print(
    "\nSelected Store-SKU combinations:",
    len(sample_series)
)


print(
    "\nActivity distribution:"
)


print(
    sample_series[
        "activity_group"
    ].value_counts()
    .to_string()
)


print(
    "\nSelected combinations:"
)


print(
    sample_series[
        [
            "store_id",
            "sku_id",
            "positive_rate",
            "total_demand",
            "positive_days",
            "activity_group"
        ]
    ]
    .sort_values(
        "positive_rate"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE SELECTED SERIES
# ============================================================

selected_path = (
    OUTPUT_PATH
    /
    "selected_arima_sarima_series.csv"
)

sample_series.to_csv(
    selected_path,
    index=False
)


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# MODEL LOOP
# ============================================================

for counter, row in enumerate(
    sample_series.itertuples(
        index=False
    ),
    start=1
):

    store_id = row.store_id

    sku_id = row.sku_id

    activity_group = (
        row.activity_group
    )


    print("\n" + "-" * 70)

    print(
        f"[{counter}/{len(sample_series)}]"
        f" Store={store_id}"
        f" SKU={sku_id}"
        f" Activity={activity_group}"
    )

    print("-" * 70)


    # ========================================================
    # EXTRACT SERIES
    # ========================================================

    series = arima_data[
        (
            arima_data["store_id"]
            == store_id
        )
        &
        (
            arima_data["sku_id"]
            == sku_id
        )
    ][
        [
            "date",
            "units_sold"
        ]
    ].copy()


    # ========================================================
    # DAILY SERIES
    # ========================================================

    series = (
        series
        .groupby(
            "date"
        )["units_sold"]
        .sum()
        .sort_index()
    )


    # Reindex to complete daily frequency

    full_index = pd.date_range(
        start=TRAIN_START,
        end=VALIDATION_END,
        freq="D"
    )

    series = (
        series
        .reindex(
            full_index,
            fill_value=0
        )
    )


    # ========================================================
    # TRAIN / VALIDATION
    # ========================================================

    train_series = series[
        series.index
        <= TRAIN_END
    ]


    validation_series = series[
        (
            series.index
            >= VALIDATION_START
        )
        &
        (
            series.index
            <= VALIDATION_END
        )
    ]


    if len(validation_series) != 30:

        print(
            "Skipping: validation does not contain 30 days."
        )

        continue


    if len(train_series) < 180:

        print(
            "Skipping: insufficient history."
        )

        continue


    actual = (
        validation_series
        .to_numpy(
            dtype=float
        )
    )


    # ========================================================
    # ARIMA
    # ========================================================

    print(
        "Training ARIMA..."
    )


    try:

        arima_model = ARIMA(
            train_series,
            order=(
                1,
                1,
                1
            )
        )

        arima_fit = (
            arima_model
            .fit()
        )

        arima_forecast = (
            arima_fit
            .forecast(
                steps=30
            )
        )

        arima_forecast = np.maximum(
            np.asarray(
                arima_forecast,
                dtype=float
            ),
            0
        )


        results.append(
            {
                "store_id":
                    store_id,

                "sku_id":
                    sku_id,

                "activity_group":
                    activity_group,

                "model":
                    "ARIMA",

                "MAE":
                    calculate_mae(
                        actual,
                        arima_forecast
                    ),

                "RMSE":
                    calculate_rmse(
                        actual,
                        arima_forecast
                    ),

                "Bias":
                    calculate_bias(
                        actual,
                        arima_forecast
                    ),

                "WAPE_pct":
                    calculate_wape(
                        actual,
                        arima_forecast
                    ),

                "Active_WAPE_pct":
                    calculate_active_wape(
                        actual,
                        arima_forecast
                    )
            }
        )


        print(
            "ARIMA completed."
        )


    except Exception as e:

        print(
            "ARIMA failed:",
            str(e)
        )


    # ========================================================
    # SARIMA
    # ========================================================

    print(
        "Training SARIMA..."
    )


    try:

        sarima_model = SARIMAX(

            train_series,

            order=(
                1,
                1,
                1
            ),

            seasonal_order=(
                1,
                0,
                1,
                7
            ),

            enforce_stationarity=False,

            enforce_invertibility=False
        )


        sarima_fit = (
            sarima_model
            .fit(
                disp=False
            )
        )


        sarima_forecast = (
            sarima_fit
            .forecast(
                steps=30
            )
        )


        sarima_forecast = np.maximum(
            np.asarray(
                sarima_forecast,
                dtype=float
            ),
            0
        )


        results.append(
            {
                "store_id":
                    store_id,

                "sku_id":
                    sku_id,

                "activity_group":
                    activity_group,

                "model":
                    "SARIMA",

                "MAE":
                    calculate_mae(
                        actual,
                        sarima_forecast
                    ),

                "RMSE":
                    calculate_rmse(
                        actual,
                        sarima_forecast
                    ),

                "Bias":
                    calculate_bias(
                        actual,
                        sarima_forecast
                    ),

                "WAPE_pct":
                    calculate_wape(
                        actual,
                        sarima_forecast
                    ),

                "Active_WAPE_pct":
                    calculate_active_wape(
                        actual,
                        sarima_forecast
                    )
            }
        )


        print(
            "SARIMA completed."
        )


    except Exception as e:

        print(
            "SARIMA failed:",
            str(e)
        )


# ============================================================
# CHECK RESULTS
# ============================================================

if len(results) == 0:

    raise RuntimeError(
        "No ARIMA/SARIMA models completed successfully."
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE ITEM RESULTS
# ============================================================

item_results_path = (
    OUTPUT_PATH
    /
    "arima_sarima_item_results.csv"
)


results_df.to_csv(
    item_results_path,
    index=False
)


# ============================================================
# AGGREGATED PERFORMANCE
# ============================================================

print("\n" + "=" * 70)

print(
    "AGGREGATED ARIMA / SARIMA PERFORMANCE"
)

print("=" * 70)


model_summary = (
    results_df
    .groupby(
        "model"
    )
    .agg(

        store_sku_count=(
            "store_id",
            "nunique"
        ),

        MAE=(
            "MAE",
            "mean"
        ),

        RMSE=(
            "RMSE",
            "mean"
        ),

        Bias=(
            "Bias",
            "mean"
        ),

        WAPE_pct=(
            "WAPE_pct",
            "mean"
        ),

        Active_WAPE_pct=(
            "Active_WAPE_pct",
            "mean"
        )
    )
    .reset_index()
)


model_summary = (
    model_summary
    .sort_values(
        [
            "MAE",
            "RMSE"
        ]
    )
    .reset_index(
        drop=True
    )
)


model_summary["rank"] = (
    np.arange(
        1,
        len(model_summary) + 1
    )
)


print(
    model_summary.to_string(
        index=False
    )
)


# ============================================================
# PERFORMANCE BY ACTIVITY GROUP
# ============================================================

print("\n" + "=" * 70)

print(
    "PERFORMANCE BY DEMAND ACTIVITY"
)

print("=" * 70)


activity_summary = (
    results_df
    .groupby(
        [
            "activity_group",
            "model"
        ]
    )
    .agg(

        store_sku_count=(
            "store_id",
            "count"
        ),

        MAE=(
            "MAE",
            "mean"
        ),

        RMSE=(
            "RMSE",
            "mean"
        ),

        Bias=(
            "Bias",
            "mean"
        ),

        WAPE_pct=(
            "WAPE_pct",
            "mean"
        ),

        Active_WAPE_pct=(
            "Active_WAPE_pct",
            "mean"
        )
    )
    .reset_index()
)


print(
    activity_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = (
    OUTPUT_PATH
    /
    "arima_sarima_model_summary.csv"
)


model_summary.to_csv(
    summary_path,
    index=False
)


activity_summary_path = (
    OUTPUT_PATH
    /
    "arima_sarima_activity_summary.csv"
)


activity_summary.to_csv(
    activity_summary_path,
    index=False
)


# ============================================================
# BEST CLASSICAL MODEL
# ============================================================

best_classical_model = (
    model_summary
    .iloc[0]["model"]
)


print("\n" + "=" * 70)

print(
    "BEST CLASSICAL TIME-SERIES MODEL"
)

print("=" * 70)


print(
    "Best model:",
    best_classical_model
)

print(
    "MAE:",
    round(
        model_summary.iloc[0]["MAE"],
        6
    )
)

print(
    "RMSE:",
    round(
        model_summary.iloc[0]["RMSE"],
        6
    )
)

print(
    "WAPE:",
    round(
        model_summary.iloc[0]["WAPE_pct"],
        6
    ),
    "%"
)

print(
    "Active WAPE:",
    round(
        model_summary
        .iloc[0]["Active_WAPE_pct"],
        6
    ),
    "%"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)

print(
    "PHASE 5.8 CORRECTED COMPLETED"
)

print("=" * 70)


print(
    "\nSelected series saved to:"
)

print(
    selected_path
)


print(
    "\nItem-level results saved to:"
)

print(
    item_results_path
)


print(
    "\nModel summary saved to:"
)

print(
    summary_path
)


print(
    "\nActivity summary saved to:"
)

print(
    activity_summary_path
)


print("\n" + "=" * 70)

print(
    "NEXT PHASE: FINAL MODEL SELECTION"
)

print("=" * 70)