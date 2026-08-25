# ============================================================
# PROJECT FORESIGHT
# Phase 5.4 - Forecast Model Evaluation
# Memory-Efficient Backtesting Engine
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

PROCESSED_PATH = BASE_PATH / "data" / "processed"

FORECASTING_PATH = PROCESSED_PATH / "forecasting"

DAILY_DEMAND_PATH = (
    FORECASTING_PATH /
    "forecast_demand_daily.csv"
)

BASELINE_PATH = (
    FORECASTING_PATH /
    "demand_forecast_baseline.csv"
)

INTERMITTENT_PATH = (
    FORECASTING_PATH /
    "intermittent_demand_forecast.csv"
)

OUTPUT_PATH = (
    FORECASTING_PATH /
    "evaluation"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_DAYS = 30

# Your latest available date
REFERENCE_DATE = pd.Timestamp("2025-10-31")

# 30-day backtest
VALIDATION_END = REFERENCE_DATE

VALIDATION_START = (
    VALIDATION_END -
    pd.Timedelta(days=VALIDATION_DAYS - 1)
)

TRAINING_END = (
    VALIDATION_START -
    pd.Timedelta(days=1)
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT - FORECAST MODEL EVALUATION")
print("=" * 70)


# ============================================================
# LOAD FORECAST DATASETS
# ============================================================

print("\nLoading baseline forecast...")

baseline = pd.read_csv(
    BASELINE_PATH,
    low_memory=False
)

print(
    "Baseline forecast shape:",
    baseline.shape
)


print("\nLoading intermittent forecast...")

intermittent = pd.read_csv(
    INTERMITTENT_PATH,
    low_memory=False
)

print(
    "Intermittent forecast shape:",
    intermittent.shape
)


# ============================================================
# VALIDATE FORECAST COLUMNS
# ============================================================

required_baseline_columns = [
    "store_id",
    "sku_id",
    "forecast_weighted",
    "forecastability",
    "baseline_trend"
]

required_intermittent_columns = [
    "store_id",
    "sku_id",
    "intermittent_forecast",
    "forecastability",
    "forecast_confidence"
]


missing_baseline = [
    col
    for col in required_baseline_columns
    if col not in baseline.columns
]

missing_intermittent = [
    col
    for col in required_intermittent_columns
    if col not in intermittent.columns
]


if missing_baseline:

    raise ValueError(
        "Missing baseline columns: "
        + str(missing_baseline)
    )


if missing_intermittent:

    raise ValueError(
        "Missing intermittent columns: "
        + str(missing_intermittent)
    )


# ============================================================
# KEEP ONLY REQUIRED FORECAST COLUMNS
# ============================================================

baseline = baseline[
    [
        "store_id",
        "sku_id",
        "forecast_weighted",
        "forecastability",
        "baseline_trend"
    ]
].copy()


intermittent = intermittent[
    [
        "store_id",
        "sku_id",
        "intermittent_forecast",
        "forecastability",
        "forecast_confidence"
    ]
].copy()


# ============================================================
# BASIC FORECAST VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FORECAST VALIDATION")
print("=" * 70)


print(
    "\nBaseline missing forecasts:",
    baseline["forecast_weighted"].isna().sum()
)

print(
    "Baseline negative forecasts:",
    (
        baseline["forecast_weighted"] < 0
    ).sum()
)


print(
    "\nIntermittent missing forecasts:",
    intermittent[
        "intermittent_forecast"
    ].isna().sum()
)

print(
    "Intermittent negative forecasts:",
    (
        intermittent[
            "intermittent_forecast"
        ] < 0
    ).sum()
)


# ============================================================
# LOAD ONLY REQUIRED DAILY DEMAND COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("LOADING VALIDATION DEMAND DATA")
print("=" * 70)

print(
    "\nValidation period:",
    VALIDATION_START.date(),
    "to",
    VALIDATION_END.date()
)

print(
    "Training end:",
    TRAINING_END.date()
)

print(
    "\nReading daily demand data in chunks..."
)


# ============================================================
# MEMORY-EFFICIENT CHUNK PROCESSING
# ============================================================

required_daily_columns = [
    "store_id",
    "sku_id",
    "date",
    "units_sold"
]


validation_parts = []

chunk_number = 0

total_rows_read = 0

total_validation_rows = 0


for chunk in pd.read_csv(
    DAILY_DEMAND_PATH,
    usecols=required_daily_columns,
    parse_dates=["date"],
    chunksize=500_000
):

    chunk_number += 1

    total_rows_read += len(chunk)

    # --------------------------------------------------------
    # Keep ONLY validation period
    # --------------------------------------------------------

    validation_chunk = chunk[
        (
            chunk["date"] >=
            VALIDATION_START
        )
        &
        (
            chunk["date"] <=
            VALIDATION_END
        )
    ]

    if not validation_chunk.empty:

        # Aggregate immediately
        validation_agg = (
            validation_chunk
            .groupby(
                [
                    "store_id",
                    "sku_id"
                ],
                as_index=False
            )[
                "units_sold"
            ]
            .sum()
        )

        validation_parts.append(
            validation_agg
        )

        total_validation_rows += len(
            validation_chunk
        )

    del chunk
    del validation_chunk


print(
    "\nChunks processed:",
    chunk_number
)

print(
    "Total rows scanned:",
    total_rows_read
)

print(
    "Validation rows found:",
    total_validation_rows
)


# ============================================================
# COMBINE VALIDATION AGGREGATIONS
# ============================================================

print(
    "\nAggregating validation demand..."
)


if len(validation_parts) == 0:

    raise ValueError(
        "No validation demand found."
    )


validation = pd.concat(
    validation_parts,
    ignore_index=True
)


del validation_parts


validation = (
    validation
    .groupby(
        [
            "store_id",
            "sku_id"
        ],
        as_index=False
    )[
        "units_sold"
    ]
    .sum()
)


validation = validation.rename(
    columns={
        "units_sold":
        "actual_units_30d"
    }
)


# ============================================================
# CREATE COMPLETE STORE-SKU EVALUATION GRID
# ============================================================

print(
    "\nCreating evaluation dataset..."
)


evaluation = baseline.merge(
    intermittent,
    on=[
        "store_id",
        "sku_id"
    ],
    how="outer",
    suffixes=(
        "_baseline",
        "_intermittent"
    )
)


evaluation = evaluation.merge(
    validation,
    on=[
        "store_id",
        "sku_id"
    ],
    how="left"
)


evaluation["actual_units_30d"] = (
    evaluation[
        "actual_units_30d"
    ]
    .fillna(0)
)


# ============================================================
# FORECAST VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION DATASET")
print("=" * 70)


print(
    "\nEvaluation shape:",
    evaluation.shape
)


print(
    "Store-SKU combinations:",
    evaluation[
        [
            "store_id",
            "sku_id"
        ]
    ].drop_duplicates().shape[0]
)


print(
    "Actual demand missing:",
    evaluation[
        "actual_units_30d"
    ].isna().sum()
)


# ============================================================
# FORECAST COLUMNS
# ============================================================

evaluation["baseline_forecast"] = (
    evaluation[
        "forecast_weighted"
    ]
    * VALIDATION_DAYS
)


evaluation["intermittent_forecast_30d"] = (
    evaluation[
        "intermittent_forecast"
    ]
    * VALIDATION_DAYS
)


# ============================================================
# ERROR CALCULATIONS
# ============================================================

evaluation[
    "baseline_error"
] = (
    evaluation[
        "baseline_forecast"
    ]
    -
    evaluation[
        "actual_units_30d"
    ]
)


evaluation[
    "intermittent_error"
] = (
    evaluation[
        "intermittent_forecast_30d"
    ]
    -
    evaluation[
        "actual_units_30d"
    ]
)


evaluation[
    "baseline_abs_error"
] = (
    evaluation[
        "baseline_error"
    ].abs()
)


evaluation[
    "intermittent_abs_error"
] = (
    evaluation[
        "intermittent_error"
    ].abs()
)


evaluation[
    "baseline_squared_error"
] = (
    evaluation[
        "baseline_error"
    ] ** 2
)


evaluation[
    "intermittent_squared_error"
] = (
    evaluation[
        "intermittent_error"
    ] ** 2
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    forecast,
    model_name
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    forecast = np.asarray(
        forecast,
        dtype=float
    )

    error = (
        forecast -
        actual
    )

    absolute_error = np.abs(
        error
    )

    squared_error = (
        error ** 2
    )

    mae = (
        absolute_error.mean()
    )

    rmse = np.sqrt(
        squared_error.mean()
    )

    bias = error.mean()

    total_actual = actual.sum()

    if total_actual > 0:

        wape = (
            absolute_error.sum()
            /
            total_actual
        ) * 100

    else:

        wape = np.nan

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "WAPE_pct": wape
    }


# ============================================================
# GLOBAL MODEL METRICS
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL MODEL PERFORMANCE")
print("=" * 70)


baseline_metrics = calculate_metrics(
    evaluation[
        "actual_units_30d"
    ],
    evaluation[
        "baseline_forecast"
    ],
    "Baseline Weighted Forecast"
)


intermittent_metrics = calculate_metrics(
    evaluation[
        "actual_units_30d"
    ],
    evaluation[
        "intermittent_forecast_30d"
    ],
    "Intermittent Forecast"
)


metrics = pd.DataFrame(
    [
        baseline_metrics,
        intermittent_metrics
    ]
)


print(
    metrics.to_string(
        index=False
    )
)


# ============================================================
# MODEL WINNER
# ============================================================

baseline_mae = (
    baseline_metrics["MAE"]
)

intermittent_mae = (
    intermittent_metrics["MAE"]
)


if baseline_mae < intermittent_mae:

    overall_winner = (
        "Baseline Weighted Forecast"
    )

elif intermittent_mae < baseline_mae:

    overall_winner = (
        "Intermittent Forecast"
    )

else:

    overall_winner = "Tie"


print(
    "\nOverall model winner based on MAE:",
    overall_winner
)


# ============================================================
# ZERO DEMAND PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("ZERO-DEMAND PERFORMANCE")
print("=" * 70)


zero_actual = (
    evaluation[
        "actual_units_30d"
    ] == 0
)


zero_demand_count = (
    zero_actual.sum()
)


print(
    "\nZero-demand store-SKU combinations:",
    zero_demand_count
)


if zero_demand_count > 0:

    baseline_zero_mae = (
        evaluation.loc[
            zero_actual,
            "baseline_abs_error"
        ].mean()
    )

    intermittent_zero_mae = (
        evaluation.loc[
            zero_actual,
            "intermittent_abs_error"
        ].mean()
    )

else:

    baseline_zero_mae = np.nan

    intermittent_zero_mae = np.nan


print(
    "Baseline zero-demand MAE:",
    baseline_zero_mae
)

print(
    "Intermittent zero-demand MAE:",
    intermittent_zero_mae
)


# ============================================================
# ACTIVE DEMAND PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("ACTIVE-DEMAND PERFORMANCE")
print("=" * 70)


active_actual = (
    evaluation[
        "actual_units_30d"
    ] > 0
)


active_count = (
    active_actual.sum()
)


print(
    "\nActive store-SKU combinations:",
    active_count
)


if active_count > 0:

    baseline_active_metrics = (
        calculate_metrics(
            evaluation.loc[
                active_actual,
                "actual_units_30d"
            ],
            evaluation.loc[
                active_actual,
                "baseline_forecast"
            ],
            "Baseline - Active"
        )
    )

    intermittent_active_metrics = (
        calculate_metrics(
            evaluation.loc[
                active_actual,
                "actual_units_30d"
            ],
            evaluation.loc[
                active_actual,
                "intermittent_forecast_30d"
            ],
            "Intermittent - Active"
        )
    )

    active_metrics = pd.DataFrame(
        [
            baseline_active_metrics,
            intermittent_active_metrics
        ]
    )

    print(
        active_metrics.to_string(
            index=False
        )
    )

else:

    active_metrics = pd.DataFrame()


# ============================================================
# STORE-SKU MODEL WINNER
# ============================================================

print("\n" + "=" * 70)
print("STORE-SKU MODEL COMPARISON")
print("=" * 70)


evaluation["baseline_better"] = (
    evaluation[
        "baseline_abs_error"
    ]
    <
    evaluation[
        "intermittent_abs_error"
    ]
)


evaluation["intermittent_better"] = (
    evaluation[
        "intermittent_abs_error"
    ]
    <
    evaluation[
        "baseline_abs_error"
    ]
)


evaluation["model_tie"] = (
    evaluation[
        "baseline_abs_error"
    ]
    ==
    evaluation[
        "intermittent_abs_error"
    ]
)


baseline_wins = (
    evaluation[
        "baseline_better"
    ].sum()
)


intermittent_wins = (
    evaluation[
        "intermittent_better"
    ].sum()
)


ties = (
    evaluation[
        "model_tie"
    ].sum()
)


print(
    "\nBaseline wins:",
    baseline_wins
)

print(
    "Intermittent model wins:",
    intermittent_wins
)

print(
    "Ties:",
    ties
)


# ============================================================
# SELECT BEST MODEL PER STORE-SKU
# ============================================================

def select_best_model(row):

    baseline_error = (
        row["baseline_abs_error"]
    )

    intermittent_error = (
        row["intermittent_abs_error"]
    )

    if baseline_error < intermittent_error:

        return "Baseline"

    elif intermittent_error < baseline_error:

        return "Intermittent"

    return "Tie"


evaluation["best_model"] = (
    evaluation.apply(
        select_best_model,
        axis=1
    )
)


# ============================================================
# RECOMMENDED FORECAST
# ============================================================

evaluation["recommended_forecast"] = (
    np.where(
        evaluation["best_model"] ==
        "Intermittent",

        evaluation[
            "intermittent_forecast_30d"
        ],

        np.where(
            evaluation["best_model"] ==
            "Baseline",

            evaluation[
                "baseline_forecast"
            ],

            (
                evaluation[
                    "baseline_forecast"
                ]
                +
                evaluation[
                    "intermittent_forecast_30d"
                ]
            )
            / 2
        )
    )
)


# ============================================================
# FORECAST ERROR OF RECOMMENDED MODEL
# ============================================================

evaluation[
    "recommended_error"
] = (
    evaluation[
        "recommended_forecast"
    ]
    -
    evaluation[
        "actual_units_30d"
    ]
)


evaluation[
    "recommended_abs_error"
] = (
    evaluation[
        "recommended_error"
    ].abs()
)


# ============================================================
# RECOMMENDED MODEL METRICS
# ============================================================

recommended_metrics = calculate_metrics(
    evaluation[
        "actual_units_30d"
    ],
    evaluation[
        "recommended_forecast"
    ],
    "Recommended Store-SKU Model"
)


print("\n" + "=" * 70)
print("RECOMMENDED MODEL PERFORMANCE")
print("=" * 70)


print(
    pd.DataFrame(
        [recommended_metrics]
    ).to_string(
        index=False
    )
)


# ============================================================
# FORECASTABILITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FORECASTABILITY DISTRIBUTION")
print("=" * 70)


forecastability_distribution = (
    evaluation[
        "forecastability_baseline"
    ]
    .value_counts(
        dropna=False
    )
)


print(
    forecastability_distribution
)


# ============================================================
# MODEL SELECTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL DISTRIBUTION")
print("=" * 70)


best_model_distribution = (
    evaluation[
        "best_model"
    ]
    .value_counts()
)


print(
    best_model_distribution
)


# ============================================================
# TOP IMPROVEMENT ITEMS
# ============================================================

evaluation[
    "improvement"
] = (
    evaluation[
        "baseline_abs_error"
    ]
    -
    evaluation[
        "intermittent_abs_error"
    ]
)


print("\n" + "=" * 70)
print("TOP 20 ITEMS WHERE INTERMITTENT MODEL IMPROVES")
print("=" * 70)


top_intermittent_improvement = (
    evaluation
    .sort_values(
        "improvement",
        ascending=False
    )
    .head(20)
)


print(
    top_intermittent_improvement[
        [
            "store_id",
            "sku_id",
            "actual_units_30d",
            "baseline_forecast",
            "intermittent_forecast_30d",
            "baseline_abs_error",
            "intermittent_abs_error",
            "improvement",
            "best_model"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# TOP ITEMS WHERE BASELINE MODEL IMPROVES
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 ITEMS WHERE BASELINE MODEL IMPROVES")
print("=" * 70)


top_baseline_improvement = (
    evaluation
    .sort_values(
        "improvement",
        ascending=True
    )
    .head(20)
)


print(
    top_baseline_improvement[
        [
            "store_id",
            "sku_id",
            "actual_units_30d",
            "baseline_forecast",
            "intermittent_forecast_30d",
            "baseline_abs_error",
            "intermittent_abs_error",
            "improvement",
            "best_model"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# CREATE EVALUATION SUMMARY
# ============================================================

evaluation_summary = pd.DataFrame(
    {
        "metric": [
            "Validation Start",
            "Validation End",
            "Validation Horizon Days",
            "Store-SKU Combinations",
            "Actual Total Units",
            "Baseline MAE",
            "Baseline RMSE",
            "Baseline Bias",
            "Baseline WAPE %",
            "Intermittent MAE",
            "Intermittent RMSE",
            "Intermittent Bias",
            "Intermittent WAPE %",
            "Recommended MAE",
            "Recommended RMSE",
            "Recommended Bias",
            "Recommended WAPE %",
            "Baseline Wins",
            "Intermittent Wins",
            "Ties",
            "Overall Winner"
        ],

        "value": [
            VALIDATION_START.date(),
            VALIDATION_END.date(),
            VALIDATION_DAYS,
            len(evaluation),
            evaluation[
                "actual_units_30d"
            ].sum(),

            baseline_metrics["MAE"],
            baseline_metrics["RMSE"],
            baseline_metrics["Bias"],
            baseline_metrics["WAPE_pct"],

            intermittent_metrics["MAE"],
            intermittent_metrics["RMSE"],
            intermittent_metrics["Bias"],
            intermittent_metrics["WAPE_pct"],

            recommended_metrics["MAE"],
            recommended_metrics["RMSE"],
            recommended_metrics["Bias"],
            recommended_metrics["WAPE_pct"],

            baseline_wins,
            intermittent_wins,
            ties,

            overall_winner
        ]
    }
)


# ============================================================
# SAVE MODEL METRICS
# ============================================================

metrics_output = (
    OUTPUT_PATH /
    "forecast_model_metrics.csv"
)


metrics.to_csv(
    metrics_output,
    index=False
)


# ============================================================
# SAVE ACTIVE METRICS
# ============================================================

if not active_metrics.empty:

    active_metrics_output = (
        OUTPUT_PATH /
        "forecast_active_demand_metrics.csv"
    )

    active_metrics.to_csv(
        active_metrics_output,
        index=False
    )


# ============================================================
# SAVE EVALUATION SUMMARY
# ============================================================

summary_output = (
    OUTPUT_PATH /
    "forecast_evaluation_summary.csv"
)


evaluation_summary.to_csv(
    summary_output,
    index=False
)


# ============================================================
# SAVE STORE-SKU EVALUATION
# ============================================================

store_sku_output = (
    OUTPUT_PATH /
    "store_sku_forecast_evaluation.csv"
)


evaluation.to_csv(
    store_sku_output,
    index=False
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


print(
    "\nEvaluation shape:",
    evaluation.shape
)


print(
    "Missing actual demand:",
    evaluation[
        "actual_units_30d"
    ].isna().sum()
)


print(
    "Missing baseline forecasts:",
    evaluation[
        "baseline_forecast"
    ].isna().sum()
)


print(
    "Missing intermittent forecasts:",
    evaluation[
        "intermittent_forecast_30d"
    ].isna().sum()
)


print(
    "Missing recommended forecasts:",
    evaluation[
        "recommended_forecast"
    ].isna().sum()
)


print(
    "Negative baseline forecasts:",
    (
        evaluation[
            "baseline_forecast"
        ] < 0
    ).sum()
)


print(
    "Negative intermittent forecasts:",
    (
        evaluation[
            "intermittent_forecast_30d"
        ] < 0
    ).sum()
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5.4 COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "\nModel metrics saved to:"
)

print(
    metrics_output
)


print(
    "\nEvaluation summary saved to:"
)

print(
    summary_output
)


print(
    "\nStore-SKU evaluation saved to:"
)

print(
    store_sku_output
)


if not active_metrics.empty:

    print(
        "\nActive-demand metrics saved to:"
    )

    print(
        active_metrics_output
    )


print("\n" + "=" * 70)
print("NEXT PHASE: FORECAST SELECTION & BUSINESS RECOMMENDATIONS")
print("=" * 70)