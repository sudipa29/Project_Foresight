# ============================================================
# PROJECT FORESIGHT
# Phase 5.9 - Final Model Selection
# ============================================================

import warnings

warnings.filterwarnings("ignore")

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
    BASE_PATH
    / "data"
    / "processed"
)

FORECASTING_PATH = (
    PROCESSED_PATH
    / "forecasting"
)

ADVANCED_ML_PATH = (
    FORECASTING_PATH
    / "advanced_ml"
)

ARIMA_PATH = (
    FORECASTING_PATH
    / "arima_sarima"
)

VALIDATION_PATH = (
    FORECASTING_PATH
    / "validation"
)

OUTPUT_PATH = (
    FORECASTING_PATH
    / "final_model_selection"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

ML_METRICS_PATH = (
    ADVANCED_ML_PATH
    / "advanced_ml_model_metrics.csv"
)

ML_SUMMARY_PATH = (
    ADVANCED_ML_PATH
    / "advanced_ml_model_summary.csv"
)

ARIMA_SUMMARY_PATH = (
    ARIMA_PATH
    / "arima_sarima_model_summary.csv"
)

ARIMA_ACTIVITY_PATH = (
    ARIMA_PATH
    / "arima_sarima_activity_summary.csv"
)

BACKTEST_SUMMARY_PATH = (
    VALIDATION_PATH
    / "rolling_backtest_summary.csv"
)

BACKTEST_RANKING_PATH = (
    VALIDATION_PATH
    / "rolling_backtest_model_ranking.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

FINAL_COMPARISON_PATH = (
    OUTPUT_PATH
    / "final_model_comparison.csv"
)

FINAL_SCORE_PATH = (
    OUTPUT_PATH
    / "final_model_scores.csv"
)

FINAL_DECISION_PATH = (
    OUTPUT_PATH
    / "final_model_decision.csv"
)

MODEL_COVERAGE_PATH = (
    OUTPUT_PATH
    / "model_coverage_summary.csv"
)

FINAL_REPORT_PATH = (
    OUTPUT_PATH
    / "final_model_selection_report.txt"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "PROJECT FORESIGHT - FINAL MODEL SELECTION"
)

print("=" * 70)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(
    dataframe,
    candidates
):

    for column in candidates:

        if column in dataframe.columns:

            return column

    return None


def safe_numeric(
    series
):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def normalize_lower_better(
    series
):

    series = safe_numeric(
        series
    )

    minimum = series.min()

    maximum = series.max()

    if pd.isna(minimum):

        return pd.Series(
            np.nan,
            index=series.index
        )

    if maximum == minimum:

        return pd.Series(
            1.0,
            index=series.index
        )

    # Best model gets 1
    # Worst model gets 0

    return (
        maximum - series
    ) / (
        maximum - minimum
    )


# ============================================================
# CHECK FILES
# ============================================================

required_files = {

    "Advanced ML metrics":
        ML_METRICS_PATH,

    "Advanced ML summary":
        ML_SUMMARY_PATH,

    "ARIMA/SARIMA summary":
        ARIMA_SUMMARY_PATH,

    "Rolling backtest summary":
        BACKTEST_SUMMARY_PATH
}


print(
    "\nChecking required files..."
)


for name, path in required_files.items():

    if path.exists():

        print(
            f"PASS: {name}"
        )

    else:

        print(
            f"WARNING: {name} not found:"
        )

        print(
            path
        )


# ============================================================
# LOAD ADVANCED ML RESULTS
# ============================================================

print("\n" + "=" * 70)

print(
    "LOADING ADVANCED ML RESULTS"
)

print("=" * 70)


if not ML_METRICS_PATH.exists():

    raise FileNotFoundError(
        f"\nAdvanced ML metrics not found:\n"
        f"{ML_METRICS_PATH}"
    )


ml = pd.read_csv(
    ML_METRICS_PATH
)


print(
    "Advanced ML rows:",
    len(ml)
)

print(
    "Advanced ML columns:",
    ml.columns.tolist()
)


# ============================================================
# NORMALIZE ML COLUMN NAMES
# ============================================================

ml_model_col = find_column(
    ml,
    [
        "model",
        "Model"
    ]
)

if ml_model_col is None:

    raise ValueError(
        "Could not identify model column in "
        "advanced_ml_model_metrics.csv"
    )


# ------------------------------------------------------------
# Convert model names
# ------------------------------------------------------------

ml["model"] = (
    ml[ml_model_col]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# Keep relevant models
# ------------------------------------------------------------

ml["model"] = (
    ml["model"]
    .replace(
        {
            "LightGBM": "LightGBM",
            "RandomForest": "RandomForest",
            "Random Forest": "RandomForest",
            "XGBoost": "XGBoost",
            "Rolling30D": "Rolling30D",
            "Rolling_30D": "Rolling30D"
        }
    )
)


# ============================================================
# METRIC COLUMNS
# ============================================================

mae_col = find_column(
    ml,
    [
        "MAE",
        "mae"
    ]
)

rmse_col = find_column(
    ml,
    [
        "RMSE",
        "rmse"
    ]
)

bias_col = find_column(
    ml,
    [
        "Bias",
        "bias"
    ]
)

wape_col = find_column(
    ml,
    [
        "WAPE_pct",
        "WAPE",
        "wape_pct"
    ]
)

active_wape_col = find_column(
    ml,
    [
        "Active_WAPE_pct",
        "Active_WAPE",
        "active_wape_pct"
    ]
)


if mae_col is None:

    raise ValueError(
        "MAE column not found in ML metrics."
    )

if rmse_col is None:

    raise ValueError(
        "RMSE column not found in ML metrics."
    )

if wape_col is None:

    raise ValueError(
        "WAPE column not found in ML metrics."
    )


# ============================================================
# CREATE ML STANDARDIZED TABLE
# ============================================================

ml_standardized = pd.DataFrame(
    {
        "model":
            ml["model"],

        "MAE":
            safe_numeric(
                ml[mae_col]
            ),

        "RMSE":
            safe_numeric(
                ml[rmse_col]
            ),

        "Bias":
            safe_numeric(
                ml[bias_col]
            )
            if bias_col
            else 0.0,

        "WAPE_pct":
            safe_numeric(
                ml[wape_col]
            ),

        "Active_WAPE_pct":
            safe_numeric(
                ml[active_wape_col]
            )
            if active_wape_col
            else np.nan,

        "source":
            "Phase_5.7"
    }
)


# ------------------------------------------------------------
# Remove duplicate models
# ------------------------------------------------------------

ml_standardized = (
    ml_standardized
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
            "Active_WAPE_pct": "mean",
            "source": "first"
        }
    )
)


# ============================================================
# LOAD ARIMA / SARIMA
# ============================================================

print("\n" + "=" * 70)

print(
    "LOADING ARIMA / SARIMA RESULTS"
)

print("=" * 70)


if not ARIMA_SUMMARY_PATH.exists():

    raise FileNotFoundError(
        f"\nARIMA/SARIMA summary not found:\n"
        f"{ARIMA_SUMMARY_PATH}"
    )


classical = pd.read_csv(
    ARIMA_SUMMARY_PATH
)


print(
    "Classical model rows:",
    len(classical)
)


print(
    classical.to_string(
        index=False
    )
)


# ============================================================
# STANDARDIZE CLASSICAL RESULTS
# ============================================================

classical_model_col = find_column(
    classical,
    [
        "model",
        "Model"
    ]
)

classical_mae_col = find_column(
    classical,
    [
        "MAE",
        "mae"
    ]
)

classical_rmse_col = find_column(
    classical,
    [
        "RMSE",
        "rmse"
    ]

)

classical_bias_col = find_column(
    classical,
    [
        "Bias",
        "bias"
    ]
)

classical_wape_col = find_column(
    classical,
    [
        "WAPE_pct",
        "WAPE",
        "wape_pct"
    ]
)

classical_active_col = find_column(
    classical,
    [
        "Active_WAPE_pct",
        "Active_WAPE",
        "active_wape_pct"
    ]
)


classical_standardized = pd.DataFrame(
    {
        "model":
            classical[
                classical_model_col
            ].astype(str),

        "MAE":
            safe_numeric(
                classical[
                    classical_mae_col
                ]
            ),

        "RMSE":
            safe_numeric(
                classical[
                    classical_rmse_col
                ]
            ),

        "Bias":
            safe_numeric(
                classical[
                    classical_bias_col
                ]
            )
            if classical_bias_col
            else 0.0,

        "WAPE_pct":
            safe_numeric(
                classical[
                    classical_wape_col
                ]
            ),

        "Active_WAPE_pct":
            safe_numeric(
                classical[
                    classical_active_col
                ]
            )
            if classical_active_col
            else np.nan,

        "source":
            "Phase_5.8"
    }
)


# ============================================================
# LOAD ROLLING BACKTEST
# ============================================================

print("\n" + "=" * 70)

print(
    "LOADING ROLLING BACKTEST RESULTS"
)

print("=" * 70)


if BACKTEST_SUMMARY_PATH.exists():

    backtest = pd.read_csv(
        BACKTEST_SUMMARY_PATH
    )

    print(
        "Rolling backtest rows:",
        len(backtest)
    )

    print(
        backtest.to_string(
            index=False
        )
    )

else:

    backtest = pd.DataFrame()


# ============================================================
# FIND ROLLING 30D
# ============================================================

rolling_result = None


if not backtest.empty:

    backtest_model_col = find_column(
        backtest,
        [
            "model",
            "Model"
        ]
    )

    if backtest_model_col:

        backtest["model"] = (
            backtest[
                backtest_model_col
            ]
            .astype(str)
            .str.strip()
        )

        rolling_rows = backtest[
            backtest["model"]
            .isin(
                [
                    "Rolling_30D",
                    "Rolling30D",
                    "Rolling 30D"
                ]
            )
        ]

        if len(rolling_rows) > 0:

            rolling_row = (
                rolling_rows
                .iloc[0]
            )

            rolling_result = {

                "model":
                    "Rolling30D",

                "MAE":
                    safe_numeric(
                        pd.Series(
                            [
                                rolling_row[
                                    "MAE"
                                ]
                            ]
                        )
                    ).iloc[0],

                "RMSE":
                    safe_numeric(
                        pd.Series(
                            [
                                rolling_row[
                                    "RMSE"
                                ]
                            ]
                        )
                    ).iloc[0],

                "Bias":
                    safe_numeric(
                        pd.Series(
                            [
                                rolling_row[
                                    "Bias"
                                ]
                            ]
                        )
                    ).iloc[0],

                "WAPE_pct":
                    safe_numeric(
                        pd.Series(
                            [
                                rolling_row[
                                    "WAPE_pct"
                                ]
                            ]
                        )
                    ).iloc[0],

                "Active_WAPE_pct":
                    np.nan,

                "source":
                    "Phase_5.6"
            }


# ============================================================
# COMBINE ALL MODELS
# ============================================================

print("\n" + "=" * 70)

print(
    "COMBINING FORECASTING MODELS"
)

print("=" * 70)


comparison_parts = [
    ml_standardized,
    classical_standardized
]


if rolling_result is not None:

    comparison_parts.append(
        pd.DataFrame(
            [rolling_result]
        )
    )


comparison = pd.concat(
    comparison_parts,
    ignore_index=True
)


# ------------------------------------------------------------
# Keep relevant models
# ------------------------------------------------------------

comparison = comparison[
    comparison["model"].isin(
        [
            "LightGBM",
            "RandomForest",
            "XGBoost",
            "Rolling30D",
            "ARIMA",
            "SARIMA"
        ]
    )
].copy()


# ------------------------------------------------------------
# Remove duplicate rows
# ------------------------------------------------------------

comparison = (
    comparison
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
            "Active_WAPE_pct": "mean",
            "source": "first"
        }
    )
)


# ============================================================
# ADD MODEL COVERAGE
# ============================================================

comparison["validation_coverage"] = 1.0

comparison["coverage_note"] = (
    "Full validation comparison"
)


comparison.loc[
    comparison["model"].isin(
        [
            "ARIMA",
            "SARIMA"
        ]
    ),
    "validation_coverage"
] = 23 / 30


comparison.loc[
    comparison["model"].isin(
        [
            "ARIMA",
            "SARIMA"
        ]
    ),
    "coverage_note"
] = (
    "23 of 30 representative "
    "Store-SKU series completed"
)


# ============================================================
# SAVE RAW COMPARISON
# ============================================================

comparison = comparison[
    [
        "model",
        "MAE",
        "RMSE",
        "Bias",
        "WAPE_pct",
        "Active_WAPE_pct",
        "validation_coverage",
        "coverage_note",
        "source"
    ]
]


comparison.to_csv(
    FINAL_COMPARISON_PATH,
    index=False
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\n" + "=" * 70)

print(
    "FINAL MODEL COMPARISON"
)

print("=" * 70)


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# NORMALIZED SCORING
# ============================================================

print("\n" + "=" * 70)

print(
    "CALCULATING FINAL MODEL SCORES"
)

print("=" * 70)


score_df = comparison.copy()


# ------------------------------------------------------------
# Metric normalization
# ------------------------------------------------------------

score_df["MAE_score"] = (
    normalize_lower_better(
        score_df["MAE"]
    )
)


score_df["RMSE_score"] = (
    normalize_lower_better(
        score_df["RMSE"]
    )
)


score_df["WAPE_score"] = (
    normalize_lower_better(
        score_df["WAPE_pct"]
    )
)


# Active WAPE can have missing values
# for Rolling30D.

active_temp = (
    score_df["Active_WAPE_pct"]
    .copy()
)

if active_temp.notna().sum() >= 2:

    score_df["Active_WAPE_score"] = (
        normalize_lower_better(
            active_temp
        )
    )

else:

    score_df["Active_WAPE_score"] = 0.0


# Bias uses absolute value
score_df["Absolute_Bias"] = (
    score_df["Bias"]
    .abs()
)


score_df["Bias_score"] = (
    normalize_lower_better(
        score_df["Absolute_Bias"]
    )
)


# ============================================================
# WEIGHTED FINAL SCORE
# ============================================================

# Weights:
#
# MAE             35%
# RMSE            25%
# WAPE            20%
# Active WAPE     10%
# Bias            10%
#
# These weights emphasize
# actual forecast accuracy.

score_df["accuracy_score"] = (

    score_df["MAE_score"]
    * 0.35

    +

    score_df["RMSE_score"]
    * 0.25

    +

    score_df["WAPE_score"]
    * 0.20

    +

    score_df["Active_WAPE_score"]
    * 0.10

    +

    score_df["Bias_score"]
    * 0.10
)


# ============================================================
# COVERAGE ADJUSTMENT
# ============================================================

# Classical models only have 23/30
# successful Store-SKU series.
#
# Apply a small coverage factor
# rather than allowing incomplete
# evaluation to dominate.

score_df["coverage_adjustment"] = (
    0.90
    +
    0.10
    *
    score_df["validation_coverage"]
)


score_df["final_score"] = (
    score_df["accuracy_score"]
    *
    score_df["coverage_adjustment"]
)


# ============================================================
# RANK MODELS
# ============================================================

score_df = (
    score_df
    .sort_values(
        "final_score",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


score_df["rank"] = (
    np.arange(
        1,
        len(score_df) + 1
    )
)


# ============================================================
# SAVE SCORES
# ============================================================

score_df.to_csv(
    FINAL_SCORE_PATH,
    index=False
)


# ============================================================
# PRINT SCORES
# ============================================================

print(
    score_df[
        [
            "rank",
            "model",
            "MAE",
            "RMSE",
            "WAPE_pct",
            "Active_WAPE_pct",
            "Absolute_Bias",
            "accuracy_score",
            "coverage_adjustment",
            "final_score"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL MODEL
# ============================================================

best_model = (
    score_df
    .iloc[0]
)


best_model_name = (
    best_model["model"]
)


# ============================================================
# BUSINESS RULE VALIDATION
# ============================================================

print("\n" + "=" * 70)

print(
    "BUSINESS / PRODUCTION MODEL VALIDATION"
)

print("=" * 70)


# ------------------------------------------------------------
# Compare LightGBM with next best
# ------------------------------------------------------------

lightgbm_row = score_df[
    score_df["model"]
    == "LightGBM"
]


if len(lightgbm_row) > 0:

    lightgbm_row = (
        lightgbm_row
        .iloc[0]
    )

    print(
        "\nLightGBM metrics:"
    )

    print(
        "MAE:",
        round(
            lightgbm_row["MAE"],
            6
        )
    )

    print(
        "RMSE:",
        round(
            lightgbm_row["RMSE"],
            6
        )
    )

    print(
        "WAPE:",
        round(
            lightgbm_row["WAPE_pct"],
            4
        ),
        "%"
    )


# ------------------------------------------------------------
# Final decision
# ------------------------------------------------------------

if best_model_name == "LightGBM":

    decision = (
        "LightGBM selected as the "
        "final production forecasting model."
    )

    reason = (
        "LightGBM provides the strongest "
        "validation performance across "
        "MAE, RMSE and WAPE while being "
        "evaluated on the full validation "
        "population."
    )

else:

    decision = (
        f"{best_model_name} selected as "
        "the final production forecasting model."
    )

    reason = (
        "The model achieved the highest "
        "weighted final evaluation score."
    )


# ============================================================
# ADD FINAL DECISION
# ============================================================

final_decision = pd.DataFrame(
    [
        {
            "final_model":
                best_model_name,

            "decision":
                decision,

            "reason":
                reason,

            "MAE":
                best_model["MAE"],

            "RMSE":
                best_model["RMSE"],

            "Bias":
                best_model["Bias"],

            "WAPE_pct":
                best_model["WAPE_pct"],

            "Active_WAPE_pct":
                best_model["Active_WAPE_pct"],

            "validation_coverage":
                best_model[
                    "validation_coverage"
                ],

            "final_score":
                best_model[
                    "final_score"
                ]
        }
    ]
)


final_decision.to_csv(
    FINAL_DECISION_PATH,
    index=False
)


# ============================================================
# MODEL COVERAGE SUMMARY
# ============================================================

coverage_summary = (
    comparison[
        [
            "model",
            "validation_coverage",
            "coverage_note"
        ]
    ]
    .copy()
)


coverage_summary.to_csv(
    MODEL_COVERAGE_PATH,
    index=False
)


# ============================================================
# PRINT FINAL DECISION
# ============================================================

print("\n" + "=" * 70)

print(
    "FINAL MODEL DECISION"
)

print("=" * 70)


print(
    "\nFINAL PRODUCTION MODEL:",
    best_model_name
)


print(
    "\nDecision:"
)

print(
    decision
)


print(
    "\nReason:"
)

print(
    reason
)


print(
    "\nFinal Score:",
    round(
        best_model["final_score"],
        6
    )
)


print(
    "\nValidation Coverage:",
    round(
        best_model[
            "validation_coverage"
        ]
        * 100,
        2
    ),
    "%"
)


# ============================================================
# WRITE TEXT REPORT
# ============================================================

report_lines = []


report_lines.append(
    "PROJECT FORESIGHT - PHASE 5.9"
)

report_lines.append(
    "FINAL MODEL SELECTION REPORT"
)

report_lines.append(
    "=" * 60
)

report_lines.append("")


report_lines.append(
    f"Final Production Model: "
    f"{best_model_name}"
)

report_lines.append("")


report_lines.append(
    "Decision:"
)

report_lines.append(
    decision
)

report_lines.append("")


report_lines.append(
    "Reason:"
)

report_lines.append(
    reason
)

report_lines.append("")


report_lines.append(
    "Final Metrics:"
)

report_lines.append(
    f"MAE: {best_model['MAE']:.6f}"
)

report_lines.append(
    f"RMSE: {best_model['RMSE']:.6f}"
)

report_lines.append(
    f"Bias: {best_model['Bias']:.6f}"
)

report_lines.append(
    f"WAPE: {best_model['WAPE_pct']:.4f}%"
)

if pd.notna(
    best_model["Active_WAPE_pct"]
):

    report_lines.append(
        f"Active WAPE: "
        f"{best_model['Active_WAPE_pct']:.4f}%"
    )


report_lines.append(
    f"Validation Coverage: "
    f"{best_model['validation_coverage'] * 100:.2f}%"
)


report_lines.append(
    f"Final Score: "
    f"{best_model['final_score']:.6f}"
)

report_lines.append("")


report_lines.append(
    "Models evaluated:"
)

for model_name in score_df["model"]:

    report_lines.append(
        f"- {model_name}"
    )


report_lines.append("")


report_lines.append(
    "Important note:"
)

report_lines.append(
    "ARIMA and SARIMA were evaluated on "
    "23 of 30 selected representative "
    "Store-SKU series because 7 series "
    "did not complete successfully."
)

report_lines.append("")


report_lines.append(
    "Next phase:"
)

report_lines.append(
    "Future 30/60/90-day forecasting "
    "using the selected production model."
)


with open(
    FINAL_REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(
            report_lines
        )
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print(
    "PHASE 5.9 COMPLETED"
)

print("=" * 70)


print(
    "\nFinal comparison saved to:"
)

print(
    FINAL_COMPARISON_PATH
)


print(
    "\nFinal scores saved to:"
)

print(
    FINAL_SCORE_PATH
)


print(
    "\nFinal decision saved to:"
)

print(
    FINAL_DECISION_PATH
)


print(
    "\nCoverage summary saved to:"
)

print(
    MODEL_COVERAGE_PATH
)


print(
    "\nFinal report saved to:"
)

print(
    FINAL_REPORT_PATH
)


print("\n" + "=" * 70)

print(
    "NEXT PHASE: FUTURE 30 / 60 / 90-DAY FORECAST"
)

print("=" * 70)