import os
import pandas as pd


print("=" * 70)
print("PROJECT FORESIGHT - MASTER MODEL COMPARISON")
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


# ============================================================
# 2. LOAD MACHINE LEARNING RESULTS
# ============================================================

print("\nLoading Machine Learning model results...")

ml_file = os.path.join(
    FORECAST_DIR,
    "advanced_model_comparison.csv"
)

ml_results = pd.read_csv(
    ml_file
)

print("\nMachine Learning models:")

print(
    ml_results.round(2).to_string(
        index=False
    )
)


# ============================================================
# 3. LOAD TUNED ARIMA / SARIMA RESULTS
# ============================================================

print("\n" + "=" * 70)
print("Loading tuned statistical model results...")
print("=" * 70)


tuning_file = os.path.join(
    ARIMA_DIR,
    "arima_sarima_tuning_results.csv"
)


if not os.path.exists(tuning_file):

    raise FileNotFoundError(
        f"\nTuning results not found:\n{tuning_file}\n\n"
        "Please run arima_sarima_tuning.py first."
    )


tuning_results = pd.read_csv(
    tuning_file
)


# ============================================================
# 4. SELECT BEST ARIMA
# ============================================================

arima_results = tuning_results[
    tuning_results["model"].str.upper() == "ARIMA"
].copy()


best_arima = arima_results.loc[
    arima_results["MAE"].idxmin()
]


# ============================================================
# 5. SELECT BEST SARIMA
# ============================================================

sarima_results = tuning_results[
    tuning_results["model"].str.upper() == "SARIMA"
].copy()


best_sarima = sarima_results.loc[
    sarima_results["MAE"].idxmin()
]


# ============================================================
# 6. CREATE STATISTICAL MODEL TABLE
# ============================================================

statistical_results = pd.DataFrame({

    "model": [
        "ARIMA",
        "SARIMA"
    ],

    "MAE": [
        best_arima["MAE"],
        best_sarima["MAE"]
    ],

    "RMSE": [
        best_arima["RMSE"],
        best_sarima["RMSE"]
    ],

    "MAPE": [
        best_arima["MAPE"],
        best_sarima["MAPE"]
    ],

    "WAPE": [
        best_arima["WAPE"],
        best_sarima["WAPE"]
    ]

})


print("\nStatistical models:")

print(
    statistical_results.round(2).to_string(
        index=False
    )
)


# ============================================================
# 7. ADD MODEL TYPE
# ============================================================

ml_results["model_type"] = "Machine Learning"

statistical_results["model_type"] = "Statistical"


# ============================================================
# 8. COMBINE ALL MODELS
# ============================================================

master_results = pd.concat(
    [
        ml_results[
            [
                "model",
                "model_type",
                "MAE",
                "RMSE",
                "MAPE",
                "WAPE"
            ]
        ],

        statistical_results[
            [
                "model",
                "model_type",
                "MAE",
                "RMSE",
                "MAPE",
                "WAPE"
            ]
        ]
    ],
    ignore_index=True
)


# ============================================================
# 9. SORT BY MAE
# ============================================================

master_results = master_results.sort_values(
    "MAE",
    ascending=True
).reset_index(
    drop=True
)


master_results.insert(
    0,
    "rank",
    range(
        1,
        len(master_results) + 1
    )
)


# ============================================================
# 10. DISPLAY MASTER COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MASTER MODEL COMPARISON")
print("=" * 70)


print(
    master_results.round(2).to_string(
        index=False
    )
)


# ============================================================
# 11. BEST OVERALL MODEL
# ============================================================

best_overall = master_results.iloc[0]


print("\n" + "=" * 70)
print("BEST OVERALL FORECASTING MODEL")
print("=" * 70)


print(
    "Model:",
    best_overall["model"]
)

print(
    "Model Type:",
    best_overall["model_type"]
)

print(
    "MAE:",
    round(
        best_overall["MAE"],
        2
    )
)

print(
    "RMSE:",
    round(
        best_overall["RMSE"],
        2
    )
)

print(
    "MAPE:",
    round(
        best_overall["MAPE"],
        2
    ),
    "%"
)

print(
    "WAPE:",
    round(
        best_overall["WAPE"],
        2
    ),
    "%"
)


# ============================================================
# 12. BEST MACHINE LEARNING MODEL
# ============================================================

ml_models = master_results[
    master_results["model_type"]
    == "Machine Learning"
]


best_ml = ml_models.iloc[0]


print("\n" + "=" * 70)
print("BEST MACHINE LEARNING MODEL")
print("=" * 70)


print(
    f"{best_ml['model']} | "
    f"MAE: {best_ml['MAE']:.2f}"
)


# ============================================================
# 13. BEST STATISTICAL MODEL
# ============================================================

statistical_models = master_results[
    master_results["model_type"]
    == "Statistical"
]


best_statistical = statistical_models.iloc[0]


print("\n" + "=" * 70)
print("BEST STATISTICAL MODEL")
print("=" * 70)


print(
    f"{best_statistical['model']} | "
    f"MAE: {best_statistical['MAE']:.2f}"
)


# ============================================================
# 14. TUNED CONFIGURATIONS
# ============================================================

print("\n" + "=" * 70)
print("BEST STATISTICAL MODEL CONFIGURATIONS")
print("=" * 70)


print(
    "\nBest ARIMA configuration:"
)

print(
    "ARIMA",
    best_arima["order"]
)


print(
    "\nBest SARIMA configuration:"
)

print(
    "SARIMA",
    best_sarima["order"],
    "x",
    best_sarima["seasonal_order"]
)


# ============================================================
# 15. PERFORMANCE GAP
# ============================================================

performance_gap = (
    best_statistical["MAE"]
    -
    best_ml["MAE"]
)


improvement_pct = (
    performance_gap
    /
    best_statistical["MAE"]
) * 100


print("\n" + "=" * 70)
print("ML VS STATISTICAL MODEL PERFORMANCE")
print("=" * 70)


print(
    "Best ML MAE:",
    round(
        best_ml["MAE"],
        2
    )
)


print(
    "Best Statistical MAE:",
    round(
        best_statistical["MAE"],
        2
    )
)


print(
    "MAE Difference:",
    round(
        performance_gap,
        2
    )
)


print(
    "ML improvement over best statistical model:",
    round(
        improvement_pct,
        2
    ),
    "%"
)


# ============================================================
# 16. SAVE MASTER RESULTS
# ============================================================

master_file = os.path.join(
    FORECAST_DIR,
    "master_model_comparison.csv"
)


master_results.to_csv(
    master_file,
    index=False
)


# ============================================================
# 17. SAVE BEST MODEL SUMMARY
# ============================================================

summary = pd.DataFrame({

    "metric": [
        "best_overall_model",
        "best_overall_model_type",
        "best_ml_model",
        "best_statistical_model",
        "best_arima_order",
        "best_sarima_order",
        "best_sarima_seasonal_order"
    ],

    "value": [
        best_overall["model"],
        best_overall["model_type"],
        best_ml["model"],
        best_statistical["model"],
        best_arima["order"],
        best_sarima["order"],
        best_sarima["seasonal_order"]
    ]

})


summary_file = os.path.join(
    FORECAST_DIR,
    "best_model_summary.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MASTER MODEL COMPARISON COMPLETED")
print("=" * 70)


print(
    "\nBest Overall Model:",
    best_overall["model"]
)


print(
    "Best ML Model:",
    best_ml["model"]
)


print(
    "Best Statistical Model:",
    best_statistical["model"]
)


print("\nFiles saved:")

print(
    master_file
)

print(
    summary_file
)

print("\n" + "=" * 70)