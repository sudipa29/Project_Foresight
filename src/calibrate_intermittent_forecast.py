# ============================================================
# PROJECT FORESIGHT
# PHASE 6.4 - INTERMITTENT FORECAST CALIBRATION
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

HISTORICAL_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "forecast_demand_daily.csv"
)

FORECAST_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
    / "intermittent_corrected"
    / "intermittent_future_30_day_forecast.csv"
)

REGIME_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
    / "validation"
    / "store_sku_demand_regimes.csv"
)

OUTPUT_DIR = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
    / "calibrated"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "calibrated_intermittent_30_day_forecast.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "calibration_summary.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT")
print("PHASE 6.4 - INTERMITTENT FORECAST CALIBRATION")
print("=" * 70)


# ============================================================
# CHECK INPUTS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

for path in [HISTORICAL_PATH, FORECAST_PATH, REGIME_PATH]:

    print("\nChecking:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print("FOUND")


# ============================================================
# LOAD FORECAST
# ============================================================

print("\n" + "=" * 70)
print("LOADING INTERMITTENT FORECAST")
print("=" * 70)

forecast = pd.read_csv(FORECAST_PATH)

print("Forecast shape:", forecast.shape)
print("Columns:", forecast.columns.tolist())

forecast["date"] = pd.to_datetime(forecast["date"])


# ============================================================
# LOAD REGIMES
# ============================================================

print("\n" + "=" * 70)
print("LOADING DEMAND REGIMES")
print("=" * 70)

regimes = pd.read_csv(REGIME_PATH)

print("Regime shape:", regimes.shape)
print("Columns:", regimes.columns.tolist())

print("\nRegime counts:")
print(regimes["demand_regime"].value_counts())


# ============================================================
# KEEP REQUIRED REGIME COLUMNS
# ============================================================

regime_cols = [
    "store_id",
    "sku_id",
    "demand_regime",
    "activity_level",
    "demand_30d",
    "demand_60d",
    "demand_90d",
    "days_since_demand"
]

available_regime_cols = [
    c for c in regime_cols
    if c in regimes.columns
]

regimes_small = regimes[available_regime_cols].copy()


# ============================================================
# MERGE REGIME WITH FORECAST
# ============================================================

print("\n" + "=" * 70)
print("MERGING FORECAST WITH DEMAND REGIMES")
print("=" * 70)

forecast = forecast.merge(
    regimes_small,
    on=["store_id", "sku_id"],
    how="left"
)

print("Merged shape:", forecast.shape)

missing_regimes = forecast["demand_regime"].isna().sum()

print("Missing demand regimes:", missing_regimes)

if missing_regimes > 0:

    raise ValueError(
        "Some Store-SKU combinations do not have demand regimes."
    )


# ============================================================
# ORIGINAL FORECAST
# ============================================================

forecast["original_forecast_units"] = (
    forecast["forecast_units"]
)


# ============================================================
# CALIBRATION FACTORS
# ============================================================

print("\n" + "=" * 70)
print("APPLYING REGIME-BASED CALIBRATION")
print("=" * 70)


# ------------------------------------------------------------
# CALIBRATION LOGIC
# ------------------------------------------------------------
#
# ACTIVE:
# Recent demand exists.
# Keep forecast unchanged.
#
# INTERMITTENT:
# No demand in last 30 days but demand exists in
# last 90 days.
# Reduce slightly because demand is less recent.
#
# DORMANT:
# No demand in last 90 days.
# Set forecast to zero.
# ------------------------------------------------------------

forecast["calibration_factor"] = np.select(
    [
        forecast["demand_regime"] == "ACTIVE",
        forecast["demand_regime"] == "INTERMITTENT",
        forecast["demand_regime"] == "DORMANT"
    ],
    [
        1.00,
        0.75,
        0.00
    ],
    default=1.00
)


# ============================================================
# CALIBRATED FORECAST
# ============================================================

forecast["calibrated_forecast_units"] = (
    forecast["original_forecast_units"]
    * forecast["calibration_factor"]
)


# ============================================================
# DAILY FORECAST
# ============================================================

forecast["calibrated_forecast_units"] = (
    forecast["calibrated_forecast_units"]
    .clip(lower=0)
)


# ============================================================
# DISPLAY CALIBRATION SUMMARY
# ============================================================

print("\nCalibration factors:")

print(
    forecast
    .groupby("demand_regime")["calibration_factor"]
    .first()
)


# ============================================================
# REGIME-LEVEL SUMMARY
# ============================================================

summary = (
    forecast
    .groupby("demand_regime")
    .agg(
        store_sku_count=("store_id", "nunique"),
        original_forecast_units=(
            "original_forecast_units",
            "sum"
        ),
        calibrated_forecast_units=(
            "calibrated_forecast_units",
            "sum"
        ),
        avg_original_forecast=(
            "original_forecast_units",
            "mean"
        ),
        avg_calibrated_forecast=(
            "calibrated_forecast_units",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# FORECAST SHARE
# ============================================================

total_calibrated = (
    summary["calibrated_forecast_units"].sum()
)

summary["forecast_share_pct"] = (
    summary["calibrated_forecast_units"]
    / total_calibrated
    * 100
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))


# ============================================================
# TOTAL FORECAST
# ============================================================

original_total = (
    forecast["original_forecast_units"].sum()
)

calibrated_total = (
    forecast["calibrated_forecast_units"].sum()
)

difference = calibrated_total - original_total

difference_pct = (
    difference / original_total * 100
)


print("\n" + "=" * 70)
print("TOTAL FORECAST COMPARISON")
print("=" * 70)

print(
    f"Original intermittent forecast: "
    f"{original_total:,.2f}"
)

print(
    f"Calibrated forecast: "
    f"{calibrated_total:,.2f}"
)

print(
    f"Difference: "
    f"{difference:,.2f}"
)

print(
    f"Difference %: "
    f"{difference_pct:.2f}%"
)


# ============================================================
# SAVE CALIBRATED FORECAST
# ============================================================

output_columns = [
    "store_id",
    "sku_id",
    "date",
    "demand_regime",
    "activity_level",
    "demand_30d",
    "demand_60d",
    "demand_90d",
    "days_since_demand",
    "occurrence_probability",
    "positive_demand_quantity",
    "original_forecast_units",
    "calibration_factor",
    "calibrated_forecast_units"
]

output_columns = [
    c for c in output_columns
    if c in forecast.columns
]

forecast[output_columns].to_csv(
    OUTPUT_PATH,
    index=False
)

summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATED FORECAST CHECK")
print("=" * 70)

print(
    "Rows:",
    len(forecast)
)

print(
    "Zero calibrated forecasts:",
    (
        forecast["calibrated_forecast_units"] == 0
    ).sum()
)

print(
    "Positive calibrated forecasts:",
    (
        forecast["calibrated_forecast_units"] > 0
    ).sum()
)

print(
    "Final 30-day calibrated forecast:",
    f"{calibrated_total:,.2f}"
)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(OUTPUT_PATH)
print(SUMMARY_PATH)

print("\n" + "=" * 70)
print("PHASE 6.4 COMPLETED")
print("=" * 70)