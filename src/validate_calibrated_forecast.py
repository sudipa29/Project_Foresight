# ============================================================
# PROJECT FORESIGHT
# PHASE 6.5 - CALIBRATED FORECAST VALIDATION
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

CALIBRATED_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
    / "calibrated"
    / "calibrated_intermittent_30_day_forecast.csv"
)

ORIGINAL_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
    / "intermittent_corrected"
    / "intermittent_future_30_day_forecast.csv"
)

LIGHTGBM_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
    / "future_30_day_forecast.csv"
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
    / "validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SUMMARY_PATH = (
    OUTPUT_DIR
    / "calibrated_forecast_validation_summary.csv"
)

STORE_SKU_PATH = (
    OUTPUT_DIR
    / "calibrated_forecast_store_sku_validation.csv"
)

REGIME_VALIDATION_PATH = (
    OUTPUT_DIR
    / "calibrated_forecast_regime_validation.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT")
print("PHASE 6.5 - CALIBRATED FORECAST VALIDATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

required_files = [
    HISTORICAL_PATH,
    CALIBRATED_PATH,
    ORIGINAL_PATH,
    LIGHTGBM_PATH,
    REGIME_PATH
]

for path in required_files:

    print("\nChecking:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    print("FOUND")


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING HISTORICAL DEMAND")
print("=" * 70)

historical = pd.read_csv(
    HISTORICAL_PATH,
    usecols=[
        "store_id",
        "sku_id",
        "date",
        "units_sold"
    ]
)

historical["date"] = pd.to_datetime(
    historical["date"]
)

historical["units_sold"] = pd.to_numeric(
    historical["units_sold"],
    errors="coerce"
).fillna(0)

last_date = historical["date"].max()

print("Historical rows:", len(historical))
print("Historical last date:", last_date)


# ============================================================
# LOAD FORECASTS
# ============================================================

print("\n" + "=" * 70)
print("LOADING FORECASTS")
print("=" * 70)

calibrated = pd.read_csv(CALIBRATED_PATH)

original = pd.read_csv(ORIGINAL_PATH)

lightgbm = pd.read_csv(LIGHTGBM_PATH)

regimes = pd.read_csv(REGIME_PATH)


# ============================================================
# DATE CONVERSION
# ============================================================

calibrated["date"] = pd.to_datetime(
    calibrated["date"]
)

original["date"] = pd.to_datetime(
    original["date"]
)

lightgbm["date"] = pd.to_datetime(
    lightgbm["date"]
)


# ============================================================
# TOTAL FORECASTS
# ============================================================

calibrated_total = (
    calibrated["calibrated_forecast_units"].sum()
)

original_total = (
    original["forecast_units"].sum()
)

lightgbm_total = (
    lightgbm["forecast_units"].sum()
)

historical_30d = historical[
    historical["date"] > (
        last_date - pd.Timedelta(days=30)
    )
]["units_sold"].sum()


# ============================================================
# TOTAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("30-DAY FORECAST COMPARISON")
print("=" * 70)

print(
    f"Historical demand:           {historical_30d:,.2f}"
)

print(
    f"LightGBM forecast:            {lightgbm_total:,.2f}"
)

print(
    f"Original intermittent:        {original_total:,.2f}"
)

print(
    f"Calibrated intermittent:      {calibrated_total:,.2f}"
)


# ============================================================
# ERROR CALCULATIONS
# ============================================================

def pct_difference(forecast, actual):

    return (
        (forecast - actual)
        / actual
        * 100
    )


lightgbm_pct = pct_difference(
    lightgbm_total,
    historical_30d
)

original_pct = pct_difference(
    original_total,
    historical_30d
)

calibrated_pct = pct_difference(
    calibrated_total,
    historical_30d
)


print("\nPercentage difference:")

print(
    f"LightGBM:          {lightgbm_pct:+.2f}%"
)

print(
    f"Original:          {original_pct:+.2f}%"
)

print(
    f"Calibrated:        {calibrated_pct:+.2f}%"
)


# ============================================================
# HORIZON SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HORIZON VALIDATION")
print("=" * 70)

horizon_rows = []

for horizon in [30, 60, 90]:

    start_date = (
        last_date
        - pd.Timedelta(days=horizon)
    )

    historical_total = historical[
        historical["date"] > start_date
    ]["units_sold"].sum()

    # --------------------------------------------------------
    # Original intermittent
    # --------------------------------------------------------

    original_file = (
        BASE_PATH
        / "data"
        / "processed"
        / "forecasting"
        / "future"
        / "intermittent_corrected"
        / f"intermittent_future_{horizon}_day_forecast.csv"
    )

    original_h = pd.read_csv(
        original_file,
        usecols=["forecast_units"]
    )

    original_h_total = (
        original_h["forecast_units"].sum()
    )

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    lightgbm_file = (
        BASE_PATH
        / "data"
        / "processed"
        / "forecasting"
        / "future"
        / f"future_{horizon}_day_forecast.csv"
    )

    lightgbm_h = pd.read_csv(
        lightgbm_file,
        usecols=["forecast_units"]
    )

    lightgbm_h_total = (
        lightgbm_h["forecast_units"].sum()
    )

    # --------------------------------------------------------
    # Calibrated
    #
    # Only 30-day calibrated forecast exists currently.
    # For 60/90 days we report NaN until calibration
    # is extended to those horizons.
    # --------------------------------------------------------

    if horizon == 30:

        calibrated_h_total = (
            calibrated[
                "calibrated_forecast_units"
            ].sum()
        )

    else:

        calibrated_h_total = np.nan

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    lightgbm_error = (
        lightgbm_h_total
        - historical_total
    )

    original_error = (
        original_h_total
        - historical_total
    )

    if pd.notna(calibrated_h_total):

        calibrated_error = (
            calibrated_h_total
            - historical_total
        )

        calibrated_pct = (
            calibrated_error
            / historical_total
            * 100
        )

    else:

        calibrated_error = np.nan
        calibrated_pct = np.nan

    row = {
        "horizon_days": horizon,
        "historical_demand": historical_total,
        "lightgbm_forecast": lightgbm_h_total,
        "original_intermittent_forecast": original_h_total,
        "calibrated_intermittent_forecast": calibrated_h_total,
        "lightgbm_difference_pct": (
            lightgbm_error
            / historical_total
            * 100
        ),
        "original_intermittent_difference_pct": (
            original_error
            / historical_total
            * 100
        ),
        "calibrated_difference_pct": calibrated_pct
    }

    horizon_rows.append(row)

    print(
        f"\n{horizon}-DAY"
    )

    print(
        f"Historical:       {historical_total:,.2f}"
    )

    print(
        f"LightGBM:          {lightgbm_h_total:,.2f}"
    )

    print(
        f"Original:          {original_h_total:,.2f}"
    )

    if pd.notna(calibrated_h_total):

        print(
            f"Calibrated:        "
            f"{calibrated_h_total:,.2f}"
        )


summary = pd.DataFrame(
    horizon_rows
)


# ============================================================
# STORE-SKU VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STORE-SKU LEVEL VALIDATION")
print("=" * 70)


# Historical recent 30 days

recent_start = (
    last_date
    - pd.Timedelta(days=30)
)

recent = historical[
    historical["date"] > recent_start
]

historical_store_sku = (
    recent
    .groupby(
        ["store_id", "sku_id"]
    )["units_sold"]
    .sum()
    .rename("historical_30d")
)


# Forecasts

original_store_sku = (
    original
    .groupby(
        ["store_id", "sku_id"]
    )["forecast_units"]
    .sum()
    .rename("original_forecast_30d")
)

calibrated_store_sku = (
    calibrated
    .groupby(
        ["store_id", "sku_id"]
    )["calibrated_forecast_units"]
    .sum()
    .rename("calibrated_forecast_30d")
)

lightgbm_store_sku = (
    lightgbm
    .groupby(
        ["store_id", "sku_id"]
    )["forecast_units"]
    .sum()
    .rename("lightgbm_forecast_30d")
)


# Merge

store_sku = pd.concat(
    [
        historical_store_sku,
        lightgbm_store_sku,
        original_store_sku,
        calibrated_store_sku
    ],
    axis=1
).fillna(0)


# ============================================================
# ABSOLUTE ERRORS
# ============================================================

store_sku["lightgbm_abs_error"] = (
    abs(
        store_sku["lightgbm_forecast_30d"]
        - store_sku["historical_30d"]
    )
)

store_sku["original_abs_error"] = (
    abs(
        store_sku["original_forecast_30d"]
        - store_sku["historical_30d"]
    )
)

store_sku["calibrated_abs_error"] = (
    abs(
        store_sku["calibrated_forecast_30d"]
        - store_sku["historical_30d"]
    )
)


print(
    "\nStore-SKU MAE:"
)

print(
    "LightGBM:",
    store_sku["lightgbm_abs_error"].mean()
)

print(
    "Original intermittent:",
    store_sku["original_abs_error"].mean()
)

print(
    "Calibrated intermittent:",
    store_sku["calibrated_abs_error"].mean()
)


# ============================================================
# ZERO-DEMAND STORE-SKU ANALYSIS
# ============================================================

zero_mask = (
    store_sku["historical_30d"] == 0
)

zero_count = zero_mask.sum()

print("\nRecent zero-demand Store-SKU:", zero_count)

print(
    "LightGBM average forecast:",
    store_sku.loc[
        zero_mask,
        "lightgbm_forecast_30d"
    ].mean()
)

print(
    "Original intermittent average:",
    store_sku.loc[
        zero_mask,
        "original_forecast_30d"
    ].mean()
)

print(
    "Calibrated intermittent average:",
    store_sku.loc[
        zero_mask,
        "calibrated_forecast_30d"
    ].mean()
)


# ============================================================
# REGIME VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("REGIME-LEVEL VALIDATION")
print("=" * 70)


regime_columns = [
    "store_id",
    "sku_id",
    "demand_regime"
]

regime_small = regimes[
    regime_columns
].drop_duplicates()


store_sku = store_sku.reset_index()

store_sku = store_sku.merge(
    regime_small,
    on=["store_id", "sku_id"],
    how="left"
)


regime_validation = (
    store_sku
    .groupby("demand_regime")
    .agg(
        store_sku_count=(
            "sku_id",
            "count"
        ),
        historical_demand_30d=(
            "historical_30d",
            "sum"
        ),
        lightgbm_forecast_30d=(
            "lightgbm_forecast_30d",
            "sum"
        ),
        original_forecast_30d=(
            "original_forecast_30d",
            "sum"
        ),
        calibrated_forecast_30d=(
            "calibrated_forecast_30d",
            "sum"
        ),
        lightgbm_mae=(
            "lightgbm_abs_error",
            "mean"
        ),
        original_mae=(
            "original_abs_error",
            "mean"
        ),
        calibrated_mae=(
            "calibrated_abs_error",
            "mean"
        )
    )
    .reset_index()
)


print(
    regime_validation.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

summary.to_csv(
    SUMMARY_PATH,
    index=False
)

store_sku.to_csv(
    STORE_SKU_PATH,
    index=False
)

regime_validation.to_csv(
    REGIME_VALIDATION_PATH,
    index=False
)


# ============================================================
# FINAL DECISION
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION DECISION")
print("=" * 70)

original_mae = (
    store_sku["original_abs_error"].mean()
)

calibrated_mae = (
    store_sku["calibrated_abs_error"].mean()
)

lightgbm_mae = (
    store_sku["lightgbm_abs_error"].mean()
)


print(
    f"LightGBM MAE:       {lightgbm_mae:.4f}"
)

print(
    f"Original MAE:       {original_mae:.4f}"
)

print(
    f"Calibrated MAE:     {calibrated_mae:.4f}"
)


if calibrated_mae < original_mae:

    decision = (
        "CALIBRATED INTERMITTENT "
        "IMPROVES STORE-SKU MAE"
    )

else:

    decision = (
        "CALIBRATION DOES NOT IMPROVE "
        "STORE-SKU MAE"
    )


print("\nDecision:")
print(decision)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION FILES SAVED")
print("=" * 70)

print(SUMMARY_PATH)
print(STORE_SKU_PATH)
print(REGIME_VALIDATION_PATH)

print("\n" + "=" * 70)
print("PHASE 6.5 COMPLETED")
print("=" * 70)