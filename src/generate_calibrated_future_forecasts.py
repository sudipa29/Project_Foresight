# ============================================================
# PROJECT FORESIGHT
# PHASE 6.6 - CALIBRATED FUTURE FORECAST GENERATION
#
# Generates:
#   30-Day
#   60-Day
#   90-Day
#
# Using:
#   Original Intermittent Forecast
#   + Demand Regime Calibration
#
# Calibration:
#   ACTIVE        -> 1.00
#   INTERMITTENT  -> 0.75
#   DORMANT       -> 0.00
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

FORECAST_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "future"
)

INTERMITTENT_PATH = (
    FORECAST_PATH
    / "intermittent_corrected"
)

REGIME_PATH = (
    FORECAST_PATH
    / "validation"
    / "store_sku_demand_regimes.csv"
)

OUTPUT_PATH = (
    FORECAST_PATH
    / "calibrated"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

FILES = {
    30: INTERMITTENT_PATH
        / "intermittent_future_30_day_forecast.csv",

    60: INTERMITTENT_PATH
        / "intermittent_future_60_day_forecast.csv",

    90: INTERMITTENT_PATH
        / "intermittent_future_90_day_forecast.csv",
}


# ============================================================
# CALIBRATION FACTORS
# ============================================================

CALIBRATION_FACTORS = {
    "ACTIVE": 1.00,
    "INTERMITTENT": 0.75,
    "DORMANT": 0.00,
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PROJECT FORESIGHT")
print("PHASE 6.6 - CALIBRATED FUTURE FORECAST GENERATION")
print("=" * 70)


# ============================================================
# CHECK INPUTS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

for horizon, path in FILES.items():

    print(f"\nChecking {horizon}-day forecast:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{path}"
        )

    print("FOUND")


print("\nChecking demand regime file:")
print(REGIME_PATH)

if not REGIME_PATH.exists():
    raise FileNotFoundError(
        f"Missing regime file:\n{REGIME_PATH}"
    )

print("FOUND")


# ============================================================
# LOAD REGIMES
# ============================================================

print("\n" + "=" * 70)
print("LOADING DEMAND REGIMES")
print("=" * 70)

regimes = pd.read_csv(REGIME_PATH)

print("Regime shape:", regimes.shape)

required_regime_columns = [
    "store_id",
    "sku_id",
    "demand_regime",
]

missing = [
    c for c in required_regime_columns
    if c not in regimes.columns
]

if missing:
    raise ValueError(
        f"Missing regime columns: {missing}"
    )


regimes = regimes[
    required_regime_columns
].copy()


print("\nDemand regime counts:")
print(
    regimes["demand_regime"]
    .value_counts()
)


# ============================================================
# PROCESS EACH HORIZON
# ============================================================

summary_rows = []


for horizon, input_file in FILES.items():

    print("\n" + "-" * 70)
    print(f"GENERATING CALIBRATED {horizon}-DAY FORECAST")
    print("-" * 70)

    # --------------------------------------------------------
    # Load forecast
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    print("Original shape:", df.shape)

    required_columns = [
        "store_id",
        "sku_id",
        "date",
        "occurrence_probability",
        "positive_demand_quantity",
        "forecast_units",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing forecast columns: {missing}"
        )

    # --------------------------------------------------------
    # Merge demand regime
    # --------------------------------------------------------

    df = df.merge(
        regimes,
        on=["store_id", "sku_id"],
        how="left",
        validate="many_to_one"
    )

    missing_regimes = df["demand_regime"].isna().sum()

    if missing_regimes > 0:
        raise ValueError(
            f"{missing_regimes} rows have missing demand regimes"
        )

    # --------------------------------------------------------
    # Apply calibration
    # --------------------------------------------------------

    df["calibration_factor"] = (
        df["demand_regime"]
        .map(CALIBRATION_FACTORS)
    )

    if df["calibration_factor"].isna().any():

        unknown_regimes = (
            df.loc[
                df["calibration_factor"].isna(),
                "demand_regime"
            ]
            .unique()
        )

        raise ValueError(
            f"Unknown demand regimes: {unknown_regimes}"
        )

    df["calibrated_forecast_units"] = (
        df["forecast_units"]
        * df["calibration_factor"]
    )

    # --------------------------------------------------------
    # Remove tiny floating-point noise
    # --------------------------------------------------------

    df.loc[
        df["calibrated_forecast_units"] < 1e-10,
        "calibrated_forecast_units"
    ] = 0.0

    # --------------------------------------------------------
    # Output columns
    # --------------------------------------------------------

    output = df[
        [
            "store_id",
            "sku_id",
            "date",
            "demand_regime",
            "calibration_factor",
            "occurrence_probability",
            "positive_demand_quantity",
            "forecast_units",
            "calibrated_forecast_units",
        ]
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_PATH
        / f"calibrated_intermittent_{horizon}_day_forecast.csv"
    )

    output.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    original_total = (
        output["forecast_units"].sum()
    )

    calibrated_total = (
        output["calibrated_forecast_units"].sum()
    )

    difference = (
        calibrated_total
        - original_total
    )

    difference_pct = (
        difference
        / original_total
        * 100
        if original_total != 0
        else 0
    )

    zero_count = (
        output["calibrated_forecast_units"]
        == 0
    ).sum()

    positive_count = (
        output["calibrated_forecast_units"]
        > 0
    ).sum()

    print("\nOriginal forecast:",
          f"{original_total:,.2f}")

    print("Calibrated forecast:",
          f"{calibrated_total:,.2f}")

    print("Difference:",
          f"{difference:,.2f}")

    print("Difference %:",
          f"{difference_pct:.2f}%")

    print("Zero calibrated rows:",
          zero_count)

    print("Positive calibrated rows:",
          positive_count)

    print("\nSaved:")
    print(output_file)

    # --------------------------------------------------------
    # Regime summary
    # --------------------------------------------------------

    regime_summary = (
        output
        .groupby("demand_regime")
        .agg(
            store_sku_count=(
                ["store_id", "sku_id"],
                lambda x: x.drop_duplicates().shape[0]
            )
            if False else
            ("store_id", "nunique"),

            original_forecast_units=(
                "forecast_units",
                "sum"
            ),

            calibrated_forecast_units=(
                "calibrated_forecast_units",
                "sum"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Store-SKU summary
    # --------------------------------------------------------

    store_sku = (
        output
        .groupby(
            ["store_id", "sku_id", "demand_regime"]
        )
        .agg(
            original_forecast_units=(
                "forecast_units",
                "sum"
            ),
            calibrated_forecast_units=(
                "calibrated_forecast_units",
                "sum"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Add summary rows
    # --------------------------------------------------------

    summary_rows.append(
        {
            "horizon_days": horizon,
            "original_forecast_units": original_total,
            "calibrated_forecast_units": calibrated_total,
            "difference_units": difference,
            "difference_pct": difference_pct,
            "zero_forecast_rows": zero_count,
            "positive_forecast_rows": positive_count,
            "store_sku_count": store_sku.shape[0],
        }
    )


# ============================================================
# SAVE OVERALL SUMMARY
# ============================================================

summary = pd.DataFrame(summary_rows)

summary_file = (
    OUTPUT_PATH
    / "calibrated_forecast_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATED FORECAST SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

print("\nSaved:")
print(summary_file)

print("\n" + "=" * 70)
print("PHASE 6.6 COMPLETED")
print("=" * 70)