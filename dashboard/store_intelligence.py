# ============================================================
# PROJECT FORESIGHT
# Store Intelligence Dashboard
#
# Purpose:
# Store-level inventory, demand, coverage and risk analysis
#
# Important:
# Store-level coverage is calculated AFTER aggregation.
# This prevents SKU-level infinite coverage values caused
# by zero-demand Store-SKU combinations from corrupting
# store-level averages.
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.title("🏬 Store Intelligence")

st.caption(
    "Store-level inventory, demand coverage, overstock risk "
    "and network inventory concentration using the calibrated "
    "forecasting pipeline."
)


# ============================================================
# BASE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

INTEGRATION_DIR = os.path.join(
    PROCESSED_DIR,
    "forecasting",
    "integration"
)

INTEGRATED_FILE = os.path.join(
    INTEGRATION_DIR,
    "calibrated_forecast_inventory_integrated.csv"
)


# ============================================================
# SAFE DATA LOADER
# ============================================================

@st.cache_data
def load_inventory_data(path):

    if not os.path.exists(path):

        return pd.DataFrame()

    try:

        df = pd.read_csv(path)

        return df

    except Exception as e:

        st.error(
            f"Unable to load inventory integration data: {e}"
        )

        return pd.DataFrame()


df = load_inventory_data(
    INTEGRATED_FILE
)


# ============================================================
# DATA VALIDATION
# ============================================================

if df.empty:

    st.error(
        "Calibrated inventory integration dataset was not found."
    )

    st.stop()


required_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
    "units_30d",
    "units_90d"
]


missing_columns = [
    c for c in required_columns
    if c not in df.columns
]


if missing_columns:

    st.error(
        "Required columns are missing from the integration dataset:"
    )

    st.code(
        ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# NUMERIC CLEANING
# ============================================================

numeric_columns = [
    "stock_on_hand",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
    "units_30d",
    "units_90d"
]


for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)


# Prevent negative forecast values from affecting
# business interpretation.

forecast_columns = [
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d"
]


for col in forecast_columns:

    df[col] = df[col].clip(
        lower=0
    )


# ============================================================
# STORE-LEVEL AGGREGATION
# ============================================================
#
# IMPORTANT:
#
# We DO NOT calculate store coverage by averaging the
# Store-SKU coverage_days column.
#
# Instead:
#
# Store Coverage =
# Total Store Inventory /
# (Total Store 30-Day Forecast / 30)
#
# This prevents zero-forecast SKU combinations from
# producing infinity.
# ============================================================

store_summary = (
    df
    .groupby("store_id", as_index=False)
    .agg(
        inventory=(
            "stock_on_hand",
            "sum"
        ),

        forecast_30d=(
            "calibrated_forecast_30d",
            "sum"
        ),

        forecast_60d=(
            "calibrated_forecast_60d",
            "sum"
        ),

        forecast_90d=(
            "calibrated_forecast_90d",
            "sum"
        ),

        units_30d=(
            "units_30d",
            "sum"
        ),

        units_90d=(
            "units_90d",
            "sum"
        ),

        store_sku_count=(
            "sku_id",
            "count"
        )
    )
)


# ============================================================
# STORE-LEVEL CALCULATIONS
# ============================================================

# Inventory / 30-day forecast

store_summary["inventory_to_forecast"] = np.where(

    store_summary["forecast_30d"] > 0,

    store_summary["inventory"]
    /
    store_summary["forecast_30d"],

    np.nan
)


# Inventory coverage in days

store_summary["coverage_days"] = np.where(

    store_summary["forecast_30d"] > 0,

    store_summary["inventory"]
    /
    (
        store_summary["forecast_30d"]
        / 30
    ),

    np.nan
)


# 60-day coverage

store_summary["coverage_60d"] = np.where(

    store_summary["forecast_60d"] > 0,

    store_summary["inventory"]
    /
    (
        store_summary["forecast_60d"]
        / 60
    ),

    np.nan
)


# 90-day coverage

store_summary["coverage_90d"] = np.where(

    store_summary["forecast_90d"] > 0,

    store_summary["inventory"]
    /
    (
        store_summary["forecast_90d"]
        / 90
    ),

    np.nan
)


# Historical demand vs forecast

store_summary["historical_to_forecast"] = np.where(

    store_summary["forecast_30d"] > 0,

    store_summary["units_30d"]
    /
    store_summary["forecast_30d"],

    np.nan
)


# ============================================================
# COUNT DORMANT / ZERO-FORECAST SKU COMBINATIONS
# ============================================================

dormant_mask = (
    df["calibrated_forecast_30d"]
    <= 0
)


dormant_summary = (
    df.loc[dormant_mask]
    .groupby(
        "store_id",
        as_index=False
    )
    .agg(
        no_forecast_sku_count=(
            "sku_id",
            "count"
        ),

        no_forecast_inventory=(
            "stock_on_hand",
            "sum"
        )
    )
)


store_summary = store_summary.merge(
    dormant_summary,
    on="store_id",
    how="left"
)


store_summary[
    "no_forecast_sku_count"
] = (
    store_summary[
        "no_forecast_sku_count"
    ]
    .fillna(0)
)


store_summary[
    "no_forecast_inventory"
] = (
    store_summary[
        "no_forecast_inventory"
    ]
    .fillna(0)
)


# ============================================================
# STORE RISK CLASSIFICATION
# ============================================================
#
# Coverage thresholds:
#
# < 30 days       = Healthy
# 30-90 days      = Elevated
# 90-180 days     = High
# 180-365 days    = Overstock
# >365 days       = Severe Overstock
#
# These are business interpretation thresholds.
# ============================================================

def classify_store_risk(days):

    if pd.isna(days):

        return "No Forecast"

    if days <= 30:

        return "Healthy"

    elif days <= 90:

        return "Elevated"

    elif days <= 180:

        return "High"

    elif days <= 365:

        return "Overstock"

    else:

        return "Severe Overstock"


store_summary["inventory_risk"] = (
    store_summary[
        "coverage_days"
    ]
    .apply(classify_store_risk)
)


# ============================================================
# STORE RANKING
# ============================================================

store_summary["inventory_rank"] = (
    store_summary[
        "inventory"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)


store_summary["forecast_rank"] = (
    store_summary[
        "forecast_30d"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)


store_summary["coverage_rank"] = (
    store_summary[
        "coverage_days"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)


# ============================================================
# CLEAN INFINITE VALUES
# ============================================================

store_summary = (
    store_summary
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


# ============================================================
# NETWORK KPIs
# ============================================================

total_inventory = (
    store_summary["inventory"]
    .sum()
)


total_forecast_30d = (
    store_summary["forecast_30d"]
    .sum()
)


total_forecast_60d = (
    store_summary["forecast_60d"]
    .sum()
)


total_forecast_90d = (
    store_summary["forecast_90d"]
    .sum()
)


network_inventory_ratio = np.nan

if total_forecast_30d > 0:

    network_inventory_ratio = (
        total_inventory
        /
        total_forecast_30d
    )


network_coverage_days = np.nan

if total_forecast_30d > 0:

    network_coverage_days = (
        total_inventory
        /
        (
            total_forecast_30d
            / 30
        )
    )


severe_store_count = (
    store_summary[
        "inventory_risk"
    ]
    .eq("Severe Overstock")
    .sum()
)


overstock_store_count = (
    store_summary[
        "inventory_risk"
    ]
    .eq("Overstock")
    .sum()
)


no_forecast_store_count = (
    store_summary[
        "no_forecast_sku_count"
    ]
    .gt(0)
    .sum()
)


total_dormant_inventory = (
    store_summary[
        "no_forecast_inventory"
    ]
    .sum()
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    ### 🏬 Network Store Position

    Store-level inventory and demand are aggregated from the
    calibrated production forecast. Coverage is calculated
    using total store inventory divided by total store
    30-day forecast demand.
    """
)


# ============================================================
# KPI ROW 1
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Current Inventory",
        f"{total_inventory:,.0f}"
    )


with col2:

    st.metric(
        "30-Day Forecast",
        f"{total_forecast_30d:,.0f}"
    )


with col3:

    st.metric(
        "Inventory / 30D Forecast",
        (
            f"{network_inventory_ratio:,.2f}×"
            if pd.notna(network_inventory_ratio)
            else "N/A"
        )
    )


with col4:

    st.metric(
        "Network Coverage",
        (
            f"{network_coverage_days:,.0f} days"
            if pd.notna(network_coverage_days)
            else "N/A"
        )
    )


# ============================================================
# KPI ROW 2
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "60-Day Forecast",
        f"{total_forecast_60d:,.0f}"
    )


with col2:

    st.metric(
        "90-Day Forecast",
        f"{total_forecast_90d:,.0f}"
    )


with col3:

    st.metric(
        "Severe Overstock Stores",
        f"{severe_store_count:,}"
    )


with col4:

    st.metric(
        "Stores With Dormant SKUs",
        f"{no_forecast_store_count:,}"
    )


# ============================================================
# CRITICAL ALERT
# ============================================================

st.divider()


if (
    pd.notna(network_inventory_ratio)
    and network_inventory_ratio > 10
):

    st.error(
        f"""
        🚨 **Severe Inventory Imbalance**

        Current network inventory is approximately
        **{network_inventory_ratio:,.2f}×**
        the calibrated 30-day forecast.

        Network inventory coverage is approximately
        **{network_coverage_days:,.0f} days**.

        Management focus should be on inventory reduction,
        replenishment control, dormant-stock review and
        redistribution analysis rather than additional
        purchasing.
        """
    )

elif (
    pd.notna(network_inventory_ratio)
    and network_inventory_ratio > 3
):

    st.warning(
        f"""
        ⚠️ **Elevated Inventory Position**

        Current inventory is approximately
        **{network_inventory_ratio:,.2f}×**
        the 30-day calibrated forecast.
        """
    )

else:

    st.success(
        "Inventory appears broadly aligned with near-term demand."
    )


# ============================================================
# STORE INVENTORY VS FORECAST
# ============================================================

st.divider()

st.subheader(
    "📦 Store Inventory vs 30-Day Forecast"
)


inventory_chart = store_summary.copy()


fig = px.bar(
    inventory_chart,
    x="store_id",
    y=[
        "inventory",
        "forecast_30d"
    ],
    barmode="group",
    title="Store Inventory vs Calibrated 30-Day Forecast",
    labels={
        "store_id": "Store",
        "value": "Units",
        "variable": "Metric"
    }
)


fig.update_layout(
    xaxis=dict(
        type="category"
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# INVENTORY COVERAGE CHART
# ============================================================

st.subheader(
    "⏱️ Store Inventory Coverage"
)


coverage_chart = (
    store_summary
    .dropna(
        subset=[
            "coverage_days"
        ]
    )
    .sort_values(
        "coverage_days",
        ascending=False
    )
)


fig = px.bar(
    coverage_chart,
    x="store_id",
    y="coverage_days",
    title="Store Inventory Coverage — Days",
    labels={
        "store_id": "Store",
        "coverage_days": "Coverage Days"
    },
    hover_data=[
        "inventory",
        "forecast_30d",
        "inventory_to_forecast"
    ]
)


fig.add_hline(
    y=365,
    line_dash="dash",
    annotation_text="365-Day Threshold"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader(
    "⚠️ Store Inventory Risk Distribution"
)


risk_distribution = (
    store_summary[
        "inventory_risk"
    ]
    .value_counts()
    .reset_index()
)


risk_distribution.columns = [
    "inventory_risk",
    "store_count"
]


fig = px.bar(
    risk_distribution,
    x="inventory_risk",
    y="store_count",
    text="store_count",
    title="Store Count by Inventory Risk"
)


fig.update_traces(
    textposition="outside"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# TOP INVENTORY STORES
# ============================================================

st.divider()

st.subheader(
    "🏆 Highest Inventory Stores"
)


top_inventory = (
    store_summary
    .sort_values(
        "inventory",
        ascending=False
    )
    .head(10)
)


display_columns = [
    "store_id",
    "inventory",
    "forecast_30d",
    "inventory_to_forecast",
    "coverage_days",
    "inventory_risk"
]


st.dataframe(
    top_inventory[
        display_columns
    ].round(2),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# TOP FORECAST STORES
# ============================================================

st.subheader(
    "📈 Highest Forecast Demand Stores"
)


top_forecast = (
    store_summary
    .sort_values(
        "forecast_30d",
        ascending=False
    )
    .head(10)
)


st.dataframe(
    top_forecast[
        display_columns
    ].round(2),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# HIGHEST COVERAGE STORES
# ============================================================

st.subheader(
    "🚨 Highest Inventory Coverage Stores"
)


top_coverage = (
    store_summary
    .dropna(
        subset=[
            "coverage_days"
        ]
    )
    .sort_values(
        "coverage_days",
        ascending=False
    )
    .head(10)
)


coverage_columns = [
    "store_id",
    "inventory",
    "forecast_30d",
    "inventory_to_forecast",
    "coverage_days",
    "no_forecast_sku_count",
    "no_forecast_inventory",
    "inventory_risk"
]


st.dataframe(
    top_coverage[
        coverage_columns
    ].round(2),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DORMANT INVENTORY
# ============================================================

st.divider()

st.subheader(
    "💤 Dormant Inventory Risk"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Store-SKU With Zero Forecast",
        f"{int(dormant_mask.sum()):,}"
    )


with col2:

    st.metric(
        "Inventory in Zero-Forecast Items",
        f"{total_dormant_inventory:,.0f}"
    )


with col3:

    st.metric(
        "Stores Affected",
        f"{no_forecast_store_count:,}"
    )


st.info(
    """
    Zero-forecast Store-SKU combinations should not be
    interpreted as automatically worthless inventory.

    They represent combinations where the current calibrated
    30-day forecast is zero. These items should be reviewed
    for dormant demand, slow movement, discontinuation,
    transfer, markdown or forecast-model limitations.
    """
)


# ============================================================
# STORE INVENTORY CONCENTRATION
# ============================================================

st.subheader(
    "🏬 Store Inventory Concentration"
)


concentration = (
    store_summary[
        [
            "store_id",
            "inventory"
        ]
    ]
    .sort_values(
        "inventory",
        ascending=False
    )
)


fig = px.pie(
    concentration.head(10),
    names="store_id",
    values="inventory",
    title="Inventory Concentration — Top 10 Stores"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# STORE DEMAND CONCENTRATION
# ============================================================

st.subheader(
    "📊 Store Forecast Concentration"
)


forecast_concentration = (
    store_summary[
        [
            "store_id",
            "forecast_30d"
        ]
    ]
    .sort_values(
        "forecast_30d",
        ascending=False
    )
    .head(10)
)


fig = px.bar(
    forecast_concentration,
    x="store_id",
    y="forecast_30d",
    text="forecast_30d",
    title="Top 10 Stores by 30-Day Forecast"
)


fig.update_traces(
    texttemplate="%{text:.1f}",
    textposition="outside"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# STORE-LEVEL MANAGEMENT VIEW
# ============================================================

st.divider()

st.subheader(
    "🎯 Store-Level Management View"
)


management_columns = [
    "store_id",
    "inventory",
    "forecast_30d",
    "inventory_to_forecast",
    "coverage_days",
    "no_forecast_sku_count",
    "no_forecast_inventory",
    "inventory_risk"
]


management_df = (
    store_summary[
        management_columns
    ]
    .sort_values(
        [
            "coverage_days",
            "inventory"
        ],
        ascending=[
            False,
            False
        ]
    )
)


st.dataframe(
    management_df.round(2),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MANAGEMENT INTERPRETATION
# ============================================================

st.divider()

st.subheader(
    "💡 Store Intelligence Interpretation"
)


highest_inventory_store = (
    store_summary
    .sort_values(
        "inventory",
        ascending=False
    )
    .iloc[0]
)


highest_forecast_store = (
    store_summary
    .sort_values(
        "forecast_30d",
        ascending=False
    )
    .iloc[0]
)


highest_coverage_store = (
    store_summary
    .dropna(
        subset=[
            "coverage_days"
        ]
    )
    .sort_values(
        "coverage_days",
        ascending=False
    )
    .iloc[0]
)


st.markdown(
    f"""
### 🚨 1. Inventory Position

The network currently holds approximately
**{total_inventory:,.0f} units** of inventory against
a calibrated 30-day forecast of approximately
**{total_forecast_30d:,.0f} units**.

This represents approximately
**{network_inventory_ratio:,.2f}×**
the 30-day forecast.


### 📦 2. Store With Highest Inventory

**Store {int(highest_inventory_store["store_id"])}**
currently has the highest inventory at approximately
**{highest_inventory_store["inventory"]:,.0f} units**.

Its calibrated 30-day forecast is approximately
**{highest_inventory_store["forecast_30d"]:,.2f} units**.


### 📈 3. Highest Demand Store

**Store {int(highest_forecast_store["store_id"])}**
has the highest calibrated 30-day forecast at approximately
**{highest_forecast_store["forecast_30d"]:,.2f} units**.

Even high-demand stores should be evaluated against their
existing inventory position before considering additional
replenishment.


### 🚨 4. Highest Coverage Store

**Store {int(highest_coverage_store["store_id"])}**
has the highest finite inventory coverage at approximately
**{highest_coverage_store["coverage_days"]:,.0f} days**.

This calculation is based on aggregated store inventory
and aggregated store-level forecast, avoiding infinite
coverage caused by zero-demand SKUs.


### 💤 5. Dormant Inventory

Approximately
**{int(dormant_mask.sum()):,} Store-SKU combinations**
have zero calibrated 30-day forecast.

These combinations should be reviewed separately rather
than allowing zero-demand records to distort store-level
coverage metrics.


### 🎯 6. Management Priority

The current store-level evidence supports the following
management sequence:

**Control replenishment → Review severe overstock →
Investigate dormant inventory → Evaluate redistribution →
Consider markdown / liquidation strategies →
Monitor actual demand → Recalculate inventory position.**
"""
)


# ============================================================
# DATA QUALITY CHECK
# ============================================================

st.divider()

st.subheader(
    "🔍 Store Intelligence Data Quality"
)


quality_df = pd.DataFrame(
    {
        "Metric": [
            "Integrated Store-SKU Rows",
            "Stores",
            "Total Inventory",
            "Positive Forecast Store-SKU",
            "Zero Forecast Store-SKU",
            "Store Coverage Infinite Values",
            "Store Coverage Missing Values"
        ],

        "Value": [
            len(df),
            store_summary["store_id"].nunique(),
            total_inventory,
            int(
                (
                    df[
                        "calibrated_forecast_30d"
                    ] > 0
                ).sum()
            ),
            int(
                (
                    df[
                        "calibrated_forecast_30d"
                    ] <= 0
                ).sum()
            ),
            int(
                np.isinf(
                    store_summary[
                        "coverage_days"
                    ]
                ).sum()
            ),
            int(
                store_summary[
                    "coverage_days"
                ].isna().sum()
            )
        ]
    }
)


st.dataframe(
    quality_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SOURCE DATA
# ============================================================

with st.expander(
    "🔍 View Source Integration Data"
):

    st.write(
        f"Source: {INTEGRATED_FILE}"
    )

    st.dataframe(
        df.head(1000),
        use_container_width=True,
        hide_index=True
    )