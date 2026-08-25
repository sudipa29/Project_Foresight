import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT FORESIGHT
# PAGE 9 - SKU PERFORMANCE
# ============================================================


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SKU Performance | Project Foresight",
    page_icon="🏷️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "future"
)

SKU_INVENTORY_PATH = (
    DATA_PATH
    / "inventory_recommendations"
    / "sku_inventory_summary.csv"
)

SKU_FORECAST_PATH = (
    DATA_PATH
    / "sku_future_demand_summary.csv"
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🏷️ SKU Performance")

st.caption(
    "SKU-level demand forecast, inventory position, risk and replenishment intelligence"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_sku_inventory():

    if not SKU_INVENTORY_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(SKU_INVENTORY_PATH)

    return df


@st.cache_data
def load_sku_forecast():

    if not SKU_FORECAST_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(SKU_FORECAST_PATH)

    return df


sku_df = load_sku_inventory()
forecast_df = load_sku_forecast()


# ============================================================
# VALIDATE DATA
# ============================================================

if sku_df.empty:

    st.error(
        "SKU inventory summary could not be loaded."
    )

    st.info(
        f"Expected file:\n{SKU_INVENTORY_PATH}"
    )

    st.stop()


# ============================================================
# MERGE FORECAST INFORMATION
# ============================================================

if not forecast_df.empty:

    forecast_columns = [
        "sku_id",
        "forecast_90d_units",
        "avg_daily_forecast",
        "max_daily_forecast"
    ]

    available_columns = [
        col
        for col in forecast_columns
        if col in forecast_df.columns
    ]

    forecast_df = forecast_df[available_columns]

    sku_df = sku_df.merge(
        forecast_df,
        on="sku_id",
        how="left"
    )


# ============================================================
# CLEAN DATA
# ============================================================

numeric_columns = [
    "total_stock",
    "forecast_30d",
    "forecast_60d",
    "forecast_90d",
    "suggested_reorder_qty",
    "stores",
    "high_risk_items",
    "critical_items",
    "overstock_items",
    "forecast_90d_units",
    "avg_daily_forecast",
    "max_daily_forecast"
]

for col in numeric_columns:

    if col in sku_df.columns:

        sku_df[col] = pd.to_numeric(
            sku_df[col],
            errors="coerce"
        ).fillna(0)


sku_df["sku_id"] = sku_df["sku_id"].astype(str)


# ============================================================
# DERIVED METRICS
# ============================================================

sku_df["inventory_coverage_days"] = np.where(
    sku_df["avg_daily_forecast"] > 0,
    sku_df["total_stock"] / sku_df["avg_daily_forecast"],
    np.inf
)


sku_df["forecast_growth_90_vs_30"] = np.where(
    sku_df["forecast_30d"] > 0,
    (
        sku_df["forecast_90d"]
        / sku_df["forecast_30d"]
    ),
    0
)


sku_df["inventory_to_30d_ratio"] = np.where(
    sku_df["forecast_30d"] > 0,
    sku_df["total_stock"]
    / sku_df["forecast_30d"],
    np.inf
)


# ============================================================
# SKU RISK CLASSIFICATION
# ============================================================

def classify_sku(row):

    if row.get("critical_items", 0) > 0:
        return "Critical"

    if row.get("high_risk_items", 0) > 0:
        return "High Risk"

    if row.get("overstock_items", 0) > 0:
        return "Overstock"

    if row.get("suggested_reorder_qty", 0) > 0:
        return "Reorder"

    return "Normal"


sku_df["sku_status"] = sku_df.apply(
    classify_sku,
    axis=1
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 SKU Filters")


# ------------------------------------------------------------
# SKU FILTER
# ------------------------------------------------------------

sku_options = sorted(
    sku_df["sku_id"].unique().tolist()
)

selected_skus = st.sidebar.multiselect(
    "Select SKU",
    options=sku_options,
    default=[]
)


# ------------------------------------------------------------
# STATUS FILTER
# ------------------------------------------------------------

status_options = [
    "All",
    "Critical",
    "High Risk",
    "Overstock",
    "Reorder",
    "Normal"
]

selected_status = st.sidebar.selectbox(
    "SKU Status",
    status_options
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = sku_df.copy()


if selected_skus:

    filtered_df = filtered_df[
        filtered_df["sku_id"].isin(selected_skus)
    ]


if selected_status != "All":

    filtered_df = filtered_df[
        filtered_df["sku_status"]
        == selected_status
    ]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_skus = filtered_df["sku_id"].nunique()

total_inventory = filtered_df["total_stock"].sum()

forecast_30 = filtered_df["forecast_30d"].sum()

forecast_60 = filtered_df["forecast_60d"].sum()

forecast_90 = filtered_df["forecast_90d"].sum()

total_reorder = filtered_df[
    "suggested_reorder_qty"
].sum()

overstock_skus = (
    filtered_df["overstock_items"] > 0
).sum()

high_risk_skus = (
    filtered_df["high_risk_items"] > 0
).sum()

critical_skus = (
    filtered_df["critical_items"] > 0
).sum()


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📊 SKU Portfolio Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total SKUs",
        f"{total_skus:,}"
    )


with col2:

    st.metric(
        "Current Inventory",
        f"{total_inventory:,.0f}"
    )


with col3:

    st.metric(
        "30-Day Forecast",
        f"{forecast_30:,.0f}"
    )


with col4:

    st.metric(
        "90-Day Forecast",
        f"{forecast_90:,.0f}"
    )


col5, col6, col7, col8 = st.columns(4)

with col5:

    st.metric(
        "60-Day Forecast",
        f"{forecast_60:,.0f}"
    )


with col6:

    st.metric(
        "Suggested Reorder",
        f"{total_reorder:,.0f}"
    )


with col7:

    st.metric(
        "Overstock SKUs",
        f"{overstock_skus:,}"
    )


with col8:

    st.metric(
        "High/Critical Risk",
        f"{high_risk_skus + critical_skus:,}"
    )


# ============================================================
# INVENTORY WARNING
# ============================================================

if forecast_30 > 0:

    portfolio_ratio = (
        total_inventory
        / forecast_30
    )

    st.warning(
        f"⚠️ Portfolio inventory is approximately "
        f"{portfolio_ratio:,.1f}× the 30-day forecast."
    )


# ============================================================
# SELECTED SKU DETAIL
# ============================================================

st.divider()

st.subheader("🔍 SKU Detail")


if selected_skus:

    detail_df = filtered_df.copy()

else:

    # Default to highest inventory SKUs
    detail_df = (
        filtered_df
        .sort_values(
            "total_stock",
            ascending=False
        )
        .head(20)
    )


# ============================================================
# SKU DETAIL TABLE
# ============================================================

display_columns = [
    "sku_id",
    "total_stock",
    "forecast_30d",
    "forecast_60d",
    "forecast_90d",
    "suggested_reorder_qty",
    "stores",
    "overstock_items",
    "high_risk_items",
    "critical_items",
    "sku_status"
]

display_columns = [
    col
    for col in display_columns
    if col in detail_df.columns
]


display_df = detail_df[
    display_columns
].copy()


display_df = display_df.rename(
    columns={
        "sku_id": "SKU",
        "total_stock": "Current Inventory",
        "forecast_30d": "30D Forecast",
        "forecast_60d": "60D Forecast",
        "forecast_90d": "90D Forecast",
        "suggested_reorder_qty": "Reorder Qty",
        "stores": "Stores",
        "overstock_items": "Overstock Items",
        "high_risk_items": "High Risk Items",
        "critical_items": "Critical Items",
        "sku_status": "Status"
    }
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CHARTS
# ============================================================

st.divider()

st.subheader("📈 Forecast Profile")


chart_df = (
    filtered_df
    .groupby("sku_id", as_index=False)[
        [
            "forecast_30d",
            "forecast_60d",
            "forecast_90d"
        ]
    ]
    .sum()
)


if not chart_df.empty:

    chart_df = (
        chart_df
        .sort_values(
            "forecast_90d",
            ascending=False
        )
        .head(20)
    )

    chart_df = chart_df.set_index("sku_id")

    st.bar_chart(
        chart_df[
            [
                "forecast_30d",
                "forecast_60d",
                "forecast_90d"
            ]
        ],
        use_container_width=True
    )


# ============================================================
# INVENTORY VS FORECAST
# ============================================================

st.subheader("📦 Inventory vs 30-Day Forecast")


inventory_chart = (
    filtered_df[
        [
            "sku_id",
            "total_stock",
            "forecast_30d"
        ]
    ]
    .sort_values(
        "total_stock",
        ascending=False
    )
    .head(20)
    .set_index("sku_id")
)


if not inventory_chart.empty:

    inventory_chart = inventory_chart.rename(
        columns={
            "total_stock": "Current Inventory",
            "forecast_30d": "30-Day Forecast"
        }
    )

    st.bar_chart(
        inventory_chart,
        use_container_width=True
    )


# ============================================================
# TOP PERFORMING / HIGH DEMAND SKUs
# ============================================================

st.divider()

col_left, col_right = st.columns(2)


with col_left:

    st.subheader("🔥 Highest Forecast SKUs")

    top_skus = (
        filtered_df[
            [
                "sku_id",
                "forecast_90d"
            ]
        ]
        .sort_values(
            "forecast_90d",
            ascending=False
        )
        .head(10)
    )

    top_skus = top_skus.rename(
        columns={
            "sku_id": "SKU",
            "forecast_90d": "90-Day Forecast"
        }
    )

    st.dataframe(
        top_skus,
        use_container_width=True,
        hide_index=True
    )


with col_right:

    st.subheader("⚠️ Highest Inventory SKUs")

    inventory_skus = (
        filtered_df[
            [
                "sku_id",
                "total_stock",
                "forecast_30d"
            ]
        ]
        .sort_values(
            "total_stock",
            ascending=False
        )
        .head(10)
    )

    inventory_skus = inventory_skus.rename(
        columns={
            "sku_id": "SKU",
            "total_stock": "Inventory",
            "forecast_30d": "30-Day Forecast"
        }
    )

    st.dataframe(
        inventory_skus,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# OVERSTOCK ANALYSIS
# ============================================================

st.divider()

st.subheader("🚨 Overstock Analysis")


overstock_df = filtered_df[
    filtered_df["overstock_items"] > 0
].copy()


if not overstock_df.empty:

    overstock_df["inventory_to_30d_ratio"] = np.where(
        overstock_df["forecast_30d"] > 0,
        overstock_df["total_stock"]
        / overstock_df["forecast_30d"],
        np.inf
    )


    overstock_df = (
        overstock_df
        .sort_values(
            "inventory_to_30d_ratio",
            ascending=False
        )
        .head(20)
    )


    overstock_display = overstock_df[
        [
            "sku_id",
            "total_stock",
            "forecast_30d",
            "inventory_to_30d_ratio",
            "overstock_items"
        ]
    ].copy()


    overstock_display = overstock_display.rename(
        columns={
            "sku_id": "SKU",
            "total_stock": "Inventory",
            "forecast_30d": "30-Day Forecast",
            "inventory_to_30d_ratio": "Inventory / 30D Forecast",
            "overstock_items": "Overstock Items"
        }
    )


    st.dataframe(
        overstock_display,
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "No overstocked SKUs found for the current filter."
    )


# ============================================================
# REORDER ANALYSIS
# ============================================================

st.subheader("🔄 Reorder Recommendations")


reorder_df = filtered_df[
    filtered_df["suggested_reorder_qty"] > 0
].copy()


if not reorder_df.empty:

    reorder_display = reorder_df[
        [
            "sku_id",
            "suggested_reorder_qty",
            "forecast_30d",
            "forecast_60d",
            "forecast_90d",
            "stores"
        ]
    ].sort_values(
        "suggested_reorder_qty",
        ascending=False
    ).head(20)


    reorder_display = reorder_display.rename(
        columns={
            "sku_id": "SKU",
            "suggested_reorder_qty": "Suggested Reorder",
            "forecast_30d": "30-Day Forecast",
            "forecast_60d": "60-Day Forecast",
            "forecast_90d": "90-Day Forecast",
            "stores": "Stores"
        }
    )


    st.dataframe(
        reorder_display,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No SKU-level replenishment is currently recommended."
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.divider()

st.subheader("💡 SKU Business Insights")


if not filtered_df.empty:

    highest_forecast_sku = (
        filtered_df
        .sort_values(
            "forecast_90d",
            ascending=False
        )
        .iloc[0]
    )


    highest_inventory_sku = (
        filtered_df
        .sort_values(
            "total_stock",
            ascending=False
        )
        .iloc[0]
    )


    st.write(
        f"**Highest projected demand:** "
        f"SKU {highest_forecast_sku['sku_id']} "
        f"with approximately "
        f"{highest_forecast_sku['forecast_90d']:,.0f} units "
        f"forecast over 90 days."
    )


    st.write(
        f"**Highest inventory exposure:** "
        f"SKU {highest_inventory_sku['sku_id']} "
        f"with approximately "
        f"{highest_inventory_sku['total_stock']:,.0f} units "
        f"currently held."
    )


    if overstock_skus > 0:

        st.write(
            f"**Inventory efficiency:** "
            f"{overstock_skus:,} SKU(s) have associated "
            f"overstock inventory positions."
        )


    if total_reorder > 0:

        st.write(
            f"**Replenishment:** "
            f"{total_reorder:,.0f} units are currently "
            f"recommended for replenishment."
        )

    else:

        st.write(
            "**Replenishment:** No additional SKU-level "
            "replenishment is currently recommended."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Project FORESIGHT | SKU Performance & Inventory Intelligence"
)