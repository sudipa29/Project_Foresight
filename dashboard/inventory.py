# ============================================================
# PROJECT FORESIGHT
# PAGE 6 - INVENTORY INTELLIGENCE
#
# Phase 6.3 Production Inventory Dashboard
#
# Uses:
# calibrated_forecast_inventory_integrated.csv
# calibrated_forecast_inventory_integration_summary.csv
# business_inventory_executive_summary.csv
# inventory_business_actions.csv
#
# ============================================================

import os
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


INTEGRATION_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "forecasting",
    "integration"
)


BUSINESS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "forecasting",
    "business_insights"
)


INTEGRATED_FILE = os.path.join(
    INTEGRATION_DIR,
    "calibrated_forecast_inventory_integrated.csv"
)


INTEGRATION_SUMMARY_FILE = os.path.join(
    INTEGRATION_DIR,
    "calibrated_forecast_inventory_integration_summary.csv"
)


EXECUTIVE_SUMMARY_FILE = os.path.join(
    BUSINESS_DIR,
    "business_inventory_executive_summary.csv"
)


ACTIONS_FILE = os.path.join(
    BUSINESS_DIR,
    "inventory_business_actions.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_inventory_data():

    inventory = pd.read_csv(
        INTEGRATED_FILE
    )

    integration_summary = pd.read_csv(
        INTEGRATION_SUMMARY_FILE
    )

    executive_summary = pd.read_csv(
        EXECUTIVE_SUMMARY_FILE
    )

    actions = pd.read_csv(
        ACTIONS_FILE
    )

    return (
        inventory,
        integration_summary,
        executive_summary,
        actions
    )


# ============================================================
# ERROR HANDLING
# ============================================================

try:

    (
        inventory,
        integration_summary,
        executive_summary,
        actions
    ) = load_inventory_data()

except Exception as e:

    st.error(
        "❌ Inventory Intelligence data could not be loaded."
    )

    st.code(
        str(e)
    )

    st.info(
        f"""
        Expected files:

        {INTEGRATED_FILE}

        {INTEGRATION_SUMMARY_FILE}

        {EXECUTIVE_SUMMARY_FILE}

        {ACTIONS_FILE}
        """
    )

    st.stop()


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "📦 Inventory Intelligence"
)

st.caption(
    "Inventory health, demand coverage, replenishment risk "
    "and excess-stock analysis using the calibrated forecast."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🔎 Inventory Filters"
)


# ============================================================
# STORE FILTER
# ============================================================

store_options = sorted(
    inventory["store_id"].unique()
)


selected_stores = st.sidebar.multiselect(
    "Select Store(s)",
    options=store_options,
    default=[]
)


# ============================================================
# SKU FILTER
# ============================================================

sku_options = sorted(
    inventory["sku_id"].unique()
)


selected_skus = st.sidebar.multiselect(
    "Select SKU(s)",
    options=sku_options,
    default=[]
)


# ============================================================
# INVENTORY STATUS FILTER
# ============================================================

status_options = sorted(
    inventory[
        "calibrated_inventory_status"
    ]
    .dropna()
    .unique()
)


selected_status = st.sidebar.multiselect(
    "Inventory Status",
    options=status_options,
    default=[]
)


# ============================================================
# PRIORITY FILTER
# ============================================================

priority_options = sorted(
    inventory[
        "priority"
    ]
    .dropna()
    .unique()
)


selected_priority = st.sidebar.multiselect(
    "Priority",
    options=priority_options,
    default=[]
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = inventory.copy()


if selected_stores:

    filtered_df = filtered_df[
        filtered_df["store_id"].isin(
            selected_stores
        )
    ]


if selected_skus:

    filtered_df = filtered_df[
        filtered_df["sku_id"].isin(
            selected_skus
        )
    ]


if selected_status:

    filtered_df = filtered_df[
        filtered_df[
            "calibrated_inventory_status"
        ].isin(
            selected_status
        )
    ]


if selected_priority:

    filtered_df = filtered_df[
        filtered_df[
            "priority"
        ].isin(
            selected_priority
        )
    ]


# ============================================================
# EMPTY CHECK
# ============================================================

if filtered_df.empty:

    st.warning(
        "No inventory records match the selected filters."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_inventory = (
    filtered_df[
        "stock_on_hand"
    ].sum()
)


forecast_30 = (
    filtered_df[
        "calibrated_forecast_30d"
    ].sum()
)


forecast_60 = (
    filtered_df[
        "calibrated_forecast_60d"
    ].sum()
)


forecast_90 = (
    filtered_df[
        "calibrated_forecast_90d"
    ].sum()
)


inventory_forecast_ratio = (
    total_inventory / forecast_30
    if forecast_30 > 0
    else float("inf")
)


reorder_quantity = (
    filtered_df[
        "suggested_reorder_qty"
    ].sum()
)


# ============================================================
# HEADER
# ============================================================

st.subheader(
    "📊 Inventory & Demand Position"
)


# ============================================================
# KPI ROW 1
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Current Inventory",
        f"{total_inventory:,.0f}"
    )


with c2:

    st.metric(
        "30-Day Forecast",
        f"{forecast_30:,.0f}"
    )


with c3:

    st.metric(
        "60-Day Forecast",
        f"{forecast_60:,.0f}"
    )


with c4:

    st.metric(
        "90-Day Forecast",
        f"{forecast_90:,.0f}"
    )


# ============================================================
# KPI ROW 2
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    if inventory_forecast_ratio != float("inf"):

        value = (
            f"{inventory_forecast_ratio:,.2f}×"
        )

    else:

        value = "N/A"


    st.metric(
        "Inventory / 30D Forecast",
        value
    )


with c2:

    st.metric(
        "Reorder Quantity",
        f"{reorder_quantity:,.0f}"
    )


with c3:

    overstock_count = (
        filtered_df[
            "calibrated_inventory_status"
        ]
        .eq("OVERSTOCK")
        .sum()
    )

    st.metric(
        "Overstock Store-SKU",
        f"{overstock_count:,}"
    )


with c4:

    no_forecast_count = (
        filtered_df[
            "calibrated_avg_daily_forecast_30d"
        ]
        .fillna(0)
        .eq(0)
        .sum()
    )

    st.metric(
        "No-Forecast Store-SKU",
        f"{no_forecast_count:,}"
    )


st.divider()


# ============================================================
# INVENTORY COVERAGE WARNING
# ============================================================

coverage_days = (
    filtered_df[
        "calibrated_inventory_coverage_days"
    ]
    .replace(
        [float("inf"), -float("inf")],
        pd.NA
    )
    .dropna()
)


if not coverage_days.empty:

    median_coverage = coverage_days.median()

else:

    median_coverage = 0


st.warning(
    f"""
    ⚠️ **Inventory Coverage Alert**

    Current inventory substantially exceeds near-term
    calibrated demand in the integrated inventory set.

    The filtered inventory currently represents approximately
    **{inventory_forecast_ratio:,.2f}×** of the 30-day calibrated
    forecast.

    Median calibrated inventory coverage is approximately
    **{median_coverage:,.0f} days**.

    This indicates a significant excess-inventory position
    under the current forecast scenario.
    """
)


# ============================================================
# INVENTORY VS FORECAST
# ============================================================

st.subheader(
    "📈 Inventory vs Forecast Demand"
)


comparison_df = pd.DataFrame(
    {
        "Horizon": [
            "30 Days",
            "60 Days",
            "90 Days"
        ],
        "Forecast Demand": [
            forecast_30,
            forecast_60,
            forecast_90
        ]
    }
)


fig_forecast = px.bar(
    comparison_df,
    x="Horizon",
    y="Forecast Demand",
    text="Forecast Demand",
    title="Calibrated Forecast Demand by Horizon"
)


fig_forecast.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)


fig_forecast.update_layout(
    height=420
)


st.plotly_chart(
    fig_forecast,
    use_container_width=True
)


# ============================================================
# INVENTORY COVERAGE
# ============================================================

st.subheader(
    "⏱️ Inventory Coverage Analysis"
)


coverage_df = (
    filtered_df[
        [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "calibrated_inventory_coverage_days",
            "calibrated_inventory_status"
        ]
    ]
    .copy()
)


coverage_df = coverage_df.replace(
    [float("inf"), -float("inf")],
    pd.NA
)


coverage_df = coverage_df.dropna(
    subset=[
        "calibrated_inventory_coverage_days"
    ]
)


top_coverage = (
    coverage_df
    .sort_values(
        "calibrated_inventory_coverage_days",
        ascending=False
    )
    .head(20)
    .sort_values(
        "calibrated_inventory_coverage_days"
    )
)


fig_coverage = px.bar(
    top_coverage,
    x="calibrated_inventory_coverage_days",
    y=top_coverage.apply(
        lambda x:
        f"Store {x['store_id']} / SKU {x['sku_id']}",
        axis=1
    ),
    orientation="h",
    text="calibrated_inventory_coverage_days",
    title="Highest Inventory Coverage Store-SKU",
    labels={
        "calibrated_inventory_coverage_days":
            "Coverage Days",
        "y":
            "Store / SKU"
    }
)


fig_coverage.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)


fig_coverage.update_layout(
    height=650
)


st.plotly_chart(
    fig_coverage,
    use_container_width=True
)


st.divider()


# ============================================================
# INVENTORY STATUS DISTRIBUTION
# ============================================================

st.subheader(
    "⚠️ Inventory Status Distribution"
)


status_df = (
    filtered_df[
        "calibrated_inventory_status"
    ]
    .fillna(
        "Unknown"
    )
    .value_counts()
    .reset_index()
)


status_df.columns = [
    "Inventory Status",
    "Store-SKU Count"
]


col1, col2 = st.columns(2)


with col1:

    fig_status = px.bar(
        status_df,
        x="Inventory Status",
        y="Store-SKU Count",
        text="Store-SKU Count",
        title="Inventory Status"
    )

    fig_status.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_status,
        use_container_width=True
    )


with col2:

    fig_status_pie = px.pie(
        status_df,
        names="Inventory Status",
        values="Store-SKU Count",
        title="Inventory Status Share"
    )

    st.plotly_chart(
        fig_status_pie,
        use_container_width=True
    )


# ============================================================
# STOCKOUT RISK
# ============================================================

st.subheader(
    "🚨 Stockout Risk"
)


risk_df = (
    filtered_df[
        "calibrated_stockout_risk_30d"
    ]
    .fillna(
        "Unknown"
    )
    .value_counts()
    .reset_index()
)


risk_df.columns = [
    "Stockout Risk",
    "Store-SKU Count"
]


fig_risk = px.bar(
    risk_df,
    x="Stockout Risk",
    y="Store-SKU Count",
    text="Store-SKU Count",
    title="30-Day Calibrated Stockout Risk"
)


fig_risk.update_traces(
    textposition="outside"
)


st.plotly_chart(
    fig_risk,
    use_container_width=True
)


# ============================================================
# STORE ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "🏬 Store Inventory Analysis"
)


store_summary = (
    filtered_df
    .groupby(
        "store_id",
        as_index=False
    )
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
        store_sku_count=(
            "sku_id",
            "nunique"
        )
    )
)


store_summary[
    "inventory_to_30d_forecast"
] = (
    store_summary[
        "inventory"
    ]
    /
    store_summary[
        "forecast_30d"
    ].replace(
        0,
        pd.NA
    )
)


top_store_inventory = (
    store_summary
    .sort_values(
        "inventory",
        ascending=False
    )
    .head(20)
)


fig_store = px.bar(
    top_store_inventory.sort_values(
        "inventory"
    ),
    x="inventory",
    y="store_id",
    orientation="h",
    text="inventory",
    title="Top Stores by Current Inventory",
    labels={
        "inventory":
            "Units in Inventory",
        "store_id":
            "Store"
    }
)


fig_store.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)


fig_store.update_layout(
    height=650
)


st.plotly_chart(
    fig_store,
    use_container_width=True
)


st.dataframe(
    store_summary.sort_values(
        "inventory",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SKU ANALYSIS
# ============================================================

st.subheader(
    "📦 SKU Inventory Analysis"
)


sku_summary = (
    filtered_df
    .groupby(
        "sku_id",
        as_index=False
    )
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
        store_count=(
            "store_id",
            "nunique"
        )
    )
)


sku_summary[
    "inventory_to_30d_forecast"
] = (
    sku_summary[
        "inventory"
    ]
    /
    sku_summary[
        "forecast_30d"
    ].replace(
        0,
        pd.NA
    )
)


top_sku_inventory = (
    sku_summary
    .sort_values(
        "inventory",
        ascending=False
    )
    .head(20)
)


fig_sku = px.bar(
    top_sku_inventory.sort_values(
        "inventory"
    ),
    x="inventory",
    y="sku_id",
    orientation="h",
    text="inventory",
    title="Top SKUs by Current Inventory",
    labels={
        "inventory":
            "Units in Inventory",
        "sku_id":
            "SKU"
    }
)


fig_sku.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)


fig_sku.update_layout(
    height=650
)


st.plotly_chart(
    fig_sku,
    use_container_width=True
)


st.dataframe(
    sku_summary.sort_values(
        "inventory",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# REPLENISHMENT
# ============================================================

st.divider()

st.subheader(
    "🔄 Replenishment Recommendation"
)


total_reorder = (
    filtered_df[
        "suggested_reorder_qty"
    ].sum()
)


reorder_count = (
    filtered_df[
        "suggested_reorder_qty"
    ]
    .fillna(0)
    .gt(0)
    .sum()
)


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Suggested Reorder",
        f"{total_reorder:,.0f}"
    )


with c2:

    st.metric(
        "Store-SKU Requiring Reorder",
        f"{reorder_count:,}"
    )


with c3:

    st.metric(
        "Current Inventory",
        f"{total_inventory:,.0f}"
    )


if total_reorder == 0:

    st.info(
        """
        **No additional replenishment is recommended under
        the current planning assumptions.**

        Existing inventory substantially exceeds the
        near-term calibrated demand forecast.
        """
    )

else:

    st.warning(
        f"""
        The current filtered inventory requires approximately
        **{total_reorder:,.0f} units** of additional replenishment
        under the existing recommendation logic.
        """
    )


# ============================================================
# BUSINESS ACTIONS
# ============================================================

st.divider()

st.subheader(
    "💡 Inventory Business Actions"
)


for _, row in actions.sort_values(
    "priority"
).iterrows():

    priority = row[
        "priority"
    ]

    category = row[
        "action_category"
    ]

    finding = row[
        "finding"
    ]

    recommendation = row[
        "recommended_action"
    ]

    reason = row[
        "business_reason"
    ]

    st.markdown(
        f"""
        ### Priority {priority} — {category}

        **Finding:** {finding}

        **Recommended Action:** `{recommendation}`

        **Business Reason:** {reason}
        """
    )


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.divider()

st.subheader(
    "📋 Executive Inventory Summary"
)


summary_display = executive_summary.copy()


st.dataframe(
    summary_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# INVENTORY DETAIL
# ============================================================

st.divider()

st.subheader(
    "🔍 Inventory Detail"
)


detail_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "reorder_point",
    "safety_stock",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
    "calibrated_inventory_coverage_days",
    "calibrated_stockout_risk_30d",
    "calibrated_inventory_status",
    "suggested_reorder_qty",
    "reorder_status",
    "priority"
]


available_columns = [
    column
    for column in detail_columns
    if column in filtered_df.columns
]


detail_df = (
    filtered_df[
        available_columns
    ]
    .sort_values(
        "stock_on_hand",
        ascending=False
    )
)


st.dataframe(
    detail_df,
    use_container_width=True,
    height=600,
    hide_index=True
)


# ============================================================
# SOURCE DATA
# ============================================================

with st.expander(
    "🔍 View Source Integration Data"
):

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=600,
        hide_index=True
    )