import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


# ============================================================
# PROJECT FORESIGHT
# PHASE 7 — RISK & INSIGHTS
# ============================================================

st.title("🚨 Risk & Insights")

st.caption(
    "Inventory risk, demand risk, forecast risk and actionable "
    "business insights using the calibrated forecasting pipeline."
)


# ============================================================
# BASE PATH
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

FORECASTING_DIR = os.path.join(
    PROCESSED_DIR,
    "forecasting"
)

INTEGRATION_DIR = os.path.join(
    FORECASTING_DIR,
    "integration"
)

BUSINESS_DIR = os.path.join(
    FORECASTING_DIR,
    "business_insights"
)

FUTURE_DIR = os.path.join(
    FORECASTING_DIR,
    "future"
)


# ============================================================
# FILE LOADER
# ============================================================

@st.cache_data
def load_csv(path):

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception:
        return pd.DataFrame()


# ============================================================
# LOAD PRODUCTION DATA
# ============================================================

inventory = load_csv(
    os.path.join(
        INTEGRATION_DIR,
        "calibrated_forecast_inventory_integrated.csv"
    )
)


integration_summary = load_csv(
    os.path.join(
        INTEGRATION_DIR,
        "calibrated_forecast_inventory_integration_summary.csv"
    )
)


executive_summary = load_csv(
    os.path.join(
        BUSINESS_DIR,
        "business_inventory_executive_summary.csv"
    )
)


business_insights = load_csv(
    os.path.join(
        BUSINESS_DIR,
        "business_inventory_insights.csv"
    )
)


business_actions = load_csv(
    os.path.join(
        BUSINESS_DIR,
        "inventory_business_actions.csv"
    )
)


dormant_inventory = load_csv(
    os.path.join(
        BUSINESS_DIR,
        "dormant_inventory_business_insights.csv"
    )
)


overstock_inventory = load_csv(
    os.path.join(
        BUSINESS_DIR,
        "overstock_business_insights.csv"
    )
)


store_insights = load_csv(
    os.path.join(
        BUSINESS_DIR,
        "store_inventory_business_insights.csv"
    )
)


sku_insights = load_csv(
    os.path.join(
        BUSINESS_DIR,
        "sku_inventory_business_insights.csv"
    )
)


future_30 = load_csv(
    os.path.join(
        FUTURE_DIR,
        "future_30_day_forecast.csv"
    )
)


# ============================================================
# DATA VALIDATION
# ============================================================

if inventory.empty:

    st.error(
        "❌ Calibrated inventory integration data could not be loaded."
    )

    st.info(
        f"""
Expected file:

{os.path.join(
    INTEGRATION_DIR,
    "calibrated_forecast_inventory_integrated.csv"
)}
"""
    )

    st.stop()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "stock_on_hand",
    "calibrated_forecast_30d",
    "calibrated_forecast_60d",
    "calibrated_forecast_90d",
    "calibrated_inventory_coverage_days",
    "suggested_reorder_qty",
    "calibrated_replenishment_gap_30d",
    "calibrated_replenishment_gap_60d",
    "calibrated_replenishment_gap_90d",
    "calibrated_vs_existing_forecast_difference"
]


for col in numeric_columns:

    if col in inventory.columns:

        inventory[col] = pd.to_numeric(
            inventory[col],
            errors="coerce"
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔎 Risk Filters")


# Store filter

if "store_id" in inventory.columns:

    stores = sorted(
        inventory["store_id"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_stores = st.sidebar.multiselect(
        "Select Store(s)",
        options=stores,
        default=[]
    )

else:

    selected_stores = []


# SKU filter

if "sku_id" in inventory.columns:

    skus = sorted(
        inventory["sku_id"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_skus = st.sidebar.multiselect(
        "Select SKU(s)",
        options=skus,
        default=[]
    )

else:

    selected_skus = []


# ============================================================
# FILTER DATA
# ============================================================

filtered_inventory = inventory.copy()


if selected_stores:

    filtered_inventory = filtered_inventory[
        filtered_inventory["store_id"].isin(
            selected_stores
        )
    ]


if selected_skus:

    filtered_inventory = filtered_inventory[
        filtered_inventory["sku_id"].isin(
            selected_skus
        )
    ]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_inventory = (
    filtered_inventory["stock_on_hand"].sum()
)


forecast_30 = (
    filtered_inventory["calibrated_forecast_30d"].sum()
)


forecast_60 = (
    filtered_inventory["calibrated_forecast_60d"].sum()
)


forecast_90 = (
    filtered_inventory["calibrated_forecast_90d"].sum()
)


inventory_forecast_ratio = (
    total_inventory / forecast_30
    if forecast_30 > 0
    else 0
)


reorder_quantity = (
    filtered_inventory["suggested_reorder_qty"].sum()
)


overstock_count = 0


if "calibrated_inventory_status" in filtered_inventory.columns:

    overstock_count = (
        filtered_inventory[
            filtered_inventory[
                "calibrated_inventory_status"
            ].astype(str).str.upper()
            == "OVERSTOCK"
        ]
        .shape[0]
    )


no_forecast_count = 0


if "calibrated_forecast_30d" in filtered_inventory.columns:

    no_forecast_count = (
        filtered_inventory[
            filtered_inventory[
                "calibrated_forecast_30d"
            ].fillna(0)
            <= 0
        ]
        .shape[0]
    )


# ============================================================
# EXECUTIVE RISK SUMMARY
# ============================================================

st.subheader(
    "🚨 Executive Risk Position"
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Current Inventory",
    f"{total_inventory:,.0f}"
)


col2.metric(
    "30-Day Forecast",
    f"{forecast_30:,.0f}"
)


col3.metric(
    "Inventory / 30D Forecast",
    f"{inventory_forecast_ratio:,.2f}×"
)


col4.metric(
    "Reorder Quantity",
    f"{reorder_quantity:,.0f}"
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "60-Day Forecast",
    f"{forecast_60:,.0f}"
)


col2.metric(
    "90-Day Forecast",
    f"{forecast_90:,.0f}"
)


col3.metric(
    "Overstock Store-SKU",
    f"{overstock_count:,}"
)


col4.metric(
    "No-Forecast Store-SKU",
    f"{no_forecast_count:,}"
)


# ============================================================
# INVENTORY RISK ALERT
# ============================================================

st.divider()

if inventory_forecast_ratio >= 10:

    st.error(
        f"""
        ### ⚠️ Severe Inventory Imbalance

        Current inventory is approximately
        **{inventory_forecast_ratio:,.2f}×**
        the 30-day calibrated demand forecast.

        This indicates a significant excess-inventory position
        under the current forecast scenario.

        **Recommended management focus:**

        • Pause unnecessary replenishment  
        • Review excess inventory  
        • Investigate dormant inventory  
        • Evaluate inter-store transfers  
        • Monitor actual demand against forecast
        """
    )

elif inventory_forecast_ratio >= 3:

    st.warning(
        f"""
        ### ⚠️ Elevated Inventory Coverage

        Current inventory is approximately
        **{inventory_forecast_ratio:,.2f}×**
        the 30-day calibrated forecast.
        """
    )

else:

    st.success(
        "Inventory coverage is within a relatively manageable range "
        "under the current forecast scenario."
    )


# ============================================================
# INVENTORY VS FORECAST
# ============================================================

st.divider()

st.subheader(
    "📦 Inventory vs Forecast Demand"
)


comparison_df = pd.DataFrame(
    {
        "Measure": [
            "Current Inventory",
            "30-Day Forecast",
            "60-Day Forecast",
            "90-Day Forecast"
        ],

        "Units": [
            total_inventory,
            forecast_30,
            forecast_60,
            forecast_90
        ]
    }
)


fig_inventory = px.bar(
    comparison_df,
    x="Measure",
    y="Units",
    text="Units",
    title="Current Inventory vs Calibrated Demand Forecast"
)


fig_inventory.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)


fig_inventory.update_layout(
    height=450,
    xaxis_title="",
    yaxis_title="Units"
)


st.plotly_chart(
    fig_inventory,
    use_container_width=True
)


# ============================================================
# INVENTORY COVERAGE
# ============================================================

st.subheader(
    "⏱️ Inventory Coverage Analysis"
)


if "calibrated_inventory_coverage_days" in filtered_inventory.columns:

    coverage = (
        filtered_inventory[
            "calibrated_inventory_coverage_days"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if not coverage.empty:

        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Median Coverage",
            f"{coverage.median():,.0f} days"
        )


        col2.metric(
            "Average Coverage",
            f"{coverage.mean():,.0f} days"
        )


        col3.metric(
            "Maximum Coverage",
            f"{coverage.max():,.0f} days"
        )


        fig_coverage = px.histogram(
            coverage,
            x=coverage,
            nbins=40,
            title="Distribution of Inventory Coverage"
        )


        fig_coverage.update_layout(
            xaxis_title="Inventory Coverage (Days)",
            yaxis_title="Store-SKU Count"
        )


        st.plotly_chart(
            fig_coverage,
            use_container_width=True
        )


# ============================================================
# INVENTORY STATUS DISTRIBUTION
# ============================================================

st.subheader(
    "⚠️ Inventory Status Distribution"
)


if "calibrated_inventory_status" in filtered_inventory.columns:

    status_df = (
        filtered_inventory[
            "calibrated_inventory_status"
        ]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )


    status_df.columns = [
        "Status",
        "Store-SKU Count"
    ]


    fig_status = px.bar(
        status_df,
        x="Status",
        y="Store-SKU Count",
        text="Store-SKU Count",
        title="Calibrated Inventory Status"
    )


    fig_status.update_traces(
        textposition="outside"
    )


    st.plotly_chart(
        fig_status,
        use_container_width=True
    )


    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# STOCKOUT RISK
# ============================================================

st.subheader(
    "🚨 Stockout Risk"
)


if "calibrated_stockout_risk_30d" in filtered_inventory.columns:

    stockout_df = (
        filtered_inventory[
            "calibrated_stockout_risk_30d"
        ]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )


    stockout_df.columns = [
        "Risk Level",
        "Store-SKU Count"
    ]


    fig_stockout = px.bar(
        stockout_df,
        x="Risk Level",
        y="Store-SKU Count",
        color="Risk Level",
        text="Store-SKU Count",
        title="30-Day Calibrated Stockout Risk"
    )


    fig_stockout.update_traces(
        textposition="outside"
    )


    st.plotly_chart(
        fig_stockout,
        use_container_width=True
    )


# ============================================================
# OVERSTOCK ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "📦 Overstock Risk Analysis"
)


if "calibrated_inventory_coverage_days" in filtered_inventory.columns:

    overstock_df = (
        filtered_inventory
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna(
            subset=[
                "calibrated_inventory_coverage_days"
            ]
        )
        .sort_values(
            "calibrated_inventory_coverage_days",
            ascending=False
        )
        .head(20)
    )


    if not overstock_df.empty:

        display_cols = [
            c for c in [
                "store_id",
                "sku_id",
                "stock_on_hand",
                "calibrated_forecast_30d",
                "calibrated_inventory_coverage_days",
                "calibrated_inventory_status",
                "suggested_reorder_qty"
            ]
            if c in overstock_df.columns
        ]


        st.dataframe(
            overstock_df[display_cols].round(2),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# DORMANT INVENTORY
# ============================================================

st.subheader(
    "💤 Dormant Inventory Risk"
)


if "calibrated_forecast_30d" in filtered_inventory.columns:

    dormant_df = filtered_inventory[
        filtered_inventory[
            "calibrated_forecast_30d"
        ].fillna(0)
        <= 0
    ].copy()


    dormant_units = (
        dormant_df["stock_on_hand"].sum()
        if not dormant_df.empty
        else 0
    )


    col1, col2 = st.columns(2)


    col1.metric(
        "No-Forecast Store-SKU",
        f"{len(dormant_df):,}"
    )


    col2.metric(
        "Inventory in No-Forecast Items",
        f"{dormant_units:,.0f}"
    )


    if not dormant_df.empty:

        display_cols = [
            c for c in [
                "store_id",
                "sku_id",
                "stock_on_hand",
                "calibrated_forecast_30d",
                "calibrated_inventory_coverage_days"
            ]
            if c in dormant_df.columns
        ]


        st.dataframe(
            dormant_df[
                display_cols
            ]
            .sort_values(
                "stock_on_hand",
                ascending=False
            )
            .head(25),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# STORE RISK
# ============================================================

st.divider()

st.subheader(
    "🏬 Store-Level Risk Concentration"
)


if "store_id" in filtered_inventory.columns:

    store_risk = (
        filtered_inventory
        .groupby("store_id", as_index=False)
        .agg(
            inventory_units=(
                "stock_on_hand",
                "sum"
            ),
            forecast_30d=(
                "calibrated_forecast_30d",
                "sum"
            ),
            store_sku_count=(
                "sku_id",
                "count"
            )
        )
    )


    store_risk["inventory_forecast_ratio"] = np.where(
        store_risk["forecast_30d"] > 0,
        store_risk["inventory_units"]
        /
        store_risk["forecast_30d"],
        np.nan
    )


    fig_store = px.bar(
        store_risk.sort_values(
            "inventory_forecast_ratio",
            ascending=False
        ).head(20),
        x="inventory_forecast_ratio",
        y="store_id",
        orientation="h",
        text="inventory_forecast_ratio",
        title="Top 20 Stores by Inventory / Forecast Ratio"
    )


    fig_store.update_traces(
        texttemplate="%{text:.1f}×",
        textposition="outside"
    )


    st.plotly_chart(
        fig_store,
        use_container_width=True
    )


    st.dataframe(
        store_risk
        .sort_values(
            "inventory_forecast_ratio",
            ascending=False
        )
        .head(20)
        .round(2),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SKU RISK
# ============================================================

st.subheader(
    "📦 SKU-Level Risk Concentration"
)


if "sku_id" in filtered_inventory.columns:

    sku_risk = (
        filtered_inventory
        .groupby("sku_id", as_index=False)
        .agg(
            inventory_units=(
                "stock_on_hand",
                "sum"
            ),
            forecast_30d=(
                "calibrated_forecast_30d",
                "sum"
            ),
            store_count=(
                "store_id",
                "count"
            )
        )
    )


    sku_risk["inventory_forecast_ratio"] = np.where(
        sku_risk["forecast_30d"] > 0,
        sku_risk["inventory_units"]
        /
        sku_risk["forecast_30d"],
        np.nan
    )


    fig_sku = px.bar(
        sku_risk.sort_values(
            "inventory_forecast_ratio",
            ascending=False
        ).head(20),
        x="inventory_forecast_ratio",
        y="sku_id",
        orientation="h",
        text="inventory_forecast_ratio",
        title="Top 20 SKUs by Inventory / Forecast Ratio"
    )


    fig_sku.update_traces(
        texttemplate="%{text:.1f}×",
        textposition="outside"
    )


    st.plotly_chart(
        fig_sku,
        use_container_width=True
    )


# ============================================================
# REPLENISHMENT RISK
# ============================================================

st.subheader(
    "🔄 Replenishment Risk"
)


reorder_required = 0


if "suggested_reorder_qty" in filtered_inventory.columns:

    reorder_required = (
        filtered_inventory[
            filtered_inventory[
                "suggested_reorder_qty"
            ] > 0
        ]
        .shape[0]
    )


col1, col2, col3 = st.columns(3)


col1.metric(
    "Suggested Reorder",
    f"{reorder_quantity:,.0f}"
)


col2.metric(
    "Store-SKU Requiring Reorder",
    f"{reorder_required:,}"
)


col3.metric(
    "Current Inventory",
    f"{total_inventory:,.0f}"
)


if reorder_quantity == 0:

    st.success(
        """
        No additional replenishment is recommended under the
        current planning assumptions.

        Existing inventory substantially exceeds the near-term
        calibrated demand forecast.
        """
    )

else:

    st.warning(
        f"""
        The current filtered inventory requires approximately
        **{reorder_quantity:,.0f} additional units**
        under the current replenishment logic.
        """
    )


# ============================================================
# BUSINESS ACTIONS
# ============================================================

st.divider()

st.subheader(
    "💡 Business Risk Actions"
)


if not business_actions.empty:

    action_columns = [
        c for c in [
            "priority",
            "action_category",
            "finding",
            "metric",
            "metric_value",
            "affected_store_sku",
            "recommended_action",
            "business_reason"
        ]
        if c in business_actions.columns
    ]


    for _, row in business_actions.iterrows():

        priority = row.get(
            "priority",
            ""
        )

        category = row.get(
            "action_category",
            "BUSINESS ACTION"
        )

        finding = row.get(
            "finding",
            ""
        )

        action = row.get(
            "recommended_action",
            ""
        )

        reason = row.get(
            "business_reason",
            ""
        )


        with st.expander(
            f"Priority {priority} — {category}"
        ):

            st.markdown(
                f"""
                **Finding**

                {finding}


                **Recommended Action**

                `{action}`


                **Business Reason**

                {reason}
                """
            )


else:

    st.info(
        "No structured business action records were found."
    )


# ============================================================
# EXECUTIVE INSIGHTS
# ============================================================

st.divider()

st.subheader(
    "📋 Executive Risk Interpretation"
)


st.markdown(
    f"""
### 🚨 1. Inventory Risk

Current inventory is approximately
**{total_inventory:,.0f} units**, while the calibrated
30-day forecast is approximately
**{forecast_30:,.0f} units**.

This represents an inventory-to-forecast ratio of
approximately **{inventory_forecast_ratio:,.2f}×**.

The current position therefore indicates substantial
excess inventory under the production forecast scenario.


### 📦 2. Overstock Risk

Approximately **{overstock_count:,} Store-SKU combinations**
are currently classified as overstock.

The business should prioritize inventory reduction,
redistribution and replenishment control rather than
additional purchasing.


### 💤 3. Dormant Inventory Risk

Approximately **{no_forecast_count:,} Store-SKU combinations**
have no current calibrated 30-day forecast.

Inventory associated with these combinations should be
reviewed for transfer, markdown, liquidation or
discontinuation decisions.


### 🔄 4. Replenishment Risk

The current planning logic recommends approximately
**{reorder_quantity:,.0f} units** of additional replenishment.

This indicates that new replenishment is not currently
the primary business requirement.


### 📈 5. Demand Monitoring

The calibrated forecast should continue to be compared
against actual sales.

If actual demand begins increasing materially, the inventory
position and replenishment recommendations should be
recalculated using updated demand observations.


### 🎯 6. Management Priority

The immediate priority should be:

**Control excess inventory → review dormant stock →
evaluate redistribution → monitor actual demand →
reassess replenishment.**
"""
)


# ============================================================
# DATA QUALITY / SOURCE STATUS
# ============================================================

st.divider()

st.subheader(
    "🔍 Risk Data Availability"
)


status_data = pd.DataFrame(
    {
        "Dataset": [
            "Calibrated Inventory Integration",
            "Integration Summary",
            "Executive Inventory Summary",
            "Business Inventory Insights",
            "Inventory Business Actions",
            "Dormant Inventory Insights",
            "Overstock Insights",
            "Store Inventory Insights",
            "SKU Inventory Insights",
            "Future 30-Day Forecast"
        ],

        "Status": [
            "Available"
            if not inventory.empty
            else "Missing",

            "Available"
            if not integration_summary.empty
            else "Missing",

            "Available"
            if not executive_summary.empty
            else "Missing",

            "Available"
            if not business_insights.empty
            else "Missing",

            "Available"
            if not business_actions.empty
            else "Missing",

            "Available"
            if not dormant_inventory.empty
            else "Missing",

            "Available"
            if not overstock_inventory.empty
            else "Missing",

            "Available"
            if not store_insights.empty
            else "Missing",

            "Available"
            if not sku_insights.empty
            else "Missing",

            "Available"
            if not future_30.empty
            else "Missing"
        ]
    }
)


st.dataframe(
    status_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SOURCE DATA
# ============================================================

with st.expander(
    "🔍 View Risk Source Data"
):

    tabs = st.tabs(
        [
            "Inventory Integration",
            "Executive Summary",
            "Business Actions",
            "Dormant Inventory",
            "Overstock",
            "Store Insights",
            "SKU Insights"
        ]
    )


    with tabs[0]:

        st.dataframe(
            inventory,
            use_container_width=True,
            hide_index=True
        )


    with tabs[1]:

        st.dataframe(
            executive_summary,
            use_container_width=True,
            hide_index=True
        )


    with tabs[2]:

        st.dataframe(
            business_actions,
            use_container_width=True,
            hide_index=True
        )


    with tabs[3]:

        st.dataframe(
            dormant_inventory,
            use_container_width=True,
            hide_index=True
        )


    with tabs[4]:

        st.dataframe(
            overstock_inventory,
            use_container_width=True,
            hide_index=True
        )


    with tabs[5]:

        st.dataframe(
            store_insights,
            use_container_width=True,
            hide_index=True
        )


    with tabs[6]:

        st.dataframe(
            sku_insights,
            use_container_width=True,
            hide_index=True
        )