# Executive Overview

import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(__file__).resolve().parents[1]

PHASE8_PATH = (
    BASE_PATH
    / "data"
    / "processed"
    / "forecasting"
    / "business_insights"
    / "phase8"
)

EXECUTIVE_FILE = PHASE8_PATH / "phase8_executive_summary.csv"
HORIZON_FILE = PHASE8_PATH / "phase8_forecast_horizon_summary.csv"
COVERAGE_FILE = PHASE8_PATH / "phase8_inventory_coverage_distribution.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    executive = pd.read_csv(EXECUTIVE_FILE)
    horizon = pd.read_csv(HORIZON_FILE)
    coverage = pd.read_csv(COVERAGE_FILE)

    return executive, horizon, coverage


executive, horizon, coverage = load_data()

e = executive.iloc[0]


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🏠 Executive Overview")

st.caption(
    "Enterprise-level view of demand, inventory, forecasting and business actions."
)


# ============================================================
# BUSINESS STATUS
# ============================================================

status = str(e["business_inventory_status"])

if status == "CRITICAL_OVERSTOCK":

    st.error(
        "🚨 CRITICAL OVERSTOCK — Inventory materially exceeds forecasted and planning demand."
    )

else:

    st.info(f"Business Status: {status}")


# ============================================================
# KPI ROW 1
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Store-SKU Combinations",
    f"{int(e['total_store_sku']):,}"
)

c2.metric(
    "Stores",
    f"{int(e['total_stores']):,}"
)

c3.metric(
    "SKUs",
    f"{int(e['total_skus']):,}"
)

c4.metric(
    "Inventory Units",
    f"{e['total_inventory_units']:,.0f}"
)


# ============================================================
# KPI ROW 2
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "30-Day Forecast",
    f"{e['forecast_30d']:,.0f}"
)

c2.metric(
    "60-Day Forecast",
    f"{e['forecast_60d']:,.0f}"
)

c3.metric(
    "90-Day Forecast",
    f"{e['forecast_90d']:,.0f}"
)

c4.metric(
    "30-Day Planning Demand",
    f"{e['planning_30d_demand']:,.0f}"
)


# ============================================================
# KPI ROW 3
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Inventory / 30D Forecast",
    f"{e['inventory_to_30d_forecast_ratio']:.2f}×"
)

c2.metric(
    "Inventory / Planning Demand",
    f"{e['inventory_to_30d_planning_ratio']:.2f}×"
)

c3.metric(
    ">365 Days Inventory",
    f"{int(e['over365_doi_store_sku']):,}"
)

c4.metric(
    "No-Forecast Store-SKU",
    f"{int(e['no_forecast_store_sku']):,}"
)


st.divider()


# ============================================================
# KEY BUSINESS INSIGHTS
# ============================================================

st.subheader("🔎 Key Business Insights")

col1, col2, col3 = st.columns(3)

with col1:

    st.warning(
        f"""
        **Extreme Inventory Coverage**

        {int(e['over365_doi_store_sku']):,} Store-SKU combinations
        have more than 365 days of inventory coverage.

        This indicates substantial excess inventory exposure.
        """
    )


with col2:

    st.warning(
        f"""
        **Dormant / No-Forecast Inventory**

        {int(e['no_forecast_store_sku']):,} Store-SKU combinations
        have no forecast.

        Inventory tied to these combinations:
        **{e['no_forecast_inventory']:,.0f} units**
        """
    )


with col3:

    st.info(
        f"""
        **Replenishment Decision**

        Suggested reorder quantity:

        **{e['suggested_reorder_quantity']:,.0f} units**

        Store-SKU requiring reorder:

        **{int(e['store_sku_requiring_reorder']):,}**
        """
    )


st.divider()


# ============================================================
# FORECAST HORIZON
# ============================================================

st.subheader("📈 Forecast Horizon")

fig = px.bar(
    horizon,
    x="horizon",
    y=["forecast_units", "planning_demand_units"],
    barmode="group",
    labels={
        "value": "Units",
        "horizon": "Forecast Horizon",
        "variable": "Measure"
    },
    title="Forecast vs Planning Demand"
)

fig.update_layout(
    legend_title_text="",
    height=450
)

st.plotly_chart(
    fig,
    width="stretch"
)


# ============================================================
# INVENTORY COVERAGE
# ============================================================

st.subheader("📦 Inventory Coverage Distribution")

coverage_display = coverage.copy()

fig2 = px.bar(
    coverage_display,
    x="inventory_coverage_category",
    y="inventory_units",
    text="inventory_percentage",
    labels={
        "inventory_coverage_category": "Inventory Coverage",
        "inventory_units": "Inventory Units"
    },
    title="Inventory Units by Coverage Category"
)

fig2.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig2.update_layout(
    height=450
)

st.plotly_chart(
    fig2,
    width="stretch"
)


# ============================================================
# EXECUTIVE CONCLUSION
# ============================================================

st.divider()

st.subheader("💡 Executive Conclusion")

st.markdown(
    f"""
### Current Business Situation

The current inventory position is classified as:

**{status}**

Total inventory stands at approximately
**{e['total_inventory_units']:,.0f} units**, while the calibrated
30-day forecast is only **{e['forecast_30d']:,.0f} units**.

This results in an inventory-to-forecast ratio of approximately
**{e['inventory_to_30d_forecast_ratio']:.2f}×**.

Furthermore, **{int(e['over365_doi_store_sku']):,} Store-SKU combinations**
have more than 365 days of inventory coverage.

The current recommendation is therefore to:

- Stop unnecessary replenishment.
- Review existing stock.
- Investigate extreme overstock.
- Review dormant/no-forecast inventory.
- Prioritize inventory liquidation, transfer or redistribution decisions.
"""
)