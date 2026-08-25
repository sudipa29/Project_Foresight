# ============================================================
# PROJECT FORESIGHT
# PAGE 2 - DEMAND & FORECAST ANALYSIS
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


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

MASTER_FILE = PHASE8_PATH / "phase8_dashboard_master.csv"

HORIZON_FILE = PHASE8_PATH / "phase8_forecast_horizon_summary.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    master = pd.read_csv(MASTER_FILE)

    horizon = pd.read_csv(HORIZON_FILE)

    return master, horizon


df, horizon = load_data()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📈 Demand & Forecast Analysis")

st.caption(
    "Explore calibrated demand forecasts across stores, SKUs and forecast horizons."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Forecast Filters")


# ------------------------------------------------------------
# STORE FILTER
# ------------------------------------------------------------

store_options = sorted(
    df["store_id"].dropna().unique()
)


selected_stores = st.sidebar.multiselect(
    "Select Store(s)",
    options=store_options,
    default=[]
)


# ------------------------------------------------------------
# SKU FILTER
# ------------------------------------------------------------

sku_options = sorted(
    df["sku_id"].dropna().unique()
)


selected_skus = st.sidebar.multiselect(
    "Select SKU(s)",
    options=sku_options,
    default=[]
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df.copy()


if selected_stores:

    filtered_df = filtered_df[
        filtered_df["store_id"].isin(selected_stores)
    ]


if selected_skus:

    filtered_df = filtered_df[
        filtered_df["sku_id"].isin(selected_skus)
    ]


# ============================================================
# EMPTY DATA CHECK
# ============================================================

if filtered_df.empty:

    st.warning(
        "No data is available for the selected Store/SKU filters."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

# ------------------------------------------------------------
# FORECAST
# ------------------------------------------------------------

forecast_30 = (
    filtered_df["calibrated_forecast_30d"].sum()
)

forecast_60 = (
    filtered_df["calibrated_forecast_60d"].sum()
)

forecast_90 = (
    filtered_df["calibrated_forecast_90d"].sum()
)


# ------------------------------------------------------------
# PLANNING DEMAND
# ------------------------------------------------------------

planning_daily = (
    filtered_df["planning_daily_demand"].sum()
)


planning_30 = (
    planning_daily * 30
)

planning_60 = (
    planning_daily * 60
)

planning_90 = (
    planning_daily * 90
)


# ------------------------------------------------------------
# FORECAST / PLANNING RATIOS
# ------------------------------------------------------------

ratio_30 = (

    forecast_30 / planning_30

    if planning_30 > 0

    else 0

)


ratio_60 = (

    forecast_60 / planning_60

    if planning_60 > 0

    else 0

)


ratio_90 = (

    forecast_90 / planning_90

    if planning_90 > 0

    else 0

)


# ============================================================
# FORECAST SUMMARY
# ============================================================

st.subheader("📊 Forecast Summary")


# ============================================================
# ROW 1
# 30D / 60D / 90D FORECAST
# ============================================================

c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        label="30-Day Forecast",
        value=f"{forecast_30:,.0f}"
    )


with c2:

    st.metric(
        label="60-Day Forecast",
        value=f"{forecast_60:,.0f}"
    )


with c3:

    st.metric(
        label="90-Day Forecast",
        value=f"{forecast_90:,.0f}"
    )


# ============================================================
# ROW 2
# 30D / 60D / 90D PLANNING DEMAND
# ============================================================

c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        label="30-Day Planning Demand",
        value=f"{planning_30:,.0f}"
    )


with c2:

    st.metric(
        label="60-Day Planning Demand",
        value=f"{planning_60:,.0f}"
    )


with c3:

    st.metric(
        label="90-Day Planning Demand",
        value=f"{planning_90:,.0f}"
    )


# ============================================================
# ROW 3
# FORECAST / PLANNING RATIO
# ============================================================

c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        label="30D Forecast / Planning",
        value=f"{ratio_30:.2f}×"
    )


with c2:

    st.metric(
        label="60D Forecast / Planning",
        value=f"{ratio_60:.2f}×"
    )


with c3:

    st.metric(
        label="90D Forecast / Planning",
        value=f"{ratio_90:.2f}×"
    )


st.divider()


# ============================================================
# FORECAST VS PLANNING DEMAND
# ============================================================

st.subheader("📈 Forecast vs Planning Demand")

st.caption(
    "Comparison of calibrated forecast volume against planning demand "
    "across the 30-day, 60-day and 90-day horizons."
)


chart_df = pd.DataFrame(
    {
        "Horizon": [
            "30 DAYS",
            "60 DAYS",
            "90 DAYS"
        ],

        "Forecast": [
            forecast_30,
            forecast_60,
            forecast_90
        ],

        "Planning Demand": [
            planning_30,
            planning_60,
            planning_90
        ]
    }
)


fig = px.bar(
    chart_df,

    x="Horizon",

    y=[
        "Forecast",
        "Planning Demand"
    ],

    barmode="group",

    text_auto=True,

    labels={
        "value": "Units",
        "variable": "Measure"
    }
)


fig.update_layout(

    height=450,

    margin=dict(
        l=40,
        r=40,
        t=50,
        b=40
    ),

    legend_title_text="",

    xaxis_title="Forecast Horizon",

    yaxis_title="Units",

    hovermode="x unified"
)


st.plotly_chart(
    fig,
    width="stretch"
)


# ============================================================
# FORECAST / PLANNING RATIO
# ============================================================

st.subheader("⚖️ Forecast-to-Planning Ratio")

st.caption(
    "A ratio of 1.0× indicates that forecast demand is aligned "
    "with the planning baseline."
)


ratio_df = pd.DataFrame(
    {
        "Horizon": [
            "30 DAYS",
            "60 DAYS",
            "90 DAYS"
        ],

        "Ratio": [
            ratio_30,
            ratio_60,
            ratio_90
        ]
    }
)


fig_ratio = px.bar(

    ratio_df,

    x="Horizon",

    y="Ratio",

    text="Ratio",

    labels={
        "Ratio": "Forecast / Planning Ratio"
    }
)


# ------------------------------------------------------------
# REFERENCE LINE
# ------------------------------------------------------------

fig_ratio.add_hline(

    y=1,

    line_dash="dash",

    annotation_text="1.0× = Planning Alignment",

    annotation_position="top left"
)


# ------------------------------------------------------------
# BAR LABELS
# ------------------------------------------------------------

fig_ratio.update_traces(

    texttemplate="%{text:.2f}×",

    textposition="outside"
)


# ------------------------------------------------------------
# CHART LAYOUT
# ------------------------------------------------------------

fig_ratio.update_layout(

    height=420,

    margin=dict(
        l=40,
        r=40,
        t=60,
        b=40
    ),

    xaxis_title="Forecast Horizon",

    yaxis_title="Forecast / Planning",

    showlegend=False
)


st.plotly_chart(

    fig_ratio,

    width="stretch"
)


st.divider()


# ============================================================
# TOP FORECASTED STORE-SKU
# ============================================================

st.subheader("🔥 Highest 30-Day Forecast Store-SKU")

st.caption(
    "Top 20 Store-SKU combinations ranked by calibrated 30-day forecast."
)


# ------------------------------------------------------------
# SELECT REQUIRED COLUMNS
# ------------------------------------------------------------

top_forecast = (

    filtered_df[
        [
            "store_id",
            "sku_id",
            "calibrated_forecast_30d",
            "calibrated_forecast_60d",
            "calibrated_forecast_90d",
            "planning_daily_demand"
        ]
    ]

    .sort_values(
        "calibrated_forecast_30d",

        ascending=False
    )

    .head(20)

    .copy()
)


# ------------------------------------------------------------
# CREATE DISPLAY LABEL
# ------------------------------------------------------------

top_forecast["Store / SKU"] = (

    "Store "

    + top_forecast["store_id"].astype(str)

    + " / SKU "

    + top_forecast["sku_id"].astype(str)

)


# ------------------------------------------------------------
# SORT FOR HORIZONTAL BAR
# ------------------------------------------------------------

top_forecast = top_forecast.sort_values(

    "calibrated_forecast_30d",

    ascending=True

)


# ============================================================
# TOP FORECAST CHART
# ============================================================

fig_top = px.bar(

    top_forecast,

    x="calibrated_forecast_30d",

    y="Store / SKU",

    orientation="h",

    text="calibrated_forecast_30d",

    labels={

        "calibrated_forecast_30d":
            "30-Day Forecast",

        "Store / SKU":
            "Store / SKU"

    }

)


fig_top.update_traces(

    texttemplate="%{text:,.0f}",

    textposition="outside"

)


fig_top.update_layout(

    height=650,

    margin=dict(

        l=40,

        r=80,

        t=50,

        b=40

    ),

    xaxis_title="30-Day Forecast",

    yaxis_title="Store / SKU"

)


st.plotly_chart(

    fig_top,

    width="stretch"

)


st.divider()


# ============================================================
# FORECAST DETAIL TABLE
# ============================================================

st.subheader("🔍 Forecast Detail")

st.caption(
    "Detailed Store-SKU forecast information for the selected filters."
)


display_columns = [

    "store_id",

    "sku_id",

    "units_30d",

    "units_90d",

    "avg_daily_demand_30d",

    "avg_daily_demand_90d",

    "calibrated_forecast_30d",

    "calibrated_forecast_60d",

    "calibrated_forecast_90d",

    "planning_daily_demand",

    "demand_available"

]


available_columns = [

    column

    for column in display_columns

    if column in filtered_df.columns

]


detail_df = (

    filtered_df[available_columns]

    .sort_values(

        "calibrated_forecast_30d",

        ascending=False

    )

)


st.dataframe(

    detail_df,

    width="stretch",

    height=500,

    hide_index=True

)


# ============================================================
# FORECAST INTERPRETATION
# ============================================================

st.divider()

st.subheader("💡 Forecast Interpretation")


# ============================================================
# INTERPRETATION BASED ON 30-DAY RATIO
# ============================================================

if ratio_30 < 1:

    st.info(

        f"""
        The 30-day calibrated forecast is approximately
        **{ratio_30:.2f}×** the planning demand.

        This indicates that the ML-calibrated forecast is below
        the planning baseline over the 30-day horizon.
        """

    )


elif ratio_30 > 1:

    st.warning(

        f"""
        The 30-day calibrated forecast is approximately
        **{ratio_30:.2f}×** the planning demand.

        This indicates that the forecast is above the planning
        baseline and should be monitored for demand acceleration.
        """

    )


else:

    st.success(

        "The 30-day forecast is approximately aligned "
        "with planning demand."

    )