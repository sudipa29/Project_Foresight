# ============================================================
# PROJECT FORESIGHT
# PAGE 5 - FUTURE DEMAND FORECAST
#
# Production 30 / 60 / 90 Day Forecast Dashboard
#
# Source:
# data/processed/forecasting/future/
#
# Files:
#   future_30_day_forecast.csv
#   future_60_day_forecast.csv
#   future_90_day_forecast.csv
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

FUTURE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "forecasting",
    "future"
)


# ============================================================
# FILES
# ============================================================

FORECAST_FILES = {

    "30 Days":
        "future_30_day_forecast.csv",

    "60 Days":
        "future_60_day_forecast.csv",

    "90 Days":
        "future_90_day_forecast.csv"
}


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_forecast(filename):

    path = os.path.join(
        FUTURE_DIR,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    df["forecast_units"] = pd.to_numeric(
        df["forecast_units"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # REMOVE INVALID VALUES
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "date",
            "forecast_units"
        ]
    )

    return df


# ============================================================
# LOAD ALL HORIZONS
# ============================================================

@st.cache_data
def load_all_forecasts():

    data = {}

    for horizon, filename in FORECAST_FILES.items():

        data[horizon] = load_forecast(
            filename
        )

    return data


# ============================================================
# LOAD
# ============================================================

try:

    forecasts = load_all_forecasts()

except Exception as e:

    st.error(
        "❌ Future forecast data could not be loaded."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🔮 Future Demand Forecast"
)

st.caption(
    "Production demand forecasts generated using the final "
    "forecasting model across 30, 60 and 90-day planning horizons."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🔎 Forecast Controls"
)


selected_horizon = st.sidebar.radio(
    "Forecast Horizon",
    [
        "30 Days",
        "60 Days",
        "90 Days"
    ]
)


# ============================================================
# SELECT DATA
# ============================================================

df = forecasts[
    selected_horizon
].copy()


# ============================================================
# STORE / SKU FILTERS
# ============================================================

store_options = sorted(
    df["store_id"].unique()
)

selected_stores = st.sidebar.multiselect(
    "Select Store(s)",
    options=store_options,
    default=[]
)


sku_options = sorted(
    df["sku_id"].unique()
)

selected_skus = st.sidebar.multiselect(
    "Select SKU(s)",
    options=sku_options,
    default=[]
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


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


# ============================================================
# EMPTY CHECK
# ============================================================

if filtered_df.empty:

    st.warning(
        "No forecast data matches the selected filters."
    )

    st.stop()


# ============================================================
# DATE RANGE
# ============================================================

start_date = filtered_df["date"].min()

end_date = filtered_df["date"].max()

number_of_days = (
    filtered_df["date"]
    .nunique()
)


# ============================================================
# DAILY AGGREGATION
# ============================================================

daily_forecast = (
    filtered_df
    .groupby(
        "date",
        as_index=False
    )["forecast_units"]
    .sum()
    .sort_values(
        "date"
    )
)


# ============================================================
# HORIZON TOTAL
# ============================================================

forecast_total = (
    filtered_df[
        "forecast_units"
    ].sum()
)


# ============================================================
# DAILY METRICS
# ============================================================

average_daily_forecast = (
    daily_forecast[
        "forecast_units"
    ].mean()
)

maximum_daily_forecast = (
    daily_forecast[
        "forecast_units"
    ].max()
)

minimum_daily_forecast = (
    daily_forecast[
        "forecast_units"
    ].min()
)


# ============================================================
# PAGE SUBHEADER
# ============================================================

st.subheader(
    f"📅 {selected_horizon} Forecast"
)

st.caption(
    f"Forecast period: "
    f"**{start_date:%d %b %Y}** "
    f"to "
    f"**{end_date:%d %b %Y}** "
    f"({number_of_days} forecast days)"
)


# ============================================================
# KPI ROW
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        f"{selected_horizon} Forecast",
        f"{forecast_total:,.0f}"
    )


with c2:

    st.metric(
        "Average Daily Forecast",
        f"{average_daily_forecast:,.1f}"
    )


with c3:

    st.metric(
        "Peak Daily Forecast",
        f"{maximum_daily_forecast:,.1f}"
    )


with c4:

    st.metric(
        "Lowest Daily Forecast",
        f"{minimum_daily_forecast:,.1f}"
    )


st.divider()


# ============================================================
# HORIZON COMPARISON
# ============================================================

st.subheader(
    "📊 Forecast Horizon Comparison"
)

st.caption(
    "Total model-generated demand across each planning horizon."
)


horizon_rows = []


for horizon_name, horizon_data in forecasts.items():

    total = (
        horizon_data[
            "forecast_units"
        ].sum()
    )

    days = (
        horizon_data[
            "date"
        ].nunique()
    )

    horizon_rows.append(
        {
            "Horizon": horizon_name,
            "Forecast Units": total,
            "Forecast Days": days
        }
    )


horizon_df = pd.DataFrame(
    horizon_rows
)


fig_horizon = px.bar(
    horizon_df,
    x="Horizon",
    y="Forecast Units",
    text="Forecast Units",
    title="Total Forecast Demand by Horizon",
    labels={
        "Forecast Units": "Forecast Units",
        "Horizon": "Forecast Horizon"
    }
)


fig_horizon.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)


fig_horizon.update_layout(
    height=430,
    margin=dict(
        l=40,
        r=40,
        t=70,
        b=40
    )
)


st.plotly_chart(
    fig_horizon,
    use_container_width=True
)


# ============================================================
# HORIZON TABLE
# ============================================================

display_horizon = horizon_df.copy()

display_horizon[
    "Forecast Units"
] = display_horizon[
    "Forecast Units"
].round(0).astype(int)


st.dataframe(
    display_horizon,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# DAILY FORECAST TREND
# ============================================================

st.subheader(
    f"📈 Daily {selected_horizon} Forecast"
)

st.caption(
    "Forecast demand aggregated across all selected Store-SKU "
    "combinations for each future date."
)


fig_daily = px.line(
    daily_forecast,
    x="date",
    y="forecast_units",
    markers=True,
    title=f"Daily Future Demand — {selected_horizon}",
    labels={
        "date": "Forecast Date",
        "forecast_units": "Forecast Units"
    }
)


fig_daily.update_layout(
    height=500,
    hovermode="x unified",
    margin=dict(
        l=40,
        r=40,
        t=70,
        b=40
    )
)


st.plotly_chart(
    fig_daily,
    use_container_width=True
)


st.divider()


# ============================================================
# STORE FORECAST
# ============================================================

st.subheader(
    f"🏬 Store Forecast — {selected_horizon}"
)


store_forecast = (
    filtered_df
    .groupby(
        "store_id",
        as_index=False
    )["forecast_units"]
    .sum()
    .sort_values(
        "forecast_units",
        ascending=False
    )
)


top_stores = (
    store_forecast
    .head(20)
    .sort_values(
        "forecast_units",
        ascending=True
    )
)


fig_store = px.bar(
    top_stores,
    x="forecast_units",
    y="store_id",
    orientation="h",
    text="forecast_units",
    title=f"Top 20 Stores by {selected_horizon} Forecast",
    labels={
        "store_id": "Store",
        "forecast_units": "Forecast Units"
    }
)


fig_store.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)


fig_store.update_layout(
    height=650,
    margin=dict(
        l=40,
        r=80,
        t=70,
        b=40
    )
)


st.plotly_chart(
    fig_store,
    use_container_width=True
)


st.divider()


# ============================================================
# SKU FORECAST
# ============================================================

st.subheader(
    f"📦 SKU Forecast — {selected_horizon}"
)


sku_forecast = (
    filtered_df
    .groupby(
        "sku_id",
        as_index=False
    )["forecast_units"]
    .sum()
    .sort_values(
        "forecast_units",
        ascending=False
    )
)


top_skus = (
    sku_forecast
    .head(20)
    .sort_values(
        "forecast_units",
        ascending=True
    )
)


fig_sku = px.bar(
    top_skus,
    x="forecast_units",
    y="sku_id",
    orientation="h",
    text="forecast_units",
    title=f"Top 20 SKUs by {selected_horizon} Forecast",
    labels={
        "sku_id": "SKU",
        "forecast_units": "Forecast Units"
    }
)


fig_sku.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)


fig_sku.update_layout(
    height=650,
    margin=dict(
        l=40,
        r=80,
        t=70,
        b=40
    )
)


st.plotly_chart(
    fig_sku,
    use_container_width=True
)


st.divider()


# ============================================================
# HIGH / LOW FORECAST STORE-SKU
# ============================================================

st.subheader(
    "🔥 Forecast Demand Concentration"
)


col_high, col_low = st.columns(2)


# ============================================================
# HIGH DEMAND
# ============================================================

with col_high:

    st.markdown(
        "#### 🔥 Highest Forecast Store-SKU"
    )

    high_items = (
        filtered_df[
            [
                "store_id",
                "sku_id",
                "forecast_units"
            ]
        ]
        .groupby(
            [
                "store_id",
                "sku_id"
            ],
            as_index=False
        )[
            "forecast_units"
        ]
        .sum()
        .sort_values(
            "forecast_units",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        high_items,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# LOW DEMAND
# ============================================================

with col_low:

    st.markdown(
        "#### 📉 Lowest Forecast Store-SKU"
    )

    low_items = (
        filtered_df[
            [
                "store_id",
                "sku_id",
                "forecast_units"
            ]
        ]
        .groupby(
            [
                "store_id",
                "sku_id"
            ],
            as_index=False
        )[
            "forecast_units"
        ]
        .sum()
        .sort_values(
            "forecast_units",
            ascending=True
        )
        .head(10)
    )

    st.dataframe(
        low_items,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ============================================================
# FORECAST ASSUMPTIONS
# ============================================================

st.subheader(
    "⚙️ Forecast Assumptions"
)


st.info(
    """
    **Forecast scenario**

    Future demand is taken directly from the production
    forecasting pipeline using the final selected model.

    The dashboard does not recalculate or modify these forecasts.

    Forecast values represent the model-generated demand scenario
    for future planning.

    Discount and promotion assumptions are based on the
    assumptions used during the forecasting pipeline.
    """
)


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_INFO_FILE = os.path.join(
    FUTURE_DIR,
    "future_forecast_model_info.csv"
)


if os.path.exists(
    MODEL_INFO_FILE
):

    try:

        model_info = pd.read_csv(
            MODEL_INFO_FILE
        )

        with st.expander(
            "🤖 Forecast Model Information"
        ):

            st.dataframe(
                model_info,
                use_container_width=True,
                hide_index=True
            )

    except Exception:

        pass


# ============================================================
# FORECAST DETAIL
# ============================================================

st.subheader(
    "🔍 Forecast Detail"
)

detail_df = filtered_df.copy()


detail_df = detail_df.sort_values(
    [
        "date",
        "store_id",
        "sku_id"
    ]
)


st.dataframe(
    detail_df,
    use_container_width=True,
    height=500,
    hide_index=True
)


# ============================================================
# SOURCE DATA
# ============================================================

with st.expander(
    "🔍 View Complete Forecast Source Data"
):

    tab30, tab60, tab90 = st.tabs(
        [
            "30 Days",
            "60 Days",
            "90 Days"
        ]
    )


    with tab30:

        st.dataframe(
            forecasts["30 Days"],
            use_container_width=True,
            hide_index=True
        )


    with tab60:

        st.dataframe(
            forecasts["60 Days"],
            use_container_width=True,
            hide_index=True
        )


    with tab90:

        st.dataframe(
            forecasts["90 Days"],
            use_container_width=True,
            hide_index=True
        )