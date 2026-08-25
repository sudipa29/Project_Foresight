# ============================================================
# PROJECT FORESIGHT
# PAGE 4 - MODEL FORECASTING
#
# Purpose:
#   Compare forecasting models, evaluate ML performance,
#   visualize validation forecasts and explain model drivers.
#
# Important:
#   This page focuses on MODEL EVALUATION.
#   Future 30/60/90-day planning forecasts are handled
#   separately on the Demand Analysis / Future Forecast pages.
# ============================================================

import os
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.title("🤖 Model Forecasting")

st.caption(
    "Compare forecasting models, evaluate prediction accuracy "
    "and understand the drivers behind the selected ML model."
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FORECAST_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "forecasting"
)


# ============================================================
# REQUIRED FILES
# ============================================================

REQUIRED_FILES = [
    "master_model_comparison.csv",
    "best_model_summary.csv",
    "advanced_model_comparison.csv",
    "advanced_model_forecasts.csv",
    "advanced_feature_importance.csv"
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_forecasting_data():

    paths = {
        "master": os.path.join(
            FORECAST_DIR,
            "master_model_comparison.csv"
        ),

        "best": os.path.join(
            FORECAST_DIR,
            "best_model_summary.csv"
        ),

        "comparison": os.path.join(
            FORECAST_DIR,
            "advanced_model_comparison.csv"
        ),

        "forecasts": os.path.join(
            FORECAST_DIR,
            "advanced_model_forecasts.csv"
        ),

        "importance": os.path.join(
            FORECAST_DIR,
            "advanced_feature_importance.csv"
        )
    }


    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    missing_files = [
        filename
        for filename in REQUIRED_FILES
        if not os.path.exists(
            os.path.join(
                FORECAST_DIR,
                filename
            )
        )
    ]


    if missing_files:

        raise FileNotFoundError(
            "Missing forecasting files:\n"
            + "\n".join(missing_files)
        )


    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    master = pd.read_csv(
        paths["master"]
    )

    best = pd.read_csv(
        paths["best"]
    )

    comparison = pd.read_csv(
        paths["comparison"]
    )

    forecasts = pd.read_csv(
        paths["forecasts"]
    )

    importance = pd.read_csv(
        paths["importance"]
    )


    # --------------------------------------------------------
    # Date conversion
    # --------------------------------------------------------

    if "date" in forecasts.columns:

        forecasts["date"] = pd.to_datetime(
            forecasts["date"],
            errors="coerce"
        )


    return (
        master,
        best,
        comparison,
        forecasts,
        importance
    )


# ============================================================
# LOAD WITH ERROR HANDLING
# ============================================================

try:

    (
        master,
        best,
        comparison,
        forecasts,
        importance
    ) = load_forecasting_data()


except Exception as e:

    st.error(
        "❌ Unable to load forecasting data."
    )

    st.code(
        str(e)
    )

    st.info(
        f"""
        Expected folder:

        {FORECAST_DIR}

        Required files:

        - master_model_comparison.csv
        - best_model_summary.csv
        - advanced_model_comparison.csv
        - advanced_model_forecasts.csv
        - advanced_feature_importance.csv
        """
    )

    st.stop()


# ============================================================
# DATA VALIDATION
# ============================================================

required_master_columns = [
    "model",
    "model_type",
    "MAE",
    "RMSE",
    "MAPE",
    "WAPE"
]


missing_master_columns = [
    column
    for column in required_master_columns
    if column not in master.columns
]


if missing_master_columns:

    st.error(
        "❌ Required model evaluation columns are missing:"
    )

    st.code(
        ", ".join(
            missing_master_columns
        )
    )

    st.stop()


# ============================================================
# CLEAN MODEL METRICS
# ============================================================

metric_columns = [
    "MAE",
    "RMSE",
    "MAPE",
    "WAPE"
]


for column in metric_columns:

    master[column] = pd.to_numeric(
        master[column],
        errors="coerce"
    )


master = master.dropna(
    subset=["MAE"]
)


if master.empty:

    st.error(
        "No valid model evaluation records were found."
    )

    st.stop()


# ============================================================
# BEST OVERALL MODEL
# ============================================================

best_overall = master.loc[
    master["MAE"].idxmin()
]


# ============================================================
# MODEL TYPE DATA
# ============================================================

ml_models = master[
    master["model_type"]
    .astype(str)
    .str.lower()
    .str.contains("machine")
]


stat_models = master[
    master["model_type"]
    .astype(str)
    .str.lower()
    .str.contains("statistical")
]


best_ml = None
best_stat = None


if not ml_models.empty:

    best_ml = ml_models.loc[
        ml_models["MAE"].idxmin()
    ]


if not stat_models.empty:

    best_stat = stat_models.loc[
        stat_models["MAE"].idxmin()
    ]


# ============================================================
# PAGE SUMMARY
# ============================================================

st.subheader(
    "🏆 Final Model Selection"
)


st.success(
    f"""
    **Selected Model: {best_overall['model']}**

    The model achieved the lowest MAE among the evaluated
    forecasting approaches.

    **Model Type:** {best_overall['model_type']}

    **MAE:** {best_overall['MAE']:.2f}

    **RMSE:** {best_overall['RMSE']:.2f}

    **WAPE:** {best_overall['WAPE']:.2f}%
    """
)


# ============================================================
# KPI ROW
# ============================================================

kpi1, kpi2, kpi3, kpi4 = st.columns(4)


with kpi1:

    st.metric(
        "Selected Model",
        str(
            best_overall["model"]
        )
    )


with kpi2:

    st.metric(
        "MAE",
        f"{best_overall['MAE']:.2f}"
    )


with kpi3:

    st.metric(
        "RMSE",
        f"{best_overall['RMSE']:.2f}"
    )


with kpi4:

    st.metric(
        "WAPE",
        f"{best_overall['WAPE']:.2f}%"
    )


st.divider()


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader(
    "📊 Model Accuracy Comparison"
)

st.caption(
    "Lower MAE, RMSE, MAPE and WAPE indicate better forecast accuracy."
)


# ------------------------------------------------------------
# MODEL TABLE
# ------------------------------------------------------------

display_columns = [
    "rank",
    "model",
    "model_type",
    "MAE",
    "RMSE",
    "MAPE",
    "WAPE"
]


available_display_columns = [
    column
    for column in display_columns
    if column in master.columns
]


display_master = master[
    available_display_columns
].copy()


format_dict = {}

for column in [
    "MAE",
    "RMSE"
]:

    if column in display_master.columns:

        format_dict[column] = "{:.2f}"


for column in [
    "MAPE",
    "WAPE"
]:

    if column in display_master.columns:

        format_dict[column] = "{:.2f}%"


if format_dict:

    styled_table = (
        display_master
        .style
        .format(format_dict)
    )

else:

    styled_table = display_master


st.dataframe(
    styled_table,
    width="stretch",
    hide_index=True
)


# ------------------------------------------------------------
# MAE CHART
# ------------------------------------------------------------

fig_mae = px.bar(

    master.sort_values(
        "MAE",
        ascending=True
    ),

    x="model",

    y="MAE",

    color="model_type",

    text="MAE",

    title="MAE Comparison — Lower is Better",

    labels={
        "MAE": "Mean Absolute Error",
        "model": "Model",
        "model_type": "Model Type"
    }
)


fig_mae.update_traces(

    texttemplate="%{text:.2f}",

    textposition="outside"
)


fig_mae.update_layout(

    height=450,

    margin=dict(
        l=40,
        r=40,
        t=70,
        b=60
    ),

    legend_title_text="Model Type"
)


st.plotly_chart(
    fig_mae,
    width="stretch"
)


# ============================================================
# ML VS STATISTICAL
# ============================================================

st.subheader(
    "🤖 Machine Learning vs Statistical Models"
)


if (
    best_ml is not None
    and best_stat is not None
):

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # BEST ML
    # --------------------------------------------------------

    with col1:

        st.info(
            f"""
            ### 🤖 Best Machine Learning Model

            **{best_ml['model']}**

            MAE: **{best_ml['MAE']:.2f}**

            RMSE: **{best_ml['RMSE']:.2f}**

            WAPE: **{best_ml['WAPE']:.2f}%**
            """
        )


    # --------------------------------------------------------
    # BEST STATISTICAL
    # --------------------------------------------------------

    with col2:

        st.warning(
            f"""
            ### 📈 Best Statistical Model

            **{best_stat['model']}**

            MAE: **{best_stat['MAE']:.2f}**

            RMSE: **{best_stat['RMSE']:.2f}**

            WAPE: **{best_stat['WAPE']:.2f}%**
            """
        )


    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    if best_stat["MAE"] != 0:

        improvement = (
            (
                best_stat["MAE"]
                -
                best_ml["MAE"]
            )
            /
            best_stat["MAE"]
        ) * 100


        if improvement >= 0:

            st.metric(
                "ML MAE Improvement vs Statistical Benchmark",
                f"{improvement:.2f}%"
            )

        else:

            st.metric(
                "ML MAE Difference vs Statistical Benchmark",
                f"{improvement:.2f}%"
            )


else:

    st.info(
        "Both Machine Learning and Statistical model groups "
        "are not available for comparison."
    )


st.divider()


# ============================================================
# WAPE COMPARISON
# ============================================================

st.subheader(
    "📊 Weighted Forecast Error"
)

if "WAPE" in master.columns:

    fig_wape = px.bar(

        master.sort_values(
            "WAPE",
            ascending=True
        ),

        x="model",

        y="WAPE",

        color="model_type",

        text="WAPE",

        title="WAPE Comparison — Lower is Better",

        labels={
            "WAPE": "Weighted Absolute Percentage Error (%)",
            "model": "Model"
        }
    )


    fig_wape.update_traces(

        texttemplate="%{text:.2f}%",

        textposition="outside"
    )


    fig_wape.update_layout(

        height=450,

        margin=dict(
            l=40,
            r=40,
            t=70,
            b=60
        )
    )


    st.plotly_chart(
        fig_wape,
        width="stretch"
    )


st.divider()


# ============================================================
# VALIDATION FORECAST
# ============================================================

st.subheader(
    "📈 Validation: Actual Demand vs ML Forecast"
)

st.caption(
    "This chart shows model validation performance. "
    "It should not be interpreted as the future 30/60/90-day forecast."
)


forecast_columns = [
    "units_sold",
    "rf_forecast",
    "xgb_forecast",
    "lgb_forecast"
]


available_forecast_columns = [
    column
    for column in forecast_columns
    if column in forecasts.columns
]


if (
    "date" in forecasts.columns
    and "units_sold" in forecasts.columns
):

    # --------------------------------------------------------
    # Aggregate by date if multiple Store-SKU observations
    # exist.
    # --------------------------------------------------------

    if (
        forecasts[
            ["date"]
        ]
        .drop_duplicates()
        .shape[0]
        <
        len(forecasts)
    ):

        numeric_forecast_columns = [
            column
            for column in available_forecast_columns
            if column != "date"
        ]


        validation_daily = (
            forecasts
            .groupby("date", as_index=False)[
                numeric_forecast_columns
            ]
            .sum()
            .sort_values("date")
        )

    else:

        validation_daily = (
            forecasts[
                ["date"]
                +
                [
                    column
                    for column in available_forecast_columns
                    if column != "date"
                ]
            ]
            .dropna(
                subset=["date"]
            )
            .sort_values("date")
        )


    validation_plot_columns = [
        column
        for column in [
            "units_sold",
            "rf_forecast",
            "xgb_forecast",
            "lgb_forecast"
        ]
        if column in validation_daily.columns
    ]


    if len(validation_plot_columns) >= 2:

        fig_actual = px.line(

            validation_daily,

            x="date",

            y=validation_plot_columns,

            title="Actual Demand vs ML Validation Forecasts",

            labels={
                "date": "Date",
                "value": "Units",
                "variable": "Series"
            }
        )


        fig_actual.update_layout(

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
            fig_actual,
            width="stretch"
        )


    else:

        st.info(
            "Insufficient validation forecast columns "
            "are available for this chart."
        )


else:

    st.info(
        "Validation forecast data is not available."
    )


# ============================================================
# LIGHTGBM FOCUS
# ============================================================

if (
    "date" in forecasts.columns
    and "units_sold" in forecasts.columns
    and "lgb_forecast" in forecasts.columns
):

    st.subheader(
        "🏆 LightGBM Validation Performance"
    )

    lgb_columns = [
        "date",
        "units_sold",
        "lgb_forecast"
    ]


    lgb_daily = (
        forecasts[
            lgb_columns
        ]
        .dropna(
            subset=["date"]
        )
        .groupby(
            "date",
            as_index=False
        )[
            [
                "units_sold",
                "lgb_forecast"
            ]
        ]
        .sum()
        .sort_values("date")
    )


    fig_lgb = px.line(

        lgb_daily,

        x="date",

        y=[
            "units_sold",
            "lgb_forecast"
        ],

        title="Actual vs LightGBM Validation Forecast",

        labels={
            "date": "Date",
            "value": "Units",
            "variable": "Series"
        }
    )


    fig_lgb.update_layout(

        height=450,

        hovermode="x unified",

        margin=dict(
            l=40,
            r=40,
            t=70,
            b=40
        )
    )


    st.plotly_chart(
        fig_lgb,
        width="stretch"
    )


st.divider()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "🎯 Top Model Predictors"
)

st.caption(
    "Feature importance shows which variables contributed most "
    "to the selected ML model's predictions. It does not establish causation."
)


if (
    not importance.empty
    and "average_importance"
    in importance.columns
    and "feature"
    in importance.columns
):

    top_features = (
        importance
        .sort_values(
            "average_importance",
            ascending=False
        )
        .head(15)
        .copy()
    )


    fig_features = px.bar(

        top_features.sort_values(
            "average_importance",
            ascending=True
        ),

        x="average_importance",

        y="feature",

        orientation="h",

        title="Top 15 ML Model Features",

        labels={
            "average_importance": "Average Importance",
            "feature": "Feature"
        }
    )


    fig_features.update_layout(

        height=600,

        margin=dict(
            l=40,
            r=60,
            t=70,
            b=40
        )
    )


    st.plotly_chart(
        fig_features,
        width="stretch"
    )


else:

    st.info(
        "Feature importance data is not available."
    )


st.divider()


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

st.subheader(
    "💡 Forecasting Interpretation"
)


overall_model_name = str(
    best_overall["model"]
)


overall_model_type = str(
    best_overall["model_type"]
)


if (
    best_ml is not None
    and best_stat is not None
):

    if best_ml["MAE"] < best_stat["MAE"]:

        comparison_message = (
            f"The best machine-learning model "
            f"({best_ml['model']}) achieved lower MAE "
            f"than the best statistical benchmark "
            f"({best_stat['model']})."
        )

    elif best_ml["MAE"] > best_stat["MAE"]:

        comparison_message = (
            f"The best statistical model "
            f"({best_stat['model']}) achieved lower MAE "
            f"than the best machine-learning model "
            f"({best_ml['model']})."
        )

    else:

        comparison_message = (
            "The best machine-learning and statistical "
            "models achieved the same MAE."
        )

else:

    comparison_message = (
        "A complete ML-versus-statistical comparison "
        "is not available in the current model results."
    )


st.markdown(
    f"""
### Key Findings

**🏆 Selected Model**

The current model-selection results identify
**{overall_model_name}** as the best-performing model
based on the lowest available MAE.

**📊 Model Type**

The selected model belongs to the
**{overall_model_type}** category.

**📉 Forecast Accuracy**

The selected model has:

- MAE: **{best_overall['MAE']:.2f}**
- RMSE: **{best_overall['RMSE']:.2f}**
- WAPE: **{best_overall['WAPE']:.2f}%**

**🤖 ML vs Statistical Benchmark**

{comparison_message}

**🎯 Model Interpretation**

The feature-importance analysis identifies the variables
that contributed most to the ML model's predictions.
These should be interpreted as predictive signals rather
than direct causal drivers.

**📌 Business Use**

The selected model can support demand planning and inventory
decisions, while the statistical model remains useful as a
benchmark for monitoring model stability.
"""
)


# ============================================================
# SOURCE DATA
# ============================================================

with st.expander(
    "🔍 View Forecasting Source Data"
):

    source_tabs = st.tabs(
        [
            "Model Comparison",
            "Best Model",
            "Advanced Comparison",
            "Validation Forecasts",
            "Feature Importance"
        ]
    )


    with source_tabs[0]:

        st.dataframe(
            master,
            width="stretch",
            hide_index=True
        )


    with source_tabs[1]:

        st.dataframe(
            best,
            width="stretch",
            hide_index=True
        )


    with source_tabs[2]:

        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True
        )


    with source_tabs[3]:

        st.dataframe(
            forecasts,
            width="stretch",
            hide_index=True
        )


    with source_tabs[4]:

        st.dataframe(
            importance,
            width="stretch",
            hide_index=True
        )