# ============================================================
# PROJECT FORESIGHT
# PAGE 3 - DEMAND DRIVERS
# ============================================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🎯 Demand Drivers")

st.caption(
    "Analyze how discounts, promotions, stores, categories and "
    "seasonality influence product demand."
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DRIVER_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "demand_driver_analysis"
)


# ============================================================
# FILE LOADER
# ============================================================

@st.cache_data
def load_csv(filename):

    path = os.path.join(
        DRIVER_DIR,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return pd.read_csv(path)


# ============================================================
# LOAD DATA
# ============================================================

try:

    category_discount = load_csv(
        "category_discount.csv"
    )

    category_promotion = load_csv(
        "category_promotion.csv"
    )

    discount_analysis = load_csv(
        "discount_analysis.csv"
    )

    high_discount_risk = load_csv(
        "high_discount_risk.csv"
    )

    month_promotion = load_csv(
        "month_promotion.csv"
    )

    promotion_analysis = load_csv(
        "promotion_analysis.csv"
    )

    store_promotion = load_csv(
        "store_promotion.csv"
    )

    weekday_promotion = load_csv(
        "weekday_promotion.csv"
    )

    correlation_matrix = load_csv(
        "correlation_matrix.csv"
    )


except Exception as e:

    st.error(
        "❌ Demand Driver data could not be loaded."
    )

    st.code(
        str(e)
    )

    st.info(
        f"Expected folder:\n{DRIVER_DIR}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🎯 Demand Driver Filters"
)

st.sidebar.caption(
    "Use the available filters to focus the analysis."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(
    dataframe,
    keywords,
    exclude_keywords=None
):

    """
    Find the first column whose name matches
    one of the supplied keywords.
    """

    exclude_keywords = exclude_keywords or []

    for column in dataframe.columns:

        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in keywords
        ):

            if not any(
                keyword in column_lower
                for keyword in exclude_keywords
            ):

                return column

    return None


def find_numeric_column(
    dataframe,
    keywords=None
):

    """
    Find a numeric column using keyword matching.
    """

    numeric_columns = dataframe.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:
        return None

    if keywords:

        for column in numeric_columns:

            column_lower = column.lower()

            if any(
                keyword in column_lower
                for keyword in keywords
            ):

                return column

    return numeric_columns[0]


# ============================================================
# OPTIONAL CATEGORY FILTER
# ============================================================

category_filter = None

if "category" in category_discount.columns:

    categories = sorted(
        category_discount["category"]
        .dropna()
        .astype(str)
        .unique()
    )

    category_filter = st.sidebar.multiselect(
        "Select Category(s)",
        options=categories,
        default=[]
    )


# ============================================================
# FILTER CATEGORY DATA
# ============================================================

filtered_category_discount = category_discount.copy()

filtered_category_promotion = category_promotion.copy()


if category_filter:

    if "category" in filtered_category_discount.columns:

        filtered_category_discount = (
            filtered_category_discount[
                filtered_category_discount["category"]
                .astype(str)
                .isin(category_filter)
            ]
        )

    if "category" in filtered_category_promotion.columns:

        filtered_category_promotion = (
            filtered_category_promotion[
                filtered_category_promotion["category"]
                .astype(str)
                .isin(category_filter)
            ]
        )


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader(
    "📊 Demand Driver Overview"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Driver Datasets",
        "9"
    )


with col2:

    st.metric(
        "Categories Analyzed",
        category_discount.shape[0]
    )


with col3:

    st.metric(
        "Stores Analyzed",
        store_promotion.shape[0]
    )


with col4:

    st.metric(
        "High Discount Records",
        high_discount_risk.shape[0]
    )


st.divider()


# ============================================================
# SECTION 1
# DISCOUNT IMPACT
# ============================================================

st.subheader(
    "🏷️ Discount Impact by Category"
)

st.caption(
    "Explore the relationship between discount levels and demand."
)


if not filtered_category_discount.empty:

    numeric_columns = (
        filtered_category_discount
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    demand_col = find_numeric_column(
        filtered_category_discount,
        [
            "unit",
            "demand",
            "sales"
        ]
    )

    discount_col = find_column(
        filtered_category_discount,
        [
            "discount"
        ]
    )


    if demand_col and discount_col:

        fig_discount = px.scatter(

            filtered_category_discount,

            x=discount_col,

            y=demand_col,

            color=(
                "category"
                if "category"
                in filtered_category_discount.columns
                else None
            ),

            size=demand_col,

            hover_data=(
                filtered_category_discount.columns.tolist()
            ),

            labels={
                discount_col: "Discount",
                demand_col: "Demand"
            },

            title="Discount vs Demand by Category"
        )


        fig_discount.update_layout(

            height=500,

            margin=dict(
                l=40,
                r=40,
                t=60,
                b=40
            ),

            legend_title_text="Category"
        )


        st.plotly_chart(
            fig_discount,
            width="stretch"
        )


    else:

        st.info(
            "Required discount and demand columns "
            "were not identified."
        )


else:

    st.info(
        "No category discount data is available "
        "for the selected filters."
    )


# ============================================================
# OPTIONAL CATEGORY DISCOUNT DATA
# ============================================================

with st.expander(
    "🔍 View Category Discount Data"
):

    st.dataframe(
        filtered_category_discount,
        width="stretch",
        hide_index=True
    )


st.divider()


# ============================================================
# SECTION 2
# PROMOTION IMPACT
# ============================================================

st.subheader(
    "📣 Promotion Impact"
)

st.caption(
    "Evaluate promotional activity across categories "
    "and overall promotion groups."
)


# ============================================================
# CATEGORY PROMOTION
# ============================================================

st.markdown(
    "#### 📂 Promotion Impact by Category"
)


if not filtered_category_promotion.empty:

    demand_col = find_numeric_column(
        filtered_category_promotion,
        [
            "unit",
            "demand",
            "sales"
        ]
    )

    promotion_col = find_column(
        filtered_category_promotion,
        [
            "promotion",
            "promo"
        ]
    )


    if (
        demand_col
        and "category"
        in filtered_category_promotion.columns
    ):

        fig_category_promotion = px.bar(

            filtered_category_promotion,

            x="category",

            y=demand_col,

            color=(
                promotion_col
                if promotion_col
                else None
            ),

            labels={
                demand_col: "Demand"
            },

            title="Category Demand by Promotion"
        )


        fig_category_promotion.update_layout(

            height=450,

            margin=dict(
                l=40,
                r=40,
                t=60,
                b=40
            )
        )


        st.plotly_chart(
            fig_category_promotion,
            width="stretch"
        )


    else:

        st.info(
            "Category promotion columns could not "
            "be identified."
        )


# ============================================================
# PROMOTION PERFORMANCE
# ============================================================

st.markdown(
    "#### 📈 Promotion Performance"
)


if not promotion_analysis.empty:

    numeric_columns = (
        promotion_analysis
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )


    if numeric_columns:

        promotion_col = find_column(
            promotion_analysis,
            [
                "promotion",
                "promo"
            ]
        )


        x_col = (
            promotion_col
            if promotion_col
            else promotion_analysis.columns[0]
        )


        y_col = find_numeric_column(
            promotion_analysis,
            [
                "unit",
                "demand",
                "sales"
            ]
        )


        fig_promotion = px.bar(

            promotion_analysis,

            x=x_col,

            y=y_col,

            text=y_col,

            labels={
                y_col: "Demand"
            },

            title="Promotion Performance"
        )


        fig_promotion.update_traces(

            texttemplate="%{text:,.0f}",

            textposition="outside"
        )


        fig_promotion.update_layout(

            height=450,

            margin=dict(
                l=40,
                r=60,
                t=60,
                b=40
            )
        )


        st.plotly_chart(
            fig_promotion,
            width="stretch"
        )


    else:

        st.info(
            "No numeric promotion metric was found."
        )


st.divider()


# ============================================================
# SECTION 3
# STORE PROMOTION
# ============================================================

st.subheader(
    "🏬 Store Promotion Performance"
)

st.caption(
    "Identify stores where promotional activity is associated "
    "with stronger demand."
)


if not store_promotion.empty:

    numeric_columns = (
        store_promotion
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )


    if numeric_columns:

        store_col = find_column(
            store_promotion,
            [
                "store",
                "branch"
            ]
        )


        y_col = find_numeric_column(
            store_promotion,
            [
                "unit",
                "demand",
                "sales"
            ]
        )


        if store_col:

            store_chart = (
                store_promotion
                .sort_values(
                    y_col,
                    ascending=True
                )
                .copy()
            )


            fig_store = px.bar(

                store_chart,

                x=y_col,

                y=store_col,

                orientation="h",

                text=y_col,

                labels={
                    y_col: "Demand",
                    store_col: "Store"
                },

                title="Store Promotion Performance"
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
                    t=60,
                    b=40
                )
            )


            st.plotly_chart(
                fig_store,
                width="stretch"
            )


        else:

            st.info(
                "Store identifier column was not found."
            )


    else:

        st.info(
            "No numeric store-promotion metric was found."
        )


st.divider()


# ============================================================
# SECTION 4
# CALENDAR EFFECT
# ============================================================

st.subheader(
    "📅 Calendar & Promotion Patterns"
)

st.caption(
    "Review weekday and monthly promotional patterns "
    "that may influence demand."
)


calendar_col1, calendar_col2 = st.columns(2)


# ============================================================
# WEEKDAY
# ============================================================

with calendar_col1:

    st.markdown(
        "#### 📅 Promotion by Weekday"
    )


    if not weekday_promotion.empty:

        x_col = (
            weekday_promotion.columns[0]
        )

        y_col = find_numeric_column(
            weekday_promotion,
            [
                "unit",
                "demand",
                "sales",
                "promotion",
                "promo"
            ]
        )


        if y_col:

            fig_weekday = px.bar(

                weekday_promotion,

                x=x_col,

                y=y_col,

                labels={
                    y_col: "Promotion / Demand"
                },

                title="Weekday Promotion Pattern"
            )


            fig_weekday.update_layout(

                height=420,

                margin=dict(
                    l=40,
                    r=40,
                    t=60,
                    b=40
                )
            )


            st.plotly_chart(
                fig_weekday,
                width="stretch"
            )


        else:

            st.info(
                "No numeric weekday metric found."
            )


# ============================================================
# MONTHLY
# ============================================================

with calendar_col2:

    st.markdown(
        "#### 🗓️ Monthly Promotion Pattern"
    )


    if not month_promotion.empty:

        x_col = (
            month_promotion.columns[0]
        )

        y_col = find_numeric_column(
            month_promotion,
            [
                "unit",
                "demand",
                "sales",
                "promotion",
                "promo"
            ]
        )


        if y_col:

            fig_month = px.line(

                month_promotion,

                x=x_col,

                y=y_col,

                markers=True,

                labels={
                    y_col: "Promotion / Demand"
                },

                title="Monthly Promotion Pattern"
            )


            fig_month.update_layout(

                height=420,

                margin=dict(
                    l=40,
                    r=40,
                    t=60,
                    b=40
                )
            )


            st.plotly_chart(
                fig_month,
                width="stretch"
            )


        else:

            st.info(
                "No numeric monthly metric found."
            )


st.divider()


# ============================================================
# SECTION 5
# DISCOUNT RISK
# ============================================================

st.subheader(
    "⚠️ High Discount Risk"
)


if not high_discount_risk.empty:

    st.warning(

        f"{len(high_discount_risk):,} records identified "
        "in the high-discount analysis."
    )


    st.dataframe(

        high_discount_risk,

        width="stretch",

        height=350,

        hide_index=True
    )


else:

    st.success(
        "No high-discount risk records were identified."
    )


st.divider()


# ============================================================
# SECTION 6
# CORRELATION MATRIX
# ============================================================

st.subheader(
    "🔗 Demand Driver Correlation"
)

st.caption(
    "Correlation provides an analytical indication of relationships "
    "between numeric demand drivers. It does not imply causation."
)


if not correlation_matrix.empty:

    numeric_corr = (
        correlation_matrix
        .select_dtypes(include=np.number)
    )


    if not numeric_corr.empty:

        fig_corr = px.imshow(

            numeric_corr,

            text_auto=".2f",

            aspect="auto",

            labels={
                "color": "Correlation"
            },

            title="Demand Driver Correlation Matrix"
        )


        fig_corr.update_layout(

            height=600,

            margin=dict(
                l=40,
                r=40,
                t=70,
                b=40
            )
        )


        st.plotly_chart(

            fig_corr,

            width="stretch"
        )


    else:

        st.info(
            "No numeric correlation data is available."
        )


else:

    st.info(
        "Correlation matrix is not available."
    )


st.divider()


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.subheader(
    "💡 Demand Driver Insights"
)


st.markdown(
    """
### Key areas to monitor

**🏷️ Discount Effect**

Analyze whether higher discounts are associated with higher
unit demand and whether the additional volume justifies the
reduction in margin.

**📣 Promotion Effect**

Compare promotional activity and demand patterns to identify
whether promotions appear to generate stronger sales activity.

**🏬 Store Effect**

Identify stores where promotional activity is associated with
stronger demand and investigate differences in store behavior.

**📅 Calendar Effect**

Weekday and monthly patterns can help determine when
promotions should be concentrated.

**⚠️ Discount Risk**

High-discount records should be monitored because increased
sales volume does not necessarily mean increased profitability.

**🔗 Correlation**

Use correlation as a screening tool for relationships between
demand drivers. Correlation alone should not be interpreted as
proof of causation.
"""
)


# ============================================================
# SOURCE DATA
# ============================================================

with st.expander(
    "🔍 View Demand Driver Source Data"
):

    tabs = st.tabs(
        [
            "Category Discount",
            "Category Promotion",
            "Promotion",
            "Store Promotion",
            "Weekday",
            "Monthly",
            "High Discount",
            "Correlation"
        ]
    )


    with tabs[0]:

        st.dataframe(
            category_discount,
            width="stretch",
            hide_index=True
        )


    with tabs[1]:

        st.dataframe(
            category_promotion,
            width="stretch",
            hide_index=True
        )


    with tabs[2]:

        st.dataframe(
            promotion_analysis,
            width="stretch",
            hide_index=True
        )


    with tabs[3]:

        st.dataframe(
            store_promotion,
            width="stretch",
            hide_index=True
        )


    with tabs[4]:

        st.dataframe(
            weekday_promotion,
            width="stretch",
            hide_index=True
        )


    with tabs[5]:

        st.dataframe(
            month_promotion,
            width="stretch",
            hide_index=True
        )


    with tabs[6]:

        st.dataframe(
            high_discount_risk,
            width="stretch",
            hide_index=True
        )


    with tabs[7]:

        st.dataframe(
            correlation_matrix,
            width="stretch",
            hide_index=True
        )