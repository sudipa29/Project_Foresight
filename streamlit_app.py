import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Project Foresight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 Project Foresight")

st.caption(
    "Demand Intelligence & Machine Learning Forecasting Platform"
)


# ============================================================
# NAVIGATION
# ============================================================

pg = st.navigation(
    [
        st.Page(
            "dashboard/overview.py",
            title="Executive Overview",
            icon="🏠"
        ),

        st.Page(
            "dashboard/demand_analysis.py",
            title="Demand Analysis",
            icon="📈"
        ),

        st.Page(
            "dashboard/demand_drivers.py",
            title="Demand Drivers",
            icon="🎯"
        ),

        st.Page(
            "dashboard/forecasting.py",
            title="Model Forecasting",
            icon="🤖"
        ),

        st.Page(
            "dashboard/future_forecast.py",
            title="Future Forecast",
            icon="🔮"
        ),

        st.Page(
            "dashboard/inventory.py",
            title="Inventory Intelligence",
            icon="📦"
        ),

        st.Page(
            "dashboard/risk_analysis.py",
            title="Risk & Insights",
            icon="🚨"
        ),

        st.Page(
            "dashboard/store_intelligence.py",
            title="Store Intelligence",
            icon="🏬"
        ),
    ]
)


# ============================================================
# RUN APPLICATION
# ============================================================

pg.run()