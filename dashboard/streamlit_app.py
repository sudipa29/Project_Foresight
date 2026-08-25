import streamlit as st
from pathlib import Path
import runpy


# ============================================================
# PROJECT FORESIGHT
# MAIN STREAMLIT APPLICATION
# ============================================================


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
# DASHBOARD DIRECTORY
# ============================================================

DASHBOARD_DIR = Path(__file__).resolve().parent


# ============================================================
# PAGE RUNNER
# ============================================================

def run_dashboard_page(filename):
    """
    Execute a dashboard page using its absolute path.
    """

    page_path = DASHBOARD_DIR / filename

    if not page_path.exists():

        st.error(
            f"Dashboard page not found:\n\n{page_path}"
        )

        st.stop()

    runpy.run_path(
        str(page_path),
        run_name="__main__"
    )


# ============================================================
# PAGE FUNCTIONS
# ============================================================

def page_overview():
    run_dashboard_page("overview.py")


def page_demand_analysis():
    run_dashboard_page("demand_analysis.py")


def page_demand_drivers():
    run_dashboard_page("demand_drivers.py")


def page_forecasting():
    run_dashboard_page("forecasting.py")


def page_future_forecast():
    run_dashboard_page("future_forecast.py")


def page_inventory():
    run_dashboard_page("inventory.py")


def page_risk_analysis():
    run_dashboard_page("risk_analysis.py")


def page_store_intelligence():
    run_dashboard_page("store_intelligence.py")


def page_sku_performance():
    run_dashboard_page("sku_performance.py")


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

        # ----------------------------------------------------
        # PAGE 1
        # ----------------------------------------------------

        st.Page(
            page_overview,
            title="Executive Overview",
            icon="📊"
        ),


        # ----------------------------------------------------
        # PAGE 2
        # ----------------------------------------------------

        st.Page(
            page_demand_analysis,
            title="Demand Analysis",
            icon="📈"
        ),


        # ----------------------------------------------------
        # PAGE 3
        # ----------------------------------------------------

        st.Page(
            page_demand_drivers,
            title="Demand Drivers",
            icon="🎯"
        ),


        # ----------------------------------------------------
        # PAGE 4
        # ----------------------------------------------------

        st.Page(
            page_forecasting,
            title="Model Forecasting",
            icon="🤖"
        ),


        # ----------------------------------------------------
        # PAGE 5
        # ----------------------------------------------------

        st.Page(
            page_future_forecast,
            title="Future Forecast",
            icon="🔮"
        ),


        # ----------------------------------------------------
        # PAGE 6
        # ----------------------------------------------------

        st.Page(
            page_inventory,
            title="Inventory Intelligence",
            icon="📦"
        ),


        # ----------------------------------------------------
        # PAGE 7
        # ----------------------------------------------------

        st.Page(
            page_risk_analysis,
            title="Risk & Insights",
            icon="🚨"
        ),


        # ----------------------------------------------------
        # PAGE 8
        # ----------------------------------------------------

        st.Page(
            page_store_intelligence,
            title="Store Intelligence",
            icon="🏬"
        ),


        # ----------------------------------------------------
        # PAGE 9
        # ----------------------------------------------------

        st.Page(
            page_sku_performance,
            title="SKU Performance",
            icon="🏷️"
        ),

    ]
)


# ============================================================
# RUN SELECTED PAGE
# ============================================================

pg.run()