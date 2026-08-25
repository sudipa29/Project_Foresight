# src/utils.py
from pathlib import Path
import streamlit as st
import pandas as pd
import sys

# Dynamic root pointing to Project_Foresight/
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data" / "processed" / "forecasting"
MODELS_DIR = ROOT_DIR / "Models"

@st.cache_data(show_spinner="Loading data...")
def load_dataset(file_path: Path, optional: bool = False):
    """Safely loads a CSV or Parquet file from a dynamic Path object."""
    if not file_path.exists():
        if optional:
            st.warning(f"⚠️ Optional file missing: `{file_path.name}`")
            return None
        else:
            st.error(f"❌ **Critical Data File Missing**\n\nPath: `{file_path}`")
            st.info("💡 Ensure this file is committed and pushed to GitHub, and not blocked by `.gitignore`.")
            st.stop()

    if file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix in [".parquet", ".pq"]:
        return pd.read_parquet(file_path)
    else:
        st.error(f"Unsupported format: `{file_path.suffix}`")
        st.stop()