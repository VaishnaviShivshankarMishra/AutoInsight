import streamlit as st
import pandas as pd

from modules.cleaning import show_cleaning
from modules.outlier_handling import show_outlier_handling
from modules.feature_engineering import show_feature_engineering
from modules.feature_selection import show_feature_selection
from modules.eda import show_eda
from modules.dashboard import show_eda_dashboard


# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="AutoInsight", layout="wide")

st.title("🚀 AutoInsight - Automated Data Cleaning & Dashboard Generator")


# ----------------------------
# Session State Initialization
# ----------------------------
def initialize_session_state():
    defaults = {
        "raw_df": None,
        "cleaned_df": None,
        "processed_df": None,
        "uploaded_file_name": None,
        "transformation_log": []
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ----------------------------
# Helper Function - File Loader
# ----------------------------
def load_file(uploaded_file):
    """
    Loads CSV or Excel file safely with encoding fallback for CSV.
    """
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding="latin1")
            except Exception:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file)

        elif file_name.endswith((".xlsx", ".xls")):
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)

        else:
            st.error("Unsupported file type. Please upload CSV or Excel.")
            return None

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# ----------------------------
# Helper Function - Reset Dataset
# ----------------------------
def reset_to_original():
    if st.session_state.raw_df is not None:
        st.session_state.cleaned_df = st.session_state.raw_df.copy()
        st.session_state.processed_df = st.session_state.raw_df.copy()
        st.session_state.transformation_log = []


# ----------------------------
# Helper Function - Get Current Dataset
# ----------------------------
def get_current_df():
    if st.session_state.processed_df is not None:
        return st.session_state.processed_df
    elif st.session_state.cleaned_df is not None:
        return st.session_state.cleaned_df
    elif st.session_state.raw_df is not None:
        return st.session_state.raw_df
    return None


# ----------------------------
# Sidebar - File Upload
# ----------------------------
st.sidebar.header("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    # Reload only when a new file is uploaded
    if st.session_state.uploaded_file_name != uploaded_file.name:
        df = load_file(uploaded_file)

        if df is not None:
            st.session_state.raw_df = df.copy()
            st.session_state.cleaned_df = df.copy()
            st.session_state.processed_df = df.copy()
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.transformation_log = []

            st.sidebar.success(f"Loaded file: {uploaded_file.name}")


# ----------------------------
# Sidebar - Reset
# ----------------------------
if st.session_state.raw_df is not None:
    if st.sidebar.button("🔄 Reset to Original Dataset"):
        reset_to_original()
        st.sidebar.success("Dataset reset to original.")


# ----------------------------
# Main App
# ----------------------------
if st.session_state.raw_df is None:
    st.info("Please upload a dataset to begin.")
else:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📁 Dataset Overview",
        "🧹 Data Cleaning",
        "📉 Outlier Handling",
        "🛠 Feature Engineering",
        "🎯 Feature Selection",
        "📊 EDA",
        "📈 Dashboard",
        "💾 Download",
        "📝 Transformation Log"
    ])

    current_df = get_current_df()

    # ----------------------------
    # Tab 1 - Dataset Overview
    # ----------------------------
    with tab1:
        st.subheader("📁 Dataset Overview")

        try:
            if current_df is None or current_df.empty:
                st.warning("No dataset available.")
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Rows", current_df.shape[0])
                col2.metric("Columns", current_df.shape[1])
                col3.metric("Missing Values", int(current_df.isnull().sum().sum()))
                col4.metric("Duplicate Rows", int(current_df.duplicated().sum()))

                st.write("### Data Preview")
                st.dataframe(current_df.head(), use_container_width=True)

                st.write("### Column Summary")
                summary_df = pd.DataFrame({
                    "Column": current_df.columns,
                    "Data Type": current_df.dtypes.astype(str).values,
                    "Missing Values": current_df.isnull().sum().values,
                    "Unique Values": current_df.nunique(dropna=True).values
                })
                st.dataframe(summary_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error displaying dataset overview: {e}")

    # ----------------------------
    # Tab 2 - Data Cleaning
    # ----------------------------
    with tab2:
        try:
            show_cleaning()
        except Exception as e:
            st.error(f"Error in Data Cleaning module: {e}")

    # ----------------------------
    # Tab 3 - Outlier Handling
    # ----------------------------
    with tab3:
        try:
            show_outlier_handling()
        except Exception as e:
            st.error(f"Error in Outlier Handling module: {e}")

    # ----------------------------
    # Tab 4 - Feature Engineering
    # ----------------------------
    with tab4:
        try:
            show_feature_engineering()
        except Exception as e:
            st.error(f"Error in Feature Engineering module: {e}")

    # ----------------------------
    # Tab 5 - Feature Selection
    # ----------------------------
    with tab5:
        try:
            show_feature_selection()
        except Exception as e:
            st.error(f"Error in Feature Selection module: {e}")

    # ----------------------------
    # Tab 6 - EDA
    # ----------------------------
    with tab6:
        try:
            latest_df_for_eda = get_current_df()

            if latest_df_for_eda is None or latest_df_for_eda.empty:
                st.warning("No dataset available for EDA.")
            else:
                show_eda(latest_df_for_eda)

        except Exception as e:
            st.error(f"Error in EDA module: {e}")

    # ----------------------------
    # Tab 7 - Dashboard
    # ----------------------------
    with tab7:
        try:
            show_eda_dashboard()
        except Exception as e:
            st.error(f"Error in Dashboard module: {e}")

    # ----------------------------
    # Tab 8 - Download
    # ----------------------------
    with tab8:
        st.subheader("💾 Download Datasets")

        try:
            if st.session_state.raw_df is not None:
                st.download_button(
                    label="⬇ Download Raw Dataset",
                    data=st.session_state.raw_df.to_csv(index=False).encode("utf-8"),
                    file_name="raw_dataset.csv",
                    mime="text/csv"
                )

            if st.session_state.processed_df is not None:
                st.download_button(
                    label="⬇ Download Processed Dataset",
                    data=st.session_state.processed_df.to_csv(index=False).encode("utf-8"),
                    file_name="processed_dataset.csv",
                    mime="text/csv"
                )
            else:
                st.info("No processed dataset available yet.")

        except Exception as e:
            st.error(f"Error while preparing download: {e}")

    # ----------------------------
    # Tab 9 - Transformation Log
    # ----------------------------
    with tab9:
        st.subheader("📝 Transformation Log")

        try:
            if st.session_state.transformation_log:
                for i, log in enumerate(st.session_state.transformation_log, 1):
                    st.write(f"{i}. {log}")
            else:
                st.info("No transformations applied yet.")

        except Exception as e:
            st.error(f"Error displaying transformation log: {e}")