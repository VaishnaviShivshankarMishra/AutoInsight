# app.py
import streamlit as st
import pandas as pd
from datetime import datetime

from modules.dashboard import show_dashboard, remove_outliers_iqr, encode_columns, scale_columns
from modules.feature_engineering import apply_pca
from modules.feature_selection import select_features

# ---------------- Project Header ----------------
st.title("🛠️ AutoInsight - Automated Data Cleaning & Dashboard Generator")
st.markdown(f"**Developed by:** Vaishnavi Mishra  |  **Date:** {datetime.today().strftime('%Y-%m-%d')}")
st.markdown("---")

# ---------------- Session State ----------------
if "start_clicked" not in st.session_state:
    st.session_state.start_clicked = False

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"Dataset '{uploaded_file.name}' loaded successfully!")

    # ---------------- Start Button ----------------
    if st.button("Start Analysis"):
        st.session_state.start_clicked = True

    if st.session_state.start_clicked:

        # ---------------- Sidebar ----------------
        st.sidebar.header("Data Processing Options")

        # Multi-select cleaning options
        cleaning_options = st.sidebar.multiselect(
            "Data Cleaning",
            ["Fill Missing Values", "Remove Duplicates", "Remove Outliers (IQR)"]
        )

        encoding_option = st.sidebar.selectbox("Encoding", ["None", "Label Encoding", "One-Hot Encoding"])
        scaling_option = st.sidebar.selectbox("Scaling", ["None", "StandardScaler", "MinMaxScaler"])
        dataset_tab = st.sidebar.radio(
            "Select Dataset for Dashboard",
            ["Raw Dataset", "Cleaned Dataset", "PCA Dataset", "Feature-Selected Dataset"]
        )

        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=["object", "category"]).columns.tolist()

        # ---------------- Apply Cleaning ----------------
        if dataset_tab != "Raw Dataset":
            if "Fill Missing Values" in cleaning_options:
                for col in numeric_cols:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                for col in categorical_cols:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

            if "Remove Duplicates" in cleaning_options:
                df_processed.drop_duplicates(inplace=True)

            if "Remove Outliers (IQR)" in cleaning_options and numeric_cols:
                df_processed = remove_outliers_iqr(df_processed, numeric_cols)

            if encoding_option != "None":
                df_processed = encode_columns(df_processed, encoding_option)

            if scaling_option != "None":
                df_processed = scale_columns(df_processed, scaling_option)

        # ---------------- Apply PCA ----------------
        if dataset_tab == "PCA Dataset":
            n_components = st.sidebar.slider(
                "PCA Components", 1, min(len(numeric_cols), 10), 2
            )
            df_processed = apply_pca(df_processed, n_components=n_components)

        # ---------------- Feature Selection ----------------
        if dataset_tab == "Feature-Selected Dataset":
            threshold = st.sidebar.slider(
                "Variance Threshold", 0.0, 1.0, 0.1
            )
            df_processed = select_features(df_processed, method="variance", threshold=threshold)

        # ---------------- Show Dashboard ----------------
        show_dashboard(df_processed, dataset_name=dataset_tab)

else:
    st.info("Upload a dataset to start analysis.")