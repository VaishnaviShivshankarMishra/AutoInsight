import streamlit as st
import pandas as pd
from datetime import datetime

from modules.dashboard import show_dashboard, remove_outliers_iqr, encode_columns, scale_columns
from modules.feature_engineering import apply_pca
from modules.feature_selection import select_features

# ---------------- File Loader Function ----------------
def load_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    try:
        # Handle CSV files with multiple encodings
        if file_name.endswith(".csv"):
            encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]

            for enc in encodings:
                try:
                    uploaded_file.seek(0)  # reset file pointer before each attempt
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    st.info(f"CSV loaded using encoding: {enc}")
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception:
                    continue

            # Final fallback
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines="skip")
            st.warning("Some problematic rows were skipped while loading the CSV.")
            return df

        # Handle Excel files
        elif file_name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)

        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


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
    df = load_file(uploaded_file)

    if df is not None:
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
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())

                    for col in categorical_cols:
                        if not df_processed[col].mode().empty:
                            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

                if "Remove Duplicates" in cleaning_options:
                    df_processed.drop_duplicates(inplace=True)

                if "Remove Outliers (IQR)" in cleaning_options and numeric_cols:
                    df_processed = remove_outliers_iqr(df_processed, numeric_cols)

                if encoding_option != "None":
                    df_processed = encode_columns(df_processed, encoding_option)

                if scaling_option != "None":
                    df_processed = scale_columns(df_processed, scaling_option)

            # Recalculate numeric columns after cleaning/encoding/scaling
            numeric_cols_processed = df_processed.select_dtypes(include=["number"]).columns.tolist()

            # ---------------- Apply PCA ----------------
            if dataset_tab == "PCA Dataset":
                if len(numeric_cols_processed) >= 1:
                    max_components = min(len(numeric_cols_processed), 10)
                    default_components = min(2, max_components)

                    n_components = st.sidebar.slider(
                        "PCA Components", 1, max_components, default_components
                    )
                    df_processed = apply_pca(df_processed, n_components=n_components)
                else:
                    st.warning("PCA requires at least one numeric column after preprocessing.")

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