import streamlit as st
import pandas as pd
from datetime import datetime

from modules.dashboard import show_dashboard, remove_outliers_iqr, encode_columns, scale_columns
from modules.feature_engineering import apply_pca
from modules.feature_selection import select_features

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AutoInsight",
    page_icon="🛠️",
    layout="wide"
)

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
        elif file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)

        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# ---------------- Helper: Convert DataFrame to CSV ----------------
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


# ---------------- Session State ----------------
if "start_clicked" not in st.session_state:
    st.session_state.start_clicked = False

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# ---------------- Project Header ----------------
st.title("🛠️ AutoInsight - Automated Data Cleaning & Dashboard Generator")
st.markdown(f"**Developed by:** Vaishnavi Mishra  |  **Date:** {datetime.today().strftime('%Y-%m-%d')}")
st.markdown("---")


# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Reset Start Analysis if a new file is uploaded
    if st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.start_clicked = False
        st.session_state.uploaded_file_name = uploaded_file.name

    df = load_file(uploaded_file)

    if df is not None:
        st.success(f"Dataset '{uploaded_file.name}' loaded successfully!")

        # ---------------- Dataset Preview ----------------
        st.subheader("📄 Dataset Preview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        with st.expander("View First 10 Rows"):
            st.dataframe(df.head(10), use_container_width=True)

        # ---------------- Start / Reset Buttons ----------------
        col_start, col_reset = st.columns([1, 1])

        with col_start:
            if st.button("▶️ Start Analysis"):
                st.session_state.start_clicked = True

        with col_reset:
            if st.button("🔄 Reset Analysis"):
                st.session_state.start_clicked = False
                st.rerun()

        # ---------------- Main Analysis ----------------
        if st.session_state.start_clicked:

            # ---------------- Sidebar ----------------
            st.sidebar.header("⚙️ Data Processing Options")

            cleaning_options = st.sidebar.multiselect(
                "Data Cleaning",
                ["Fill Missing Values", "Remove Duplicates", "Remove Outliers (IQR)"]
            )

            encoding_option = st.sidebar.selectbox(
                "Encoding",
                ["None", "Label Encoding", "One-Hot Encoding"]
            )

            scaling_option = st.sidebar.selectbox(
                "Scaling",
                ["None", "StandardScaler", "MinMaxScaler"]
            )

            dataset_tab = st.sidebar.radio(
                "Select Dataset for Dashboard",
                ["Raw Dataset", "Cleaned Dataset", "PCA Dataset", "Feature-Selected Dataset"]
            )

            df_processed = df.copy()
            numeric_cols = df_processed.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df_processed.select_dtypes(include=["object", "category"]).columns.tolist()

            # ---------------- Apply Cleaning / Encoding / Scaling ----------------
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

            # Recalculate numeric columns after preprocessing
            numeric_cols_processed = df_processed.select_dtypes(include=["number"]).columns.tolist()

            # ---------------- Apply PCA ----------------
            if dataset_tab == "PCA Dataset":
                if len(numeric_cols_processed) >= 1:
                    max_components = min(df_processed.shape[0], len(numeric_cols_processed), 10)
                    default_components = min(2, max_components)

                    if max_components >= 1:
                        n_components = st.sidebar.slider(
                            "PCA Components",
                            min_value=1,
                            max_value=max_components,
                            value=default_components
                        )

                        try:
                            df_processed = apply_pca(df_processed, n_components=n_components)
                            st.info("PCA applied successfully. Missing numeric values (if any) were handled automatically.")
                        except Exception as e:
                            st.error(f"PCA could not be applied: {e}")
                    else:
                        st.warning("PCA cannot be applied because the dataset does not have enough rows/columns.")
                else:
                    st.warning("PCA requires at least one numeric column after preprocessing.")

            # ---------------- Apply Feature Selection ----------------
            if dataset_tab == "Feature-Selected Dataset":
                if len(numeric_cols_processed) >= 1:
                    threshold = st.sidebar.slider(
                        "Variance Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.1
                    )

                    try:
                        df_processed = select_features(
                            df_processed,
                            method="variance",
                            threshold=threshold
                        )

                        if df_processed.shape[1] == 0:
                            st.warning("No features met the selected variance threshold. Try lowering the threshold.")
                        else:
                            st.info("Feature selection applied successfully. Missing numeric values (if any) were handled automatically.")

                    except Exception as e:
                        st.error(f"Feature selection could not be applied: {e}")
                else:
                    st.warning("Feature selection requires at least one numeric column after preprocessing.")

            # ---------------- Processed Dataset Summary ----------------
            st.subheader("📊 Processed Dataset Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Processed Rows", df_processed.shape[0])
            col2.metric("Processed Columns", df_processed.shape[1])
            col3.metric("Remaining Missing Values", int(df_processed.isnull().sum().sum()))

            with st.expander("View Processed Dataset (First 10 Rows)"):
                st.dataframe(df_processed.head(10), use_container_width=True)

            # ---------------- Download Processed Dataset ----------------
            csv_data = convert_df_to_csv(df_processed)

            st.download_button(
                label="⬇️ Download Processed Dataset as CSV",
                data=csv_data,
                file_name=f"processed_{dataset_tab.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

            st.markdown("---")

            # ---------------- Show Dashboard ----------------
            if not df_processed.empty and df_processed.shape[1] > 0:
                show_dashboard(df_processed, dataset_name=dataset_tab)
            else:
                st.warning("No data available to display in dashboard after processing.")

else:
    st.info("📂 Upload a dataset to start analysis.")