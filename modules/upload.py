import streamlit as st
import pandas as pd


def show_data_upload():
    st.subheader("📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        key="csv_uploader"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.session_state.raw_df = df.copy()

            # Initialize processed_df only if not already created
            if "processed_df" not in st.session_state or st.session_state.processed_df is None:
                st.session_state.processed_df = df.copy()

            if "transformation_log" not in st.session_state:
                st.session_state.transformation_log = []

            st.success("Dataset uploaded successfully!")

            st.write("### Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)

            st.write("### Dataset Info")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Missing Values", int(df.isnull().sum().sum()))

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")