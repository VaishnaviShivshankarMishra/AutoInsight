import streamlit as st
import pandas as pd


def upload_file():
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx"],
        key="dataset_uploader"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("❌ Unsupported file format.")
                return None

            st.success("✅ File uploaded successfully!")
            return df

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return None

    return None