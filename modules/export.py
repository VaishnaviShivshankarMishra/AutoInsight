import streamlit as st


def show_download_button(df, file_name="processed_dataset.csv", button_label="Download CSV", key="download_btn"):
    if df is None or df.empty:
        st.warning("⚠️ No data available to download.")
        return

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=button_label,
        data=csv,
        file_name=file_name,
        mime="text/csv",
        key=key
    )