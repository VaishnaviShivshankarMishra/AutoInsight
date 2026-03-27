import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_eda(df):
    # Do NOT use st.header here because app.py already shows the main section heading

    if df is None or df.empty:
        st.warning("⚠️ No dataset available for EDA.")
        return

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    # ---------------- Statistical Summary ----------------
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    # ---------------- Missing Values ----------------
    st.subheader("Missing Values")
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })
    st.dataframe(missing_df, use_container_width=True)

    # ---------------- Correlation Heatmap ----------------
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] >= 2:
        st.subheader("Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ---------------- Distribution Plot ----------------
    if numeric_df.shape[1] > 0:
        st.subheader("Distribution Plot")

        selected_col = st.selectbox(
            "Select numeric column for distribution analysis",
            numeric_df.columns.tolist(),
            key="eda_distribution_column"
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[selected_col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)

    # ---------------- Box Plot ----------------
    if numeric_df.shape[1] > 0:
        st.subheader("Box Plot")

        selected_box_col = st.selectbox(
            "Select numeric column for box plot",
            numeric_df.columns.tolist(),
            key="eda_boxplot_column"
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df[selected_box_col], ax=ax)
        ax.set_title(f"Box Plot of {selected_box_col}")
        st.pyplot(fig)