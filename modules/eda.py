import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
    try:
        summary = df.describe(include="all").transpose()
        st.dataframe(summary, use_container_width=True)
    except Exception:
        # Fallback in rare mixed-type edge cases
        fallback_summary = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": [df[col].nunique(dropna=True) for col in df.columns]
        })
        st.dataframe(fallback_summary, use_container_width=True)

    # ---------------- Missing Values ----------------
    st.subheader("Missing Values")
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })
    st.dataframe(missing_df, use_container_width=True)

    # ---------------- Correlation Heatmap ----------------
    numeric_df = df.select_dtypes(include=["number"]).copy()

    # Remove columns that are entirely null
    numeric_df = numeric_df.dropna(axis=1, how="all")

    if numeric_df.shape[1] >= 2:
        st.subheader("Correlation Heatmap")

        try:
            corr = numeric_df.corr()

            # If correlation matrix is valid
            if corr.shape[0] >= 2 and not corr.isnull().all().all():
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Not enough valid numeric variation for correlation heatmap.")
        except Exception as e:
            st.warning(f"Could not generate correlation heatmap: {e}")

    # ---------------- Distribution Plot ----------------
    if numeric_df.shape[1] > 0:
        st.subheader("Distribution Plot")

        valid_dist_cols = []
        for col in numeric_df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 0:
                valid_dist_cols.append(col)

        if valid_dist_cols:
            selected_col = st.selectbox(
                "Select numeric column for distribution analysis",
                valid_dist_cols,
                key="eda_distribution_column"
            )

            try:
                plot_series = pd.to_numeric(df[selected_col], errors="coerce").dropna()

                if len(plot_series) > 0:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.histplot(plot_series, kde=True, ax=ax)
                    ax.set_title(f"Distribution of {selected_col}")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Selected column has no valid numeric values to plot.")
            except Exception as e:
                st.warning(f"Could not generate distribution plot: {e}")
        else:
            st.info("No valid numeric columns available for distribution plot.")

    # ---------------- Box Plot ----------------
    if numeric_df.shape[1] > 0:
        st.subheader("Box Plot")

        valid_box_cols = []
        for col in numeric_df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 0:
                valid_box_cols.append(col)

        if valid_box_cols:
            selected_box_col = st.selectbox(
                "Select numeric column for box plot",
                valid_box_cols,
                key="eda_boxplot_column"
            )

            try:
                plot_series = pd.to_numeric(df[selected_box_col], errors="coerce").dropna()

                if len(plot_series) > 0:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.boxplot(x=plot_series, ax=ax)
                    ax.set_title(f"Box Plot of {selected_box_col}")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Selected column has no valid numeric values to plot.")
            except Exception as e:
                st.warning(f"Could not generate box plot: {e}")