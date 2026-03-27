import streamlit as st
import pandas as pd
import numpy as np


def clean_data(df):
    cleaned_df = df.copy()

    st.subheader("🧹 Cleaning Process Report")

    # ---------------- Missing Values ----------------
    missing_before = cleaned_df.isnull().sum().sum()
    st.write(f"**Missing values before cleaning:** {missing_before}")

    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns

    # Fill numeric columns with median
    for col in numeric_cols:
        if cleaned_df[col].isnull().sum() > 0:
            median_value = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_value)

    # Fill categorical columns with mode
    for col in categorical_cols:
        if cleaned_df[col].isnull().sum() > 0:
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_value[0])

    missing_after = cleaned_df.isnull().sum().sum()
    st.write(f"**Missing values after cleaning:** {missing_after}")

    # ---------------- Duplicate Removal ----------------
    duplicates_before = cleaned_df.duplicated().sum()
    st.write(f"**Duplicate rows before removal:** {duplicates_before}")

    cleaned_df = cleaned_df.drop_duplicates()

    duplicates_after = cleaned_df.duplicated().sum()
    st.write(f"**Duplicate rows after removal:** {duplicates_after}")

    # ---------------- Outlier Handling ----------------
    st.subheader("📌 Outlier Handling")

    remove_outliers = st.checkbox("Remove outliers using IQR method", key="remove_outliers_checkbox")

    if remove_outliers and len(numeric_cols) > 0:
        rows_before = cleaned_df.shape[0]

        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            cleaned_df = cleaned_df[
                (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
            ]

        rows_after = cleaned_df.shape[0]
        removed_rows = rows_before - rows_after

        st.write(f"**Rows before outlier removal:** {rows_before}")
        st.write(f"**Rows after outlier removal:** {rows_after}")
        st.write(f"**Outlier rows removed:** {removed_rows}")
    else:
        st.info("ℹ️ Outlier removal not applied.")

    # ---------------- Final Summary ----------------
    st.subheader("✅ Cleaning Summary")
    st.write(f"**Final dataset shape:** {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")

    return cleaned_df