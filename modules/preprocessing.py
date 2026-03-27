import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


def show_preprocessing(df):
    # Do NOT use st.header here because app.py already shows the main section heading

    if df is None or df.empty:
        st.warning("⚠️ No dataset available for preprocessing.")
        return None

    processed_df = df.copy()

    st.subheader("Preprocessing Options")

    # ---------------- Encoding ----------------
    categorical_cols = processed_df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(categorical_cols) > 0:
        st.write("### Categorical Columns Detected")
        st.write(categorical_cols)

        encoding_method = st.radio(
            "Choose Encoding Method",
            ["None", "Label Encoding", "One-Hot Encoding"],
            key="preprocessing_encoding_method"
        )

        if encoding_method == "Label Encoding":
            for col in categorical_cols:
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))

            st.success("✅ Label Encoding applied successfully!")

        elif encoding_method == "One-Hot Encoding":
            processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
            st.success("✅ One-Hot Encoding applied successfully!")

        else:
            st.info("ℹ️ No encoding applied.")
    else:
        st.info("ℹ️ No categorical columns found for encoding.")

    # ---------------- Scaling ----------------
    numeric_cols = processed_df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) > 0:
        st.write("### Numeric Columns Available for Scaling")
        st.write(numeric_cols)

        scaling_method = st.radio(
            "Choose Scaling Method",
            ["None", "Standard Scaling", "Min-Max Scaling"],
            key="preprocessing_scaling_method"
        )

        if scaling_method == "Standard Scaling":
            scaler = StandardScaler()
            processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
            st.success("✅ Standard Scaling applied successfully!")

        elif scaling_method == "Min-Max Scaling":
            scaler = MinMaxScaler()
            processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
            st.success("✅ Min-Max Scaling applied successfully!")

        else:
            st.info("ℹ️ No scaling applied.")
    else:
        st.warning("⚠️ No numeric columns found for scaling.")

    # ---------------- Preview ----------------
    st.subheader("Preprocessed Data Preview")
    st.dataframe(processed_df.head(), use_container_width=True)

    return processed_df