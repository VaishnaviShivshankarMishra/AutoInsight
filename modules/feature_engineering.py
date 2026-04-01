import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def _log_action(message):
    if "transformation_log" not in st.session_state:
        st.session_state.transformation_log = []
    st.session_state.transformation_log.append(message)


def _get_current_df():
    if "processed_df" in st.session_state and st.session_state.processed_df is not None:
        return st.session_state.processed_df.copy()
    elif "raw_df" in st.session_state and st.session_state.raw_df is not None:
        return st.session_state.raw_df.copy()
    return None


def _save_df(df):
    st.session_state.processed_df = df.copy()


def show_feature_engineering():
    st.subheader("⚙️ Feature Engineering")

    df = _get_current_df()

    if df is None:
        st.warning("Please upload a dataset first.")
        return

    st.write("### Current Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # -------------------------
    # Auto Feature Engineering
    # -------------------------
    st.write("### ⚡ Auto Feature Engineering")

    if st.button("Run Auto Feature Engineering"):
        working_df = df.copy()
        logs = []

        # Auto encoding
        cat_cols = working_df.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in cat_cols:
            if working_df[col].nunique(dropna=True) <= 20:
                le = LabelEncoder()
                working_df[col] = working_df[col].astype(str)
                working_df[col] = le.fit_transform(working_df[col])
                logs.append(f"Auto-encoded column '{col}' using Label Encoding.")

        # Auto scaling
        num_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            scaler = StandardScaler()
            working_df[num_cols] = scaler.fit_transform(working_df[num_cols])
            logs.append(f"Auto-scaled numeric columns using StandardScaler: {num_cols}")

        _save_df(working_df)

        for log in logs:
            _log_action(log)

        if logs:
            st.success("Auto Feature Engineering completed successfully.")
            for log in logs:
                st.write(f"✅ {log}")
        else:
            st.info("No automatic feature engineering actions were applied.")

    st.markdown("---")

    # -------------------------
    # Encoding
    # -------------------------
    st.write("### Encode Categorical Columns")

    current_cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if current_cat_cols:
        selected_col = st.selectbox("Select categorical column", current_cat_cols)
        st.caption("💡 Suggested: Label Encoding for binary/low-cardinality columns")

        encoding_option = st.selectbox("Select encoding method", ["Label Encoding"])

        if st.button("Apply Encoding"):
            working_df = df.copy()

            try:
                if encoding_option == "Label Encoding":
                    le = LabelEncoder()
                    working_df[selected_col] = working_df[selected_col].astype(str)
                    working_df[selected_col] = le.fit_transform(working_df[selected_col])

                    _save_df(working_df)
                    _log_action(f"Applied Label Encoding on '{selected_col}'.")
                    st.success(f"Label Encoding applied successfully on '{selected_col}'.")

            except Exception as e:
                st.error(f"Error while encoding: {e}")
    else:
        st.info("No categorical columns available for encoding.")

    st.markdown("---")

    # -------------------------
    # Scaling
    # -------------------------
    st.write("### Scale Numeric Columns")

    current_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if current_num_cols:
        selected_scale_cols = st.multiselect("Select numeric columns to scale", current_num_cols)
        st.caption("💡 Suggested: StandardScaler for most ML models")

        scaling_option = st.selectbox("Select scaling method", ["StandardScaler", "MinMaxScaler"])

        if st.button("Apply Scaling"):
            if selected_scale_cols:
                working_df = df.copy()

                try:
                    if scaling_option == "StandardScaler":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()

                    working_df[selected_scale_cols] = scaler.fit_transform(working_df[selected_scale_cols])

                    _save_df(working_df)
                    _log_action(f"Applied {scaling_option} on columns: {selected_scale_cols}")
                    st.success(f"{scaling_option} applied successfully.")
                except Exception as e:
                    st.error(f"Error while scaling: {e}")
            else:
                st.warning("Please select at least one numeric column.")
    else:
        st.info("No numeric columns available for scaling.")

    st.markdown("---")

    # -------------------------
    # PCA
    # -------------------------
    st.write("### PCA (Dimensionality Reduction)")

    current_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(current_num_cols) >= 2:
        st.caption("💡 Suggested: Use PCA when many numeric columns are correlated")

        pca_cols = st.multiselect("Select numeric columns for PCA", current_num_cols, key="pca_cols")

        # Safe handling to avoid slider min=max error
        if len(pca_cols) == 0:
            st.info("Select at least 2 numeric columns to enable PCA.")
            n_components = 1
        elif len(pca_cols) == 1:
            st.info("Only 1 column selected. Please select at least 2 columns for PCA.")
            n_components = 1
        else:
            n_components = st.slider(
                "Select number of PCA components",
                min_value=1,
                max_value=len(pca_cols),
                value=min(2, len(pca_cols)),
                key="pca_slider"
            )

        if st.button("Apply PCA"):
            if len(pca_cols) >= 2:
                try:
                    working_df = df.copy()

                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(working_df[pca_cols])

                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(scaled_data)

                    # Drop original columns
                    working_df = working_df.drop(columns=pca_cols)

                    # Add PCA columns
                    for i in range(n_components):
                        working_df[f"PCA_{i+1}"] = pca_result[:, i]

                    _save_df(working_df)
                    _log_action(
                        f"Applied PCA on columns {pca_cols} with {n_components} components. "
                        f"Explained variance: {np.round(pca.explained_variance_ratio_, 4).tolist()}"
                    )

                    st.success("PCA applied successfully!")
                    st.write("Explained Variance Ratio:", np.round(pca.explained_variance_ratio_, 4))

                except Exception as e:
                    st.error(f"Error while applying PCA: {e}")
            else:
                st.warning("Please select at least 2 numeric columns for PCA.")
    else:
        st.info("At least 2 numeric columns are required for PCA.")

    st.markdown("---")
    st.write("### Updated Dataset Preview")
    latest_df = _get_current_df()
    st.dataframe(latest_df.head(), use_container_width=True)