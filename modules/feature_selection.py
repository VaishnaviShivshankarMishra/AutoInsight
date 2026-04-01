import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def _log_action(message):
    if "transformation_log" not in st.session_state:
        st.session_state.transformation_log = []
    st.session_state.transformation_log.append(message)


def _get_current_df():
    if "processed_df" in st.session_state and st.session_state.processed_df is not None:
        return st.session_state.processed_df.copy()
    elif "cleaned_df" in st.session_state and st.session_state.cleaned_df is not None:
        return st.session_state.cleaned_df.copy()
    elif "raw_df" in st.session_state and st.session_state.raw_df is not None:
        return st.session_state.raw_df.copy()
    return None


def _save_df(df):
    # Feature selection should only update the latest working dataset
    # Do NOT overwrite cleaned_df here
    st.session_state.processed_df = df.copy()


def _remove_high_correlation(df, threshold=0.9):
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return df, [], None

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if to_drop:
        df = df.drop(columns=to_drop)

    return df, to_drop, corr_matrix


def _apply_variance_threshold(df, threshold=0.0):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return df, []

    numeric_df = df[numeric_cols].copy().fillna(0)

    # Safe guard: if numeric_df is empty after processing
    if numeric_df.shape[1] == 0:
        return df, []

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric_df)

    kept_mask = selector.get_support()
    kept_cols = list(np.array(numeric_cols)[kept_mask])
    dropped_cols = [col for col in numeric_cols if col not in kept_cols]

    if dropped_cols:
        df = df.drop(columns=dropped_cols)

    return df, dropped_cols


def show_feature_selection():
    st.subheader("🎯 Feature Selection")

    df = _get_current_df()

    if df is None:
        st.warning("No dataset available. Please upload a dataset first.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.write("### Current Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    if not numeric_cols:
        st.info("No numeric columns found for feature selection.")
        return

    st.write("### ⚡ Auto Feature Selection Suggestions")
    st.info("Recommended order: 1) Remove low variance features → 2) Remove highly correlated features")

    st.markdown("---")

    # ----------------------------
    # Variance Threshold
    # ----------------------------
    st.write("### Variance Threshold")

    variance_threshold = st.slider(
        "Select variance threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01
    )

    if st.button("Apply Variance Threshold"):
        working_df = df.copy()

        try:
            current_numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()

            if not current_numeric_cols:
                st.warning("No numeric columns available for variance threshold.")
            else:
                working_df, dropped_cols = _apply_variance_threshold(working_df, threshold=variance_threshold)
                _save_df(working_df)

                if dropped_cols:
                    _log_action(
                        f"Feature Selection: Removed low-variance numeric columns using VarianceThreshold({variance_threshold}). Dropped columns: {', '.join(dropped_cols)}."
                    )
                    st.success(f"Dropped columns: {', '.join(dropped_cols)}")
                else:
                    st.info("No columns were dropped by variance threshold.")

        except Exception as e:
            st.error(f"Error while applying variance threshold: {e}")

    st.markdown("---")

    # ----------------------------
    # Correlation Filter
    # ----------------------------
    st.write("### Correlation-Based Feature Removal")

    corr_threshold = st.slider(
        "Select correlation threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.9,
        step=0.01
    )

    if st.button("Remove Highly Correlated Features"):
        working_df = df.copy()

        try:
            current_numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()

            if len(current_numeric_cols) < 2:
                st.warning("At least 2 numeric columns are required for correlation-based feature removal.")
            else:
                working_df, dropped_cols, corr_matrix = _remove_high_correlation(working_df, threshold=corr_threshold)
                _save_df(working_df)

                if dropped_cols:
                    _log_action(
                        f"Feature Selection: Removed highly correlated columns using correlation threshold {corr_threshold}. Dropped columns: {', '.join(dropped_cols)}."
                    )
                    st.success(f"Dropped columns: {', '.join(dropped_cols)}")
                else:
                    st.info("No highly correlated columns found above threshold.")

        except Exception as e:
            st.error(f"Error while removing correlated features: {e}")

    st.markdown("---")

    # ----------------------------
    # Auto Feature Selection
    # ----------------------------
    st.write("### ⚡ Auto Feature Selection")

    if st.button("Run Auto Feature Selection"):
        working_df = df.copy()
        logs = []

        try:
            current_numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()

            if not current_numeric_cols:
                st.info("No numeric columns available for auto feature selection.")
            else:
                working_df, dropped_var = _apply_variance_threshold(working_df, threshold=0.0)
                if dropped_var:
                    logs.append(
                        f"Feature Selection (Auto): Removed low-variance numeric columns. Dropped: {', '.join(dropped_var)}."
                    )

                current_numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()

                if len(current_numeric_cols) >= 2:
                    working_df, dropped_corr, _ = _remove_high_correlation(working_df, threshold=0.9)
                    if dropped_corr:
                        logs.append(
                            f"Feature Selection (Auto): Removed highly correlated columns with threshold 0.9. Dropped: {', '.join(dropped_corr)}."
                        )

                _save_df(working_df)

                for msg in logs:
                    _log_action(msg)

                if logs:
                    st.success("Auto Feature Selection completed successfully.")
                    for msg in logs:
                        st.write(f"✅ {msg}")
                else:
                    st.info("No feature selection actions were needed.")

        except Exception as e:
            st.error(f"Error while running auto feature selection: {e}")

    st.markdown("---")
    st.write("### Updated Dataset Preview")
    latest_df = _get_current_df()
    st.dataframe(latest_df.head(), use_container_width=True)