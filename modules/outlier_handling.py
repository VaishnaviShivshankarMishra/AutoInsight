import streamlit as st
import pandas as pd
import numpy as np


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
    st.session_state.cleaned_df = df.copy()
    st.session_state.processed_df = df.copy()


def _get_outlier_bounds(series):
    """
    Safe IQR bounds calculation.
    Returns None values if series is not suitable.
    """
    series = pd.to_numeric(series, errors="coerce").dropna()

    if len(series) < 5:
        return None, None, None, None, None

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    # Constant / near-constant columns
    if pd.isna(iqr):
        return None, None, q1, q3, iqr

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return lower, upper, q1, q3, iqr


def _count_outliers(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    lower, upper, q1, q3, iqr = _get_outlier_bounds(series)

    if lower is None or upper is None:
        return 0, None, None

    count = ((series < lower) | (series > upper)).sum()
    return int(count), lower, upper


def _suggest_outlier_strategy(df, numeric_cols):
    suggestions = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(series) < 5:
            continue

        outlier_count, lower, upper = _count_outliers(series)

        if lower is None or upper is None or outlier_count == 0:
            continue

        outlier_pct = (outlier_count / len(series)) * 100

        if outlier_pct <= 5:
            suggestions[col] = "Cap Outliers (IQR Winsorization)"
        elif outlier_pct <= 15:
            suggestions[col] = "Remove Outlier Rows"
        else:
            suggestions[col] = "Cap Outliers (Too many outliers to safely remove rows)"

    return suggestions


def _cap_outliers(df, col):
    series = pd.to_numeric(df[col], errors="coerce")
    lower, upper, q1, q3, iqr = _get_outlier_bounds(series.dropna())

    if lower is None or upper is None:
        return df, 0, None, None, q1, q3, iqr

    outlier_mask = (series < lower) | (series > upper)
    outlier_count = int(outlier_mask.sum())

    # Preserve NaNs, clip only numeric values
    df[col] = series.clip(lower=lower, upper=upper)

    return df, outlier_count, lower, upper, q1, q3, iqr


def _remove_outliers(df, col):
    series = pd.to_numeric(df[col], errors="coerce")
    lower, upper, q1, q3, iqr = _get_outlier_bounds(series.dropna())

    if lower is None or upper is None:
        return df, 0, None, None, q1, q3, iqr

    before_rows = len(df)

    # Keep NaNs; remove only rows where numeric values are outside bounds
    mask_keep = series.isna() | ((series >= lower) & (series <= upper))
    df = df.loc[mask_keep].copy()

    removed_rows = before_rows - len(df)

    return df, int(removed_rows), lower, upper, q1, q3, iqr


def _auto_handle_outliers(df, numeric_cols):
    logs = []
    suggestions = _suggest_outlier_strategy(df, numeric_cols)

    for col, suggestion in suggestions.items():
        if col not in df.columns:
            continue

        if suggestion.startswith("Cap Outliers"):
            df, outlier_count, lower, upper, q1, q3, iqr = _cap_outliers(df, col)

            if lower is not None and upper is not None and outlier_count > 0:
                logs.append(
                    f"Outliers detected in '{col}' ({outlier_count} outliers). Auto Outlier Handling capped values using IQR method. "
                    f"Q1={round(q1,4)}, Q3={round(q3,4)}, IQR={round(iqr,4)}, lower={round(lower,4)}, upper={round(upper,4)}."
                )

        elif suggestion == "Remove Outlier Rows":
            df, removed_rows, lower, upper, q1, q3, iqr = _remove_outliers(df, col)

            if lower is not None and upper is not None and removed_rows > 0:
                logs.append(
                    f"Outliers detected in '{col}'. Auto Outlier Handling removed {removed_rows} rows using IQR rule. "
                    f"Q1={round(q1,4)}, Q3={round(q3,4)}, IQR={round(iqr,4)}, lower={round(lower,4)}, upper={round(upper,4)}."
                )

    return df, logs


def show_outlier_handling():
    st.subheader("📉 Outlier Handling")

    df = _get_current_df()

    if df is None or df.empty:
        st.warning("No dataset available. Please upload a dataset first.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns found for outlier detection.")
        return

    st.write("### Numeric Columns Overview")

    outlier_summary = []
    for col in numeric_cols:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(series) < 5:
            outlier_summary.append([col, "Insufficient Data", "-", "-"])
            continue

        outlier_count, lower, upper = _count_outliers(series)

        if lower is None or upper is None:
            outlier_summary.append([col, 0, "-", "-"])
        else:
            outlier_summary.append([col, outlier_count, round(lower, 4), round(upper, 4)])

    summary_df = pd.DataFrame(
        outlier_summary,
        columns=["Column", "Outlier Count", "Lower Bound", "Upper Bound"]
    )
    st.dataframe(summary_df, use_container_width=True)

    st.write("### 💡 Smart Suggestions")
    suggestions = _suggest_outlier_strategy(df, numeric_cols)

    if suggestions:
        for col, suggestion in suggestions.items():
            st.write(f"- **{col}** → {suggestion}")
    else:
        st.success("No significant outliers detected in numeric columns.")

    st.write("### ⚡ Auto Outlier Handling")
    if st.button("Run Auto Outlier Handling"):
        working_df = df.copy()
        working_df, logs = _auto_handle_outliers(working_df, numeric_cols)
        _save_df(working_df)

        for msg in logs:
            _log_action(msg)

        if logs:
            st.success("Auto Outlier Handling completed successfully.")
            for msg in logs:
                st.write(f"✅ {msg}")
        else:
            st.info("No outlier actions were needed.")

    st.markdown("---")

    st.write("### Handle Outliers (Manual)")
    selectable_cols = [col for col in numeric_cols if col in df.columns]

    if selectable_cols:
        selected_col = st.selectbox("Select numeric column", selectable_cols)

        series = pd.to_numeric(df[selected_col], errors="coerce").dropna()

        if len(series) >= 5:
            outlier_count, lower, upper = _count_outliers(series)

            st.write(f"Outliers detected in **{selected_col}**: **{outlier_count}**")
            st.caption(f"💡 Suggested: **{suggestions.get(selected_col, 'No action needed')}**")

            if lower is not None and upper is not None:
                st.caption(f"IQR Bounds → Lower: **{round(lower,4)}**, Upper: **{round(upper,4)}**")
            else:
                st.caption("IQR Bounds → Not available for this column.")

            method = st.selectbox(
                "Choose outlier handling method",
                ["Cap Outliers (IQR Winsorization)", "Remove Outlier Rows"]
            )

            if st.button("Apply Outlier Handling"):
                working_df = df.copy()

                try:
                    if method == "Cap Outliers (IQR Winsorization)":
                        working_df, outlier_count, lower, upper, q1, q3, iqr = _cap_outliers(working_df, selected_col)

                        if lower is not None and upper is not None:
                            _log_action(
                                f"Outliers detected in '{selected_col}' ({outlier_count} outliers). Manually capped values using IQR method. "
                                f"Q1={round(q1,4)}, Q3={round(q3,4)}, IQR={round(iqr,4)}, lower={round(lower,4)}, upper={round(upper,4)}."
                            )

                    elif method == "Remove Outlier Rows":
                        working_df, removed_rows, lower, upper, q1, q3, iqr = _remove_outliers(working_df, selected_col)

                        if lower is not None and upper is not None:
                            _log_action(
                                f"Outliers detected in '{selected_col}'. Manually removed {removed_rows} rows using IQR rule. "
                                f"Q1={round(q1,4)}, Q3={round(q3,4)}, IQR={round(iqr,4)}, lower={round(lower,4)}, upper={round(upper,4)}."
                            )

                    _save_df(working_df)
                    st.success(f"Applied '{method}' on '{selected_col}' successfully.")

                except Exception as e:
                    st.error(f"Error while handling outliers: {e}")
        else:
            st.info("Selected column does not have enough non-null values for outlier detection.")
    else:
        st.info("No numeric columns available.")

    st.markdown("---")
    st.write("### Updated Dataset Preview")
    latest_df = _get_current_df()

    if latest_df is not None and not latest_df.empty:
        st.dataframe(latest_df.head(), use_container_width=True)
    else:
        st.info("No updated dataset available.")