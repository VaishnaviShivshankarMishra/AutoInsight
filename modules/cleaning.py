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
    elif "raw_df" in st.session_state and st.session_state.raw_df is not None:
        return st.session_state.raw_df.copy()
    return None


def _save_df(df):
    st.session_state.processed_df = df.copy()


def _suggest_missing_strategy(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    suggestions = {}

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        missing_pct = (df[col].isnull().sum() / len(df)) * 100 if len(df) > 0 else 0

        if missing_pct > 40:
            suggestions[col] = "Drop Column"
        else:
            if col in numeric_cols:
                try:
                    skewness = df[col].dropna().skew()
                    suggestions[col] = "Fill with Median" if abs(skewness) > 1 else "Fill with Mean"
                except:
                    suggestions[col] = "Fill with Median"
            else:
                suggestions[col] = "Fill with Mode"

    return suggestions


def _auto_clean(df):
    logs = []

    # Remove duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        logs.append(f"Removed {dup_count} duplicate rows automatically.")

    # Handle missing values
    suggestions = _suggest_missing_strategy(df)
    for col, strategy in suggestions.items():
        missing_count = df[col].isnull().sum()

        if strategy == "Drop Column":
            df = df.drop(columns=[col])
            logs.append(f"Dropped column '{col}' automatically due to high missing values ({missing_count}).")

        elif strategy == "Fill with Mean":
            fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
            logs.append(f"Filled missing values in '{col}' using Mean ({round(fill_val, 4)}).")

        elif strategy == "Fill with Median":
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            logs.append(f"Filled missing values in '{col}' using Median ({round(fill_val, 4)}).")

        elif strategy == "Fill with Mode":
            mode_series = df[col].mode()
            if not mode_series.empty:
                fill_val = mode_series[0]
                df[col] = df[col].fillna(fill_val)
                logs.append(f"Filled missing values in '{col}' using Mode ({fill_val}).")

    return df, logs


def show_cleaning():
    st.subheader("🧹 Data Cleaning")

    df = _get_current_df()

    if df is None:
        st.warning("Please upload a dataset first.")
        return

    st.write("### Current Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write("### Missing Values Summary")
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing %": ((df.isnull().sum() / len(df)) * 100).round(2).values
    })
    st.dataframe(missing_df, use_container_width=True)

    st.write("### 💡 Smart Suggestions")
    suggestions = _suggest_missing_strategy(df)

    if suggestions:
        for col, suggestion in suggestions.items():
            st.write(f"- **{col}** → {suggestion}")
    else:
        st.success("No missing values found.")

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        st.info(f"Recommended: Remove Duplicates ({dup_count} duplicate rows found)")
    else:
        st.success("No duplicate rows found.")

    st.write("### ⚡ Auto Clean")
    if st.button("Run Auto Clean"):
        new_df, logs = _auto_clean(df.copy())
        _save_df(new_df)
        for log in logs:
            _log_action(log)

        if logs:
            st.success("Auto Clean completed successfully.")
            for log in logs:
                st.write(f"✅ {log}")
        else:
            st.info("No cleaning was needed.")

    st.markdown("---")

    st.write("### Handle Missing Values (Manual)")
    missing_cols = df.columns[df.isnull().sum() > 0].tolist()

    if missing_cols:
        selected_col = st.selectbox("Select column with missing values", missing_cols)
        st.caption(f"💡 Suggested: {suggestions.get(selected_col, 'No suggestion available')}")

        option = st.selectbox(
            "Choose method",
            ["Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows", "Drop Column"]
        )

        if st.button("Apply Missing Value Handling"):
            working_df = df.copy()
            missing_count = working_df[selected_col].isnull().sum()

            try:
                if option == "Fill with Mean":
                    fill_val = working_df[selected_col].mean()
                    working_df[selected_col] = working_df[selected_col].fillna(fill_val)
                    _log_action(f"Filled missing values in '{selected_col}' using Mean ({round(fill_val, 4)}).")

                elif option == "Fill with Median":
                    fill_val = working_df[selected_col].median()
                    working_df[selected_col] = working_df[selected_col].fillna(fill_val)
                    _log_action(f"Filled missing values in '{selected_col}' using Median ({round(fill_val, 4)}).")

                elif option == "Fill with Mode":
                    mode_series = working_df[selected_col].mode()
                    if not mode_series.empty:
                        fill_val = mode_series[0]
                        working_df[selected_col] = working_df[selected_col].fillna(fill_val)
                        _log_action(f"Filled missing values in '{selected_col}' using Mode ({fill_val}).")

                elif option == "Drop Rows":
                    before_rows = len(working_df)
                    working_df = working_df.dropna(subset=[selected_col])
                    removed_rows = before_rows - len(working_df)
                    _log_action(f"Dropped {removed_rows} rows with missing values in '{selected_col}'.")

                elif option == "Drop Column":
                    working_df = working_df.drop(columns=[selected_col])
                    _log_action(f"Dropped column '{selected_col}' manually.")

                _save_df(working_df)
                st.success(f"Applied '{option}' on '{selected_col}' successfully.")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.success("No missing values found.")

    st.markdown("---")

    st.write("### Handle Duplicate Rows (Manual)")
    st.write(f"Duplicate Rows Found: **{dup_count}**")

    if dup_count > 0:
        if st.button("Remove Duplicates"):
            working_df = df.copy()
            before_rows = len(working_df)
            working_df = working_df.drop_duplicates()
            removed_rows = before_rows - len(working_df)

            _save_df(working_df)
            _log_action(f"Removed {removed_rows} duplicate rows manually.")
            st.success(f"Removed {removed_rows} duplicate rows successfully.")
    else:
        st.success("No duplicate rows found.")

    st.markdown("---")
    st.write("### Updated Dataset Preview")
    latest_df = _get_current_df()
    st.dataframe(latest_df.head(), use_container_width=True)