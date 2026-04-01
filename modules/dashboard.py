import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _get_current_df():
    if "processed_df" in st.session_state and st.session_state.processed_df is not None:
        return st.session_state.processed_df.copy()
    elif "cleaned_df" in st.session_state and st.session_state.cleaned_df is not None:
        return st.session_state.cleaned_df.copy()
    elif "raw_df" in st.session_state and st.session_state.raw_df is not None:
        return st.session_state.raw_df.copy()
    return None


def _suggest_chart(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    suggestions = []

    if len(numeric_cols) >= 1:
        suggestions.append("Histogram for numeric distribution")

    if len(categorical_cols) >= 1:
        suggestions.append("Bar Chart for categorical frequency")

    if len(numeric_cols) >= 2:
        suggestions.append("Scatter Plot for relationship between numeric features")
        suggestions.append("Correlation Heatmap for numeric columns")

    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        suggestions.append("Box Plot for numeric values grouped by category")

    return suggestions


def show_eda_dashboard():
    st.subheader("📊 EDA Dashboard")

    df = _get_current_df()

    if df is None or df.empty:
        st.warning("Please upload a dataset first.")
        return

    st.write("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # -------------------------
    # Basic Metrics
    # -------------------------
    st.write("### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    # -------------------------
    # Data Types
    # -------------------------
    st.write("### Column Data Types")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Unique Values": [df[col].nunique(dropna=True) for col in df.columns]
    })
    st.dataframe(dtype_df, use_container_width=True)

    # -------------------------
    # Summary Stats
    # -------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        st.write("### Numeric Summary")
        try:
            numeric_summary = df[numeric_cols].describe().T
            st.dataframe(numeric_summary, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate numeric summary: {e}")

    if categorical_cols:
        st.write("### Categorical Summary")
        try:
            cat_summary = pd.DataFrame({
                "Column": categorical_cols,
                "Unique Values": [df[col].nunique(dropna=True) for col in categorical_cols],
                "Top Value": [
                    df[col].mode(dropna=True)[0] if not df[col].mode(dropna=True).empty else None
                    for col in categorical_cols
                ]
            })
            st.dataframe(cat_summary, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate categorical summary: {e}")

    # -------------------------
    # Auto Chart Suggestions
    # -------------------------
    st.write("### 💡 Auto Chart Suggestions")
    suggestions = _suggest_chart(df)

    if suggestions:
        for s in suggestions:
            st.write(f"- {s}")
    else:
        st.info("No chart suggestions available for this dataset.")

    st.markdown("---")

    # -------------------------
    # Manual Chart Builder
    # -------------------------
    st.write("### Create Charts")

    chart_type_options = ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Correlation Heatmap"]
    chart_type = st.selectbox("Select chart type", chart_type_options)

    if chart_type == "Histogram":
        if numeric_cols:
            valid_hist_cols = [col for col in numeric_cols if pd.to_numeric(df[col], errors="coerce").dropna().shape[0] > 0]

            if valid_hist_cols:
                col = st.selectbox("Select numeric column", valid_hist_cols, key="hist_col")

                if st.button("Generate Histogram"):
                    try:
                        plot_series = pd.to_numeric(df[col], errors="coerce").dropna()

                        if len(plot_series) == 0:
                            st.warning("Selected column has no valid numeric values.")
                        else:
                            fig, ax = plt.subplots()
                            ax.hist(plot_series, bins=20)
                            ax.set_title(f"Histogram of {col}")
                            ax.set_xlabel(col)
                            ax.set_ylabel("Frequency")
                            st.pyplot(fig)
                            plt.close(fig)
                    except Exception as e:
                        st.error(f"Could not generate histogram: {e}")
            else:
                st.warning("No valid numeric columns available.")
        else:
            st.warning("No numeric columns available.")

    elif chart_type == "Bar Chart":
        if categorical_cols:
            valid_cat_cols = []
            for col in categorical_cols:
                non_null_count = df[col].dropna().shape[0]
                if non_null_count > 0:
                    valid_cat_cols.append(col)

            if valid_cat_cols:
                col = st.selectbox("Select categorical column", valid_cat_cols, key="bar_col")

                if st.button("Generate Bar Chart"):
                    try:
                        value_counts = df[col].astype(str).replace("nan", np.nan).dropna().value_counts().head(20)

                        if value_counts.empty:
                            st.warning("Selected column has no valid categories to plot.")
                        else:
                            fig, ax = plt.subplots()
                            value_counts.plot(kind="bar", ax=ax)
                            ax.set_title(f"Bar Chart of {col}")
                            ax.set_xlabel(col)
                            ax.set_ylabel("Count")
                            st.pyplot(fig)
                            plt.close(fig)
                    except Exception as e:
                        st.error(f"Could not generate bar chart: {e}")
            else:
                st.warning("No valid categorical columns available.")
        else:
            st.warning("No categorical columns available.")

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X-axis column", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Select Y-axis column", numeric_cols, key="scatter_y")

            if st.button("Generate Scatter Plot"):
                try:
                    plot_df = df[[x_col, y_col]].copy()
                    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
                    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
                    plot_df = plot_df.dropna()

                    if plot_df.empty:
                        st.warning("Not enough valid numeric data for scatter plot.")
                    else:
                        fig, ax = plt.subplots()
                        ax.scatter(plot_df[x_col], plot_df[y_col])
                        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        st.pyplot(fig)
                        plt.close(fig)
                except Exception as e:
                    st.error(f"Could not generate scatter plot: {e}")
        else:
            st.warning("At least 2 numeric columns are required.")

    elif chart_type == "Box Plot":
        if categorical_cols and numeric_cols:
            valid_cat_cols = []
            for col in categorical_cols:
                if df[col].dropna().shape[0] > 0:
                    valid_cat_cols.append(col)

            valid_num_cols = []
            for col in numeric_cols:
                if pd.to_numeric(df[col], errors="coerce").dropna().shape[0] > 0:
                    valid_num_cols.append(col)

            if valid_cat_cols and valid_num_cols:
                cat_col = st.selectbox("Select categorical column", valid_cat_cols, key="box_cat")
                num_col = st.selectbox("Select numeric column", valid_num_cols, key="box_num")

                if st.button("Generate Box Plot"):
                    try:
                        plot_df = df[[cat_col, num_col]].copy()
                        plot_df[num_col] = pd.to_numeric(plot_df[num_col], errors="coerce")
                        plot_df = plot_df.dropna(subset=[cat_col, num_col])

                        if plot_df.empty:
                            st.warning("Not enough valid data for box plot.")
                        else:
                            # Limit categories for readability
                            top_categories = plot_df[cat_col].astype(str).value_counts().head(20).index
                            plot_df = plot_df[plot_df[cat_col].astype(str).isin(top_categories)].copy()
                            plot_df[cat_col] = plot_df[cat_col].astype(str)

                            if plot_df.empty:
                                st.warning("No valid categories available after filtering.")
                            else:
                                fig, ax = plt.subplots(figsize=(10, 5))
                                plot_df.boxplot(column=num_col, by=cat_col, ax=ax)
                                plt.suptitle("")
                                ax.set_title(f"Box Plot of {num_col} by {cat_col}")
                                ax.set_xlabel(cat_col)
                                ax.set_ylabel(num_col)
                                plt.xticks(rotation=45, ha="right")
                                st.pyplot(fig)
                                plt.close(fig)
                    except Exception as e:
                        st.error(f"Could not generate box plot: {e}")
            else:
                st.warning("Need at least 1 valid categorical and 1 valid numeric column.")
        else:
            st.warning("Need at least 1 categorical and 1 numeric column.")

    elif chart_type == "Correlation Heatmap":
        if len(numeric_cols) >= 2:
            if st.button("Generate Correlation Heatmap"):
                try:
                    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
                    numeric_df = numeric_df.dropna(axis=1, how="all")

                    if numeric_df.shape[1] < 2:
                        st.warning("Not enough valid numeric columns for correlation heatmap.")
                    else:
                        corr = numeric_df.corr()

                        if corr.shape[0] < 2 or corr.isnull().all().all():
                            st.warning("Could not compute a valid correlation matrix.")
                        else:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            cax = ax.matshow(corr, aspect="auto")
                            fig.colorbar(cax)

                            ax.set_xticks(range(len(corr.columns)))
                            ax.set_yticks(range(len(corr.columns)))
                            ax.set_xticklabels(corr.columns, rotation=90)
                            ax.set_yticklabels(corr.columns)
                            ax.set_title("Correlation Heatmap", pad=20)

                            st.pyplot(fig)
                            plt.close(fig)
                except Exception as e:
                    st.error(f"Could not generate correlation heatmap: {e}")
        else:
            st.warning("At least 2 numeric columns are required.")