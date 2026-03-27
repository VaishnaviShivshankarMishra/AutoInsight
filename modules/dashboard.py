# modules/dashboard.py
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ----------------- Data Cleaning / Preprocessing Helpers -----------------
def remove_outliers_iqr(df, numeric_cols):
    df_clean = df.copy()
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

def encode_columns(df, method):
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()
    if method == "Label Encoding":
        le = LabelEncoder()
        for col in categorical_cols:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    elif method == "One-Hot Encoding":
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    return df_encoded

def scale_columns(df, method):
    df_scaled = df.copy()
    numeric_cols = df_scaled.select_dtypes(include=["number"]).columns.tolist()
    if method == "StandardScaler":
        scaler = StandardScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled

# ----------------- Dashboard Visualization -----------------
def show_dashboard(df, dataset_name="Processed Dataset"):
    st.subheader(f"📊 Dashboard - {dataset_name}")

    if df is None or df.empty:
        st.warning("⚠️ No data available for dashboard.")
        return

    # Dataset Preview
    with st.expander("👀 Preview Dataset"):
        st.dataframe(df.head(), use_container_width=True)

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Histogram", "Box Plot"],
        key=f"{dataset_name}_chart_type"
    )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = df.columns.tolist()

    # ---------------- Chart Logic ----------------
    if chart_type == "Bar Chart":
        if numeric_cols and categorical_cols:
            x_col = st.selectbox("X-axis (Categorical)", categorical_cols, key=f"{dataset_name}_bar_x")
            y_col = st.selectbox("Y-axis (Numeric)", numeric_cols, key=f"{dataset_name}_bar_y")
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 1 numeric and 1 categorical column.")

    elif chart_type == "Line Chart":
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols, key=f"{dataset_name}_line_x")
            y_options = [c for c in numeric_cols if c != x_col]
            if y_options:
                y_col = st.selectbox("Y-axis", y_options, key=f"{dataset_name}_line_y")
                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid Y-axis options.")
        else:
            st.warning("Need at least 2 numeric columns.")

    elif chart_type == "Pie Chart":
        if categorical_cols:
            names_col = st.selectbox("Category Column", categorical_cols, key=f"{dataset_name}_pie_names")
            values_col = numeric_cols[0] if numeric_cols else None
            fig = px.pie(df, names=names_col, values=values_col, title=f"Pie Chart of {names_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 1 categorical column.")

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols, key=f"{dataset_name}_scatter_x")
            y_options = [c for c in numeric_cols if c != x_col]
            y_col = st.selectbox("Y-axis", y_options, key=f"{dataset_name}_scatter_y")
            color_col_options = ["None"] + all_cols
            color_col = st.selectbox("Color Grouping (Optional)", color_col_options, key=f"{dataset_name}_scatter_color")
            if color_col == "None":
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            else:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns.")

    elif chart_type == "Histogram":
        if numeric_cols:
            hist_col = st.selectbox("Select Numeric Column", numeric_cols, key=f"{dataset_name}_hist")
            fig = px.histogram(df, x=hist_col, title=f"Histogram of {hist_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 1 numeric column.")

    elif chart_type == "Box Plot":
        if numeric_cols:
            y_col = st.selectbox("Numeric Column", numeric_cols, key=f"{dataset_name}_box_y")
            x_options = ["None"] + categorical_cols
            x_col = st.selectbox("Grouping Column (Optional)", x_options, key=f"{dataset_name}_box_x")
            if x_col == "None":
                fig = px.box(df, y=y_col, title=f"Box Plot of {y_col}")
            else:
                fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 1 numeric column.")

    # ---------------- Quick Metrics ----------------
    st.subheader("Quick Dataset Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric Columns", len(numeric_cols))