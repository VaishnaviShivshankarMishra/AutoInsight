import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def _get_current_df():
    if "processed_df" in st.session_state and st.session_state.processed_df is not None:
        return st.session_state.processed_df.copy()
    elif "raw_df" in st.session_state and st.session_state.raw_df is not None:
        return st.session_state.raw_df.copy()
    return None


def show_modeling():
    st.subheader("🤖 Modeling")

    df = _get_current_df()

    if df is None:
        st.warning("Please upload a dataset first.")
        return

    st.write("### Current Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    if df.shape[1] < 2:
        st.warning("Dataset must have at least 2 columns for modeling.")
        return

    target_col = st.selectbox("Select Target Column", df.columns.tolist())

    problem_type = st.selectbox("Select Problem Type", ["Classification", "Regression"])

    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode non-numeric X columns
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Encode y if classification and non-numeric
    if problem_type == "Classification" and not pd.api.types.is_numeric_dtype(y):
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))

    # Drop rows with NaN (safe baseline)
    combined = pd.concat([X, pd.Series(y, name=target_col)], axis=1).dropna()
    X = combined.drop(columns=[target_col])
    y = combined[target_col]

    if len(X) < 5:
        st.warning("Not enough clean rows available for modeling after preprocessing.")
        return

    model_name = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Random Forest Classifier"] if problem_type == "Classification"
        else ["Linear Regression", "Random Forest Regressor"]
    )

    if st.button("Train Model"):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if problem_type == "Classification":
                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                else:
                    model = RandomForestClassifier(random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                st.success("Model trained successfully!")
                st.write(f"**Accuracy:** {round(acc, 4)}")

            else:
                if model_name == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.success("Model trained successfully!")
                st.write(f"**MSE:** {round(mse, 4)}")
                st.write(f"**R² Score:** {round(r2, 4)}")

        except Exception as e:
            st.error(f"Error during model training: {e}")