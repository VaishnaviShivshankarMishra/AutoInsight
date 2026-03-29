# modules/feature_selection.py

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def select_features(df, method="variance", threshold=0.1):
    df_fs = df.copy()

    # Select only numeric columns
    numeric_cols = df_fs.select_dtypes(include=["number"]).columns.tolist()

    # If no numeric columns, return original dataframe
    if len(numeric_cols) == 0:
        return df_fs

    # Fill missing numeric values with median
    for col in numeric_cols:
        if df_fs[col].isnull().sum() > 0:
            df_fs[col] = df_fs[col].fillna(df_fs[col].median())

    if method == "variance":
        selector = VarianceThreshold(threshold=threshold)
        selected_data = selector.fit_transform(df_fs[numeric_cols])

        selected_columns = [
            col for col, keep in zip(numeric_cols, selector.get_support()) if keep
        ]

        if len(selected_columns) == 0:
            return pd.DataFrame(index=df_fs.index)

        return pd.DataFrame(selected_data, columns=selected_columns, index=df_fs.index)

    return df_fs