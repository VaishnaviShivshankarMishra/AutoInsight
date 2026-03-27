# modules/feature_selection.py
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def select_features(df, method="variance", threshold=0.1):
    df_fs = df.copy()
    numeric_cols = df_fs.select_dtypes(include=["number"]).columns.tolist()
    if method == "variance" and numeric_cols:
        selector = VarianceThreshold(threshold=threshold)
        selected = selector.fit_transform(df_fs[numeric_cols])
        selected_cols = [col for col, keep in zip(numeric_cols, selector.get_support()) if keep]
        df_fs = df_fs[selected_cols]
    return df_fs