# modules/feature_engineering.py
import pandas as pd
from sklearn.decomposition import PCA

def apply_pca(df, n_components=2):
    df_pca = df.copy()
    numeric_cols = df_pca.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) == 0:
        return df_pca

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_pca[numeric_cols])
    pca_cols = [f"PC{i+1}" for i in range(pca_result.shape[1])]
    df_pca = pd.DataFrame(pca_result, columns=pca_cols)
    return df_pca