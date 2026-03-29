# modules/feature_engineering.py

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca(df, n_components=2):
    """
    Apply PCA on numeric columns of the dataframe.
    
    Steps:
    1. Select numeric columns only
    2. Fill missing numeric values with median
    3. Scale numeric data before PCA (important for correct PCA)
    4. Apply PCA
    5. Return dataframe containing principal components only
    """

    # Make a copy to avoid modifying original dataframe
    df_pca = df.copy()

    # Select numeric columns only
    numeric_cols = df_pca.select_dtypes(include=["number"]).columns.tolist()

    # If no numeric columns exist, return empty dataframe
    if len(numeric_cols) == 0:
        return pd.DataFrame(index=df_pca.index)

    # Keep only numeric data
    X = df_pca[numeric_cols].copy()

    # Fill missing values with median
    for col in numeric_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())

    # If after filling there are still NaNs (rare edge case), fill with 0
    X = X.fillna(0)

    # Adjust n_components safely
    max_components = min(X.shape[0], X.shape[1])  # min(rows, columns)
    n_components = min(n_components, max_components)

    # If n_components becomes invalid, return original numeric data
    if n_components < 1:
        return X

    # Scale data before PCA (best practice)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)

    # Create PCA dataframe
    pca_columns = [f"PC{i+1}" for i in range(n_components)]
    df_pca_result = pd.DataFrame(pca_result, columns=pca_columns, index=df_pca.index)

    return df_pca_result