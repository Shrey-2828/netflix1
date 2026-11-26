import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

def prepare_clustering_features(df: pd.DataFrame, n_pca_components: int = 15):
    """
    Step 1: Prepare numerical and categorical features.
    Extracts multi-faceted features across genres, countries, ratings, and release era,
    followed by variance-retaining PCA dimensionality reduction.
    """
    df_feat = df.copy()
    
    # Clean genres
    df_feat['genres_text'] = df_feat['listed_in'].apply(
        lambda x: " ".join([g.strip().lower().replace(" ", "_") for g in str(x).split(',') if g.strip()])
    )
    
    # Top countries
    top_countries = df_feat['country'].value_counts().head(12).index.tolist()
    df_feat['country_top'] = df_feat['country'].apply(lambda c: c if c in top_countries else 'Other')
    
    # Top ratings
    top_ratings = df_feat['rating'].value_counts().head(8).index.tolist()
    df_feat['rating_top'] = df_feat['rating'].apply(lambda r: r if r in top_ratings else 'Other')
    
    # Numerical release year
    df_feat['release_year_num'] = pd.to_numeric(df_feat['release_year'], errors='coerce').fillna(2018)
    
    categorical_cols = ['type', 'country_top', 'rating_top']
    numerical_cols = ['release_year_num']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('genre', TfidfVectorizer(max_features=100, binary=False), 'genres_text')
        ],
        remainder='drop'
    )
    
    X_raw = preprocessor.fit_transform(df_feat)
    
    # Apply PCA for dimensionality reduction (curse of dimensionality mitigation)
    pca_cluster = PCA(n_components=n_pca_components, random_state=42)
    X_pca = pca_cluster.fit_transform(X_raw)
    
    # Also 2D PCA for spatial visualization
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_raw)
    
    return X_pca, X_2d, preprocessor, pca_cluster, pca_2d
