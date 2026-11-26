import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_features(df: pd.DataFrame):
    """
    Step 1 & Step 2: Feature Selection and Categorical Encoding.
    Extracts informative content attributes while strictly preventing data leakage
    (raw duration unit 'min'/'Season' is intentionally excluded to ensure true predictive modeling).
    """
    df_feat = df.copy()
    
    # Feature 1: Director Presence (Movies much more frequently report dedicated individual directors)
    df_feat['has_director'] = df_feat['director'].apply(lambda d: 1 if d and str(d).lower() not in ['', 'not given', 'unknown'] else 0)
    
    # Feature 2: Release Year & Historical Age
    df_feat['release_year'] = pd.to_numeric(df_feat['release_year'], errors='coerce').fillna(2018)
    
    # Feature 3: Year & Month Added to Netflix
    dates = pd.to_datetime(df_feat['date_added'], errors='coerce')
    df_feat['year_added'] = dates.dt.year.fillna(df_feat['release_year'])
    df_feat['month_added'] = dates.dt.month.fillna(6)
    
    # Feature 4: Title Word Count & Character Count
    df_feat['title_word_count'] = df_feat['title'].apply(lambda t: len(str(t).split()))
    df_feat['title_char_count'] = df_feat['title'].apply(lambda t: len(str(t)))
    
    # Feature 5: Top Countries
    top_countries = df_feat['country'].value_counts().head(15).index.tolist()
    df_feat['country_top'] = df_feat['country'].apply(lambda c: c if c in top_countries else 'Other')
    
    # Feature 6: Rating categories
    top_ratings = df_feat['rating'].value_counts().head(10).index.tolist()
    df_feat['rating_top'] = df_feat['rating'].apply(lambda r: r if r in top_ratings else 'Other')
    
    # Feature 7: Genres / Listed_in tokenization
    df_feat['genres_text'] = df_feat['listed_in'].apply(
        lambda x: " ".join([g.strip().lower().replace(" ", "_").replace("-", "_") for g in str(x).split(',') if g.strip()])
    )
    
    # Binary Target: Movie -> 1, TV Show -> 0
    y = (df_feat['type'] == 'Movie').astype(int).values
    
    # Define Column Transformer for Encoding
    categorical_cols = ['country_top', 'rating_top']
    numerical_cols = ['release_year', 'year_added', 'month_added', 'title_word_count', 'title_char_count', 'has_director']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('genre', CountVectorizer(token_pattern=r'(?u)\b\w+\b', binary=True), 'genres_text')
        ],
        remainder='drop'
    )
    
    X = preprocessor.fit_transform(df_feat)
    
    # Get feature names for interpretability
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    genre_feature_names = [f"genre_{g}" for g in preprocessor.named_transformers_['genre'].get_feature_names_out().tolist()]
    feature_names = numerical_cols + cat_feature_names + genre_feature_names
    
    return X, y, preprocessor, feature_names
