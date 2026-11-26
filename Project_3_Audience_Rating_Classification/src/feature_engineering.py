import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer

# Audience Maturity Tier Mapping
RATING_MAPPING = {
    'TV-MA': 'Mature_Adults',
    'R': 'Mature_Adults',
    'NC-17': 'Mature_Adults',
    'UR': 'Mature_Adults',
    
    'TV-14': 'Teens_14_Plus',
    'PG-13': 'Teens_14_Plus',
    
    'TV-PG': 'Older_Kids_Family',
    'PG': 'Older_Kids_Family',
    'TV-G': 'Older_Kids_Family',
    'G': 'Older_Kids_Family',
    
    'TV-Y': 'Young_Children',
    'TV-Y7': 'Young_Children',
    'TV-Y7-FV': 'Young_Children'
}

def analyze_and_map_ratings(df: pd.DataFrame):
    """
    Step 1: Analyze rating categories and establish standardized audience maturity tiers.
    Filters unrated/sparse edge cases (e.g., 'NR') and returns mapped dataset.
    """
    df_clean = df.copy()
    
    # Map raw ratings to standardized audience maturity tiers
    df_clean['target_tier'] = df_clean['rating'].map(RATING_MAPPING)
    
    # Drop titles where rating is unmapped / unrated (e.g. 'NR' or blank)
    df_clean = df_clean.dropna(subset=['target_tier']).reset_index(drop=True)
    
    tier_distribution = df_clean['target_tier'].value_counts().to_dict()
    raw_distribution = df_clean['rating'].value_counts().to_dict()
    
    return df_clean, tier_distribution, raw_distribution

def extract_duration_value(val: str, content_type: str) -> float:
    """Extract numeric value of duration."""
    m = re.search(r'(\d+)', str(val))
    if m:
        num = float(m.group(1))
        # If it's a TV Show season, approximate runtime or scale
        if 'season' in str(val).lower() or content_type == 'TV Show':
            return num * 10.0  # approximate season scale
        return num
    return 90.0

def prepare_features(df_clean: pd.DataFrame):
    """
    Step 2: Prepare training datasets.
    Extracts informative content attributes and constructs feature pipeline.
    """
    df_feat = df_clean.copy()
    
    # Numerical Duration
    df_feat['duration_num'] = [
        extract_duration_value(row['duration'], row['type']) 
        for _, row in df_feat.iterrows()
    ]
    
    # Release Year
    df_feat['release_year'] = pd.to_numeric(df_feat['release_year'], errors='coerce').fillna(2018)
    
    # Has Director Flag
    df_feat['has_director'] = df_feat['director'].apply(
        lambda d: 1 if d and str(d).lower() not in ['', 'not given', 'unknown'] else 0
    )
    
    # Top Countries
    top_countries = df_feat['country'].value_counts().head(12).index.tolist()
    df_feat['country_top'] = df_feat['country'].apply(lambda c: c if c in top_countries else 'Other')
    
    # Cleaned genre tokens
    df_feat['genres_text'] = df_feat['listed_in'].apply(
        lambda x: " ".join([g.strip().lower().replace(" ", "_").replace("-", "_") for g in str(x).split(',') if g.strip()])
    )
    
    # Encode target tiers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_feat['target_tier'])
    
    categorical_cols = ['type', 'country_top']
    numerical_cols = ['release_year', 'duration_num', 'has_director']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('genre', CountVectorizer(token_pattern=r'(?u)\b\w+\b', binary=True), 'genres_text')
        ],
        remainder='drop'
    )
    
    X = preprocessor.fit_transform(df_feat)
    
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    genre_names = [f"genre_{g}" for g in preprocessor.named_transformers_['genre'].get_feature_names_out().tolist()]
    feature_names = numerical_cols + cat_names + genre_names
    
    return X, y, preprocessor, label_encoder, feature_names
