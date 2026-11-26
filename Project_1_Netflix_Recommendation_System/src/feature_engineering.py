import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text: str) -> str:
    """Lowercase and remove non-alphanumeric noise while preserving key tokens."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_content_soup(df: pd.DataFrame) -> pd.Series:
    """
    Step 1: Prepare content-related features.
    Synthesize genres, categories, director, country, rating, and content type into a single rich text feature.
    Genre features ('listed_in') are given higher emphasis through token repetition.
    """
    soups = []
    for _, row in df.iterrows():
        # Clean individual tokens
        genres = clean_text(row.get('listed_in', ''))
        # Repeat genres 2x to ensure content genre affinity has strong weighting
        weighted_genres = f"{genres} {genres}"
        director = clean_text(row.get('director', ''))
        country = clean_text(row.get('country', ''))
        rating = clean_text(row.get('rating', ''))
        content_type = clean_text(row.get('type', ''))
        
        soup = f"{weighted_genres} {director} {country} {rating} {content_type}"
        soups.append(soup)
        
    return pd.Series(soups, index=df.index)

def extract_tfidf_features(corpus: pd.Series, max_features: int = 5000):
    """
    Step 2: Convert text data into machine-readable format.
    Transforms text corpus into TF-IDF numerical matrix with unigrams and bigrams.
    """
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix
