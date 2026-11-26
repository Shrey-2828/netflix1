import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import joblib

class NetflixRecommender:
    """Content-Based Recommender System for Netflix titles."""
    
    def __init__(self, df: pd.DataFrame, tfidf_matrix, vectorizer):
        self.df = df.copy().reset_index(drop=True)
        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = vectorizer
        # Build quick lookup dictionary for title case-insensitive indices
        self.title_indices = {
            str(title).strip().lower(): idx 
            for idx, title in enumerate(self.df['title'])
        }
        
    def get_title_index(self, query_title: str):
        """Case-insensitive search for a title index."""
        normalized = str(query_title).strip().lower()
        if normalized in self.title_indices:
            return self.title_indices[normalized]
        # Partial match fallback
        for title, idx in self.title_indices.items():
            if normalized in title:
                return idx
        return None

    def recommend(self, title: str, top_n: int = 10, content_type_filter: str = 'All') -> pd.DataFrame:
        """
        Step 3 & Step 4: Calculate similarity scores and generate top-N recommendations.
        
        Args:
            title (str): Seed Netflix title.
            top_n (int): Number of recommendations to generate.
            content_type_filter (str): 'All', 'Movie', or 'TV Show'.
            
        Returns:
            pd.DataFrame: Recommended titles with similarity score and metadata.
        """
        idx = self.get_title_index(title)
        if idx is None:
            raise ValueError(f"Title '{title}' not found in Netflix catalogue.")
            
        # Calculate cosine similarity of target against all items (linear kernel on normalized TF-IDF)
        seed_vector = self.tfidf_matrix[idx]
        sim_scores = linear_kernel(seed_vector, self.tfidf_matrix).flatten()
        
        # Rank descending
        ranked_indices = np.argsort(sim_scores)[::-1]
        
        recommendations = []
        seed_row = self.df.iloc[idx]
        seed_genres = set([g.strip().lower() for g in seed_row['listed_in'].split(',') if g.strip()])
        
        for rank_idx in ranked_indices:
            if rank_idx == idx:
                continue  # skip the seed item itself
                
            candidate = self.df.iloc[rank_idx]
            
            # Apply content type filtering if requested
            if content_type_filter != 'All' and candidate['type'] != content_type_filter:
                continue
                
            cand_genres = set([g.strip().lower() for g in candidate['listed_in'].split(',') if g.strip()])
            shared_genres = seed_genres.intersection(cand_genres)
            
            similarity_pct = round(float(sim_scores[rank_idx]) * 100, 2)
            
            recommendations.append({
                'title': candidate['title'],
                'type': candidate['type'],
                'similarity_score': similarity_pct,
                'rating': candidate['rating'],
                'release_year': candidate['release_year'],
                'genres': candidate['listed_in'],
                'shared_genres': ", ".join(shared_genres) if shared_genres else "General Affinity",
                'director': candidate['director'],
                'country': candidate['country']
            })
            
            if len(recommendations) >= top_n:
                break
                
        return pd.DataFrame(recommendations)

    def save_model(self, model_dir: str):
        """Persist metadata and vectorizer for reproducible serving."""
        joblib.dump(self.df[['show_id', 'type', 'title', 'director', 'country', 'rating', 'listed_in', 'release_year', 'duration']], f"{model_dir}/title_metadata.joblib")
        joblib.dump(self.vectorizer, f"{model_dir}/tfidf_vectorizer.joblib")
        joblib.dump(self.tfidf_matrix, f"{model_dir}/tfidf_matrix.joblib")
