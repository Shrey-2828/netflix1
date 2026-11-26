import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Persona naming template based on cluster dominant attributes
PERSONA_TEMPLATES = {
    'international_drama': 'International Dramas & Romance',
    'hollywood_action': 'Hollywood Action, Thrillers & Blockbusters',
    'family_animation': 'Family, Animated & Kids Adventures',
    'documentary_factual': 'Documentaries & Culture Docuseries',
    'crime_mystery_tv': 'Crime Mysteries & Suspense Series'
}

def find_optimal_clusters(X_pca, k_range=range(2, 9), random_state: int = 42):
    """
    Step 2: Apply clustering algorithms across a range of k to evaluate Elbow and Silhouette scores.
    """
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=random_state)
        km.fit(X_pca)
        inertias.append(float(km.inertia_))
        
        # Compute silhouette on sample for speed and stability
        sample_indices = np.random.choice(X_pca.shape[0], size=min(2000, X_pca.shape[0]), replace=False)
        sil = silhouette_score(X_pca[sample_indices], km.labels_[sample_indices])
        silhouette_scores.append(float(sil))
        
    return list(k_range), inertias, silhouette_scores

def fit_kmeans(X_pca, n_clusters: int = 5, random_state: int = 42):
    """Fit final K-Means model with chosen cluster count."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=15, max_iter=400, random_state=random_state)
    labels = kmeans.fit_predict(X_pca)
    return kmeans, labels

def profile_clusters(df: pd.DataFrame, labels: np.ndarray):
    """
    Step 3: Identify content groups and analyze semantic cluster characteristics.
    """
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    
    cluster_profiles = []
    
    for c in sorted(df_clustered['cluster'].unique()):
        sub = df_clustered[df_clustered['cluster'] == c]
        total_items = len(sub)
        
        # Dominant genre tokens
        all_genres = []
        for g_list in sub['listed_in'].dropna():
            all_genres.extend([g.strip() for g in g_list.split(',') if g.strip()])
        top_genres = pd.Series(all_genres).value_counts().head(3).to_dict()
        
        # Top country
        top_countries = sub['country'].value_counts().head(3).to_dict()
        
        # Dominant type
        type_dist = sub['type'].value_counts(normalize=True).to_dict()
        
        # Dominant rating
        top_ratings = sub['rating'].value_counts().head(2).to_dict()
        
        # Average release year
        mean_release = round(float(pd.to_numeric(sub['release_year'], errors='coerce').mean()), 1)
        
        # Name cluster based on top genres & type
        primary_genre = list(top_genres.keys())[0] if top_genres else "General"
        primary_country = list(top_countries.keys())[0] if top_countries else "Global"
        
        cluster_name = f"Cluster {c}: {primary_genre} ({primary_country})"
        
        cluster_profiles.append({
            'cluster_id': int(c),
            'cluster_name': cluster_name,
            'size': int(total_items),
            'percentage': round(float(total_items / len(df_clustered) * 100), 2),
            'type_distribution': {k: round(v*100, 1) for k, v in type_dist.items()},
            'top_genres': top_genres,
            'top_countries': top_countries,
            'top_ratings': top_ratings,
            'avg_release_year': mean_release,
            'sample_titles': sub['title'].head(4).tolist()
        })
        
    return df_clustered, cluster_profiles
