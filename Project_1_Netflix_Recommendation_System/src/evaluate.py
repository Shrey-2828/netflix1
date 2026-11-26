import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0

def evaluate_recommendations(recommender, sample_size: int = 200, top_k: int = 10, reports_dir: str = 'reports'):
    """
    Step 5: Evaluate recommendation quality.
    Evaluates system on sample titles measuring:
    1. Average Genre Jaccard Similarity (Relevance)
    2. Genre Hit Rate @ K (Accuracy Proxy)
    3. Intra-List Diversity (1 - pairwise similarity among recommended items)
    4. Average Top-K Similarity Score
    """
    df = recommender.df
    # Sample across movies and TV shows
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    jaccard_scores = []
    hit_rates = []
    intra_list_diversities = []
    avg_similarity_scores = []
    
    for _, row in sample_df.iterrows():
        title = row['title']
        seed_genres = set([g.strip().lower() for g in row['listed_in'].split(',') if g.strip()])
        
        try:
            recs = recommender.recommend(title, top_n=top_k, content_type_filter='All')
            if recs.empty:
                continue
                
            # Metric 1: Genre Jaccard Overlap
            overlaps = []
            rec_genre_sets = []
            for _, rec_row in recs.iterrows():
                cand_genres = set([g.strip().lower() for g in rec_row['genres'].split(',') if g.strip()])
                rec_genre_sets.append(cand_genres)
                overlaps.append(jaccard_similarity(seed_genres, cand_genres))
                
            avg_jaccard = np.mean(overlaps)
            jaccard_scores.append(avg_jaccard)
            
            # Metric 2: Hit Rate @ K (at least 1 shared genre)
            hits = sum(1 for o in overlaps if o > 0)
            hit_rates.append(hits / len(recs))
            
            # Metric 3: Intra-List Diversity (average pairwise 1 - Jaccard between recs)
            pair_diversities = []
            for i in range(len(rec_genre_sets)):
                for j in range(i + 1, len(rec_genre_sets)):
                    pair_diversities.append(1.0 - jaccard_similarity(rec_genre_sets[i], rec_genre_sets[j]))
            if pair_diversities:
                intra_list_diversities.append(np.mean(pair_diversities))
                
            # Metric 4: Average similarity score
            avg_similarity_scores.append(np.mean(recs['similarity_score']))
            
        except Exception:
            continue

    metrics = {
        "evaluation_sample_size": len(jaccard_scores),
        "top_k": top_k,
        "mean_genre_jaccard_similarity": round(float(np.mean(jaccard_scores)), 4),
        "genre_hit_rate_at_k": round(float(np.mean(hit_rates)), 4),
        "intra_list_diversity": round(float(np.mean(intra_list_diversities)), 4),
        "avg_top_k_similarity_pct": round(float(np.mean(avg_similarity_scores)), 2)
    }
    
    # Save metrics JSON
    with open(f"{reports_dir}/metrics_summary.json", 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Generate Visualizations
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(jaccard_scores, kde=True, color='#E50914', bins=15)
    plt.title(f'Genre Overlap Distribution (Mean Jaccard: {metrics["mean_genre_jaccard_similarity"]})', fontsize=11, fontweight='bold')
    plt.xlabel('Jaccard Similarity Score')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    metrics_display = {
        'Genre Hit Rate @ 10': metrics['genre_hit_rate_at_k'],
        'Intra-List Diversity': metrics['intra_list_diversity'],
        'Mean Genre Jaccard': metrics['mean_genre_jaccard_similarity']
    }
    bars = plt.bar(metrics_display.keys(), metrics_display.values(), color=['#221F1F', '#E50914', '#B81D24'])
    plt.ylim(0, 1.1)
    plt.title('Content Quality & Diversity Benchmark', fontsize=11, fontweight='bold')
    plt.ylabel('Score (0 - 1.0)')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f"{reports_dir}/genre_overlap_evaluation.png", dpi=300)
    plt.close()
    
    return metrics
