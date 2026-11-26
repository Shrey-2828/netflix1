import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.feature_engineering import prepare_clustering_features
from src.clustering import find_optimal_clusters, fit_kmeans, profile_clusters
from src.visualize import plot_elbow_and_silhouette, plot_2d_pca_clusters, plot_cluster_breakdown
from src.evaluate import interpret_and_save_summary

def main():
    print("=================================================================")
    print("    TASK 4: NETFLIX CONTENT SEGMENTATION & CLUSTERING PIPELINE   ")
    print("=================================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Step 1: Prepare numerical and categorical features
    print("\n[Step 1/5] Extracting numerical, categorical & TF-IDF features with PCA reduction...")
    df = load_data()
    X_pca, X_2d, preprocessor, pca_cluster, pca_2d = prepare_clustering_features(df, n_pca_components=15)
    print(f" Dataset loaded: {len(df)} titles.")
    print(f" Dimensionality reduced feature shape: {X_pca.shape}")
    
    # Step 2: Apply clustering algorithms & determine optimal k
    print("\n[Step 2/5] Evaluating K-Means across k in [2..8] via Elbow & Silhouette metrics...")
    k_range, inertias, silhouette_scores = find_optimal_clusters(X_pca, k_range=range(2, 8))
    for k, inert, sil in zip(k_range, inertias, silhouette_scores):
        print(f"   k={k} | Inertia (WCSS): {inert:,.1f} | Silhouette Score: {sil:.4f}")
        
    optimal_k = 5
    print(f"\n Fitting production K-Means with optimal k={optimal_k}...")
    kmeans_model, labels = fit_kmeans(X_pca, n_clusters=optimal_k)
    
    # Step 3: Identify content groups
    print("\n[Step 3/5] Identifying content groups and profiling cluster archetypes...")
    df_clustered, cluster_profiles = profile_clusters(df, labels)
    for p in cluster_profiles:
        print(f"\n   ==> {p['cluster_name']} ({p['size']} titles, {p['percentage']}%)")
        print(f"       Top Genres: {list(p['top_genres'].keys())}")
        print(f"       Top Countries: {list(p['top_countries'].keys())}")
        print(f"       Avg Release Year: {p['avg_release_year']}")
        print(f"       Sample Titles: {p['sample_titles']}")
        
    # Step 4: Visualize clusters
    print("\n[Step 4/5] Generating 2D PCA cluster visualizations and Elbow curves...")
    plot_elbow_and_silhouette(k_range, inertias, silhouette_scores, reports_dir=reports_dir)
    plot_2d_pca_clusters(X_2d, labels, cluster_profiles, reports_dir=reports_dir)
    plot_cluster_breakdown(cluster_profiles, reports_dir=reports_dir)
    
    # Step 5: Interpret cluster characteristics
    print("\n[Step 5/5] Interpreting cluster characteristics and synthesizing business insights...")
    summary = interpret_and_save_summary(cluster_profiles, silhouette_scores, chosen_k=optimal_k, reports_dir=reports_dir)
    
    # Save Artifacts
    print("\n Persisting clustering models & preprocessors...")
    joblib.dump(kmeans_model, os.path.join(models_dir, 'kmeans_model.joblib'))
    joblib.dump(pca_cluster, os.path.join(models_dir, 'pca_transformer.joblib'))
    joblib.dump(pca_2d, os.path.join(models_dir, 'pca_2d.joblib'))
    joblib.dump(preprocessor, os.path.join(models_dir, 'clustering_preprocessor.joblib'))
    df_clustered.to_csv(os.path.join(reports_dir, 'clustered_netflix_dataset.csv'), index=False)
    
    print(f" Models saved to '{models_dir}'.")
    print(f" Diagnostic plots, CSV and summary saved to '{reports_dir}'.")
    print("\n>>> Task 4 Pipeline Completed Successfully! <<<")

if __name__ == '__main__':
    main()
