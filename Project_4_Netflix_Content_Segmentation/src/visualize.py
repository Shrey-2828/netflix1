import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_elbow_and_silhouette(k_values, inertias, silhouette_scores, reports_dir: str = 'reports'):
    """Plot Elbow Method and Silhouette Analysis dual curves."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = '#E50914'
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Inertia (WCSS)', color=color, fontsize=11, fontweight='bold')
    ax1.plot(k_values, inertias, marker='o', color=color, lw=2.5, label='Inertia (Elbow)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color2 = '#1f77b4'
    ax2.set_ylabel('Silhouette Score', color=color2, fontsize=11, fontweight='bold')
    ax2.plot(k_values, silhouette_scores, marker='s', color=color2, lw=2.5, linestyle='--', label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Optimal Cluster Evaluation: Elbow Curve & Silhouette Analysis', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f"{reports_dir}/elbow_and_silhouette_curve.png", dpi=300)
    plt.close()

def plot_2d_pca_clusters(X_2d, labels, cluster_profiles, reports_dir: str = 'reports'):
    """Plot 2D PCA projection of content clusters with centroids."""
    df_pca = pd.DataFrame({
        'PCA1': X_2d[:, 0],
        'PCA2': X_2d[:, 1],
        'Cluster': [f"Cluster {l}" for l in labels]
    })
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_pca, x='PCA1', y='PCA2', hue='Cluster',
        palette='tab10', alpha=0.6, s=25, edgecolor=None
    )
    
    # Calculate and plot centroids in 2D
    for c in np.unique(labels):
        center_x = np.mean(X_2d[labels == c, 0])
        center_y = np.mean(X_2d[labels == c, 1])
        plt.scatter(center_x, center_y, s=150, c='black', marker='X', edgecolor='white', lw=1.5)
        plt.text(center_x, center_y + 0.15, f"C{c}", fontsize=11, fontweight='bold', ha='center',
                 bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7))
        
    plt.title('Netflix Content Segmentation: 2D PCA Cluster Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Principal Component 1 (Primary Genre & Production Axis)')
    plt.ylabel('Principal Component 2 (Format & Release Era Axis)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{reports_dir}/pca_2d_clusters.png", dpi=300)
    plt.close()

def plot_cluster_breakdown(cluster_profiles, reports_dir: str = 'reports'):
    """Plot cluster size and dominant format breakdown."""
    names = [p['cluster_name'] for p in cluster_profiles]
    sizes = [p['size'] for p in cluster_profiles]
    
    plt.figure(figsize=(10, 5))
    bars = plt.barh(names, sizes, color=sns.color_palette('rocket', len(names)))
    plt.title('Cluster Volume & Content Group Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Titles')
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 20, bar.get_y() + bar.get_height()/2.0, f"{int(w)} titles", va='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f"{reports_dir}/cluster_genre_profiles.png", dpi=300)
    plt.close()
