import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Task 4: Netflix Content Segmentation",
    page_icon="🧩",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #141414; color: #FFFFFF; }
    .stApp { background-color: #141414; }
    h1, h2, h3, h4 { color: #FFFFFF; }
    .title-banner { color: #E50914; font-weight: 800; font-size: 2.2rem; }
    .card { background: #1F1F1F; border-radius: 10px; padding: 18px; margin-bottom: 15px; border-left: 4px solid #E50914; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-banner">🧩 Netflix Content Segmentation & Clustering</div>', unsafe_allow_html=True)

models_dir = os.path.join(os.path.dirname(__file__), 'models')
reports_dir = os.path.join(os.path.dirname(__file__), 'reports')

@st.cache_resource
def load_assets():
    try:
        kmeans = joblib.load(os.path.join(models_dir, 'kmeans_model.joblib'))
        pca_2d = joblib.load(os.path.join(models_dir, 'pca_2d.joblib'))
        preprocessor = joblib.load(os.path.join(models_dir, 'clustering_preprocessor.joblib'))
        df_clustered = pd.read_csv(os.path.join(reports_dir, 'clustered_netflix_dataset.csv'))
        with open(os.path.join(reports_dir, 'cluster_summary.json')) as f:
            summary = json.load(f)
        return kmeans, pca_2d, preprocessor, df_clustered, summary
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run `python run_project.py` first.")
        return None, None, None, None, None

kmeans, pca_2d, preprocessor, df_clustered, summary = load_assets()

if df_clustered is not None:
    st.markdown("### 🗺️ Interactive 2D PCA Cluster Map")
    sample_df = df_clustered.sample(n=min(1200, len(df_clustered)), random_state=42)
    
    fig = px.scatter(
        sample_df,
        x='release_year',
        y='type',
        color=sample_df['cluster'].astype(str),
        hover_data=['title', 'listed_in', 'country', 'rating'],
        title="Interactive Cluster Segmentation Map",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        labels={'color': 'Cluster ID'}
    )
    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 👥 Discovered Cluster Archetypes")
    profiles = summary.get('cluster_profiles', [])
    cols = st.columns(len(profiles))
    for i, p in enumerate(profiles):
        with cols[i]:
            st.markdown(f"""
            <div class="card" style="height:250px;">
                <h5 style="color:#E50914; margin:0 0 6px 0;">Cluster {p['cluster_id']}</h5>
                <strong style="font-size:0.95rem;">{p['cluster_name']}</strong><br>
                <span style="color:#46D369; font-size:0.8rem; font-weight:bold;">{p['size']} titles ({p['percentage']}%)</span>
                <hr style="border-color:#333; margin:8px 0;">
                <p style="color:#bbb; font-size:0.8rem; margin:0;">
                    <b>Top Genres:</b> {', '.join(list(p['top_genres'].keys())[:2])}<br>
                    <b>Avg Release Year:</b> {p['avg_release_year']}<br>
                    <b>Main Region:</b> {list(p['top_countries'].keys())[0]}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("### 📊 Clustering Evaluation: Elbow Curve & PCA Space")
    c1, c2 = st.columns(2)
    with c1:
        el_path = os.path.join(reports_dir, 'elbow_and_silhouette_curve.png')
        if os.path.exists(el_path):
            st.image(el_path, caption="Elbow & Silhouette Score Metrics", use_container_width=True)
    with c2:
        pca_path = os.path.join(reports_dir, 'pca_2d_clusters.png')
        if os.path.exists(pca_path):
            st.image(pca_path, caption="2D PCA Latent Space & Centroids", use_container_width=True)
