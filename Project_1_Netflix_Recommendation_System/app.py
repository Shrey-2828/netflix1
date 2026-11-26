import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Task 1: Netflix Content Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #141414; color: #FFFFFF; }
    .stApp { background-color: #141414; }
    h1, h2, h3, h4 { color: #FFFFFF; }
    .title-banner { color: #E50914; font-weight: 800; font-size: 2.2rem; }
    .card { background: #1F1F1F; border-radius: 10px; padding: 18px; margin-bottom: 15px; border-left: 4px solid #E50914; }
    .badge-movie { background-color: #E50914; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }
    .badge-tv { background-color: #0071EB; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }
    .badge-score { background-color: #46D369; color: black; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-banner">🎬 Netflix Content Recommendation System</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#aaa;'>Task 1 (Easy) | NLP TF-IDF Cosine Similarity Engine</p>", unsafe_allow_html=True)

models_dir = os.path.join(os.path.dirname(__file__), 'models')
reports_dir = os.path.join(os.path.dirname(__file__), 'reports')

@st.cache_resource
def load_recommender():
    try:
        from src.recommender import NetflixRecommender
        df_meta = joblib.load(os.path.join(models_dir, 'title_metadata.joblib'))
        tfidf_mat = joblib.load(os.path.join(models_dir, 'tfidf_matrix.joblib'))
        vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
        return NetflixRecommender(df_meta, tfidf_mat, vectorizer)
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run `python run_project.py` first.")
        return None

recommender = load_recommender()

if recommender is not None:
    all_titles = recommender.df['title'].tolist()
    default_idx = all_titles.index("Stranger Things") if "Stranger Things" in all_titles else 0
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_title = st.selectbox("Select a Netflix Title:", all_titles, index=default_idx)
    with col2:
        top_n = st.slider("Number of Recommendations:", 3, 20, 6)
    with col3:
        fmt_filter = st.selectbox("Format Filter:", ["All", "Movie", "TV Show"])
        
    seed = recommender.df[recommender.df['title'] == selected_title].iloc[0]
    st.markdown(f"""
    <div style="background:#222; padding:12px 18px; border-radius:8px; margin-bottom:15px; border-left:4px solid #E50914;">
        <span class="badge-{'movie' if seed['type']=='Movie' else 'badge-tv'}">{seed['type']}</span>
        <strong style="color:white; font-size:1.15rem; margin-left:8px;">{seed['title']}</strong> ({seed['release_year']}) 
        <span style="color:#888; margin-left:12px;">Rating: {seed['rating']} | Country: {seed['country']}</span><br>
        <span style="color:#bbb; font-size:0.85rem;">Genres: {seed['listed_in']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    recs = recommender.recommend(selected_title, top_n=top_n, content_type_filter=fmt_filter)
    
    st.markdown(f"### 🎯 Top {len(recs)} Recommended Titles")
    grid = st.columns(2)
    for i, r in recs.iterrows():
        with grid[i % 2]:
            badge = "badge-movie" if r['type'] == 'Movie' else "badge-tv"
            st.markdown(f"""
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span class="{badge}">{r['type']}</span>
                    <span class="badge-score">Match: {r['similarity_score']}%</span>
                </div>
                <h4 style="margin:0 0 4px 0;">{r['title']} <span style="font-size:0.85rem; color:#888;">({r['release_year']})</span></h4>
                <p style="color:#ddd; font-size:0.85rem; margin:0 0 4px 0;"><b>Shared:</b> {r['shared_genres']}</p>
                <p style="color:#888; font-size:0.8rem; margin:0;"><b>Genres:</b> {r['genres']}<br><b>Country:</b> {r['country']} | <b>Rating:</b> {r['rating']}</p>
            </div>
            """, unsafe_allow_html=True)
            
    # Evaluation metric summary
    m_file = os.path.join(reports_dir, 'metrics_summary.json')
    if os.path.exists(m_file):
        st.markdown("---")
        st.markdown("### 📊 Model Quality & Diversity Evaluation")
        with open(m_file) as f:
            m = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Genre Hit Rate @ 10", f"{m.get('genre_hit_rate_at_k', 0)*100:.1f}%")
        c2.metric("Mean Genre Jaccard", f"{m.get('mean_genre_jaccard_similarity', 0):.3f}")
        c3.metric("Intra-List Diversity", f"{m.get('intra_list_diversity', 0):.3f}")
        c4.metric("Avg Similarity Score", f"{m.get('avg_top_k_similarity_pct', 0)}%")
        
        plot_path = os.path.join(reports_dir, 'genre_overlap_evaluation.png')
        if os.path.exists(plot_path):
            st.image(plot_path, caption="Quality & Jaccard Overlap Distribution", use_container_width=True)
