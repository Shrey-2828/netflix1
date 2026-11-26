import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Task 2: Content Type Classifier",
    page_icon="🏷️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #141414; color: #FFFFFF; }
    .stApp { background-color: #141414; }
    h1, h2, h3, h4 { color: #FFFFFF; }
    .title-banner { color: #E50914; font-weight: 800; font-size: 2.2rem; }
    .card { background: #1F1F1F; border-radius: 10px; padding: 18px; margin-bottom: 15px; border-left: 4px solid #0071EB; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-banner">🏷️ Netflix Content Type Prediction Model</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#aaa;'>Task 2 (Easy) | Binary Classification Benchmarking (Leakage-Free)</p>", unsafe_allow_html=True)

models_dir = os.path.join(os.path.dirname(__file__), 'models')
reports_dir = os.path.join(os.path.dirname(__file__), 'reports')

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(os.path.join(models_dir, 'best_model.joblib'))
        preprocessor = joblib.load(os.path.join(models_dir, 'feature_pipeline.joblib'))
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names.joblib'))
        return model, preprocessor, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run `python run_project.py` first.")
        return None, None, None

model, preprocessor, feature_names = load_assets()

if model is not None:
    st.markdown("### 🧪 Live Inference Form")
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            title_input = st.text_input("Title:", "Breaking Bad")
            country_input = st.selectbox("Country of Origin:", ["United States", "India", "United Kingdom", "Japan", "South Korea", "Spain", "Other"])
        with col2:
            year_input = st.slider("Release Year:", 1970, 2025, 2013)
            rating_input = st.selectbox("Maturity Rating:", ["TV-MA", "TV-14", "R", "PG-13", "TV-PG", "TV-Y", "Other"])
        with col3:
            has_director_input = st.selectbox("Has Named Director?", ["No", "Yes"])
            genres_input = st.text_input("Genre Tokens:", "Crime TV Shows, TV Dramas, TV Thrillers")
            
        submitted = st.form_submit_button("Predict Content Type")
        
    if submitted:
        sample_df = pd.DataFrame([{
            'title': title_input,
            'country_top': country_input,
            'release_year': year_input,
            'year_added': 2021,
            'month_added': 8,
            'title_word_count': len(title_input.split()),
            'title_char_count': len(title_input),
            'rating_top': rating_input,
            'has_director': 1 if has_director_input == "Yes" else 0,
            'genres_text': " ".join([g.strip().lower().replace(" ", "_") for g in genres_input.split(',') if g.strip()])
        }])
        
        X_sample = preprocessor.transform(sample_df)
        pred_class = model.predict(X_sample)[0]
        pred_probs = model.predict_proba(X_sample)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
        
        result_type = "Movie" if pred_class == 1 else "TV Show"
        conf = pred_probs[1] if pred_class == 1 else pred_probs[0]
        
        st.success(f"### 🎉 Result: **{result_type}** (Confidence: {conf*100:.2f}%)")
        
    st.markdown("---")
    st.markdown("### 📈 Multi-Model Benchmark Results")
    metrics_file = os.path.join(reports_dir, 'model_comparison_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics_data = json.load(f)
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
    c1, c2 = st.columns(2)
    with c1:
        cm_path = os.path.join(reports_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
    with c2:
        roc_path = os.path.join(reports_dir, 'roc_curve.png')
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curves Comparison", use_container_width=True)
