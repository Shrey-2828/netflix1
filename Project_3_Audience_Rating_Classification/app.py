import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Task 3: Audience Rating Classifier",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #141414; color: #FFFFFF; }
    .stApp { background-color: #141414; }
    h1, h2, h3, h4 { color: #FFFFFF; }
    .title-banner { color: #E50914; font-weight: 800; font-size: 2.2rem; }
    .card { background: #1F1F1F; border-radius: 10px; padding: 18px; margin-bottom: 15px; border-left: 4px solid #46D369; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-banner">🎯 Netflix Audience Rating Classification</div>', unsafe_allow_html=True)

models_dir = os.path.join(os.path.dirname(__file__), 'models')
reports_dir = os.path.join(os.path.dirname(__file__), 'reports')

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(os.path.join(models_dir, 'best_rating_model.joblib'))
        preprocessor = joblib.load(os.path.join(models_dir, 'rating_preprocessor.joblib'))
        encoder = joblib.load(os.path.join(models_dir, 'rating_encoder.joblib'))
        return model, preprocessor, encoder
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run `python run_project.py` first.")
        return None, None, None

model, preprocessor, encoder = load_assets()

if model is not None:
    st.markdown("### 🛡️ Live Audience Maturity Tier Prediction")
    with st.form("rating_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            title_input = st.text_input("Title:", "Stranger Things")
            format_input = st.selectbox("Format:", ["TV Show", "Movie"])
        with col2:
            country_input = st.selectbox("Country of Origin:", ["United States", "United Kingdom", "India", "Japan", "South Korea", "Other"])
            duration_input = st.number_input("Duration (Minutes or Season Index):", 5, 240, 50)
        with col3:
            year_input = st.slider("Release Year:", 1980, 2025, 2016)
            genres_input = st.text_input("Genres:", "TV Sci-Fi & Fantasy, TV Horror, TV Mysteries")
            
        submitted = st.form_submit_button("Predict Maturity Rating Tier")
        
    if submitted:
        sample_df = pd.DataFrame([{
            'type': format_input,
            'country_top': country_input,
            'duration_num': float(duration_input),
            'release_year': year_input,
            'has_director': 1 if format_input == "Movie" else 0,
            'genres_text': " ".join([g.strip().lower().replace(" ", "_") for g in genres_input.split(',') if g.strip()])
        }])
        
        X_sample = preprocessor.transform(sample_df)
        pred_code = model.predict(X_sample)[0]
        tier_label = encoder.inverse_transform([pred_code])[0]
        
        st.success(f"### 🛡️ Predicted Maturity Tier: **{tier_label.replace('_', ' ')}**")
        
    st.markdown("---")
    st.markdown("### ⚙️ Hyperparameter Tuning Gains & Model Diagnostic Reports")
    rep_file = os.path.join(reports_dir, 'classification_report.json')
    if os.path.exists(rep_file):
        with open(rep_file) as f:
            rep_data = json.load(f)
        st.markdown(f"**Identified Best Parameters (GridSearchCV):** `{rep_data.get('best_hyperparameters')}`")
        st.dataframe(pd.DataFrame(rep_data.get('model_comparison', [])), use_container_width=True)
        
    c1, c2 = st.columns(2)
    with c1:
        cm_path = os.path.join(reports_dir, 'multiclass_confusion_matrix.png')
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Multi-Class Confusion Matrix", use_container_width=True)
    with c2:
        fi_path = os.path.join(reports_dir, 'audience_rating_feature_importance.png')
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Top Feature Importance for Audience Ratings", use_container_width=True)
