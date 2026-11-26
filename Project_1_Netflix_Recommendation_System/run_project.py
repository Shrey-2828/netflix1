import os
import sys

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.feature_engineering import create_content_soup, extract_tfidf_features
from src.recommender import NetflixRecommender
from src.evaluate import evaluate_recommendations

def main():
    print("=================================================================")
    print("   TASK 1: NETFLIX CONTENT RECOMMENDATION SYSTEM PIPELINE       ")
    print("=================================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Step 1: Prepare content-related features
    print("\n[Step 1/5] Loading dataset and preparing content-related features...")
    df = load_data()
    print(f" Loaded {len(df)} titles successfully.")
    soup = create_content_soup(df)
    print(f" Content soup generated for {len(soup)} titles.")
    
    # Step 2: Convert text data into machine-readable format
    print("\n[Step 2/5] Converting text features to TF-IDF matrix...")
    vectorizer, tfidf_matrix = extract_tfidf_features(soup)
    print(f" TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Step 3: Instantiate recommendation engine
    print("\n[Step 3/5] Initializing recommender and similarity engine...")
    recommender = NetflixRecommender(df, tfidf_matrix, vectorizer)
    
    # Step 4: Generate recommendations for selected sample titles
    print("\n[Step 4/5] Generating recommendations for test titles...")
    test_titles = ["Dick Johnson Is Dead", "Blood & Water", "Stranger Things", "Narcos"]
    for test_title in test_titles:
        try:
            print(f"\n Recommendations for: '{test_title}'")
            recs = recommender.recommend(test_title, top_n=5)
            for idx, r in recs.iterrows():
                print(f"   {idx+1}. [{r['type']}] {r['title']} ({r['release_year']}) - Match: {r['similarity_score']}% | Shared: {r['shared_genres']}")
        except Exception as e:
            print(f"   Skipped '{test_title}': {e}")
            
    # Step 5: Evaluate recommendation quality
    print("\n[Step 5/5] Evaluating recommendation quality across diversity & hit metrics...")
    metrics = evaluate_recommendations(recommender, sample_size=150, top_k=10, reports_dir=reports_dir)
    print(" Evaluation Complete. Metrics Summary:")
    for k, v in metrics.items():
        print(f"   - {k}: {v}")
        
    # Save model artifacts
    print("\n Persisting recommendation model artifacts...")
    recommender.save_model(models_dir)
    print(f" Models successfully saved to '{models_dir}'.")
    print(f" Reports and evaluation charts saved to '{reports_dir}'.")
    print("\n>>> Task 1 Pipeline Completed Successfully! <<<")

if __name__ == '__main__':
    main()
