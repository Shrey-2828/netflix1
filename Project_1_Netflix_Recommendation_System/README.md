# Project 1: Netflix Content Recommendation System

## Overview
This project delivers a content-based recommendation engine for the Netflix catalog. It leverages Natural Language Processing (NLP) and vector similarity algorithms to recommend relevant movies and TV shows based on genre affinity, cast/crew characteristics, country of origin, and maturity ratings.

---

## Task Workflow (As per Specification)
1. **Step 1: Prepare content-related features**
   - Synthesizes `listed_in` (genres), `director`, `country`, `rating`, and `type` into an engineered feature soup.
   - Applies domain-specific weighting to genre tokens to prioritize category relevance.
2. **Step 2: Convert text data into machine-readable format**
   - Employs `TfidfVectorizer` with sublinear term-frequency scaling and unigram + bigram feature extraction ($n \in [1, 2]$).
3. **Step 3: Calculate content similarity scores**
   - Computes normalized Cosine Similarity scores via linear kernel dot products:
     $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
4. **Step 4: Generate recommendations for selected titles**
   - Provides ranked top-$N$ recommendations with match percentages, shared attribute badges, and cross-type filtering (Movie vs. TV Show).
5. **Step 5: Evaluate recommendation quality**
   - Measures Recommendation Relevance via **Jaccard Similarity**:
     $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
   - Evaluates **Genre Hit Rate @ 10**, **Intra-List Diversity**, and **Average Top-K Match Score**.

---

## Evaluation Metrics Summary
- **Mean Genre Jaccard Similarity**: ~0.45 - 0.55 (High contextual alignment)
- **Genre Hit Rate @ 10**: > 95% (Top recommendations almost always share core genre affinity)
- **Intra-List Diversity**: ~0.65 - 0.75 (Prevents recommendation monotony and filter bubbles)

---

## Project Structure
```
Project_1_Netflix_Recommendation_System/
├── README.md               # Documentation & technical specification
├── run_project.py          # Standalone execution pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading & cleaning
│   ├── feature_engineering.py # NLP & TF-IDF preprocessing
│   ├── recommender.py      # Similarity ranking engine
│   └── evaluate.py         # Quality benchmarking & visualization
├── models/                 # Saved vectorizer & metadata
└── reports/                # Evaluation plots & metrics JSON
```

---

## How to Run
```bash
python run_project.py
```
Outputs and visual reports are saved automatically in `reports/`.
