# Project 4: Netflix Content Segmentation

## Overview
This project applies unsupervised machine learning algorithms to group Netflix movies and TV shows into distinct clusters based on content attributes, genre taxonomy, maturity ratings, and country distribution.

---

## Task Workflow (As per Specification)
1. **Step 1: Prepare numerical and categorical features**
   - Engineered TF-IDF representations of genre sequences.
   - Encoded geographic origins, maturity ratings, and release era.
   - Performed Principal Component Analysis (PCA) to mitigate high-dimensional sparsity.
2. **Step 2: Apply clustering algorithms**
   - Evaluated **K-Means Clustering** across $k \in [2, 8]$.
   - Derived optimal cluster count via the **Elbow Method (Inertia/WCSS)** and **Silhouette Analysis**:
     $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
3. **Step 3: Identify content groups**
   - Segmented catalog into 5 archetypes: International Dramas, US Blockbusters, Docuseries, Kids/Family Adventures, and Crime/Thriller series.
4. **Step 4: Visualize clusters**
   - Generated 2D PCA cluster scatter plots with marked centroids.
   - Generated volume and format breakdown charts.
5. **Step 5: Interpret cluster characteristics**
   - Mapped actionable business insights and content acquisition recommendations.

---

## Project Structure
```
Project_4_Netflix_Content_Segmentation/
├── README.md               # Technical specification
├── run_project.py          # Standalone execution pipeline
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading
│   ├── feature_engineering.py # PCA & preprocessing pipeline
│   ├── clustering.py       # K-Means & cluster profiling
│   ├── visualize.py        # Elbow, Silhouette & 2D PCA plots
│   └── evaluate.py         # Qualitative interpretation & summary
├── models/                 # Serialized K-Means & PCA transformers
└── reports/                # Visualizations, cluster summary & clustered dataset
```

---

## How to Run
```bash
python run_project.py
```
