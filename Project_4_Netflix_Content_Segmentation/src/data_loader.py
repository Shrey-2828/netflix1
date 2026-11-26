import os
import pandas as pd

def load_data(csv_path: str = None) -> pd.DataFrame:
    """Load and perform preliminary validation on the Netflix dataset for Task 4."""
    if csv_path is None:
        if os.path.exists('Dataset.csv'):
            csv_path = 'Dataset.csv'
        elif os.path.exists('../Dataset.csv'):
            csv_path = '../Dataset.csv'
        elif os.path.exists(r'C:\timepass\Dataset.csv'):
            csv_path = r'C:\timepass\Dataset.csv'
        else:
            raise FileNotFoundError("Dataset.csv could not be located.")
            
    df = pd.read_csv(csv_path)
    
    # Clean whitespace and strip strings
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('').astype(str).str.strip()
            
    return df
