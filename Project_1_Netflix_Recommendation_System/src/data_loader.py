import os
import pandas as pd

def load_data(csv_path: str = None) -> pd.DataFrame:
    """Load and perform preliminary validation on the Netflix dataset.
    
    Args:
        csv_path (str, optional): Path to the dataset CSV file. Defaults to 'Dataset.csv'.
        
    Returns:
        pd.DataFrame: Cleaned and validated dataset.
    """
    if csv_path is None:
        # Check current dir or parent dir
        if os.path.exists('Dataset.csv'):
            csv_path = 'Dataset.csv'
        elif os.path.exists('../Dataset.csv'):
            csv_path = '../Dataset.csv'
        elif os.path.exists(r'C:\timepass\Dataset.csv'):
            csv_path = r'C:\timepass\Dataset.csv'
        else:
            raise FileNotFoundError("Dataset.csv could not be located.")
            
    df = pd.read_csv(csv_path)
    
    # Normalize string columns
    text_cols = ['title', 'director', 'country', 'rating', 'duration', 'listed_in', 'type']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()
            
    # Drop duplicates if any
    df = df.drop_duplicates(subset=['title', 'type', 'release_year']).reset_index(drop=True)
    return df
