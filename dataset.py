import pandas as pd
import os

def load_and_clean_dataset():
    dataset_path = os.path.join('data', '/Users/noortamer/Desktop/newtask/data/cleaned_for_embedding.csv')
    
    df = pd.read_csv(dataset_path)
    
    df = df[['embedding_text', 'category']].dropna()
    
    return df