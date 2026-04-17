"""Model building module for computing similarity matrices."""

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_model(df):
    """
    Build TF-IDF vectorizer and compute cosine similarity matrix.
    
    Args:
        df: Dataframe with 'tags' column containing combined features
        
    Returns:
        Tuple of (vectorizer, similarity_matrix)
    """
    # Initialize TfidfVectorizer
    # Using a reasonable set of parameters:
    # - max_features: limit to top 5000 features
    # - stop_words: remove common English words
    # - ngram_range: use both unigrams and bigrams
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform the tags
    tfidf_matrix = vectorizer.fit_transform(df['tags'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    return vectorizer, similarity_matrix


def save_model(vectorizer, similarity_matrix, df, output_dir='models/'):
    """
    Save vectorizer, similarity matrix, and dataframe to disk.
    
    Args:
        vectorizer: Fitted TfidfVectorizer
        similarity_matrix: Computed cosine similarity matrix
        df: Processed dataframe with movie metadata
        output_dir: Directory to save model artifacts
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save vectorizer
    with open(f'{output_dir}vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Saved vectorizer to {output_dir}vectorizer.pkl")
    
    # Save similarity matrix
    with open(f'{output_dir}similarity_matrix.pkl', 'wb') as f:
        pickle.dump(similarity_matrix, f)
    print(f"Saved similarity matrix to {output_dir}similarity_matrix.pkl")
    
    # Save processed dataframe
    df.to_pickle(f'{output_dir}movies_df.pkl')
    print(f"Saved dataframe to {output_dir}movies_df.pkl")
    
    return f'{output_dir}'


def load_model(model_dir='models/'):
    """
    Load vectorizer, similarity matrix, and dataframe from disk.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Tuple of (vectorizer, similarity_matrix, dataframe)
    """
    # Load vectorizer
    with open(f'{model_dir}vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load similarity matrix
    with open(f'{model_dir}similarity_matrix.pkl', 'rb') as f:
        similarity_matrix = pickle.load(f)
    
    # Load dataframe
    df = pd.read_pickle(f'{model_dir}movies_df.pkl')
    
    return vectorizer, similarity_matrix, df


# Import pandas for load_model function
import pandas as pd
