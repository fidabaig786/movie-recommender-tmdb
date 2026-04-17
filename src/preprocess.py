"""Data preprocessing module for cleaning and preparing the movie dataset."""

import pandas as pd
import numpy as np
from utils.helpers import extract_cast, extract_director, parse_json_column


def load_data(movies_path):
    """
    Load movies dataset from CSV.
    
    Args:
        movies_path: Path to tmdb_5000_movies.csv
        
    Returns:
        Loaded dataframe
    """
    df = pd.read_csv(movies_path)
    print(f"Loaded {len(df)} movies from {movies_path}")
    return df


def extract_cast_and_director(df):
    """
    Extract cast and director from movies dataframe.
    
    Assumes the dataframe has a 'credits_data' column with crew information.
    For the TMDB dataset, we'll parse cast from 'cast' if available.
    
    Args:
        df: Movies dataframe
        
    Returns:
        Dataframe with 'top_cast' and 'director' columns added
    """
    df['top_cast'] = ''
    df['director'] = ''
    
    # Note: The tmdb_5000_movies.csv doesn't include cast/crew directly
    # We would need to merge with tmdb_5000_credits.csv for this
    
    return df


def parse_genres(df):
    """
    Parse genres column from JSON string.
    
    Args:
        df: Movies dataframe
        
    Returns:
        Dataframe with 'genre_list' column added
    """
    df['genre_list'] = df['genres'].apply(lambda x: parse_json_column(x))
    return df


def parse_keywords(df):
    """
    Parse keywords column from JSON string.
    
    Args:
        df: Movies dataframe
        
    Returns:
        Dataframe with 'keyword_list' column added
    """
    df['keyword_list'] = df['keywords'].apply(lambda x: parse_json_column(x))
    return df


def handle_missing_values(df):
    """
    Handle missing values in key columns.
    
    Args:
        df: Movies dataframe
        
    Returns:
        Dataframe with missing values handled
    """
    # Fill missing overview with empty string
    df['overview'] = df['overview'].fillna('')
    
    # Fill missing genres, keywords with empty JSON arrays
    df['genres'] = df['genres'].fillna('[]')
    df['keywords'] = df['keywords'].fillna('[]')
    
    # Ensure title exists
    df = df.dropna(subset=['title'])
    
    print(f"After handling missing values: {len(df)} movies remain")
    return df


def preprocess_data(movies_path, credits_path=None):
    """
    Complete preprocessing pipeline.
    
    Args:
        movies_path: Path to tmdb_5000_movies.csv
        credits_path: Optional path to tmdb_5000_credits.csv
        
    Returns:
        Cleaned dataframe
    """
    # Load movies data
    df = load_data(movies_path)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Parse JSON columns
    df = parse_genres(df)
    df = parse_keywords(df)
    
    # Load and merge credits if provided
    if credits_path:
        try:
            credits_df = pd.read_csv(credits_path)
            print(f"Loaded {len(credits_df)} credit records from {credits_path}")
            
            # Rename movie_id to id for merging
            credits_df = credits_df.rename(columns={'movie_id': 'id'})
            
            # Merge on movie ID
            df = df.merge(credits_df, on='id', how='left', suffixes=('', '_credits'))
            
            # Extract cast and director from merged credits
            df['top_cast'] = df['cast'].apply(lambda x: extract_cast(x, top_n=3))
            df['director'] = df['crew'].apply(extract_director)
            
            # Convert lists to comma-separated strings for easier concatenation
            df['top_cast'] = df['top_cast'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
            
        except Exception as e:
            print(f"Warning: Could not load credits data: {e}")
            df['top_cast'] = ''
            df['director'] = ''
    else:
        df['top_cast'] = ''
        df['director'] = ''
    
    # Convert genre and keyword lists to space-separated strings
    df['genres_str'] = df['genre_list'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    df['keywords_str'] = df['keyword_list'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    
    print("Preprocessing complete!")
    return df
