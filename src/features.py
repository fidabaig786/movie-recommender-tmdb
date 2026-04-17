"""Feature engineering module for creating recommendation features."""

from utils.helpers import clean_string


def create_tags(df):
    """
    Create combined 'tags' column from multiple features.
    
    Combines genres, keywords, cast, director, and overview into a single
    feature column, with proper cleaning (lowercase, remove spaces).
    
    Args:
        df: Preprocessed dataframe with the following columns:
            - genres_str: Space-separated genre names
            - keywords_str: Space-separated keywords
            - top_cast: Space-separated cast names
            - director: Director name
            - overview: Movie overview text
        
    Returns:
        Dataframe with new 'tags' column
    """
    def combine_features(row):
        """Combine all features into a single tag string."""
        components = [
            row.get('genres_str', ''),
            row.get('keywords_str', ''),
            row.get('top_cast', ''),
            row.get('director', ''),
            row.get('overview', '')
        ]
        
        # Join all non-empty components
        combined = ' '.join([str(c) for c in components if c])
        
        # Clean: lowercase and remove spaces in multi-word names
        cleaned = clean_string(combined)
        
        return cleaned
    
    df['tags'] = df.apply(combine_features, axis=1)
    
    print(f"Created 'tags' column for {len(df)} movies")
    return df


def engineer_features(df):
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Preprocessed dataframe
        
    Returns:
        Dataframe with engineered features
    """
    # Ensure required columns exist with default values
    if 'genres_str' not in df.columns:
        df['genres_str'] = ''
    if 'keywords_str' not in df.columns:
        df['keywords_str'] = ''
    if 'top_cast' not in df.columns:
        df['top_cast'] = ''
    if 'director' not in df.columns:
        df['director'] = ''
    if 'overview' not in df.columns:
        df['overview'] = ''
    
    # Create tags
    df = create_tags(df)
    
    print("Feature engineering complete!")
    return df
