"""Recommendation logic module for generating movie suggestions."""

import numpy as np


def recommend(movie_title, similarity_matrix, df, top_n=10):
    """
    Recommend top N movies similar to the input movie.
    
    Args:
        movie_title: Title of the movie to get recommendations for
        similarity_matrix: Cosine similarity matrix (n_movies x n_movies)
        df: Dataframe containing movie metadata
        top_n: Number of recommendations to return
        
    Returns:
        List of top N recommended movie titles (excluding the input movie)
        
    Raises:
        ValueError: If movie_title is not found in the dataset
    """
    # Search for the movie (case-insensitive)
    matching_movies = df[df['title'].str.lower() == movie_title.lower()]
    
    if matching_movies.empty:
        raise ValueError(f"Movie '{movie_title}' not found in the dataset.")
    
    # Get the index of the movie
    movie_idx = matching_movies.index[0]
    
    # Get similarity scores for this movie
    similarity_scores = similarity_matrix[movie_idx]
    
    # Get indices of top similar movies (excluding the movie itself)
    similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    
    # Get the titles of recommended movies
    recommended_titles = df.iloc[similar_indices]['title'].tolist()
    
    return recommended_titles


def recommend_with_scores(movie_title, similarity_matrix, df, top_n=10):
    """
    Recommend top N movies with similarity scores.
    
    Args:
        movie_title: Title of the movie to get recommendations for
        similarity_matrix: Cosine similarity matrix (n_movies x n_movies)
        df: Dataframe containing movie metadata
        top_n: Number of recommendations to return
        
    Returns:
        List of tuples (movie_title, similarity_score) sorted by score descending
        
    Raises:
        ValueError: If movie_title is not found in the dataset
    """
    # Search for the movie (case-insensitive)
    matching_movies = df[df['title'].str.lower() == movie_title.lower()]
    
    if matching_movies.empty:
        raise ValueError(f"Movie '{movie_title}' not found in the dataset.")
    
    # Get the index of the movie
    movie_idx = matching_movies.index[0]
    
    # Get similarity scores for this movie
    similarity_scores = similarity_matrix[movie_idx]
    
    # Get indices of top similar movies (excluding the movie itself)
    similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    
    # Create list of tuples (title, score)
    recommendations = [
        (df.iloc[idx]['title'], similarity_scores[idx])
        for idx in similar_indices
    ]
    
    return recommendations


def get_movie_by_index(df, index):
    """
    Get movie details by index.
    
    Args:
        df: Movies dataframe
        index: Index of the movie
        
    Returns:
        Dictionary with movie details
    """
    if index < 0 or index >= len(df):
        return None
    
    movie = df.iloc[index]
    return {
        'title': movie['title'],
        'overview': movie.get('overview', ''),
        'genres': movie.get('genre_list', []),
        'vote_average': movie.get('vote_average', 0),
        'release_date': movie.get('release_date', '')
    }
