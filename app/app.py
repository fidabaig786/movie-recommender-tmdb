"""Streamlit web application for movie recommendations."""

import streamlit as st
import pandas as pd
import pickle
import os
import sys

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add parent directory to path for imports
sys.path.insert(0, PROJECT_ROOT)

# Change to project root for data file access
os.chdir(PROJECT_ROOT)

from src.recommend import recommend, recommend_with_scores
from src.model import build_model, save_model, load_model
from src.preprocess import preprocess_data
from src.features import engineer_features


# Set page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 30px;
    }
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .similarity-score {
        color: #2E8B57;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_or_build_model():
    """Load or build the recommendation model."""
    model_dir = 'models/'
    
    # Check if model files exist
    if (os.path.exists(f'{model_dir}vectorizer.pkl') and 
        os.path.exists(f'{model_dir}similarity_matrix.pkl') and
        os.path.exists(f'{model_dir}movies_df.pkl')):
        
        st.info("📦 Loading pre-built model...")
        vectorizer, similarity_matrix, df = load_model(model_dir)
        return vectorizer, similarity_matrix, df
    
    else:
        st.info("🔨 Building model for the first time (this may take a moment)...")
        
        # Preprocess data
        st.info("📊 Preprocessing data...")
        movies_path = 'data/tmdb_5000_movies.csv'
        credits_path = 'data/tmdb_5000_credits.csv'
        
        df = preprocess_data(movies_path, credits_path)
        
        # Engineer features
        st.info("⚙️ Engineering features...")
        df = engineer_features(df)
        
        # Build model
        st.info("🤖 Building similarity matrix...")
        vectorizer, similarity_matrix = build_model(df)
        
        # Save model
        st.info("💾 Saving model...")
        save_model(vectorizer, similarity_matrix, df, model_dir)
        
        st.success("✅ Model built and saved!")
        return vectorizer, similarity_matrix, df


def main():
    """Main Streamlit application."""
    
    # Title
    st.markdown("<h1 class='main-title'>🎬 Movie Recommendation System</h1>", 
                unsafe_allow_html=True)
    st.markdown("Find similar movies based on genres, keywords, cast, and plot!")
    st.divider()
    
    # Load or build model
    try:
        vectorizer, similarity_matrix, df = load_or_build_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    num_recommendations = st.sidebar.slider(
        "Number of recommendations",
        min_value=3,
        max_value=20,
        value=10,
        step=1
    )
    
    show_scores = st.sidebar.checkbox("Show similarity scores", value=False)
    show_details = st.sidebar.checkbox("Show movie details", value=True)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Movie selection
        movie_list = sorted(df['title'].unique().tolist())
        selected_movie = st.selectbox(
            "🎥 Select a movie:",
            movie_list,
            help="Choose the movie you want recommendations for"
        )
    
    with col2:
        # Recommendation button
        get_recommendations = st.button(
            "🔍 Recommend",
            use_container_width=True,
            type="primary"
        )
    
    st.divider()
    
    # Generate recommendations
    if get_recommendations or selected_movie:
        try:
            if show_scores:
                recommendations_with_scores = recommend_with_scores(
                    selected_movie,
                    similarity_matrix,
                    df,
                    top_n=num_recommendations
                )
                
                # Display recommendations with scores
                st.subheader(f"📋 Top {num_recommendations} Recommendations for '{selected_movie}'")
                
                for i, (title, score) in enumerate(recommendations_with_scores, 1):
                    st.markdown(f"""
                    <div class="recommendation-card">
                    <b>{i}. {title}</b><br>
                    <span class="similarity-score">Similarity: {score:.2%}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if show_details:
                        movie_data = df[df['title'] == title].iloc[0]
                        with st.expander(f"ℹ️ Details"):
                            st.write(f"**Overview:** {movie_data['overview']}")
                            st.write(f"**Rating:** ⭐ {movie_data.get('vote_average', 'N/A')}/10")
                            st.write(f"**Release Date:** {movie_data.get('release_date', 'N/A')}")
                            genres = movie_data.get('genre_list', [])
                            if genres:
                                st.write(f"**Genres:** {', '.join(genres)}")
            
            else:
                recommendations = recommend(
                    selected_movie,
                    similarity_matrix,
                    df,
                    top_n=num_recommendations
                )
                
                # Display recommendations
                st.subheader(f"📋 Top {num_recommendations} Recommendations for '{selected_movie}'")
                
                for i, title in enumerate(recommendations, 1):
                    st.markdown(f"""
                    <div class="recommendation-card">
                    <b>{i}. {title}</b>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if show_details:
                        movie_data = df[df['title'] == title].iloc[0]
                        with st.expander(f"ℹ️ Details"):
                            st.write(f"**Overview:** {movie_data['overview']}")
                            st.write(f"**Rating:** ⭐ {movie_data.get('vote_average', 'N/A')}/10")
                            st.write(f"**Release Date:** {movie_data.get('release_date', 'N/A')}")
                            genres = movie_data.get('genre_list', [])
                            if genres:
                                st.write(f"**Genres:** {', '.join(genres)}")
        
        except ValueError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    
    # Footer
    st.divider()
    st.markdown("""
    ---
    **About:** This recommendation system uses TF-IDF vectorization and cosine similarity
    to find movies based on genres, keywords, cast, director, and plot overview.
    """)


if __name__ == "__main__":
    main()
