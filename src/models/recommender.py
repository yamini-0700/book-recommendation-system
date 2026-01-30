import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the content-based recommender
try:
    from src.models.content_based import ContentBasedRecommender
    CONTENT_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Could not import ContentBasedRecommender: {e}")
    CONTENT_MODEL_AVAILABLE = False

# Try to import other models (optional)
try:
    from src.models.collaborative import CollaborativeRecommender
    from src.models.hybrid import HybridRecommender
    COLLAB_MODEL_AVAILABLE = True
except ImportError:
    COLLAB_MODEL_AVAILABLE = False

class SimpleRecommendationEngine:
    """Simple fallback recommendation engine"""
    def __init__(self, books_data):
        self.books = books_data
    
    def get_trending_books(self, filters=None, n=10):
        """Get trending books based on popularity"""
        books = self.books.copy()
        
        if filters:
            if filters.get('language') and filters['language'] != 'All':
                books = books[books['language'] == filters['language']]
            if filters.get('genre') and filters['genre'] != 'All':
                books = books[books['genres'].str.contains(filters['genre'], case=False, na=False)]
            if filters.get('min_rating'):
                books = books[books['avg_rating'] >= filters['min_rating']]
        
        return books.sort_values('popularity_score', ascending=False).head(n)
    
    def get_similar_books(self, book_id, n=5):
        """Get books similar to a given book (simple version)"""
        if book_id not in self.books['bookId'].values:
            return pd.DataFrame()
        
        book = self.books[self.books['bookId'] == book_id].iloc[0]
        
        # Get other books by same author
        same_author = self.books[
            (self.books['author'] == book['author']) & 
            (self.books['bookId'] != book_id)
        ].head(n)
        
        if not same_author.empty:
            same_author['similarity_score'] = 0.8
            return same_author
        
        # If no same author, get books with similar title/keywords
        title_words = set(str(book['title']).lower().split())
        def title_similarity(row_title):
            row_words = set(str(row_title).lower().split())
            common = len(title_words.intersection(row_words))
            return common / max(len(title_words), 1)
        
        self.books['title_similarity'] = self.books['title'].apply(title_similarity)
        similar_titles = self.books[
            (self.books['bookId'] != book_id) & 
            (self.books['title_similarity'] > 0)
        ].sort_values('title_similarity', ascending=False).head(n)
        
        if not similar_titles.empty:
            similar_titles = similar_titles.drop(columns=['title_similarity'])
            similar_titles['similarity_score'] = 0.5
            return similar_titles
        
        # Final fallback: random high-rated books
        random_books = self.books[
            (self.books['bookId'] != book_id) & 
            (self.books['avg_rating'] >= 4.0)
        ].sample(n=min(n, len(self.books)-1), random_state=42)
        
        if not random_books.empty:
            random_books['similarity_score'] = 0.3
            return random_books
        
        # Ultimate fallback: any random books
        random_fallback = self.books[self.books['bookId'] != book_id].sample(
            n=min(n, len(self.books)-1), random_state=42
        )
        random_fallback['similarity_score'] = 0.1
        return random_fallback
    
    def get_personalized_recommendations(self, liked_books, viewed_books=None, n=10):
        """Get personalized recommendations"""
        if not liked_books and (viewed_books is None or not viewed_books):
            return self.get_trending_books(n=n)
        
        # Combine liked and viewed books
        user_books = list(set(liked_books + (viewed_books or [])))
        
        # Get recommendations based on user books
        all_recs = []
        for book_id in user_books[:3]:  # Limit to 3 books
            similar = self.get_similar_books(book_id, n=10)
            if not similar.empty:
                all_recs.append(similar)
        
        if all_recs:
            combined = pd.concat(all_recs).drop_duplicates(subset=['bookId'])
            combined = combined[~combined['bookId'].isin(user_books)]
            
            if 'similarity_score' in combined.columns:
                combined = combined.sort_values('similarity_score', ascending=False)
            
            return combined.head(n)
        
        return self.get_trending_books(n=n)

class RecommendationEngine:
    def __init__(self, books_data, ratings_data=None):
        self.books = books_data
        self.ratings = ratings_data
        
        print("Initializing Recommendation Engine...")
        
        # Always create simple engine as fallback
        self.simple_engine = SimpleRecommendationEngine(books_data)
        
        # Try to use advanced models if available
        if CONTENT_MODEL_AVAILABLE:
            try:
                print("Initializing content-based recommender...")
                self.content_recommender = ContentBasedRecommender(books_data)
                
                # Try to load existing models
                content_loaded = self.content_recommender.load_models(self.books)
                
                if not content_loaded:
                    print("Training new content-based models...")
                    self.content_recommender.prepare_models()
                
                print("✓ Content-based models initialized")
                self.use_advanced = True
                
            except Exception as e:
                print(f"Error with content-based models: {e}")
                print("Falling back to simple engine...")
                self.use_advanced = False
        else:
            print("Content-based model not available, using simple engine")
            self.use_advanced = False
    
    def get_trending_books(self, filters=None, n=10):
        """Get trending books based on popularity"""
        return self.simple_engine.get_trending_books(filters, n)
    
    def get_similar_books(self, book_id, n=5):
        """Get books similar to a given book"""
        if self.use_advanced:
            try:
                result = self.content_recommender.get_similar_books(book_id, n)
                if not result.empty:
                    return result
            except Exception as e:
                print(f"Advanced similar books failed: {e}")
        
        # Fallback to simple engine
        return self.simple_engine.get_similar_books(book_id, n)
    
    def get_personalized_recommendations(self, liked_books, viewed_books=None, n=10):
        """Get personalized recommendations based on user's liked books"""
        if self.use_advanced:
            try:
                result = self.content_recommender.recommend_for_user(liked_books, n)
                if not result.empty:
                    return result
            except Exception as e:
                print(f"Advanced personalized recommendations failed: {e}")
        
        # Fallback to simple engine
        return self.simple_engine.get_personalized_recommendations(liked_books, viewed_books, n)
    
    def get_recommended_books(self, filters=None, n=10):
        """Get recommended books (mix of high-rated and trending)"""
        if filters is None:
            filters = {'language': 'All', 'genre': 'All', 'min_rating': 3.5}
        
        # Get filtered books
        filtered = self.books.copy()
        if filters['language'] != 'All':
            filtered = filtered[filtered['language'] == filters['language']]
        if filters['genre'] != 'All':
            filtered = filtered[filtered['genres'].str.contains(filters['genre'], case=False, na=False)]
        filtered = filtered[filtered['avg_rating'] >= filters['min_rating']]
        
        if filtered.empty:
            return filtered
        
        # Diversify recommendations
        results = []
        
        # 1. High-rated books (4.0+)
        high_rated = filtered[filtered['avg_rating'] >= 4.0]
        if not high_rated.empty:
            results.append(high_rated.sample(min(3, len(high_rated)), random_state=42))
        
        # 2. Popular books
        popular = filtered.sort_values('popularity_score', ascending=False).head(10)
        if not popular.empty:
            results.append(popular.sample(min(4, len(popular)), random_state=42))
        
        # 3. Diverse genres
        unique_genres = filtered['genres'].dropna().unique()
        for genre in unique_genres[:3]:  # Try up to 3 different genres
            genre_books = filtered[filtered['genres'].str.contains(genre, case=False, na=False)]
            if not genre_books.empty:
                results.append(genre_books.sample(1, random_state=42))
        
        # Combine results
        if results:
            result = pd.concat(results).drop_duplicates()
            if len(result) < n:
                # Add more random books
                remaining = filtered[~filtered['bookId'].isin(result['bookId'])]
                if not remaining.empty:
                    more = remaining.sample(min(n - len(result), len(remaining)), random_state=42)
                    result = pd.concat([result, more])
            return result.head(n)
        
        return filtered.head(n)