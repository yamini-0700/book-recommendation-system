import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import pickle
from pathlib import Path
import joblib

class CollaborativeRecommender:
    def __init__(self, ratings_data=None, model_dir="models"):
        self.base_dir = Path(__file__).resolve().parents[2]
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.user_ids = None
        self.item_ids = None
        
        if ratings_data is not None:
            self.prepare_models(ratings_data)
    
    def prepare_models(self, ratings_df):
        """Create collaborative filtering model using SVD"""
        # Create user-item matrix
        self.user_ids = ratings_df['user_id'].unique()
        self.item_ids = ratings_df['book_id'].unique()
        
        user_to_idx = {user: idx for idx, user in enumerate(self.user_ids)}
        item_to_idx = {item: idx for idx, item in enumerate(self.item_ids)}
        
        rows = ratings_df['user_id'].map(user_to_idx)
        cols = ratings_df['book_id'].map(item_to_idx)
        values = ratings_df['rating'].values
        
        self.user_item_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        )
        
        # Apply SVD
        k = min(50, min(self.user_item_matrix.shape) - 1)
        U, sigma, Vt = svds(self.user_item_matrix, k=k)
        
        sigma = np.diag(sigma)
        self.user_factors = U
        self.item_factors = Vt.T
        
        # Save mappings and models
        self.save_models()
    
    def predict_ratings(self, user_idx):
        """Predict ratings for a user"""
        user_pred = self.user_factors[user_idx, :].dot(
            sigma.dot(self.item_factors.T) # type: ignore
        )
        return user_pred
    
    def recommend_for_user(self, user_id, books_df, n=10):
        """Recommend books for a specific user"""
        if user_id not in self.user_ids:
            # Cold start: return popular books
            return books_df.sort_values('popularity_score', ascending=False).head(n)
        
        user_idx = np.where(self.user_ids == user_id)[0][0]
        pred_ratings = self.predict_ratings(user_idx)
        
        # Get top N predictions
        top_indices = np.argsort(pred_ratings)[::-1][:n]
        
        # Map back to book IDs
        recommended_book_ids = self.item_ids[top_indices]
        
        # Get book details
        recommendations = books_df[books_df['bookId'].isin(recommended_book_ids)].copy()
        
        # Add predicted rating
        rating_dict = {self.item_ids[i]: pred_ratings[i] for i in top_indices}
        recommendations['predicted_rating'] = recommendations['bookId'].map(rating_dict)
        
        return recommendations.sort_values('predicted_rating', ascending=False).head(n)
    
    def recommend_for_new_user(self, liked_books, books_df, n=10):
        """Recommend for users without rating history (using item-item similarity)"""
        if not liked_books:
            return books_df.sort_values('popularity_score', ascending=False).head(n)
        
        # Simple weighted average of item factors for liked books
        liked_indices = [np.where(self.item_ids == book_id)[0][0] 
                        for book_id in liked_books 
                        if book_id in self.item_ids]
        
        if not liked_indices:
            return books_df.sort_values('popularity_score', ascending=False).head(n)
        
        # Average the item factors of liked books
        avg_item_factor = np.mean(self.item_factors[liked_indices, :], axis=0)
        
        # Compute similarity to all items
        similarities = self.item_factors.dot(avg_item_factor)
        
        # Get top N (excluding liked books)
        top_indices = np.argsort(similarities)[::-1]
        recommended_ids = []
        
        for idx in top_indices:
            book_id = self.item_ids[idx]
            if book_id not in liked_books:
                recommended_ids.append(book_id)
            if len(recommended_ids) >= n:
                break
        
        # Get book details
        recommendations = books_df[books_df['bookId'].isin(recommended_ids)].copy()
        
        return recommendations.head(n)
    
    def save_models(self):
        """Save trained models"""
        np.save(self.model_dir / "user_factors.npy", self.user_factors)
        np.save(self.model_dir / "item_factors.npy", self.item_factors)
        np.save(self.model_dir / "user_ids.npy", self.user_ids)
        np.save(self.model_dir / "item_ids.npy", self.item_ids)
    
    def load_models(self):
        """Load trained models"""
        try:
            self.user_factors = np.load(self.model_dir / "user_factors.npy")
            self.item_factors = np.load(self.model_dir / "item_factors.npy")
            self.user_ids = np.load(self.model_dir / "user_ids.npy")
            self.item_ids = np.load(self.model_dir / "item_ids.npy")
            return True
        except Exception as e:
            print(f"Error loading collaborative models: {e}")
            return False