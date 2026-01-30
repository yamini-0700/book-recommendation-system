import pandas as pd
import numpy as np
from pathlib import Path

class HybridRecommender:
    def __init__(self, content_recommender, collaborative_recommender):
        self.content_rec = content_recommender
        self.collab_rec = collaborative_recommender
    
    def recommend_books(self, user_context, books_df, n=10):
        """
        Hybrid recommendation combining content-based and collaborative filtering
        
        Parameters:
        -----------
        user_context: dict with keys:
            - 'liked_books': list of book IDs user liked
            - 'user_id': user ID (if available)
            - 'search_history': list of searched/bookmarked books
        
        Returns:
        --------
        DataFrame with recommended books
        """
        liked_books = user_context.get('liked_books', [])
        user_id = user_context.get('user_id')
        
        # Get recommendations from both models
        content_recs = pd.DataFrame()
        collab_recs = pd.DataFrame()
        
        # Content-based recommendations
        if liked_books:
            content_recs = self.content_rec.recommend_for_user(liked_books, n=n*2)
        
        # Collaborative filtering recommendations
        if user_id and self.collab_rec.user_ids is not None:
            collab_recs = self.collab_rec.recommend_for_user(user_id, books_df, n=n*2)
        elif liked_books:
            # Use item-based collaborative for new users
            collab_recs = self.collab_rec.recommend_for_new_user(liked_books, books_df, n=n*2)
        
        # Combine recommendations
        all_recs = self._combine_recommendations(content_recs, collab_recs, n)
        
        # Fallback: If no recommendations, return popular books
        if all_recs.empty:
            all_recs = books_df.sort_values('popularity_score', ascending=False).head(n)
        
        return all_recs
    
    def _combine_recommendations(self, content_recs, collab_recs, n):
        """Combine recommendations with weighted scoring"""
        combined = pd.DataFrame()
        
        if not content_recs.empty:
            content_recs['source'] = 'content'
            if 'similarity_score' in content_recs.columns:
                content_recs['score'] = content_recs['similarity_score']
            else:
                content_recs['score'] = 0.5
        
        if not collab_recs.empty:
            collab_recs['source'] = 'collaborative'
            if 'predicted_rating' in collab_recs.columns:
                collab_recs['score'] = collab_recs['predicted_rating']
            else:
                collab_recs['score'] = 0.5
        
        # Combine both sets
        all_recs = pd.concat([content_recs, collab_recs], ignore_index=True)
        
        if all_recs.empty:
            return all_recs
        
        # Group by book and create hybrid score
        # Weight: 60% content-based, 40% collaborative
        def calculate_hybrid_score(group):
            if len(group) == 2:
                # Has both content and collaborative scores
                content_score = group[group['source'] == 'content']['score'].values[0]
                collab_score = group[group['source'] == 'collaborative']['score'].values[0]
                return 0.6 * content_score + 0.4 * collab_score
            elif 'content' in group['source'].values:
                return group['score'].values[0] * 0.8  # Penalize for missing collaborative
            else:
                return group['score'].values[0] * 0.7  # Penalize for missing content
        
        hybrid_scores = all_recs.groupby('bookId').apply(calculate_hybrid_score)
        hybrid_scores.name = 'hybrid_score'
        
        # Get unique books with best score from each group
        unique_recs = all_recs.drop_duplicates(subset=['bookId'], keep='first')
        unique_recs = unique_recs.merge(hybrid_scores, on='bookId')
        
        # Sort by hybrid score
        unique_recs = unique_recs.sort_values('hybrid_score', ascending=False)
        
        return unique_recs.head(n)