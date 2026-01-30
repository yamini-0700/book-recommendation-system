import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix, save_npz, load_npz
from pathlib import Path
import joblib

class ContentBasedRecommender:
    def __init__(self, books_data=None, model_dir="models"):
        self.base_dir = Path(__file__).resolve().parents[2]
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.vectorizer = None
        self.similarity_matrix = None
        self.scaler = MinMaxScaler()
        self.books = books_data
        
        if books_data is not None:
            self.prepare_models()
    
    def prepare_models(self):
        """Create content features and similarity matrix"""
        print("Preparing content-based models...")
        
        # Take a sample for memory efficiency (first 5000 books)
        sample_size = min(5000, len(self.books))  # Change 5000 to 20000
        if len(self.books) > 5000:
            print(f"  Using sample of {sample_size} books for memory efficiency")
            books_sample = self.books.sample(n=sample_size, random_state=42).copy()
        else:
            books_sample = self.books.copy()
        
        # Create feature text
        print("  Creating feature text...")
        books_sample['feature_text'] = (
            books_sample['genres'].fillna('') + ' ' +
            books_sample['author'].fillna('') + ' ' +
            books_sample.get('description', '').fillna('')
        )
        
        # Create TF-IDF features with limited features
        print("  Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(books_sample['feature_text'])
        print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Calculate cosine similarity on sample
        print("  Calculating similarity matrix...")
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Map indices back to original books
        self.sample_indices = books_sample.index.tolist()
        self.sample_book_ids = books_sample['bookId'].tolist()
        
        # Save models
        self.save_models(books_sample)
        
        print(f"  ✓ Content-based models prepared for {sample_size} books")
    
    def get_similar_books(self, book_id, n=10):
        """Get similar books based on content"""
        if book_id not in self.books['bookId'].values:
            return pd.DataFrame()
        
        try:
            # Check if book is in our sample
            if book_id in self.sample_book_ids:
                idx = self.sample_book_ids.index(book_id)
                similarity_scores = list(enumerate(self.similarity_matrix[idx]))
                similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:n+1]
                
                book_indices = [self.sample_indices[i[0]] for i in similarity_scores]
                
                # Return with similarity scores
                result = self.books.loc[book_indices].copy()
                result['similarity_score'] = [i[1] for i in similarity_scores]
                
                return result
            else:
                # Fallback for books not in sample
                return self._fallback_similar_books(book_id, n)
                
        except Exception as e:
            print(f"Error getting similar books: {e}")
            return self._fallback_similar_books(book_id, n)
    
    def _fallback_similar_books(self, book_id, n=10):
        """Fallback method using simple genre/author matching"""
        if book_id not in self.books['bookId'].values:
            return pd.DataFrame()
        
        book = self.books[self.books['bookId'] == book_id].iloc[0]
        
        # Get other books by same author
        same_author = self.books[
            (self.books['author'] == book['author']) & 
            (self.books['bookId'] != book_id)
        ].head(n)
        
        if not same_author.empty:
            same_author['similarity_score'] = 0.7  # High score for same author
            return same_author
        
        # If no same author, get books with similar genres
        if 'genres' in book and pd.notna(book['genres']):
            book_genres = str(book['genres']).lower()
            similar_genres = self.books[
                (self.books['bookId'] != book_id) & 
                self.books['genres'].str.contains(book_genres.split(',')[0], case=False, na=False)
            ].head(n)
            
            if not similar_genres.empty:
                similar_genres['similarity_score'] = 0.5
                return similar_genres
        
        # Final fallback: random books
        random_books = self.books[self.books['bookId'] != book_id].sample(n=min(n, len(self.books)-1), random_state=42)
        random_books['similarity_score'] = 0.3
        return random_books
    
    def recommend_for_user(self, liked_books, n=10):
        """Recommend books for user based on liked books"""
        if not liked_books:
            return pd.DataFrame()
        
        # Limit to first 3 books
        liked_books = liked_books[:3]
        
        all_recommendations = []
        for book_id in liked_books:
            similar = self.get_similar_books(book_id, n=15)
            if not similar.empty:
                all_recommendations.append(similar)
        
        if not all_recommendations:
            return pd.DataFrame()
        
        # Combine recommendations
        combined = pd.concat(all_recommendations)
        
        if 'similarity_score' in combined.columns:
            # Group by book and average similarity scores
            recommendations = combined.groupby('bookId').agg({
                'similarity_score': 'mean',
                'title': 'first',
                'author': 'first',
                'genres': 'first',
                'avg_rating': 'first',
                'popularity_score': 'first'
            }).reset_index()
        else:
            recommendations = combined.drop_duplicates(subset=['bookId'])
            recommendations['similarity_score'] = 0.5
        
        # Remove books user already liked
        recommendations = recommendations[~recommendations['bookId'].isin(liked_books)]
        
        return recommendations.sort_values('similarity_score', ascending=False).head(n)
    
    def save_models(self, books_sample):
        """Save trained models"""
        try:
            joblib.dump(self.vectorizer, self.model_dir / "content_vectorizer.pkl")
            np.save(self.model_dir / "content_similarity.npy", self.similarity_matrix)
            
            # Save sample information
            sample_info = {
                'sample_indices': self.sample_indices,
                'sample_book_ids': self.sample_book_ids
            }
            joblib.dump(sample_info, self.model_dir / "sample_info.pkl")
            
            print(f"  ✓ Models saved to {self.model_dir}")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self, books_data):
        """Load trained models"""
        self.books = books_data
        
        try:
            print("Loading content-based models...")
            self.vectorizer = joblib.load(self.model_dir / "content_vectorizer.pkl")
            self.similarity_matrix = np.load(self.model_dir / "content_similarity.npy")
            
            sample_info = joblib.load(self.model_dir / "sample_info.pkl")
            self.sample_indices = sample_info['sample_indices']
            self.sample_book_ids = sample_info['sample_book_ids']
            
            print(f"  ✓ Content-based models loaded for {len(self.sample_book_ids)} books")
            return True
        except Exception as e:
            print(f"Error loading content models: {e}")
            return False