import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.recommender import RecommendationEngine

def prepare_data():
    """Load and prepare data for training"""
    print("Loading data...")
    
    # Load books features
    books_path = Path("data/processed/books_features.csv")
    books_df = pd.read_csv(books_path)
    
    print(f"Original shape: {books_df.shape}")
    
    # Remove the Unnamed: 0 column if it exists
    if 'Unnamed: 0' in books_df.columns:
        books_df = books_df.drop(columns=['Unnamed: 0'])
    
    # Fill missing values
    books_df['description'] = books_df['description'].fillna('No description available')
    books_df['genres'] = books_df['genres'].fillna('Fiction')
    books_df['language'] = books_df['language'].fillna('English')
    books_df['author'] = books_df['author'].fillna('Unknown')
    
    # Convert numeric columns
    numeric_cols = ['num_pages', 'num_ratings', 'num_reviews', 'avg_rating']
    for col in numeric_cols:
        if col in books_df.columns:
            books_df[col] = pd.to_numeric(books_df[col], errors='coerce').fillna(0)
    
    # Ensure bookId exists
    if 'bookId' not in books_df.columns:
        books_df['bookId'] = range(1, len(books_df) + 1)
    
    # Create popularity score if not exists
    if 'popularity_score' not in books_df.columns:
        if 'num_ratings' in books_df.columns:
            books_df['popularity_score'] = (
                books_df['num_ratings'] * 0.7 + 
                books_df['avg_rating'] * 30 +
                np.log1p(books_df['num_reviews']) * 10
            )
        else:
            books_df['popularity_score'] = books_df['avg_rating'] * 10
    
    print(f"After cleaning: {books_df.shape}")
    print(f"Total books: {len(books_df)}")
    
    return books_df

def test_recommendations(recommender, books_df):
    """Test the recommendation engine"""
    print("\n" + "="*60)
    print("Testing Recommendations")
    print("="*60)
    
    # Test 1: Get similar books
    print("\n1. Testing Similar Books:")
    if len(books_df) > 10:
        # Find a book with a clear title for testing
        test_book = books_df.iloc[0]
        test_book_id = test_book['bookId']
        test_title = test_book['title']
        
        similar_books = recommender.get_similar_books(test_book_id, n=5)
        
        print(f"\nBooks similar to '{test_title}':")
        if not similar_books.empty:
            for idx, (_, book) in enumerate(similar_books.iterrows(), 1):
                similarity = book.get('similarity_score', 0)
                print(f"  {idx}. {book['title']} by {book['author']} (Score: {similarity:.3f})")
        else:
            print("  No similar books found")
    
    # Test 2: Get personalized recommendations
    print("\n2. Testing Personalized Recommendations:")
    if len(books_df) >= 3:
        # Use first 3 books as "liked" books
        test_liked_books = books_df.head(3)['bookId'].tolist()
        liked_titles = books_df.head(3)['title'].tolist()
        
        print(f"\nUser liked {len(test_liked_books)} books:")
        for title in liked_titles:
            print(f"  - {title}")
        
        personalized = recommender.get_personalized_recommendations(
            liked_books=test_liked_books,
            viewed_books=test_liked_books,
            n=5
        )
        
        print("\nPersonalized recommendations:")
        if not personalized.empty:
            for idx, (_, book) in enumerate(personalized.iterrows(), 1):
                score = book.get('similarity_score', 0)
                print(f"  {idx}. {book['title']} by {book['author']} (Score: {score:.3f})")
        else:
            print("  No personalized recommendations found")
    
    # Test 3: Get trending books
    print("\n3. Testing Trending Books:")
    trending = recommender.get_trending_books(n=5)
    if not trending.empty:
        print("\nTrending books:")
        for idx, (_, book) in enumerate(trending.iterrows(), 1):
            print(f"  {idx}. {book['title']} (Rating: {book['avg_rating']:.1f}, Popularity: {book['popularity_score']:.1f})")
    
    # Test 4: Test with filters
    print("\n4. Testing Filtered Recommendations:")
    filters = {
        'language': 'English',
        'genre': 'All',
        'min_rating': 4.0
    }
    
    filtered_recs = recommender.get_recommended_books(filters, n=3)
    if not filtered_recs.empty:
        print(f"\nRecommended books with filters (language={filters['language']}, min_rating={filters['min_rating']}):")
        for idx, (_, book) in enumerate(filtered_recs.iterrows(), 1):
            print(f"  {idx}. {book['title']} (Rating: {book['avg_rating']:.1f})")
    
    print("\n" + "="*60)
    print("✓ Testing Complete!")
    print("="*60)

def main():
    print("\n" + "="*60)
    print("Book Recommendation System - Model Training")
    print("="*60)
    
    # Prepare data
    books_df = prepare_data()
    
    # Check if models already exist
    model_dir = Path("models")
    if not model_dir.exists():
        model_dir.mkdir()
        print(f"\nCreated models directory: {model_dir}")
    
    # Try to load ratings if they exist (optional)
    ratings_df = None
    ratings_path = Path("data/processed/ratings.csv")
    if os.path.exists(ratings_path):
        try:
            ratings_df = pd.read_csv(ratings_path)
            print(f"\nLoaded {len(ratings_df)} ratings")
            
            # Basic cleaning of ratings
            required_cols = ['user_id', 'book_id', 'rating']
            if all(col in ratings_df.columns for col in required_cols):
                ratings_df = ratings_df.dropna(subset=required_cols)
                ratings_df = ratings_df[ratings_df['rating'] > 0]
                print(f"After cleaning: {len(ratings_df)} ratings")
            else:
                print("Ratings file doesn't have required columns")
                ratings_df = None
        except Exception as e:
            print(f"Could not load ratings: {e}")
            ratings_df = None
    
    # Create recommendation engine
    print("\nCreating recommendation engine...")
    recommender = RecommendationEngine(books_df, ratings_df)
    
    # Test recommendations
    test_recommendations(recommender, books_df)
    
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    
    # Display statistics
    print(f"\nStatistics:")
    print(f"- Total books: {len(books_df):,}")
    if ratings_df is not None:
        print(f"- Total ratings: {len(ratings_df):,}")
        print(f"- Average ratings per book: {len(ratings_df)/len(books_df):.1f}")
    
    # Save cleaned data for app (optional)
    try:
        books_df.to_csv("data/processed/books_clean_for_app.csv", index=False)
        print(f"\n✓ Cleaned data saved to: data/processed/books_clean_for_app.csv")
    except Exception as e:
        print(f"\nNote: Could not save cleaned data: {e}")
    
    print(f"\nModels directory: {model_dir.resolve()}")

if __name__ == "__main__":
    main()