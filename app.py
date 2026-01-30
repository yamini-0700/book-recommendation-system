import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
import sys
import time
from collections import Counter
from functools import lru_cache
import hashlib
import base64

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import recommendation engine with error handling
try:
    from src.models.recommender import RecommendationEngine
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False

# ============================================================================
# 1. APP CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="BookHub",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 2. CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #ffffff;
    }
    .nav-container {
        background: rgba(15, 23, 42, 0.95);
        border-bottom: 1px solid #334155;
        padding: 15px 30px;
        margin-bottom: 20px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 15px;
        font-size: 13px;
        font-weight: 500;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
    }
    .book-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 15px;
        transition: all 0.3s;
        height: 100%;
        margin-bottom: 20px;
    }
    .book-card:hover {
        border-color: #60a5fa;
        box-shadow: 0 10px 30px rgba(96, 165, 250, 0.15);
        transform: translateY(-2px);
    }
    .book-cover {
        width: 100%;
        height: 220px;
        object-fit: cover;
        margin-bottom: 15px;
        border-radius: 8px;
        border: 2px solid #334155;
    }
    .book-title {
        font-size: 14px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 5px;
        line-height: 1.3;
        height: 36px;
        overflow: hidden;
    }
    .book-author {
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 5px;
    }
    .book-rating {
        color: #fbbf24;
        font-size: 12px;
        margin-bottom: 8px;
    }
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #60a5fa;
        margin: 25px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #334155;
    }
    .filter-bar {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
    }
    .genre-tag {
        display: inline-block;
        background: #3b82f6;
        color: white;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 12px;
        margin: 3px;
        font-weight: 500;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    .quick-back-btn {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 3. SESSION STATE MANAGEMENT
# ============================================================================
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Home"
if 'user_liked' not in st.session_state:
    st.session_state.user_liked = []
if 'user_history' not in st.session_state:
    st.session_state.user_history = []
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = None
if 'filters' not in st.session_state:
    st.session_state.filters = {
        'language': 'All',
        'genre': 'All',
        'min_rating': 3.5
    }
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False
if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False
if 'last_action' not in st.session_state:
    st.session_state.last_action = {"type": None, "book_id": None}
if 'books_df' not in st.session_state:
    st.session_state.books_df = None
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'cover_cache' not in st.session_state:
    st.session_state.cover_cache = {}
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'previous_tab' not in st.session_state:
    st.session_state.previous_tab = "Home"

# ============================================================================
# 4. REAL BOOK COVERS - OPTIMIZED
# ============================================================================
def get_no_cover_image():
    """Get no_cover.png from assets folder"""
    no_cover_path = Path("assets/no_cover.png")
    
    if no_cover_path.exists():
        try:
            with open(no_cover_path, "rb") as f:
                image_data = f.read()
                encoded = base64.b64encode(image_data).decode()
                return f"data:image/png;base64,{encoded}"
        except Exception as e:
            print(f"Error loading no_cover.png: {e}")
    
    # Fallback
    return "https://placehold.co/180x270/3b82f6/FFFFFF?text=No+Cover"

NO_COVER_IMAGE = get_no_cover_image()

@lru_cache(maxsize=10000)
def get_real_book_cover(book_id, title, author, isbn=None):
    """Get real book cover with caching - FAST implementation"""
    # Generate unique cache key
    cache_key = f"{book_id}_{title}_{author}"
    
    # Check cache first
    if cache_key in st.session_state.cover_cache:
        return st.session_state.cover_cache[cache_key]
    
    # Try OpenLibrary with ISBN first (fastest)
    if isbn and pd.notna(isbn):
        isbn_clean = str(isbn).replace('-', '').strip()
        if len(isbn_clean) in [10, 13]:
            ol_url = f"https://covers.openlibrary.org/b/isbn/{isbn_clean}-M.jpg"
            # Don't check if exists, just return URL - let browser handle 404
            st.session_state.cover_cache[cache_key] = ol_url
            return ol_url
    
    # Try Google Books with title and author
    if title and author:
        # Clean title and author for URL
        title_clean = re.sub(r'[^a-zA-Z0-9 ]', '', title)[:50].replace(' ', '+')
        author_clean = re.sub(r'[^a-zA-Z0-9 ]', '', author)[:30].replace(' ', '+')
        
        # Generate Google Books URL (they often have covers)
        google_url = f"https://books.google.com/books/content?id={hashlib.md5(f'{title}{author}'.encode()).hexdigest()[:12]}&printsec=frontcover&img=1&zoom=1&source=gbs_api"
        st.session_state.cover_cache[cache_key] = google_url
        return google_url
    
    # Try OpenLibrary with title only
    if title:
        title_clean = re.sub(r'[^a-zA-Z0-9 ]', '', title)[:50].replace(' ', '_')
        ol_title_url = f"https://covers.openlibrary.org/b/title/{title_clean}-M.jpg"
        st.session_state.cover_cache[cache_key] = ol_title_url
        return ol_title_url
    
    # Fallback to no_cover.png
    st.session_state.cover_cache[cache_key] = NO_COVER_IMAGE
    return NO_COVER_IMAGE

# ============================================================================
# 5. DATA LOADING - FAST
# ============================================================================
def load_books_data_fast():
    """Load books data quickly"""
    try:
        print("📚 Loading books data...")
        start_time = time.time()
        
        possible_paths = [
            "data/processed/books_clean_for_app.csv",
            "data/processed/books_features.csv"
        ]
        
        books = None
        for path in possible_paths:
            if os.path.exists(path):
                books = pd.read_csv(path, low_memory=False)
                print(f"✓ Loaded from {path}")
                break
        
        if books is None:
            print("Creating sample data...")
            return create_sample_data()
        
        # Clean columns
        if 'Unnamed: 0' in books.columns:
            books = books.drop(columns=['Unnamed: 0'])
        
        # Rename columns
        column_mapping = {
            'book_id': 'bookId', 'Title': 'title', 'Author': 'author',
            'Genre': 'genres', 'genre': 'genres', 'Language': 'language',
            'average_rating': 'avg_rating', 'rating': 'avg_rating',
            'isbn': 'isbn', 'isbn13': 'isbn', 'isbn10': 'isbn'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in books.columns and new_col not in books.columns:
                books = books.rename(columns={old_col: new_col})
        
        # Ensure required columns
        if 'bookId' not in books.columns:
            books['bookId'] = range(1, len(books)+1)
        
        for col in ['title', 'author', 'genres', 'language', 'avg_rating', 'description']:
            if col not in books.columns:
                if col == 'title':
                    books['title'] = [f"Book {i}" for i in range(1, len(books)+1)]
                elif col == 'author':
                    books['author'] = 'Unknown Author'
                elif col == 'genres':
                    books['genres'] = 'Fiction'
                elif col == 'language':
                    books['language'] = 'English'
                elif col == 'avg_rating':
                    books['avg_rating'] = 3.5
                elif col == 'description':
                    books['description'] = 'A compelling book worth reading.'
        
        # Clean data
        for col in ['title', 'author', 'genres', 'language', 'description']:
            if col in books.columns:
                books[col] = books[col].fillna('Unknown').astype(str)
        
        if 'avg_rating' in books.columns:
            books['avg_rating'] = pd.to_numeric(books['avg_rating'], errors='coerce').fillna(3.5)
        
        # Add popularity score
        if 'popularity_score' not in books.columns:
            books['popularity_score'] = books['avg_rating'] * 10
        
        # Extract primary genre for display (first genre from comma-separated list)
        if 'genres' in books.columns:
            books['primary_genre'] = books['genres'].apply(
                lambda x: str(x).split(',')[0].strip() if pd.notna(x) and ',' in str(x) else str(x).strip()
            )
        else:
            books['primary_genre'] = 'Unknown'
        
        elapsed = time.time() - start_time
        print(f"✓ Loaded {len(books)} books in {elapsed:.2f}s")
        return books
    
    except Exception as e:
        print(f"Error loading books: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample data with comma-separated genres"""
    np.random.seed(42)
    sample_size = 100
    
    languages = ['English', 'Spanish', 'French', 'German', 'Italian', 'Japanese', 'Russian', 'Chinese']
    base_genres = ['Fantasy', 'Science Fiction', 'Mystery', 'Romance', 'Thriller', 
                  'Historical Fiction', 'Biography', 'Self-Help', 'Science', 'Poetry']
    
    # Create comma-separated genres for testing
    def create_genres():
        num_genres = np.random.randint(1, 4)
        selected = np.random.choice(base_genres, num_genres, replace=False)
        return ', '.join(selected)
    
    data = {
        'bookId': range(1, sample_size + 1),
        'title': [f"Book {i}" for i in range(1, sample_size + 1)],
        'author': [f"Author {i}" for i in range(1, sample_size + 1)],
        'genres': [create_genres() for _ in range(sample_size)],
        'language': [languages[i % len(languages)] for i in range(sample_size)],
        'avg_rating': np.round(np.random.uniform(3.0, 4.8, sample_size), 1),
        'description': 'An interesting book about various topics that will captivate readers.',
        'popularity_score': np.random.uniform(10, 50, sample_size),
        'isbn': [f"978{str(i).zfill(10)}" for i in range(sample_size)]
    }
    
    books = pd.DataFrame(data)
    # Add primary genre
    books['primary_genre'] = books['genres'].apply(
        lambda x: str(x).split(',')[0].strip()
    )
    
    return books

# ============================================================================
# 6. INITIALIZE DATA
# ============================================================================
if not st.session_state.data_loaded:
    with st.spinner("Loading BookHub..."):
        # Load books data
        books_df = load_books_data_fast()
        st.session_state.books_df = books_df
        
        # Initialize recommender in background if available
        if MODELS_AVAILABLE:
            try:
                ratings_path = "data/processed/ratings.csv"
                ratings_df = None
                if os.path.exists(ratings_path):
                    ratings_df = pd.read_csv(ratings_path, low_memory=False)
                
                st.session_state.recommender = RecommendationEngine(books_df, ratings_df)
                print("✓ Recommender initialized")
            except Exception as e:
                print(f"Warning: Recommender not available: {e}")
                st.session_state.recommender = None
        
        st.session_state.data_loaded = True
        print("✅ App ready")

books_df = st.session_state.books_df
recommender = st.session_state.recommender

# ============================================================================
# 7. FILTER OPTIONS - FIXED: Show only top 6-7 most used genres
# ============================================================================
@st.cache_data(ttl=3600)
def get_filter_options(_books_df):
    """Get filter options - FIXED: Show only top 6-7 most used genres"""
    # Languages
    languages = ['All'] + sorted([str(x) for x in _books_df['language'].dropna().unique() 
                                 if str(x) != 'nan' and str(x) != ''])
    
    # Genres - Extract ALL genres from comma-separated strings
    all_genres = []
    for genres_str in _books_df['genres'].dropna():
        if isinstance(genres_str, str):
            # Split by comma and clean each genre
            genres_list = [genre.strip() for genre in genres_str.split(',')]
            # Filter out empty strings and 'Unknown'
            clean_genres = [g for g in genres_list if g and g != 'Unknown' and g != 'nan']
            all_genres.extend(clean_genres)
    
    # Count frequencies and get top 6-7 most used genres
    if all_genres:
        genre_counter = Counter(all_genres)
        # Get top 7 genres by frequency (or fewer if less available)
        top_genres = [genre for genre, _ in genre_counter.most_common(7)]
        
        # Make sure we have at least 'All' + some genres
        if top_genres:
            genres = ['All'] + sorted(top_genres)
        else:
            genres = ['All']
    else:
        genres = ['All']
    
    return languages, genres

available_languages, available_genres = get_filter_options(books_df)

# ============================================================================
# 8. RECOMMENDATION FUNCTIONS
# ============================================================================
def get_filtered_books(filters):
    """Get books filtered by BOTH language AND genre"""
    filtered = books_df.copy()
    
    # Apply language filter
    if filters['language'] != 'All':
        filtered = filtered[filtered['language'] == filters['language']]
    
    # Apply genre filter
    if filters['genre'] != 'All':
        # Check if selected genre is in the comma-separated genres string
        genre_mask = filtered['genres'].apply(
            lambda x: filters['genre'].lower() in str(x).lower()
        )
        filtered = filtered[genre_mask]
    
    # Apply rating filter
    filtered = filtered[filtered['avg_rating'] >= filters['min_rating']]
    
    return filtered

def get_filter_based_recommendations(filters, n=10):
    """Get recommendations based on current filters"""
    filtered_books = get_filtered_books(filters)
    
    if filtered_books.empty:
        return pd.DataFrame()
    
    # Return top-rated books from filtered pool
    recommendations = filtered_books.sort_values(
        by=['avg_rating', 'popularity_score'], 
        ascending=[False, False]
    ).head(n)
    
    return recommendations

def get_you_may_like_recommendations():
    """Get 'You May Like This' recommendations based on wishlist"""
    if not st.session_state.user_liked:
        return pd.DataFrame()
    
    wishlist_books = books_df[books_df['bookId'].isin(st.session_state.user_liked)]
    if wishlist_books.empty:
        return pd.DataFrame()
    
    # Analyze wishlist preferences
    liked_genres = []
    liked_authors = []
    liked_languages = []
    
    for _, book in wishlist_books.iterrows():
        # Extract all genres from comma-separated string
        if 'genres' in book and pd.notna(book['genres']):
            genres = str(book['genres']).split(',')
            liked_genres.extend([g.strip().lower() for g in genres])
        
        if 'author' in book and pd.notna(book['author']):
            liked_authors.append(str(book['author']).lower())
        
        if 'language' in book and pd.notna(book['language']):
            liked_languages.append(str(book['language']).lower())
    
    # Get ALL unique languages from wishlist
    all_wishlist_languages = list(set(liked_languages))
    
    if not all_wishlist_languages:
        return pd.DataFrame()
    
    # Get top preferences
    top_genres = [genre for genre, _ in Counter(liked_genres).most_common(3)]
    top_authors = [author for author, _ in Counter(liked_authors).most_common(2)]
    
    # Get books that match ANY language from wishlist
    language_mask = books_df['language'].apply(
        lambda x: str(x).lower() in all_wishlist_languages
    )
    
    candidate_books = books_df[language_mask].copy()
    candidate_books = candidate_books[~candidate_books['bookId'].isin(st.session_state.user_liked)]
    
    if candidate_books.empty:
        return pd.DataFrame()
    
    # Score books based on preferences
    def score_book(row):
        score = 0
        
        # Check genres (any match in comma-separated list)
        if 'genres' in row and pd.notna(row['genres']):
            book_genres = [g.strip().lower() for g in str(row['genres']).split(',')]
            for genre in top_genres:
                if genre in book_genres:
                    score += 4
        
        # Check author
        if 'author' in row and pd.notna(row['author']):
            if str(row['author']).lower() in top_authors:
                score += 3
        
        # Language bonus
        if 'language' in row and pd.notna(row['language']):
            lang = str(row['language']).lower()
            if lang in liked_languages:
                score += liked_languages.count(lang)
        
        # Rating factor
        if 'avg_rating' in row and pd.notna(row['avg_rating']):
            score += row['avg_rating']
        
        return score
    
    candidate_books['recommendation_score'] = candidate_books.apply(score_book, axis=1)
    recommended = candidate_books.sort_values('recommendation_score', ascending=False).head(10)
    
    return recommended

def get_similar_books(book_id, n=5):
    """Get similar books based on current book"""
    if book_id not in books_df['bookId'].values:
        return pd.DataFrame()
    
    book = books_df[books_df['bookId'] == book_id].iloc[0]
    
    # Find similar books by genre, author, or language
    similar_books = books_df[books_df['bookId'] != book_id].copy()
    
    # Score similarity
    def score_similarity(row):
        score = 0
        
        # Same author (highest weight)
        if 'author' in row and pd.notna(row['author']) and 'author' in book:
            if str(row['author']).lower() == str(book['author']).lower():
                score += 5
        
        # Genre overlap (check if any genre matches)
        if 'genres' in row and pd.notna(row['genres']) and 'genres' in book:
            row_genres = set([g.strip().lower() for g in str(row['genres']).split(',')])
            book_genres = set([g.strip().lower() for g in str(book['genres']).split(',')])
            if row_genres.intersection(book_genres):
                score += 3
        
        # Same language
        if 'language' in row and pd.notna(row['language']) and 'language' in book:
            if str(row['language']).lower() == str(book['language']).lower():
                score += 2
        
        # Similar rating (within 0.5)
        if 'avg_rating' in row and pd.notna(row['avg_rating']) and 'avg_rating' in book:
            if abs(row['avg_rating'] - book['avg_rating']) <= 0.5:
                score += 1
        
        return score
    
    similar_books['similarity_score'] = similar_books.apply(score_similarity, axis=1)
    similar_books = similar_books.sort_values('similarity_score', ascending=False).head(n)
    
    return similar_books

# ============================================================================
# 9. UI COMPONENTS
# ============================================================================
def quick_back_button():
    """Quick back button that doesn't cause lag"""
    if st.button("← Back", key="quick_back", use_container_width=True, type="primary"):
        # Store current tab before changing
        current = st.session_state.current_tab
        st.session_state.current_tab = st.session_state.previous_tab
        st.session_state.previous_tab = current
        st.session_state.selected_book = None
        st.rerun()

def render_navigation():
    """Render navigation tabs"""
    tabs = ["Home", "Discover", "Wishlist", "History"]
    
    # Store current tab before changing
    current_tab = st.session_state.current_tab
    
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    cols = st.columns([2, 1, 1, 1, 1, 3])
    
    with cols[0]:
        st.markdown("## 📚 BookHub")
    
    for i, tab in enumerate(tabs):
        with cols[i + 1]:
            if st.button(tab, key=f"nav_{tab}", use_container_width=True):
                if tab != current_tab:
                    st.session_state.previous_tab = current_tab
                    st.session_state.current_tab = tab
                    st.session_state.selected_book = None
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_filter_bar():
    """Render filter bar"""
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    st.markdown("### 🎯 Filter Books")
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        language = st.selectbox(
            "Language",
            available_languages,
            index=available_languages.index(st.session_state.filters['language']) 
            if st.session_state.filters['language'] in available_languages else 0,
            key="filter_lang"
        )
    
    with col2:
        genre = st.selectbox(
            "Genre",
            available_genres,
            index=available_genres.index(st.session_state.filters['genre']) 
            if st.session_state.filters['genre'] in available_genres else 0,
            key="filter_gen"
        )
    
    with col3:
        min_rating = st.slider(
            "Min Rating",
            1.0, 5.0, st.session_state.filters['min_rating'], 0.5,
            key="filter_rate"
        )
    
    with col4:
        apply_col, reset_col = st.columns(2)
        with apply_col:
            if st.button("Apply", key="apply_filters", use_container_width=True):
                st.session_state.filters = {
                    'language': language,
                    'genre': genre,
                    'min_rating': min_rating
                }
                st.session_state.filters_applied = True
                st.rerun()
        
        with reset_col:
            if st.button("Reset", key="reset_filters", use_container_width=True):
                st.session_state.filters = {
                    'language': 'All',
                    'genre': 'All',
                    'min_rating': 3.5
                }
                st.session_state.filters_applied = False
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_book_card(book, show_like_button=True, unique_suffix=""):
    """Render a single book card with real covers"""
    try:
        book_id = int(book.get('bookId', 0))
        title = str(book.get('title', 'Unknown Title'))
        author = str(book.get('author', 'Unknown Author'))
        
        # Get ISBN if available
        isbn = None
        for col in ['isbn13', 'isbn', 'isbn10', 'ISBN', 'isbn_13', 'isbn_10']:
            if col in book and pd.notna(book[col]):
                isbn = book[col]
                break
        
        # Get REAL book cover
        cover_url = get_real_book_cover(book_id, title, author, isbn)
        
        # Create unique key
        unique_key = f"{book_id}_{hashlib.md5(title.encode()).hexdigest()[:8]}_{unique_suffix}"
        
        # Truncate for display
        display_title = title[:40] + ('...' if len(title) > 40 else '')
        display_author = author[:20] + ('...' if len(author) > 20 else '')
        
        # Get rating
        rating = float(book.get('avg_rating', 3.5)) if pd.notna(book.get('avg_rating', 3.5)) else 3.5
        
        # Create card
        st.markdown(f"""
        <div class="book-card">
            <img src="{cover_url}" class="book-cover" alt="{title}" loading="lazy" 
                 onerror="this.src='{NO_COVER_IMAGE}'">
            <div class="book-title" title="{title}">{display_title}</div>
            <div class="book-author">by {display_author}</div>
            <div class="book-rating">{"⭐" * int(rating)} {rating:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for buttons
        col1, col2 = st.columns(2)
        
        # VIEW BUTTON
        with col1:
            if st.button("📖 View", key=f"view_{unique_key}", use_container_width=True):
                if book_id not in st.session_state.user_history:
                    st.session_state.user_history.append(book_id)
                
                st.session_state.previous_tab = st.session_state.current_tab
                st.session_state.selected_book = book_id
                st.session_state.current_tab = "View Book"
                st.session_state.last_action = {"type": "view", "book_id": book_id}
                st.rerun()
        
        # LIKE BUTTON
        with col2:
            if show_like_button:
                is_liked = book_id in st.session_state.user_liked
                button_text = "❤️ Liked" if is_liked else "🤍 Like"
                
                if st.button(button_text, key=f"like_{unique_key}", use_container_width=True):
                    if is_liked:
                        st.session_state.user_liked.remove(book_id)
                        st.session_state.last_action = {"type": "unliked", "book_id": book_id}
                    else:
                        st.session_state.user_liked.append(book_id)
                        st.session_state.last_action = {"type": "liked", "book_id": book_id}
                    st.rerun()
    
    except Exception as e:
        print(f"Error rendering book card: {e}")

def render_book_grid(books, title, cols=5, show_like_button=True, grid_id=""):
    """Render a grid of books"""
    if books.empty:
        st.info(f"No books found for '{title}'")
        return
    
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    
    max_books = cols * 2
    books_display = books.head(max_books)
    
    grid_cols = st.columns(cols)
    
    for idx, (_, book) in enumerate(books_display.iterrows()):
        with grid_cols[idx % cols]:
            render_book_card(book, show_like_button=show_like_button, unique_suffix=f"{grid_id}_{idx}")

# ============================================================================
# 10. HOME PAGE
# ============================================================================
def home_page():
    """Home page"""
    render_navigation()
    
    # Show action messages
    if st.session_state.last_action["type"]:
        action = st.session_state.last_action
        if action["type"] in ["liked", "unliked"]:
            book_id = action["book_id"]
            book_info = books_df[books_df['bookId'] == book_id]
            if not book_info.empty:
                book_title = book_info.iloc[0]['title']
                if action["type"] == "liked":
                    st.success(f"✓ Added '{book_title}' to wishlist!")
                else:
                    st.success(f"✓ Removed '{book_title}' from wishlist!")
        
        st.session_state.last_action = {"type": None, "book_id": None}
    
    st.markdown("## 📖 Welcome to BookHub")
    st.markdown("Discover your next favorite book")
    
    # FILTER BAR
    render_filter_bar()
    
    # Get current filters
    current_filters = st.session_state.filters
    
    # Show what filters are applied
    if st.session_state.filters_applied:
        filter_info = []
        if current_filters['language'] != 'All':
            filter_info.append(f"Language: {current_filters['language']}")
        if current_filters['genre'] != 'All':
            filter_info.append(f"Genre: {current_filters['genre']}")
        
        if filter_info:
            st.info(" | ".join(filter_info))
    
    # 1. FILTER-BASED RECOMMENDATIONS
    if st.session_state.filters_applied:
        st.markdown("---")
        st.markdown("### 🎯 Filter-Based Recommendations")
        
        with st.spinner(f"Finding {current_filters['genre'] if current_filters['genre'] != 'All' else 'best'} books..."):
            filter_recs = get_filter_based_recommendations(current_filters, n=10)
        
        if not filter_recs.empty:
            # Create descriptive title
            title_parts = []
            if current_filters['language'] != 'All':
                title_parts.append(f"in {current_filters['language']}")
            if current_filters['genre'] != 'All':
                title_parts.insert(0, f"{current_filters['genre']} Books")
            
            section_title = " ".join(title_parts) if title_parts else "Recommended Books"
            render_book_grid(filter_recs, section_title, show_like_button=True, grid_id="filter_recs")
    
    # 2. TRENDING BOOKS
    st.markdown("---")
    
    if st.session_state.filters_applied:
        trending_title = f"🔥 Trending in {current_filters['language'] if current_filters['language'] != 'All' else 'All Languages'}"
        trending_books = get_filtered_books(current_filters)
        if not trending_books.empty:
            trending_books = trending_books.sort_values('popularity_score', ascending=False).head(10)
    else:
        trending_title = "🔥 Trending Books"
        trending_books = books_df.sort_values('popularity_score', ascending=False).head(10)
    
    if not trending_books.empty:
        render_book_grid(trending_books, trending_title, show_like_button=True, grid_id="trending")
    
    # 3. YOU MAY LIKE THIS
    if st.session_state.user_liked:
        st.markdown("---")
        st.markdown("### 📚 You May Like This")
        
        you_may_like_recs = get_you_may_like_recommendations()
        
        if not you_may_like_recs.empty:
            render_book_grid(you_may_like_recs, "Based on your wishlist", 
                           show_like_button=True, grid_id="you_may_like")

# ============================================================================
# 11. DISCOVER PAGE
# ============================================================================
def discover_page():
    """Discover page with similar book recommendations"""
    render_navigation()
    st.markdown("## 🔍 Discover Books")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Cache search titles
        @st.cache_data(ttl=3600)
        def get_search_titles():
            return books_df['title'].dropna().unique()[:200]
        
        search_titles = get_search_titles()
        selected_title = st.selectbox(
            "Search for a book:",
            [""] + list(search_titles),
            key="discover_search"
        )
    
    with col2:
        st.write("")
        st.write("")
        if st.button("Find Book", key="find_book", use_container_width=True):
            st.session_state.search_triggered = True
            st.rerun()
    
    if selected_title and st.session_state.get('search_triggered', False):
        book_match = books_df[books_df['title'] == selected_title]
        
        if not book_match.empty:
            book = book_match.iloc[0]
            book_id = int(book['bookId'])
            
            if book_id not in st.session_state.user_history:
                st.session_state.user_history.append(book_id)
                if len(st.session_state.user_history) > 20:
                    st.session_state.user_history = st.session_state.user_history[-20:]
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Get real cover
                isbn = None
                for col in ['isbn13', 'isbn', 'isbn10', 'ISBN', 'isbn_13', 'isbn_10']:
                    if col in book and pd.notna(book[col]):
                        isbn = book[col]
                        break
                
                cover_url = get_real_book_cover(book_id, book['title'], book['author'], isbn)
                st.image(cover_url, width=250)
                
                is_liked = book_id in st.session_state.user_liked
                like_text = "❤️ Remove from Wishlist" if is_liked else "🤍 Add to Wishlist"
                
                if st.button(like_text, key=f"discover_like_{book_id}", use_container_width=True):
                    if is_liked:
                        st.session_state.user_liked.remove(book_id)
                        st.success("✓ Removed from wishlist!")
                    else:
                        st.session_state.user_liked.append(book_id)
                        st.success("✓ Added to wishlist!")
                    st.rerun()
            
            with col2:
                st.markdown(f"### {book['title']}")
                st.markdown(f"**Author:** {book['author']}")
                
                # Show primary genre only (first genre)
                if 'primary_genre' in book and pd.notna(book['primary_genre']):
                    primary_genre = str(book['primary_genre'])
                    if primary_genre and primary_genre != 'Unknown':
                        st.markdown(f"**Genre:** <span class='genre-tag'>{primary_genre}</span>", unsafe_allow_html=True)
                
                st.markdown(f"**Language:** {book['language']}")
                st.markdown(f"**Rating:** ⭐ {book['avg_rating']:.1f}")
                
                if 'description' in book and pd.notna(book['description']):
                    description = str(book['description'])
                    if description and description != 'Unknown':
                        if len(description) > 300:
                            description = description[:300] + "..."
                        st.markdown("**Description:**")
                        st.write(description)
            
            # SIMILAR BOOKS RECOMMENDATIONS
            st.markdown("---")
            st.markdown("### 📚 Similar Books You Might Like")
            
            similar_books = get_similar_books(book_id, n=8)
            
            if not similar_books.empty:
                render_book_grid(similar_books, f"Books similar to '{book['title']}'", 
                               show_like_button=True, grid_id="similar_discover")
            else:
                st.info("No similar books found. Try searching for another book!")

# ============================================================================
# 12. OTHER PAGES
# ============================================================================
def wishlist_page():
    """Wishlist page"""
    render_navigation()
    st.markdown("## ❤️ Your Wishlist")
    
    if st.session_state.user_liked:
        liked_books = books_df[books_df['bookId'].isin(st.session_state.user_liked)]
        
        if not liked_books.empty:
            st.markdown(f"**{len(liked_books)} books in your wishlist:**")
            render_book_grid(liked_books, "Your Liked Books", show_like_button=False, grid_id="wishlist")
            
            if st.button("Clear All", key="clear_wishlist", use_container_width=True):
                st.session_state.user_liked = []
                st.success("✓ Wishlist cleared!")
                st.rerun()
        else:
            st.info("No books found in your wishlist.")
    else:
        st.info("Your wishlist is empty. Click 🤍 on books to add them!")

def history_page():
    """History page"""
    render_navigation()
    st.markdown("## 📖 Your History")
    
    if st.session_state.user_history:
        seen = set()
        unique_history = []
        for book_id in reversed(st.session_state.user_history):
            if book_id not in seen:
                seen.add(book_id)
                unique_history.append(book_id)
                if len(unique_history) >= 10:
                    break
        
        for idx, book_id in enumerate(unique_history):
            book = books_df[books_df['bookId'] == book_id]
            
            if not book.empty:
                book = book.iloc[0]
                
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    isbn = None
                    for col in ['isbn13', 'isbn', 'isbn10', 'ISBN', 'isbn_13', 'isbn_10']:
                        if col in book and pd.notna(book[col]):
                            isbn = book[col]
                            break
                    
                    cover_url = get_real_book_cover(book_id, book['title'], book['author'], isbn)
                    st.image(cover_url, width=60)
                
                with col2:
                    st.markdown(f"**{book['title']}**")
                    st.markdown(f"by {book['author']} | ⭐ {book['avg_rating']:.1f}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("View", key=f"hist_view_{book_id}_{idx}", use_container_width=True):
                            st.session_state.previous_tab = st.session_state.current_tab
                            st.session_state.selected_book = book_id
                            st.session_state.current_tab = "View Book"
                            st.rerun()
                    
                    with col_b:
                        is_liked = book_id in st.session_state.user_liked
                        btn_text = "❤️ Liked" if is_liked else "🤍 Like"
                        
                        if st.button(btn_text, key=f"hist_like_{book_id}_{idx}", use_container_width=True):
                            if is_liked:
                                st.session_state.user_liked.remove(book_id)
                                st.success("✓ Removed from wishlist!")
                            else:
                                st.session_state.user_liked.append(book_id)
                                st.success("✓ Added to wishlist!")
                            st.rerun()
                
                st.markdown("---")
        
        if st.button("Clear History", key="clear_history", use_container_width=True):
            st.session_state.user_history = []
            st.success("✓ History cleared!")
            st.rerun()
    else:
        st.info("No browsing history yet.")

# ============================================================================
# 13. BOOK DETAIL PAGE (FIXED - Quick back, primary genre only)
# ============================================================================
def render_book_detail_page():
    """Render book detail view - Fast with quick back button"""
    book_id = st.session_state.selected_book
    book = books_df[books_df['bookId'] == book_id]
    
    if book.empty:
        st.error("Book not found!")
        quick_back_button()
        return
    
    book = book.iloc[0]
    
    # QUICK BACK BUTTON (no lag)
    quick_back_button()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Get real cover
        isbn = None
        for col in ['isbn13', 'isbn', 'isbn10', 'ISBN', 'isbn_13', 'isbn_10']:
            if col in book and pd.notna(book[col]):
                isbn = book[col]
                break
        
        cover_url = get_real_book_cover(book_id, book['title'], book['author'], isbn)
        st.image(cover_url, width=300)
        
        is_liked = book['bookId'] in st.session_state.user_liked
        like_text = "❤️ Remove from Wishlist" if is_liked else "🤍 Add to Wishlist"
        
        if st.button(like_text, use_container_width=True):
            if is_liked:
                st.session_state.user_liked.remove(book['bookId'])
                st.success("✓ Removed from wishlist!")
            else:
                st.session_state.user_liked.append(book['bookId'])
                st.success("✓ Added to wishlist!")
            st.rerun()
        
        st.markdown("---")
        rating = float(book['avg_rating']) if pd.notna(book['avg_rating']) else 0.0
        st.markdown(f"**Rating:** ⭐ **{rating:.1f}**/5.0")
        st.markdown(f"**Language:** {book['language']}")
        
        # Show only primary genre (not all comma-separated genres)
        if 'primary_genre' in book and pd.notna(book['primary_genre']):
            primary_genre = str(book['primary_genre'])
            if primary_genre and primary_genre != 'Unknown':
                st.markdown(f"**Genre:** {primary_genre}")
    
    with col2:
        st.markdown(f"# {book['title']}")
        st.markdown(f"### by {book['author']}")
        
        st.markdown("---")
        st.markdown("#### 📖 Description")
        if 'description' in book and pd.notna(book['description']):
            description = str(book['description'])
            if description and description != 'Unknown':
                if len(description) > 1000:
                    description = description[:1000] + "..."
                st.write(description)
        else:
            st.write("No description available for this book.")
    
    # SIMILAR BOOKS
    st.markdown("---")
    st.markdown("### 📚 You Might Also Like")
    
    similar_books = get_similar_books(book_id, n=6)
    
    if not similar_books.empty:
        render_book_grid(similar_books, f"Similar to '{book['title']}'", 
                       cols=6, show_like_button=False, grid_id="detail_similar")
    else:
        st.info("No similar books found.")

# ============================================================================
# 14. MAIN APP
# ============================================================================
def main():
    # Show book detail if on View Book tab
    if st.session_state.current_tab == "View Book" and st.session_state.selected_book is not None:
        render_book_detail_page()
        return
    
    # Show appropriate page based on current tab
    if st.session_state.current_tab == "Home":
        home_page()
    elif st.session_state.current_tab == "Discover":
        discover_page()
    elif st.session_state.current_tab == "Wishlist":
        wishlist_page()
    elif st.session_state.current_tab == "History":
        history_page()

if __name__ == "__main__":
    main()