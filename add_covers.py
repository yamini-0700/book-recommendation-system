import requests
import pandas as pd
from pathlib import Path
import time

class CoverFetcher:
    def __init__(self):
        self.api_url = "https://www.googleapis.com/books/v1/volumes"
        self.covers_dir = Path("generated_covers")
        self.covers_dir.mkdir(exist_ok=True)
    
    def get_cover_url(self, title, author=""):
        """Get book cover URL from Google Books API"""
        try:
            query = f"{title} {author}".strip()
            params = {"q": query, "maxResults": 1}
            response = requests.get(self.api_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    volume_info = data['items'][0]['volumeInfo']
                    if 'imageLinks' in volume_info:
                        return volume_info['imageLinks'].get('thumbnail', '')
            
            return "https://via.placeholder.com/150x200?text=No+Cover"
            
        except Exception as e:
            print(f"Error fetching cover for {title}: {e}")
            return "https://via.placeholder.com/150x200?text=Error"
    
    def add_covers_to_dataframe(self, books_df):
        """Add cover URLs to books dataframe"""
        print("Adding cover URLs to books...")
        books_df['cover_url'] = ""
        
        for idx, row in books_df.iterrows():
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(books_df)} books...")
            
            cover_url = self.get_cover_url(row['title'], row['author'])
            books_df.at[idx, 'cover_url'] = cover_url
            
            # Be nice to the API
            time.sleep(0.1)
        
        return books_df

# Quick test
if __name__ == "__main__":
    # Load a few books to test
    books = pd.read_csv("data/processed/books_clean.csv", nrows=5)
    fetcher = CoverFetcher()
    
    print("Testing cover fetcher...")
    for _, book in books.iterrows():
        url = fetcher.get_cover_url(book['title'], book['author'])
        print(f"{book['title'][:30]}... -> {url[:50]}...")