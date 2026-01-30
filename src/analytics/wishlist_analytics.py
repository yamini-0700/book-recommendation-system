def build_user_profile(wishlist_df, books_df, user_id):
    user_wishlist = wishlist_df[wishlist_df['user_id'] == user_id]
    
    merged = user_wishlist.merge(
        books_df,
        left_on='book_id',
        right_on='bookId',
        how='left'
    )
    
    if merged.empty:
        return None
    
    top_genre = merged['genres'].value_counts().idxmax()
    top_author = merged['author'].value_counts().idxmax()
    
    return {
        "preferred_genre": top_genre,
        "preferred_author": top_author
    }


def recommend_from_wishlist(profile, books_df, top_n=5):
    if profile is None:
        return books_df.head(top_n)
    
    return books_df[
        (books_df['genres'] == profile['preferred_genre']) |
        (books_df['author'] == profile['preferred_author'])
    ].sort_values(
        by='popularity_score',
        ascending=False
    ).head(top_n)
