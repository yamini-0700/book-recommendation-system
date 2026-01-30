def filter_by_language(books_df, language):
    return books_df[books_df['language'] == language]