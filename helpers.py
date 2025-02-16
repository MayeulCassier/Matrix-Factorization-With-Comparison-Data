"""
# Imports
"""

import pandas as pd
import os

"""
# Functions for Data Managing
"""



def load_movielens_data(folder_path='Data'):
    # Load user data
    user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users = pd.read_csv(os.path.join(folder_path, 'u.user'), sep='|', names=user_columns, encoding='latin-1')

    # Load item (movie) data
    item_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
                    'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv(os.path.join(folder_path, 'u.item'), sep='|', names=item_columns, encoding='latin-1')

    # Load ratings data
    rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(os.path.join(folder_path, 'u.data'), sep='\t', names=rating_columns, engine='python')

    # Display dataframes for clarity
    print("Users DataFrame:")
    print(users.head())
    print(f"Users shape: {users.shape}")

    print("\nMovies DataFrame:")
    print(items[['movie_id', 'title']].head())
    print(f"Movies shape: {items.shape}")

    print("\nRatings DataFrame:")
    print(ratings.head())
    print(f"Ratings shape: {ratings.shape}")

    return users, items, ratings


"""
# Functions for SVM Model
"""