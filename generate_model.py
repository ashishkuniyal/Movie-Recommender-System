"""
Model Generation Script for Movie Recommender System
Processes TMDB 5000 dataset, extracts features, computes cosine similarity,
and saves artifacts to model/movie_list.pkl and model/similarity.pkl.
"""

import ast
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_features(text):
    """Safely parse JSON-like string into list of names."""
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except Exception:
        return []


def fetch_director(text):
    """Extract director name from crew JSON string."""
    try:
        for i in ast.literal_eval(text):
            if i.get('job') == 'Director':
                return [i['name']]
    except Exception:
        pass
    return []


def collapse(items):
    """Remove spaces inside multi-word names to prevent false token collisions."""
    return [item.replace(" ", "") for item in items]


def build_model(data_dir="data", model_dir="model"):
    start_time = time.time()
    movies_csv = os.path.join(data_dir, "tmdb_5000_movies.csv")
    credits_csv = os.path.join(data_dir, "tmdb_5000_credits.csv")

    if not os.path.exists(movies_csv) or not os.path.exists(credits_csv):
        print(f"[ERROR] Required dataset files not found in '{data_dir}/'.", file=sys.stderr)
        print(f"Please ensure '{movies_csv}' and '{credits_csv}' exist.", file=sys.stderr)
        sys.exit(1)

    print("1. Loading raw TMDB datasets...")
    movies_df = pd.read_csv(movies_csv)
    credits_df = pd.read_csv(credits_csv)

    print("2. Merging datasets cleanly...")
    # Merge on both id and title to avoid cross-product duplicate rows for films with identical titles
    movies = movies_df.merge(credits_df, left_on=['id', 'title'], right_on=['movie_id', 'title'])
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    # Drop missing values and reset index to guarantee 0..N-1 alignment with matrix indices
    movies = movies.dropna().reset_index(drop=True)
    print(f"   Cleaned movie records: {len(movies)}")

    print("3. Extracting and parsing metadata features...")
    movies['genres'] = movies['genres'].apply(parse_features).apply(collapse)
    movies['keywords'] = movies['keywords'].apply(parse_features).apply(collapse)
    movies['cast'] = movies['cast'].apply(parse_features).apply(lambda x: x[:3]).apply(collapse)
    movies['crew'] = movies['crew'].apply(fetch_director).apply(collapse)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    print("4. Building composite tags...")
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

    print("5. Applying Porter Stemmer to normalize vocabulary...")
    ps = PorterStemmer()

    def stem(text):
        return " ".join(ps.stem(word) for word in text.split())

    new_df['tags'] = new_df['tags'].apply(stem)

    # Ensure index is 0..N-1
    new_df = new_df.reset_index(drop=True)

    print("6. Vectorizing text (Bag-of-Words, 5000 max features)...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    print("7. Computing Cosine Similarity matrix (float16 for fast loading)...")
    similarity = cosine_similarity(vectors).astype(np.float16)

    print(f"8. Saving artifacts to '{model_dir}/'...")
    os.makedirs(model_dir, exist_ok=True)
    movie_list_path = os.path.join(model_dir, "movie_list.pkl")
    similarity_path = os.path.join(model_dir, "similarity.pkl")

    with open(movie_list_path, "wb") as f:
        pickle.dump(new_df, f)

    with open(similarity_path, "wb") as f:
        pickle.dump(similarity, f)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Model built in {elapsed:.2f}s!")
    print(f"  - movie_list.pkl: {len(new_df)} movies")
    print(f"  - similarity.pkl: {similarity.shape} matrix")

    # Quick sanity check
    test_titles = ["Avatar", "Devil", "Signed, Sealed, Delivered"]
    print("\nSanity Check Recommendations:")
    for title in test_titles:
        matches = new_df[new_df['title'] == title]
        if not matches.empty:
            idx = matches.index[0]
            sim_scores = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
            top_rec = [new_df.iloc[i[0]].title for i in sim_scores[1:4]]
            print(f"  {title} -> {', '.join(top_rec)}")


if __name__ == "__main__":
    build_model()

