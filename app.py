import os
import pickle
import random
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Load environment variables ───────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants & Configuration ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MOVIE_LIST_PATH = os.path.join(MODEL_DIR, "movie_list.pkl")
SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity.pkl")

# TMDB API Key validation
RAW_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
TMDB_API_KEY = "" if RAW_API_KEY.lower() in ("your_tmdb_api_key_here", "none", "") else RAW_API_KEY

# Fallback SVG Poster Data URI (works 100% offline without external CDN dependency)
PLACEHOLDER_POSTER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='450' viewBox='0 0 300 450'>"
    "<rect width='300' height='450' fill='%231a1a2e'/>"
    "<circle cx='150' cy='180' r='45' fill='%2316213e'/>"
    "<polygon points='150,155 175,195 125,195' fill='%23e94560'/>"
    "<text x='150' y='270' font-family='Arial, sans-serif' font-size='18' fill='%23ffffff' font-weight='bold' text-anchor='middle'>No Poster</text>"
    "<text x='150' y='295' font-family='Arial, sans-serif' font-size='13' fill='%239a9a9a' text-anchor='middle'>Available</text>"
    "</svg>"
)


# ── Load Model & Data (Cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading recommendation models...")
def load_data():
    """Load precomputed movie list and cosine similarity matrix with safety checks."""
    if not os.path.exists(MOVIE_LIST_PATH) or not os.path.exists(SIMILARITY_PATH):
        return None, None

    with open(MOVIE_LIST_PATH, "rb") as f:
        movies_df = pickle.load(f)

    with open(SIMILARITY_PATH, "rb") as f:
        similarity_matrix = pickle.load(f)

    # Guarantee index is strictly 0..N-1 to align perfectly with matrix rows
    movies_df = movies_df.reset_index(drop=True)

    return movies_df, similarity_matrix


# ── Poster Fetching (Cached) ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_poster(movie_id):
    """Fetch movie poster URL from TMDB API with timeout and graceful fallback."""
    if not TMDB_API_KEY or not movie_id:
        return PLACEHOLDER_POSTER

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        headers = {"User-Agent": "MovieRecommenderApp/1.0"}
        response = requests.get(url, params=params, headers=headers, timeout=(3, 5))

        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass

    return PLACEHOLDER_POSTER


# ── Recommendation Engine ────────────────────────────────────────────────────
def recommend(movie_title, movies_df, similarity_matrix, top_n=5):
    """Return top_n recommended movies with poster URLs and similarity scores."""
    matches = movies_df[movies_df["title"] == movie_title]
    if matches.empty:
        return []

    # Get integer position of the query movie
    movie_idx = matches.index[0]

    # Validate index within similarity matrix bounds
    if movie_idx >= len(similarity_matrix):
        return []

    # Calculate similarity rankings
    sim_scores = similarity_matrix[movie_idx]
    ranked_indices = sorted(list(enumerate(sim_scores)), reverse=True, key=lambda x: x[1])

    recommendations = []
    # Skip index 0 as it represents the query movie itself
    for row_idx, score in ranked_indices[1 : top_n + 1]:
        rec_row = movies_df.iloc[row_idx]
        movie_id = rec_row["movie_id"]
        recommendations.append({
            "title": rec_row["title"],
            "movie_id": movie_id,
            "poster": fetch_poster(movie_id),
            "score": round(float(score) * 100, 1),
        })

    return recommendations


# ── Main Application ─────────────────────────────────────────────────────────
def main():
    movies, similarity = load_data()

    # Handle missing model files
    if movies is None or similarity is None:
        st.title("🎬 Movie Recommender System")
        st.error(
            "⚠️ **Model files not found!**\n\n"
            "The model artifacts (`model/movie_list.pkl` and `model/similarity.pkl`) are missing.\n\n"
            "To generate them, run the preprocessing script in your terminal:\n\n"
            "```bash\npython generate_model.py\n```\n\n"
            "Then refresh this page."
        )
        return

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5, step=1)

        st.markdown("---")
        st.subheader("🔑 TMDB API Status")
        if TMDB_API_KEY:
            st.success("API Key Active (Posters enabled)")
        else:
            st.warning("API Key not found or default")
            st.info(
                "To display movie posters, add your free key to `.env`:\n\n"
                "```env\nTMDB_API_KEY=your_key_here\n```\n\n"
                "Get one at [themoviedb.org](https://www.themoviedb.org/settings/api)."
            )

        st.markdown("---")
        st.subheader("📊 Dataset Info")
        st.write(f"• **Catalog Size:** {len(movies):,} movies")
        st.write("• **Features:** Genre, Keywords, Cast, Director, Overview")
        st.write("• **Algorithm:** Bag-of-Words & Cosine Similarity")

    # ── Main Header ──────────────────────────────────────────────────────────
    st.title("🎬 Movie Recommender System")
    st.caption("Content-based recommendation engine powered by Bag-of-Words & Cosine Similarity")
    st.markdown("---")

    movie_titles = list(movies["title"].values)

    # Initialize session state for random movie picker
    if "selected_movie" not in st.session_state:
        st.session_state.selected_movie = movie_titles[0] if movie_titles else ""

    col_select, col_rand = st.columns([4, 1])

    with col_rand:
        st.write("")
        st.write("")
        if st.button("🎲 Random Movie", use_container_width=True):
            st.session_state.selected_movie = random.choice(movie_titles)
            st.rerun()

    with col_select:
        # Determine selectbox index safely
        try:
            current_index = movie_titles.index(st.session_state.selected_movie)
        except ValueError:
            current_index = 0

        selected_movie = st.selectbox(
            "Select or search for a movie you like:",
            movie_titles,
            index=current_index,
            key="movie_dropdown",
        )
        st.session_state.selected_movie = selected_movie

    # Recommendation Trigger
    if st.button("🔍 Recommend Similar Movies", type="primary", use_container_width=True):
        with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
            recommendations = recommend(selected_movie, movies, similarity, top_n=top_n)

        if not recommendations:
            st.warning(f"Could not find recommendations for '{selected_movie}'.")
            return

        st.markdown(f"### Movies Similar to **{selected_movie}**")

        # Dynamic responsive layout
        cols = st.columns(len(recommendations))
        for col, rec in zip(cols, recommendations):
            with col:
                st.image(rec["poster"], use_container_width=True)
                st.markdown(f"**{rec['title']}**")
                st.caption(f"Match: `{rec['score']}%`")
                tmdb_url = f"https://www.themoviedb.org/movie/{rec['movie_id']}"
                st.markdown(f"[View on TMDB ↗]({tmdb_url})")


if __name__ == "__main__":
    main()
