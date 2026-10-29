---
title: Movie Recommender System
emoji: 🎬
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
---

# 🎬 Movie Recommender System

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-recommender-system-a72rdga7zesmada4si8vxx.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![TMDB API](https://img.shields.io/badge/TMDB-API%20v3-01d277?logo=themoviedatabase&logoColor=white)](https://www.themoviedb.org/settings/api)

> 🚀 **Live Demo:** Try the app online at **[movie-recommender-system.streamlit.app](https://movie-recommender-system-a72rdga7zesmada4si8vxx.streamlit.app)**

A content-based movie recommender system that suggests movies based on metadata similarity (plot overview, genres, keywords, top cast, and director). Built using Natural Language Processing (NLP), Bag-of-Words vectorization, Cosine Similarity, and an interactive Streamlit web interface integrated with the TMDB API for live poster art.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Building the Model](#-building-the-model)
- [Running the App](#-running-the-app)
- [TMDB API Configuration](#-tmdb-api-configuration)
- [Resolved Bugs & Technical Improvements](#-resolved-bugs--technical-improvements)
- [Tech Stack](#-tech-stack)

---

## 🌟 Overview

Recommendation systems are widely used in streaming platforms like Netflix and Prime Video. This project implements a **Content-Based Filtering** engine on the TMDB 5,000 Movie Dataset.

Instead of relying on collaborative user ratings, this system analyzes the intrinsic characteristics of films:
- **Plot Overview**: Semantic keywords from the synopsis.
- **Genres**: e.g., Action, Sci-Fi, Thriller.
- **Keywords**: Curated plot tags (e.g., *space travel*, *superhero*, *dystopia*).
- **Cast**: Top 3 billed actors.
- **Crew**: Movie Director.

---

## ⚙️ How It Works

```
                                  DATA PIPELINE
┌─────────────────────────┐     ┌─────────────────────────┐
│ tmdb_5000_movies.csv    │     │ tmdb_5000_credits.csv   │
└────────────┬────────────┘     └────────────┬────────────┘
             │                               │
             └───────────────┬───────────────┘
                             ▼
         [Clean Merge on 'id' and 'title']
                             ▼
         [Feature Extraction & Space Collapse]
     (e.g., "Sam Worthington" -> "SamWorthington")
                             ▼
         [Composite 'tags' String & Porter Stemming]
                             ▼
         [CountVectorizer (5,000 max features, stop words)]
                             ▼
         [Cosine Similarity Matrix Calculation: (4800, 4800)]
                             ▼
         [Export: movie_list.pkl & similarity.pkl]
                             │
                             ▼
                   RECOMMENDATION ENGINE
               ┌─────────────────────────────┐
               │     Streamlit Web UI        │
               │  - User selects movie       │
               │  - Computes top-N matches   │
               │  - Fetches posters via TMDB │
               └─────────────────────────────┘
```

### The Math: Cosine Similarity
Text data is transformed into a 5,000-dimensional vector space using Bag-of-Words. The similarity score between two movies $A$ and $B$ is measured by the cosine of the angle between their vector representations:

$$\text{Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$

Scores range from `0.0` (completely dissimilar) to `1.0` (identical).

---

## ✨ Key Features

- **Accurate Content Matching**: Multi-attribute tagging combines director, lead cast, genres, keywords, and synopsis.
- **Space Collapsing**: Removes spaces within names (e.g., `"Sam Worthington"` $\rightarrow$ `"SamWorthington"` vs `"Sam Mendes"` $\rightarrow$ `"SamMendes"`) to prevent false token collisions.
- **Live Poster Fetching**: Integrates with The Movie Database (TMDB) API to fetch high-resolution official movie posters.
- **Zero-Dependency Fallback**: Built-in SVG poster placeholder ensures the UI never breaks when offline or if an API key is not supplied.
- **Blazing Fast Performance**: Leverages `@st.cache_resource` and `@st.cache_data` to keep models in memory, eliminating redundant disk I/O on user interactions.
- **Interactive UI**:
  - Searchable dropdown with fast autocomplete.
  - 🎲 **Random Movie** discovery button.
  - Customizable number of recommendations slider (3 to 10).
  - Match percentage indicator and direct TMDB movie links.
- **Standalone Model Generator**: Clean CLI script (`generate_model.py`) to build the entire model in ~13 seconds without opening Jupyter.

---

## 📁 Project Structure

```plaintext
Movie Recommended System/
├── data/
│   ├── tmdb_5000_movies.csv       # TMDB 5000 movies metadata
│   └── tmdb_5000_credits.csv      # TMDB 5000 cast and crew metadata
├── model/
│   ├── movie_list.pkl             # Preprocessed movies DataFrame (generated)
│   └── similarity.pkl             # Cosine similarity matrix (generated)
├── .env.example                   # Template for environment variables
├── .env                           # Local environment variables (API keys - gitignored)
├── .gitignore                     # Git ignore rules (virtualenv, models, data)
├── app.py                         # Streamlit web application
├── generate_model.py              # CLI script to process data & build models
├── notebook86c26b4f17.ipynb       # Jupyter notebook with exploratory analysis
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+ installed
- Git (optional)

### 1. Clone or Open the Workspace
```bash
cd "c:/Users/Acer/Documents/Projects/Movie Recommended System"
```

### 2. Create and Activate a Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔨 Building the Model

The large model artifacts (`similarity.pkl` and `movie_list.pkl`) are typically excluded from Git repositories. You can generate them at any time:

### Option A: Using the CLI Generator (Recommended)
Run the dedicated build script:
```bash
python generate_model.py
```
*Builds both `model/movie_list.pkl` and `model/similarity.pkl` in ~13 seconds with automatic verification.*

### Option B: Using the Jupyter Notebook
Open `notebook86c26b4f17.ipynb` in VS Code or Jupyter Lab and run all cells sequentially.

---

## 🎬 Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🔑 TMDB API Configuration

To enable movie poster artwork:

1. Create a free account at [The Movie Database (TMDB)](https://www.themoviedb.org/).
2. Navigate to **Settings** > **API** to generate an API key (v3 auth).
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Add your API key into `.env`:
   ```env
   TMDB_API_KEY=your_actual_tmdb_api_key_here
   ```
5. Restart `app.py`. The app will detect the key and display poster art automatically.

> **Note:** If no key is provided, the application will continue to work seamlessly using the built-in SVG placeholder posters.

---

## 🐛 Resolved Bugs & Technical Improvements

This repository has undergone a comprehensive code review and bug remediation:

| Issue | Root Cause | Resolution |
| :--- | :--- | :--- |
| **IndexError: out of bounds** | `movies.dropna(inplace=True)` dropped 3 rows, leaving gaps at indices `[2658, 4145, 4437]`. The matrix has 4,800 rows (0..4799), but DataFrame indices reached 4,808. Accessing `similarity[index]` crashed on movies at the end of the catalog. | Added `.reset_index(drop=True)` immediately after `dropna()`, ensuring DataFrame index and similarity matrix rows are strictly 0..N-1 aligned. |
| **Silent Recommendation Drift** | For every movie after index 2,658, the missing indices shifted the row lookup by 1 to 3 positions. Searching for a horror movie like *Devil* retrieved the similarity vector for *Friday After Next* (stoner comedy). | Re-aligned index and positional indexing so every movie precisely queries its own vector. |
| **Cross-Contaminated Duplicate Rows** | `movies.merge(credits, on='title')` performed a cross-join on movies sharing titles (e.g., *Batman* 1989 and 1966), creating duplicate rows and mixing up cast/crew between unrelated films. | Changed merge to `movies.merge(credits, left_on=['id', 'title'], right_on=['movie_id', 'title'])`, guaranteeing unique 1:1 mapping. |
| **Severe UI Latency** | `pickle.load` of the 185 MB `similarity.pkl` was executed unconditionally at top-level on every single Streamlit rerun / interaction. | Cached models with `@st.cache_resource` and cached TMDB poster API calls with `@st.cache_data`. |
| **Crash on Missing Models** | If model files did not exist upon a fresh clone, the app crashed with an unhandled `FileNotFoundError`. | Added path resolution and friendly in-app setup instructions when model files are missing. |
| **Hardcoded Secret Instruction** | UI previously prompted users to "set TMDB_API_KEY in app.py", encouraging insecure hardcoded secrets. | Updated to load exclusively from `.env` via `python-dotenv` with clear status feedback in the sidebar. |
| **Broken Kaggle File Paths** | Legacy Kaggle code referenced non-existent paths `/kaggle/input/...`. | Refactored notebook and generator to resolve local `data/` paths reliably. |

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning & NLP**: [scikit-learn](https://scikit-learn.org/) (`CountVectorizer`, `cosine_similarity`), [NLTK](https://www.nltk.org/) (`PorterStemmer`)
- **API & Networking**: [Requests](https://requests.readthedocs.io/)
- **Environment Management**: [python-dotenv](https://github.com/theskumar/python-dotenv)

---

## 📄 License & Attribution

- Movie metadata and posters provided by [The Movie Database (TMDB)](https://www.themoviedb.org/).
- Dataset sourced from the [Kaggle TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
