
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. File paths and configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "tmdb_5000_movies.csv"

EMBEDDINGS_PATH = (
    PROJECT_ROOT / "data" / "cache" / "movie_embeddings.npy"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# 2. Load movie data
# ==========================================

# Reuse the same dataset and preprocessing as BM25.
# This ensures both retrieval methods search
# the same collection of movies.

from src.retrieval.bm25 import movies

# Reset the index so embedding row i always
# corresponds to movies.iloc[i].
movies = movies.reset_index(drop=True)


# ==========================================
# 3. Load semantic embedding model
# ==========================================

model = SentenceTransformer(MODEL_NAME)


# ==========================================
# 4. Generate or load movie embeddings
# ==========================================

def get_movie_embeddings():
    """
    Load cached embeddings if available and valid.
    Otherwise, generate embeddings for all movies.

    Returns:
        numpy.ndarray: Normalized movie embeddings.
    """

    EMBEDDINGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Fingerprint the searchable movie text.
    # If the dataset changes, regenerate embeddings.
    corpus_text = "\n".join(
        movies["search_text"].astype(str)
    )

    corpus_hash = hashlib.sha256(
        corpus_text.encode("utf-8")
    ).hexdigest()

    hash_path = EMBEDDINGS_PATH.with_suffix(".sha256")

    if EMBEDDINGS_PATH.exists() and hash_path.exists():
        saved_hash = hash_path.read_text().strip()

        if saved_hash == corpus_hash:
            embeddings = np.load(
                EMBEDDINGS_PATH,
                allow_pickle=False
            )

            if (
                embeddings.shape[0] == len(movies)
                and embeddings.ndim == 2
                and embeddings.shape[1] == 384
            ):
                print("Loaded cached movie embeddings!")
                return embeddings

    print("Generating movie embeddings...")
    print("This may take a few minutes on the first run.")

    embeddings = model.encode(
        movies["search_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    hash_path.write_text(corpus_hash)

    print("Movie embeddings generated and saved!")

    return embeddings


# ==========================================
# 5. Initialize embeddings
# ==========================================

movie_embeddings = get_movie_embeddings()

print(f"Semantic index ready: {len(movies)} movies!")


# ==========================================
# 6. Semantic movie search
# ==========================================

def search_movies_semantic(query, top_k=5):
    """
    Search movies using semantic similarity.

    Args:
        query (str): Natural-language search query.
        top_k (int): Maximum number of results.

    Returns:
        list[dict]: Ranked movie results with
        metadata and cosine similarity scores.
    """

    if not query.strip() or top_k <= 0:
        return []

    # Convert query to a normalized embedding.
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Dot product of normalized vectors
    # is equivalent to cosine similarity.
    scores = movie_embeddings @ query_embedding

    # Rank movies from highest to lowest score.
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in ranked_indices:
        movie = movies.iloc[int(idx)]

        results.append({
            "id": int(movie["id"]),
            "title": movie["title"],
            "overview": movie["overview"],
            "genres": movie["genres_text"],
            "keywords": movie["keywords_text"],
            "score": float(scores[idx]),
        })

    return results


# ==========================================
# 7. Interactive search loop
# ==========================================

if __name__ == "__main__":

    while True:
        query = input(
            "\nEnter a movie query (or 'quit'): "
        ).strip()

        if query.lower() == "quit":
            break

        if not query:
            continue

        results = search_movies_semantic(query)

        if not results:
            print("No matching movies found.")
            continue

        for rank, movie in enumerate(results, start=1):
            print(f"\n{rank}. {movie['title']}")
            print(f"Semantic Score: {movie['score']:.4f}")
            print(f"Genres: {movie['genres']}")
            print(f"Overview: {movie['overview'][:250]}...")
