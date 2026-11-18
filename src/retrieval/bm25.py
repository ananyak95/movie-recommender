
import json
import re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi


# ==========================================
# 1. Load movie dataset
# ==========================================

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "tmdb_5000_movies.csv"
)

movies = pd.read_csv(DATA_PATH)

movies = movies.fillna({
    "title": "",
    "overview": "",
    "genres": "",
    "keywords": "",
})


# ==========================================
# 2. Extract names from JSON columns
# ==========================================

def extract_names(value):
    try:
        items = json.loads(value)
        return " ".join(item["name"] for item in items)
    except (ValueError, TypeError, KeyError):
        return ""


movies["genres_text"] = movies["genres"].apply(extract_names)
movies["keywords_text"] = movies["keywords"].apply(extract_names)


# ==========================================
# 3. Combine movie metadata into searchable documents
# ==========================================

movies["search_text"] = (
    movies["title"] + " "
    + movies["overview"] + " "
    + movies["genres_text"] + " "
    + movies["keywords_text"]
)


# ==========================================
# 4. Tokenize text
# ==========================================

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


tokenized_corpus = [
    tokenize(text) for text in movies["search_text"]
]


# ==========================================
# 5. Build BM25 index
# ==========================================

bm25 = BM25Okapi(tokenized_corpus)

print(f"Indexed {len(movies)} movies!")


# ==========================================
# 6. Search and rank movies
# ==========================================

def search_movies(query, top_k=5):
    """
    Search for movies using BM25.

    Args:
        query (str): Natural-language search query.
        top_k (int): Maximum number of results.

    Returns:
        list[dict]: Ranked movie results containing
        metadata and BM25 relevance scores.
    """

    query_tokens = tokenize(query)

    if not query_tokens or top_k <= 0:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    # Return only movies with positive BM25 scores
    top_indices = [
        idx for idx in ranked_indices
        if scores[idx] > 0
    ][:top_k]

    results = []

    for idx in top_indices:
        movie = movies.iloc[idx]

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

        results = search_movies(query)

        if not results:
            print("No matching movies found.")
            continue

        for rank, movie in enumerate(results, start=1):
            print(f"\n{rank}. {movie['title']}")
            print(f"BM25 Score: {movie['score']:.2f}")
            print(f"Genres: {movie['genres']}")
            print(f"Overview: {movie['overview'][:250]}...")
