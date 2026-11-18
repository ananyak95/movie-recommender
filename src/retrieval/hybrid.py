from src.retrieval.bm25 import search_movies
from src.retrieval.semantic import search_movies_semantic

# ==========================================
# 1. Hybrid retrieval function
# ==========================================

def hybrid_search(query, top_k=5, rrf_k=60):
    if not query.strip() or top_k <= 0:
        return []

    # retrieve candidate movies
    candidate_k = max(50, top_k)

    bm25_results = search_movies(query, top_k = candidate_k)
    semantic_results = search_movies_semantic(query, top_k = candidate_k)

    # TODO: Combine ranking using RRF
    combined_results = {}

    
    # Loop through BM25 results
    for rank, movie in enumerate(bm25_results, start=1):
        movie_id = movie["id"]
        rrf_score = 1/(rrf_k + rank)
        combined_results[movie_id] = movie.copy()
        combined_results[movie_id]["score"] = rrf_score

    # Loop through semantic results
    for rank, movie in enumerate(semantic_results, start=1):

        movie_id = movie["id"]
        rrf_score = 1/(rrf_k + rank)
    # TODO: 
    # - If already in combined_results, add to its score
    # - Otherwise, add the movie as a new entry

    # TODO: Sort movies by combined RRF score


    # TODO: Sort combined_results from highest to lowest score


    # TODO: Return only the top_k movies



# ==========================================
# 2. Interactive search loop
# ==========================================

if __name__ == "__main__":
    while True:
        query = input("\nEnter a movie query (or 'quit'): ").strip()

        if query.lower() == "quit":
            break

        if query:
            results = hybrid_search(query)

            # TODO: Print the ranked results
