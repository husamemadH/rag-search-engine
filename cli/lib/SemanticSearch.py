from sentence_transformers import SentenceTransformer
import numpy as np
import os
from search_utils import Movie, load_movies


class SemanticSearch:
    model: SentenceTransformer
    embeddings: np.ndarray
    documents: list[Movie]
    document_map: dict[int, Movie]

    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = np.array([])
        self.documents = []
        self.document_map = {}

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError

        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        list = []
        for document in documents:
            self.document_map[document["id"]] = document
            list.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(list, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
        if len(documents) == len(self.embeddings):
            return self.embeddings
        else:
            return self.build_embeddings(documents)

    def search(self, query, limit):

        if len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        scores = []
        res = []
        embedded_query = self.generate_embedding(query)

        for i in range(len(self.embeddings)):
            cs = cosine_similarity(embedded_query, self.embeddings[i])
            scores.append((cs, self.documents[i]))

        top_k = sorted(scores, key=lambda pair: pair[0], reverse=True)[:limit]

        for k in top_k:
            res.append(
                {
                    "score": k[0],
                    "title": k[1]["title"],
                    "description": k[1]["description"],
                }
            )
        return res


def verify_model() -> None:
    semanticSearch = SemanticSearch()
    model = semanticSearch.model

    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def verify_embeddings() -> None:
    semantic_search = SemanticSearch()
    documents = load_movies()

    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text: str) -> None:
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
