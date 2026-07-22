import os
from lib.SemanticSearch import SemanticSearch
import numpy as np
from search_utils import Movie
from lib.SemanticSearch import semantic_chunk
import json


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        chunks = []
        chunk_metadata = []
        for document in documents:
            if document["description"] == "":
                continue
            self.document_map[document["id"]] = document

            chunks.extend(semantic_chunk(document["description"], 4, 1))
            for i, chunk in enumerate(chunks):
                chunk_metadata.append(
                    {
                        "movie_idx": document["id"],
                        "chunk_idx": i,
                        "total_chunks": len(chunks),
                    }
                )

        self.embeddings = self.model.encode(chunks, show_progress_bar=True)
        np.save("cache/chunk_embeddings.npy", self.embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(chunks)}, f, indent=2
            )
        return self.embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists(
            "cache/chunk_metadata.json"
        ):
            self.embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)
            return self.embeddings
        else:
            return self.build_chunk_embeddings(documents)
