import json
from typing import TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str


BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "cache"


def load_movies() -> list[Movie]:
    with open("data/movies.json", "r") as f:
        movies = json.load(f)

    return movies["movies"]
