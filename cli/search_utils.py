import json
from typing import TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str


DEFAULT_SEARCH_LIMIT = 5


def load_movies() -> list[Movie]:
    with open("data/movies.json", "r") as f:
        movies = json.load(f)

    return movies["movies"]
