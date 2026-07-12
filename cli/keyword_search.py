import string
from search_utils import DEFAULT_SEARCH_LIMIT, load_movies

movies = load_movies()
results = []


def search_command(query: str) -> list[str]:

    result = []
    movie_titles = []

    for movie in movies:
        movie_titles.append(movie["title"])
    preprocessed_query = preprocess(query)
    preprocessed_query = preprocessed_query.split()

    for title in movie_titles:
        preprocessed_title = preprocess(title)
        for query in preprocessed_query:
            if query in preprocessed_title:
                result.append(title)
    return result


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text
