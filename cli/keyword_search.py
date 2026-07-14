from collections import Counter
import string
import os
from search_utils import DEFAULT_SEARCH_LIMIT, Movie, load_movies
from nltk.stem import PorterStemmer
from stop_words import STOP_WORDS
import pickle
import math


class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, Movie]
    term_frequencies: dict[int, Counter[str]]

    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokenized_text = tokenize(text)

        self.term_frequencies[doc_id] = Counter()

        for token in tokenized_text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, []))

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def build(self) -> None:
        movies = load_movies()

        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

    def save(self) -> None:
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        if not os.path.exists("cache/index.pkl"):
            raise FileNotFoundError("index.pkl doesnt exist")
        if not os.path.exists("cache/docmap.pkl"):
            raise FileNotFoundError("docmap.pkl doesnt exist")

        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)


movies = load_movies()


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def tf_command(docid: int, term: str) -> None:
    idx = InvertedIndex()
    idx.load()

    tokenized_term = tokenize_single_term(term)
    print(f"{idx.get_tf(docid, tokenized_term)}")


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    tokenized_term = tokenize_single_term(term)
    total_doc_count = len(idx.docmap)
    term_match_doc_count = len(idx.get_documents(tokenized_term))

    idf_value = math.log(
        (total_doc_count + 1) / (term_match_doc_count + 1)
    )  # +1 to avoid division by zero

    return idf_value


def search_command(query: str) -> list[Movie]:
    results = []

    idx = InvertedIndex()
    idx.load()

    tokenized_query = tokenize(query)

    for token in tokenized_query:
        documents = idx.get_documents(token)
        for document_id in documents:
            results.append(idx.docmap[document_id])
            if len(results) >= DEFAULT_SEARCH_LIMIT:
                return results

    return results


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def tokenize(text: str) -> list[str]:
    text = preprocess(text)
    tokens = text.split()
    tokens = filter_stop_words(tokens)
    tokens = stem_tokens(tokens)

    return tokens


def tokenize_single_term(term: str) -> str:
    if len(tokenize(term)) > 1:
        raise ValueError("Tokenizer output should be 1 token")

    return tokenize(term)[0]


def filter_stop_words(tokens: list[str]) -> list[str]:
    tokens_without_stop_words = []

    for token in tokens:
        if token not in STOP_WORDS:
            tokens_without_stop_words.append(token)

    return tokens_without_stop_words


def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    for i in range(len(tokens)):
        tokens[i] = stemmer.stem(tokens[i])
    return tokens
