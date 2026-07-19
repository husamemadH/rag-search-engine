from collections import Counter
import string
import os
from search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    Movie,
    load_movies,
)
from nltk.stem import PorterStemmer
from stop_words import STOP_WORDS
import pickle
import math


class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, Movie]
    term_frequencies: dict[int, Counter[str]]
    doc_lengths: dict[int, int]

    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokenized_text = tokenize(text)

        self.term_frequencies[doc_id] = Counter()
        self.doc_lengths[doc_id] = len(tokenized_text)

        for token in tokenized_text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, []))

    def bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        tokenized_query = tokenize(query)
        score = {}

        for doc_id in self.docmap:
            bm25_score = 0
            for token in tokenized_query:
                bm25_score += self.bm25(doc_id, token)
            score[doc_id] = bm25_score
        return sorted(score.items(), key=lambda pair: pair[1], reverse=True)[:limit]

    def __get_avg_doc_length(self) -> float:
        total_doc_count = len(self.docmap)
        total_length = 0
        for doc_length in self.doc_lengths.values():
            total_length += doc_length

        return 0 if total_doc_count == 0 else total_length / total_doc_count

    def get_bm25_tf(self, doc_id: int, term: str, b=BM25_B, k1=BM25_K1) -> float:

        doc_length = self.doc_lengths[doc_id]

        # Length normalization factor
        length_norm = 1 - b + b * (doc_length / self.__get_avg_doc_length())

        tf = self.get_tf(doc_id, term)

        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return tf_component

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = self.get_df(term)
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:

        tokenized_term = tokenize_single_term(term)

        df = self.get_df(tokenized_term)
        N = len(self.docmap)

        IDF = math.log((N - df + 0.5) / (df + 0.5) + 1)

        return IDF

    def bm25(self, doc_id, term) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def get_df(self, term: str) -> int:
        return len(self.get_documents(term))

    def build(self) -> None:
        movies = load_movies()

        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

    def save(self) -> None:
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        for path in (self.index_path, self.docmap_path, self.tf_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} doesnt exist")

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)


movies = load_movies()


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    tokenized_term = tokenize_single_term(term)

    return idx.get_bm25_tf(doc_id, tokenized_term, b=b, k1=k1)


def bm25_idf_command(term: str):
    idx = InvertedIndex()
    idx.load()

    return idx.get_bm25_idf(term)


def tf_command(docid: int, term: str) -> None:
    idx = InvertedIndex()
    idx.load()

    tokenized_term = tokenize_single_term(term)
    print(f"{idx.get_tf(docid, tokenized_term)}")


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    tokenized_term = tokenize_single_term(term)
    return idx.get_idf(tokenized_term)


def tfidf_command(docid: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    tokenized_term = tokenize_single_term(term)
    return idx.get_tf(docid, tokenized_term) * idx.get_idf(tokenized_term)


def bm25_search_command(query: str, limit: int) -> list[tuple[Movie, float]]:
    idx = InvertedIndex()
    idx.load()
    return [
        (idx.docmap[doc_id], score) for doc_id, score in idx.bm25_search(query, limit)
    ]


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text


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
