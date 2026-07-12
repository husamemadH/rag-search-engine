import string
from search_utils import DEFAULT_SEARCH_LIMIT, load_movies
from nltk.stem import PorterStemmer

movies = load_movies()

stop_words = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "ain",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren",
    "arent",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "couldn",
    "couldnt",
    "d",
    "did",
    "didn",
    "didnt",
    "do",
    "does",
    "doesn",
    "doesnt",
    "doing",
    "don",
    "dont",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn",
    "hadnt",
    "has",
    "hasn",
    "hasnt",
    "have",
    "haven",
    "havent",
    "having",
    "he",
    "hed",
    "hell",
    "hes",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "id",
    "ill",
    "im",
    "ive",
    "if",
    "in",
    "into",
    "is",
    "isn",
    "isnt",
    "it",
    "itd",
    "itll",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "mightn",
    "mightnt",
    "more",
    "most",
    "mustn",
    "mustnt",
    "my",
    "myself",
    "needn",
    "neednt",
    "no",
    "nor",
    "not",
    "now",
    "o",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "re",
    "s",
    "same",
    "shan",
    "shant",
    "she",
    "shed",
    "shell",
    "shes",
    "should",
    "shouldve",
    "shouldn",
    "shouldnt",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "thatll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "theyd",
    "theyll",
    "theyre",
    "theyve",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "ve",
    "very",
    "was",
    "wasn",
    "wasnt",
    "we",
    "wed",
    "well",
    "were",
    "weve",
    "weren",
    "werent",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "wont",
    "wouldn",
    "wouldnt",
    "y",
    "you",
    "youd",
    "youll",
    "youre",
    "youve",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def search_command(query: str) -> list[str]:

    result = []
    tokenized_query = tokenize(query)

    for movie in movies:
        tokenized_title = tokenize(movie["title"])
        if has_matching_token(tokenized_query, tokenized_title):
            result.append(movie["title"])
            if len(result) >= DEFAULT_SEARCH_LIMIT:
                break

    return result


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


def filter_stop_words(tokens: list[str]) -> list[str]:
    tokens_without_stop_words = []

    for token in tokens:
        if token not in stop_words:
            tokens_without_stop_words.append(token)

    return tokens_without_stop_words


def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    for i in range(len(tokens)):
        tokens[i] = stemmer.stem(tokens[i])
    return tokens
