import argparse
from lib.SemanticSearch import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    SemanticSearch,
    chunk,
    semantic_chunk,
)
from lib.ChunkedSemanticSearch import ChunkedSemanticSearch
from search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify", help="verify model name")
    embed_parser = subparsers.add_parser("embed_text")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings")
    embed_query_parser = subparsers.add_parser("embed_query")
    search_parser = subparsers.add_parser("search")
    chunk_parser = subparsers.add_parser("chunk")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk")
    subparsers.add_parser("embed_chunks")

    chunk_parser.add_argument("text", type=str, help="text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="size of chunk in words"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="number of words to overlap between chunks",
    )
    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0)
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results")
    embed_parser.add_argument("text", type=str)
    embed_query_parser.add_argument("query", type=str)

    args = parser.parse_args()

    match args.command:
        case "search":
            semantic_search = SemanticSearch()
            movies = load_movies()
            semantic_search.load_or_create_embeddings(movies)
            results = semantic_search.search(args.query, args.limit)

            for i, result in enumerate(results):
                print(
                    f"{i + 1}. {result['title']} (score: {result['score']})  {result['description']}"
                )
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "chunk":
            chunk(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            movies = load_movies()
            chunked_search = ChunkedSemanticSearch()
            embeddings = chunked_search.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
