import argparse
import json
from re import search
import string
from keyword_search import search_command, build_command, tf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Use term frequency counting")

    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser.add_argument("docid", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Term to search for")

    args = parser.parse_args()

    match args.command:
        case "search":
            result = search_command(args.query)

            for i, movie in enumerate(result):
                print(f"{i + 1} {movie['id']} {movie['title']}")
            pass
        case "build":
            build_command()
            pass
        case "tf":
            tf_command(args.docid, args.term)
            pass

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
