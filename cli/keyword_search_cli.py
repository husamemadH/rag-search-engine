import argparse
import json
from re import search
import string
from keyword_search import search_command, build_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            result = search_command(args.query)

            for i, title in enumerate(result):
                print(f"{i + 1} {title}")
            pass
        case "build":
            build_command()
            pass

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
