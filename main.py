"""main.py — one command to run the whole app.

    python main.py          -> start the web UI at http://127.0.0.1:5000
    python main.py --index  -> (re)build the index from data/images, then start

Indexing is only needed when your photos change. After that, the UI loads
the saved index and searches are instant.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Gallery Assistant")
    parser.add_argument(
        "--index",
        action="store_true",
        help="(re)build the index from data/images before starting",
    )
    args = parser.parse_args()

    if args.index:
        from src.indexer import build_index

        build_index()

    # Preload the CLIP model now (takes ~15-20 s once) so the first search
    # doesn't stall while the 600 MB model loads. Searches are then ~20 ms.
    from src.model import get_model

    get_model()

    from src.ui import app

    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
