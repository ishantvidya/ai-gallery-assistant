"""ui.py — the local web interface (Flask).

This is the "display" layer of the architecture. The browser talks to this
small web server, which simply calls the search() function from Stage 6.
Nothing here touches the model or the index directly.

Why a web UI at all?
    Showing photos in a grid is what browsers are great at, and Flask needs
    nothing beyond what we already installed. The server binds to 127.0.0.1
    (localhost only), so the app never accepts connections from the network
    and your photos never leave your computer.

Run it:
    python -m src.ui     (or the friendlier entry point: python main.py)
"""

import time

from flask import Flask, abort, jsonify, render_template, request, send_file

from src import search, storage

app = Flask(__name__)

# The only folder we are allowed to serve photos from.
IMAGES_ROOT = (storage.ROOT / "data" / "images").resolve()


@app.route("/")
def index():
    """Serve the search page itself."""
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    """Handle a query:  /api/search?q=my+query&k=6  ->  JSON results."""
    query = request.args.get("q", "").strip()
    try:
        k = int(request.args.get("k", 6))
    except ValueError:
        k = 6

    if not query:
        return jsonify({"error": "Type something to search."}), 400

    started = time.perf_counter()
    try:
        results = search.search(query, k=k)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    return jsonify(
        {
            "query": query,
            "results": [
                {
                    "path": path,
                    "name": storage.resolve_image_path(path).name,
                    "score": score,
                }
                for path, score in results
            ],
            "elapsed_ms": elapsed_ms,
            "technique": "query-expansion + re-ranking",
        }
    )


@app.route("/api/stats")
def api_stats():
    """Return how many photos are indexed vs on disk (for the footer)."""
    indexed, on_disk = search.index_info()
    return jsonify({"indexed": indexed, "on_disk": on_disk})


@app.route("/image/<path:rel>")
def serve_image(rel):
    """Serve one photo file, safely.

    Security note: `rel` comes from the browser, so we never trust it. We
    resolve it (defeating any ../ tricks) and refuse to serve anything that
    is not a file directly inside data/images/.
    """
    abs_path = storage.resolve_image_path(rel).resolve()
    if abs_path.parent != IMAGES_ROOT or not abs_path.is_file():
        abort(404)
    return send_file(abs_path)


if __name__ == "__main__":
    # debug=False is deliberate: Flask's auto-reloader runs TWO copies of the
    # app, and we do not want two copies of the 600 MB model in RAM.
    search.warn_if_stale()
    # Preload the CLIP model before serving so the first search is instant
    # instead of stalling ~15-20 s while the model loads.
    from src.model import get_model

    get_model()
    app.run(host="127.0.0.1", port=5000, debug=False)
