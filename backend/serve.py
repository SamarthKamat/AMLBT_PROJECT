"""Production entry point: serves the API and the built frontend with waitress (works on Windows, macOS and Linux).

    python serve.py            # http://0.0.0.0:5000 (FLASK_PORT)

UDI keeps sessions and trained models in process memory (backed by SQLite for datasets and conversations), so run a
single process with several threads rather than several worker processes.
"""
import logging
import os

from waitress import serve

import config
from app import app

if __name__ == "__main__":
    logging.getLogger("waitress").setLevel(logging.INFO)
    print(f"UDI {config.VERSION} on http://localhost:{config.PORT}")
    serve(app, host="0.0.0.0", port=config.PORT, threads=int(os.getenv("UDI_THREADS", "8")), channel_timeout=600)
