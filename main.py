"""
NOCTURNA Trading System — Development entry point.

This file is a thin shim for local development only.
The canonical app factory lives in src/main.py.

Production deployments use:
    gunicorn 'src.main:create_app()' --workers 4 --bind 0.0.0.0:5000

Local development:
    python main.py
"""

import os
import sys
import logging

# Ensure the project root is on sys.path so `src.*` imports resolve.
sys.path.insert(0, os.path.dirname(__file__))

from src.main import create_app  # noqa: E402 — path setup above

app = create_app()

if __name__ == "__main__":
    if os.environ.get("FLASK_ENV") == "production":
        logging.critical(
            "=" * 80 + "\n"
            "WARNING: Running Flask dev server in PRODUCTION mode!\n"
            "Use: gunicorn 'src.main:create_app()' --workers 4 --bind 0.0.0.0:5000\n"
            + "=" * 80
        )

    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 5000)),
        debug=app.config["DEBUG"],
        threaded=True,
    )
