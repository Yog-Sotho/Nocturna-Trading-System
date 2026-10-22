"""
NOCTURNA Trading System — Development entry point.

This file is a thin shim for local development only.
The canonical app factory lives in src/main.py.

Production deployments use:
    gunicorn -c gunicorn.conf.py 'src.main:create_app()'
    (single worker — the engine keeps state in memory; see gunicorn.conf.py)

Local development:
    python main.py
"""

import logging
import os
import sys

# Ensure the project root is on sys.path so `src.*` imports resolve.
sys.path.insert(0, os.path.dirname(__file__))

from src.main import create_app, start_serving  # noqa: E402 — path setup above

app = create_app()

if __name__ == "__main__":
    if os.environ.get("FLASK_ENV") == "production":
        logging.critical(
            "=" * 80 + "\n"
            "WARNING: Running Flask dev server in PRODUCTION mode!\n"
            "Use: gunicorn -c gunicorn.conf.py 'src.main:create_app()'\n"
            + "=" * 80
        )

    # With the debug reloader the module runs in two processes; only the serving
    # child may start the engine, or two engines would trade one account
    if not app.config["DEBUG"] or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_serving(app)
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 5000)),
        debug=app.config["DEBUG"],
        threaded=True,
    )
