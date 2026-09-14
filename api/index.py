"""
Vercel Serverless Function entry point for Synaptia Flask Application.
Handles Vercel URL rewrites and normalizes WSGI PATH_INFO.
"""

import os
import sys

# Ensure root directory is on the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app import app


class VercelWSGIWrapper:
    """Normalizes PATH_INFO when Vercel rewrites requests to /api/index or /api/index.py."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path.startswith("/api/index.py"):
            new_path = path[len("/api/index.py") :] or "/"
            environ["PATH_INFO"] = new_path
        elif path.startswith("/api/index"):
            new_path = path[len("/api/index") :] or "/"
            environ["PATH_INFO"] = new_path
        return self.wsgi_app(environ, start_response)


# Wrap Flask WSGI app for seamless Vercel deployment
app.wsgi_app = VercelWSGIWrapper(app.wsgi_app)
