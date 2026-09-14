"""
Vercel Serverless Function entry point for Synaptia Flask Application.
Handles Vercel URL rewrites and normalizes WSGI PATH_INFO from request headers.
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
        # Read the real requested path from Vercel proxy headers
        real_path = (
            environ.get("HTTP_X_FORWARDED_PATH")
            or environ.get("HTTP_X_NOW_ROUTE_MATCHES")
            or environ.get("HTTP_X_VERCEL_PATH")
            or environ.get("REQUEST_URI")
            or environ.get("PATH_INFO", "")
        )

        # Strip query parameters if present
        if "?" in real_path:
            real_path = real_path.split("?", 1)[0]

        # Normalize rewrite prefixes
        if real_path.startswith("/api/index.py"):
            real_path = real_path[len("/api/index.py") :] or "/"
        elif real_path.startswith("/api/index"):
            real_path = real_path[len("/api/index") :] or "/"

        environ["PATH_INFO"] = real_path
        return self.wsgi_app(environ, start_response)


# Wrap Flask WSGI app for seamless Vercel deployment
app.wsgi_app = VercelWSGIWrapper(app.wsgi_app)
