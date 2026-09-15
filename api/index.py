"""
Vercel Serverless Function entry point for Synaptia Flask Application.
Handles Vercel URL rewrites and normalizes WSGI PATH_INFO from request headers.
"""

import os
import sys
import urllib.parse

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
        # 1. Check if Vercel passed __path__ in the rewritten query string
        query_string = environ.get("QUERY_STRING", "")
        extracted_path = None
        if "__path__=" in query_string:
            try:
                params = urllib.parse.parse_qs(query_string)
                if "__path__" in params and params["__path__"]:
                    extracted_path = params["__path__"][0]
                    # Clean __path__ out of QUERY_STRING so application request.args is pristine
                    clean_params = {k: v for k, v in params.items() if k != "__path__"}
                    environ["QUERY_STRING"] = urllib.parse.urlencode(clean_params, doseq=True)
            except Exception:
                pass

        # 2. Check if Vercel provided route regex capture matches in headers
        if not extracted_path:
            matches_str = environ.get("HTTP_X_NOW_ROUTE_MATCHES") or environ.get("HTTP_X_VERCEL_MATCHES", "")
            if matches_str:
                try:
                    parsed = urllib.parse.parse_qs(matches_str)
                    # Match group 1 from /(.*)
                    if "1" in parsed and parsed["1"]:
                        captured = parsed["1"][0]
                        if not captured.startswith("/"):
                            captured = "/" + captured
                        extracted_path = captured
                except Exception:
                    pass

        real_path = (
            extracted_path
            or environ.get("HTTP_X_FORWARDED_PATH")
            or environ.get("HTTP_X_FORWARDED_URI")
            or environ.get("HTTP_X_MATCHED_PATH")
            or environ.get("HTTP_X_VERCEL_PATH")
            or environ.get("REQUEST_URI")
            or environ.get("RAW_URI")
            or environ.get("PATH_INFO", "/")
        )

        # Strip query parameters if present
        if "?" in real_path:
            real_path = real_path.split("?", 1)[0]

        # Normalize rewrite prefixes if the path itself is pointing to the handler file
        if real_path.startswith("/api/index.py"):
            real_path = real_path[len("/api/index.py") :] or "/"
        elif real_path.startswith("/api/index") and (
            len(real_path) == len("/api/index") or real_path[len("/api/index")] == "/"
        ):
            real_path = real_path[len("/api/index") :] or "/"

        if not real_path.startswith("/"):
            real_path = "/" + real_path

        environ["PATH_INFO"] = real_path
        return self.wsgi_app(environ, start_response)


# Wrap Flask WSGI app for seamless Vercel deployment
app.wsgi_app = VercelWSGIWrapper(app.wsgi_app)
