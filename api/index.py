"""
Vercel Serverless Function entry point for Synaptia Flask Application.
Handles Vercel URL rewrites and normalizes WSGI PATH_INFO from request headers.
"""

import os
import sys
import json
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
        # ── RAW DEBUG ENDPOINT ────────────────────────────────────────────
        # Hit /__vercel_debug to see exactly what Vercel passes
        raw_path = environ.get("PATH_INFO", "")
        raw_qs = environ.get("QUERY_STRING", "")
        if raw_path.rstrip("/") == "/__vercel_debug" or raw_qs.startswith("__path__=/__vercel_debug"):
            debug_data = {}
            for key in sorted(environ.keys()):
                val = environ[key]
                if isinstance(val, str) and len(val) < 1000:
                    debug_data[key] = val
            body = json.dumps(debug_data, indent=2).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ])
            return [body]

        # ── PATH EXTRACTION ───────────────────────────────────────────────

        extracted_path = None

        # 1. Check QUERY_STRING for __path__ param (from vercel.json rewrite)
        query_string = environ.get("QUERY_STRING", "")
        if "__path__=" in query_string:
            try:
                params = urllib.parse.parse_qs(query_string)
                if "__path__" in params and params["__path__"]:
                    extracted_path = urllib.parse.unquote(params["__path__"][0])
                    # Clean __path__ from query string
                    clean_params = {k: v for k, v in params.items() if k != "__path__"}
                    environ["QUERY_STRING"] = urllib.parse.urlencode(clean_params, doseq=True)
            except Exception:
                pass

        # 2. Check Vercel route match headers
        if not extracted_path:
            matches_str = environ.get("HTTP_X_NOW_ROUTE_MATCHES") or environ.get("HTTP_X_VERCEL_MATCHES", "")
            if matches_str:
                try:
                    parsed = urllib.parse.parse_qs(matches_str)
                    if "1" in parsed and parsed["1"]:
                        captured = urllib.parse.unquote(parsed["1"][0])
                        if not captured.startswith("/"):
                            captured = "/" + captured
                        extracted_path = captured
                except Exception:
                    pass

        # 3. Check other common proxy headers
        if not extracted_path:
            for header in [
                "HTTP_X_FORWARDED_PATH",
                "HTTP_X_FORWARDED_URI",
                "HTTP_X_MATCHED_PATH",
                "HTTP_X_INVOKE_PATH",
                "HTTP_X_VERCEL_PATH",
            ]:
                val = environ.get(header)
                if val and val != "/api/index.py" and val != "/api/index":
                    extracted_path = val
                    break

        # 4. Fall back to PATH_INFO
        real_path = extracted_path or environ.get("PATH_INFO", "/")

        # Strip query parameters if present
        if "?" in real_path:
            real_path = real_path.split("?", 1)[0]

        # Normalize rewrite prefixes
        if real_path.startswith("/api/index.py"):
            real_path = real_path[len("/api/index.py"):] or "/"
        elif real_path.startswith("/api/index") and (
            len(real_path) == len("/api/index") or real_path[len("/api/index")] == "/"
        ):
            real_path = real_path[len("/api/index"):] or "/"

        if not real_path.startswith("/"):
            real_path = "/" + real_path

        environ["PATH_INFO"] = real_path
        return self.wsgi_app(environ, start_response)


# Wrap Flask WSGI app for seamless Vercel deployment
app.wsgi_app = VercelWSGIWrapper(app.wsgi_app)
