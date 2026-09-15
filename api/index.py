"""
Vercel Serverless Function entry point for Synaptia Flask Application.
Imports the configured Flask app and handler from the main application.
"""

import os
import sys

# Ensure root directory is on the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app import app, handler
