"""
Main FastAPI application - backward-compatible entry point.

The primary server entry point is ``server.app``, which uses the OpenEnv
``create_app()`` factory and registers all custom endpoints.
This module re-exports the app for legacy ``python app/main.py`` usage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.app import app  # re-export the canonical app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
