"""Pytest config — ensures mcp_server is importable both inside the
container (at /app/) and locally (project root).

pytest may run from /app/ or /app/tests/, so we add both /app and
the parent of the tests directory to sys.path.
"""
import os
import sys

# Inside Docker, mcp_server.py is at /app/mcp_server.py
# Locally, mcp_server.py is at the project root (parent of tests/)
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_tests_dir)

# Add project root first so `import mcp_server` finds it
for path in (_project_root, "/app"):
    if path not in sys.path and os.path.isdir(path):
        sys.path.insert(0, path)