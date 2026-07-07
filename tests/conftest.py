"""Pytest config for tests/ — ensures mcp_server is importable.

This mirrors the root conftest.py so pytest finds mcp_server whether
it runs from /app/ or /app/tests/.
"""
import os
import sys

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_tests_dir)

for path in (_project_root, "/app"):
    if path not in sys.path and os.path.isdir(path):
        sys.path.insert(0, path)