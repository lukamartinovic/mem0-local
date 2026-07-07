"""Pytest config — ensures mcp_server is importable both inside the
container (at /app/) and locally (project root)."""
import os
import sys

if os.path.isdir("/app"):
    sys.path.insert(0, "/app")
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))