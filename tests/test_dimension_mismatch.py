"""
Tests for _check_and_fix_collection_dims() — the dimension mismatch detection
logic extracted from init_memory().

Only needs Qdrant — no mem0, no Ollama, no embedder. Creates collections with
wrong dimensions via the Qdrant REST API and verifies they get deleted.

Run inside container:
    docker compose run --rm mcp-server pytest tests/test_dimension_mismatch.py -v
"""

import json
import os
import urllib.request
import urllib.error

import pytest

import mcp_server

QDRANT_HOST = os.environ.get("MEM0_QDRANT_HOST", "qdrant")
QDRANT_PORT = os.environ.get("MEM0_QDRANT_PORT", "6333")
QDRANT_BASE = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
EMBED_DIMS = int(os.environ.get("MEM0_EMBED_DIMS", "768"))


def _is_reachable(url, timeout=3):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


needs_qdrant = pytest.mark.skipif(
    not _is_reachable(QDRANT_BASE),
    reason=f"Qdrant not reachable at {QDRANT_BASE}",
)


# ── Qdrant REST helpers ──────────────────────────────────────────────────────

def _create_collection(name: str, dims: int):
    """Create a Qdrant collection with given dimensions."""
    payload = json.dumps({
        "vectors": {"size": dims, "distance": "Cosine"},
    }).encode()
    req = urllib.request.Request(
        f"{QDRANT_BASE}/collections/{name}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    urllib.request.urlopen(req, timeout=10)


def _get_collection_dims(name: str):
    """Get the dimension of a Qdrant collection, or None if it doesn't exist."""
    try:
        resp = urllib.request.urlopen(
            f"{QDRANT_BASE}/collections/{name}", timeout=5
        )
        data = json.loads(resp.read())
        vectors = (
            data.get("result", {})
            .get("config", {})
            .get("params", {})
            .get("vectors", {})
        )
        if isinstance(vectors, dict):
            for key, val in vectors.items():
                if isinstance(val, dict) and "size" in val:
                    return val["size"]
            if "size" in vectors:
                return vectors["size"]
        return None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _delete_collection(name: str):
    """Delete a Qdrant collection if it exists."""
    try:
        req = urllib.request.Request(
            f"{QDRANT_BASE}/collections/{name}", method="DELETE"
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


# ── Tests ────────────────────────────────────────────────────────────────────

@needs_qdrant
class TestDimensionMismatch:
    """Test the dimension mismatch detection and fix logic.

    This is the code that prevents the cryptic 'Vector dimension error' at
    runtime by detecting and fixing stale collections on startup.
    """

    def setup_method(self):
        """Clean up collections before each test."""
        _delete_collection("mem0")
        _delete_collection("mem0migrations")

    def teardown_method(self):
        """Clean up collections after each test."""
        _delete_collection("mem0")
        _delete_collection("mem0migrations")

    def test_no_collection_exists(self):
        """When no collection exists, function returns without error."""
        actions = mcp_server._check_and_fix_collection_dims(
            QDRANT_BASE, EMBED_DIMS
        )
        assert isinstance(actions, list)
        # No collections to report on
        assert len(actions) == 0

    def test_correct_dims_no_action(self):
        """Collection with correct dimensions is left alone."""
        _create_collection("mem0", EMBED_DIMS)
        actions = mcp_server._check_and_fix_collection_dims(
            QDRANT_BASE, EMBED_DIMS
        )
        # Should report OK, not delete
        assert any("OK" in a and "mem0" in a for a in actions)
        # Collection should still exist with correct dims
        dims = _get_collection_dims("mem0")
        assert dims == EMBED_DIMS

    def test_wrong_dims_deleted(self):
        """Collection with wrong dimensions is deleted."""
        wrong_dims = EMBED_DIMS + 256  # e.g., 1024 instead of 768
        _create_collection("mem0", wrong_dims)
        actions = mcp_server._check_and_fix_collection_dims(
            QDRANT_BASE, EMBED_DIMS
        )
        assert any("Deleted" in a and "mem0" in a for a in actions), \
            f"Expected deletion action, got: {actions}"
        # Collection should be gone
        dims = _get_collection_dims("mem0")
        assert dims is None, f"Collection should be deleted, still has dims: {dims}"

    def test_multiple_collections_checked(self):
        """Both mem0 and mem0migrations collections are checked."""
        wrong_dims = EMBED_DIMS + 256
        _create_collection("mem0", wrong_dims)
        _create_collection("mem0migrations", wrong_dims)
        actions = mcp_server._check_and_fix_collection_dims(
            QDRANT_BASE, EMBED_DIMS
        )
        # Both should be deleted
        assert any("mem0" in a and "Deleted" in a for a in actions)
        assert any("mem0migrations" in a and "Deleted" in a for a in actions)
        assert _get_collection_dims("mem0") is None
        assert _get_collection_dims("mem0migrations") is None

    def test_correct_dims_mem0migrations_left_alone(self):
        """mem0migrations with correct dims is not deleted."""
        _create_collection("mem0migrations", EMBED_DIMS)
        actions = mcp_server._check_and_fix_collection_dims(
            QDRANT_BASE, EMBED_DIMS
        )
        assert any("mem0migrations" in a and "OK" in a for a in actions)
        dims = _get_collection_dims("mem0migrations")
        assert dims == EMBED_DIMS

    def test_mixed_dims(self):
        """One collection correct, one wrong — only the wrong one is deleted."""
        _create_collection("mem0", EMBED_DIMS)  # correct
        _create_collection("mem0migrations", EMBED_DIMS + 256)  # wrong
        actions = mcp_server._check_and_fix_collection_dims(
            QDRANT_BASE, EMBED_DIMS
        )
        # mem0 should be OK, mem0migrations should be deleted
        assert any("mem0" in a and "OK" in a for a in actions)
        assert any("mem0migrations" in a and "Deleted" in a for a in actions)
        # Verify state
        assert _get_collection_dims("mem0") == EMBED_DIMS
        assert _get_collection_dims("mem0migrations") is None

    def test_qdrant_not_reachable_no_crash(self):
        """Function does not crash when Qdrant is not reachable."""
        # Use a bogus URL — function should handle connection errors gracefully
        actions = mcp_server._check_and_fix_collection_dims(
            "http://nonexistent:9999", EMBED_DIMS
        )
        assert isinstance(actions, list)
        # Should be empty (all exceptions caught)
        assert len(actions) == 0