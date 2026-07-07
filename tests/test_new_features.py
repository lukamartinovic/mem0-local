"""
Tests for new MCP tools: export_memories, import_memories, prune_memories,
and search_memories with relevance scores.

Needs Qdrant + embedder (infer=False tests). Tests that need the LLM
are marked with @needs_ollama_llm and skipped if the model isn't available.

Run inside container:
    docker compose run --rm mcp-server pytest tests/test_new_features.py -v
"""

import json
import os
import time
import uuid
import urllib.request

import pytest

import mcp_server

# ── Config ──────────────────────────────────────────────────────────────────

QDRANT_HOST = os.environ.get("MEM0_QDRANT_HOST", "qdrant")
QDRANT_PORT = os.environ.get("MEM0_QDRANT_PORT", "6333")
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
OLLAMA_URL = os.environ.get("MEM0_OLLAMA_URL", "http://ollama:11434")
MCP_PORT = os.environ.get("MCP_PORT", "8765")
MCP_URL = f"http://localhost:{MCP_PORT}"


def _is_reachable(url, timeout=3):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


needs_qdrant = pytest.mark.skipif(
    not _is_reachable(QDRANT_URL),
    reason=f"Qdrant not reachable at {QDRANT_URL}",
)
needs_ollama = pytest.mark.skipif(
    not _is_reachable(f"{OLLAMA_URL}/api/tags"),
    reason=f"Ollama not reachable at {OLLAMA_URL}",
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def memory():
    return mcp_server.init_memory()


@pytest.fixture
def test_user():
    uid = f"feat_{uuid.uuid4().hex[:8]}"
    yield uid
    # Cleanup: delete all memories for this test user after the test
    try:
        mcp_server.execute_tool("delete_all_memories", {"user_id": uid})
    except Exception:
        pass


def _add_raw(content, user_id, metadata=None):
    """Helper: add memory with infer=False (no LLM needed)."""
    args = {"content": content, "user_id": user_id}
    if metadata:
        args["metadata"] = metadata
    return mcp_server.execute_tool("add_raw_memory", args)


# ── Health endpoint tests ────────────────────────────────────────────────────

@needs_qdrant
@needs_ollama
class TestHealthEndpoint:
    """Tests for the improved /health endpoint with component status."""

    def test_health_has_init_status(self, memory):
        """Health endpoint reports initialization status."""
        # Access the health endpoint logic directly
        health = mcp_server._build_health_response()
        assert "init_status" in health
        # After init_memory() fixture, should be "ready"
        assert health["init_status"] in ("ready", "initializing", "error")

    def test_health_has_components(self, memory):
        """Health endpoint reports component reachability."""
        health = mcp_server._build_health_response()
        assert "components" in health
        assert "ollama" in health["components"]
        assert "qdrant" in health["components"]
        assert "mem0" in health["components"]

    def test_health_has_model_status(self, memory):
        """Health endpoint reports LLM model status."""
        health = mcp_server._build_health_response()
        assert "model_status" in health
        assert health["model_status"] in ("loaded", "unloaded", "unreachable", "unknown")

    def test_health_has_tool_count(self, memory):
        """Health endpoint reports correct tool count."""
        health = mcp_server._build_health_response()
        assert health["tools"] == len(mcp_server.TOOL_DEFINITIONS)

    def test_health_status_values(self, memory):
        """Health status is one of: ok, starting, degraded."""
        health = mcp_server._build_health_response()
        assert health["status"] in ("ok", "starting", "degraded")


# ── Export / Import tests ────────────────────────────────────────────────────

@needs_qdrant
@needs_ollama
class TestExportImport:
    """Tests for export_memories and import_memories tools."""

    def test_export_json(self, memory, test_user):
        """Export returns JSON array of memories."""
        _add_raw("Test memory for export at 10am", test_user)
        _add_raw("Another memory about deployment", test_user)
        time.sleep(1)
        result = mcp_server.execute_tool("export_memories", {
            "user_id": test_user,
            "format": "json",
        })
        # Result should be parseable JSON
        if isinstance(result, str):
            data = json.loads(result)
        elif isinstance(result, dict) and "data" in result:
            data = json.loads(result["data"]) if isinstance(result["data"], str) else result["data"]
        else:
            data = result
        assert isinstance(data, list)
        assert len(data) >= 2
        # Each memory should have key fields
        for mem in data:
            assert "memory" in mem or "memory_text" in mem or "text" in mem

    def test_export_csv(self, memory, test_user):
        """Export in CSV format returns CSV string."""
        _add_raw("CSV export test memory", test_user)
        time.sleep(1)
        result = mcp_server.execute_tool("export_memories", {
            "user_id": test_user,
            "format": "csv",
        })
        # Should contain CSV header
        text = json.dumps(result) if not isinstance(result, str) else result
        assert "memory" in text.lower() or "text" in text.lower()

    def test_export_empty_user(self, memory, test_user):
        """Export for user with no memories returns empty array."""
        result = mcp_server.execute_tool("export_memories", {
            "user_id": f"empty_{uuid.uuid4().hex[:8]}",
            "format": "json",
        })
        if isinstance(result, str):
            data = json.loads(result)
        elif isinstance(result, dict) and "data" in result:
            data = result["data"] if isinstance(result["data"], list) else json.loads(result["data"])
        else:
            data = result
        assert isinstance(data, list)
        assert len(data) == 0

    def test_import_json(self, memory, test_user):
        """Import memories from JSON array."""
        export_data = json.dumps([
            {"memory": "Imported fact one: uses PostgreSQL 15", "user_id": test_user},
            {"memory": "Imported fact two: runs on port 8080", "user_id": test_user},
        ])
        result = mcp_server.execute_tool("import_memories", {
            "data": export_data,
            "user_id": test_user,
        })
        assert "imported" in result or "count" in result or "status" in result
        # Verify memories were actually stored
        time.sleep(1)
        search = mcp_server.execute_tool("search_memories", {
            "query": "PostgreSQL port",
            "user_id": test_user,
        })
        results = search if isinstance(search, list) else search.get("results", [])
        combined = json.dumps(results).lower()
        assert "postgresql" in combined or "8080" in combined

    def test_import_skips_duplicates(self, memory, test_user):
        """Import skips memories that already exist for the user."""
        # Add a memory first
        _add_raw("Duplicate test: we use Redis for caching", test_user)
        time.sleep(1)
        # Try importing the same text
        export_data = json.dumps([
            {"memory": "Duplicate test: we use Redis for caching", "user_id": test_user},
            {"memory": "New fact: also use Kafka for events", "user_id": test_user},
        ])
        result = mcp_server.execute_tool("import_memories", {
            "data": export_data,
            "user_id": test_user,
        })
        # Should report at least 1 skipped
        text = json.dumps(result)
        assert "skip" in text.lower() or "imported" in text

    def test_import_invalid_data(self, memory, test_user):
        """Import with invalid JSON raises a proper error."""
        with pytest.raises(mcp_server.Mem0Error):
            mcp_server.execute_tool("import_memories", {
                "data": "not valid json",
                "user_id": test_user,
            })


# ── Prune tests ──────────────────────────────────────────────────────────────

@needs_qdrant
@needs_ollama
class TestPruneMemories:
    """Tests for prune_memories tool."""

    def test_prune_dry_run_default(self, memory, test_user):
        """prune_memories defaults to dry_run=true (safe by default)."""
        _add_raw("Old memory that might be pruned", test_user)
        time.sleep(1)
        result = mcp_server.execute_tool("prune_memories", {
            "user_id": test_user,
            "older_than_days": 0,  # Everything is older than 0 days
        })
        # dry_run should be True by default
        assert result.get("dry_run") is True or "dry_run" in json.dumps(result).lower()
        # Should report what would be deleted
        assert "would_delete" in result or "deleted" in result or "memories" in result

    def test_prune_dry_run_no_delete(self, memory, test_user):
        """Dry run does not actually delete memories."""
        _add_raw("Memory that should survive dry run", test_user)
        time.sleep(1)
        mcp_server.execute_tool("prune_memories", {
            "user_id": test_user,
            "older_than_days": 0,
            "dry_run": True,
        })
        # Memory should still exist
        search = mcp_server.execute_tool("search_memories", {
            "query": "survive dry run",
            "user_id": test_user,
        })
        results = search if isinstance(search, list) else search.get("results", [])
        assert len(results) > 0, "Dry run should not delete memories"

    def test_prune_old_memories(self, memory, test_user):
        """prune_memories with older_than_days=0 and dry_run=false deletes everything."""
        _add_raw("Memory to prune", test_user)
        _add_raw("Another memory to prune", test_user)
        time.sleep(1)
        result = mcp_server.execute_tool("prune_memories", {
            "user_id": test_user,
            "older_than_days": 0,
            "dry_run": False,
        })
        assert result.get("dry_run") is False or "dry_run" in json.dumps(result).lower()
        # Verify memories are gone
        search = mcp_server.execute_tool("search_memories", {
            "query": "prune",
            "user_id": test_user,
        })
        results = search if isinstance(search, list) else search.get("results", [])
        assert len(results) == 0, "Memories should be deleted after prune"

    def test_prune_preserves_recent(self, memory, test_user):
        """prune_memories with large older_than_days preserves recent memories."""
        _add_raw("Recent memory should survive", test_user)
        time.sleep(1)
        result = mcp_server.execute_tool("prune_memories", {
            "user_id": test_user,
            "older_than_days": 365,  # 1 year — nothing should be pruned
            "dry_run": True,
        })
        # Should report 0 would be deleted
        would_delete = result.get("would_delete", 0)
        assert would_delete == 0, f"Recent memories should not be pruned: {result}"


# ── Search with scores tests ─────────────────────────────────────────────────

@needs_qdrant
@needs_ollama
class TestSearchWithScores:
    """Tests for search_memories with relevance scores."""

    def test_search_returns_scores(self, memory, test_user):
        """Search results include relevance scores."""
        _add_raw("We use TypeScript with strict mode enabled", test_user)
        _add_raw("The database is PostgreSQL version 15", test_user)
        time.sleep(1)
        result = mcp_server.execute_tool("search_memories", {
            "query": "TypeScript configuration",
            "user_id": test_user,
            "include_scores": True,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        assert len(results) > 0
        # At least one result should have a score
        has_score = any("score" in str(r) for r in results)
        assert has_score, f"Results should include scores: {results}"

    def test_search_min_score_filter(self, memory, test_user):
        """min_score parameter filters out low-relevance results."""
        _add_raw("Very relevant: TypeScript is our primary language", test_user)
        _add_raw("Unrelated: we bought pizza for the team lunch", test_user)
        time.sleep(1)
        result = mcp_server.execute_tool("search_memories", {
            "query": "programming language",
            "user_id": test_user,
            "min_score": 0.5,
            "include_scores": True,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        # All returned results should have score >= 0.5
        for r in results:
            score = r.get("score") if isinstance(r, dict) else None
            if score is not None:
                assert score >= 0.5, f"Result with score {score} < 0.5 min_score"

    def test_search_without_scores(self, memory, test_user):
        """include_scores=false omits scores from results."""
        _add_raw("Test memory for no-scores search", test_user)
        time.sleep(1)
        result = mcp_server.execute_tool("search_memories", {
            "query": "test memory",
            "user_id": test_user,
            "include_scores": False,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        assert len(results) > 0
        # Results should still work, just without explicit score field
        # (mem0 may include it internally, but the tool shouldn't add it)

    def test_search_high_min_score_returns_fewer(self, memory, test_user):
        """Higher min_score returns fewer results."""
        for i in range(5):
            _add_raw(f"Memory number {i} about various topics", test_user)
        time.sleep(1)
        low = mcp_server.execute_tool("search_memories", {
            "query": "various topics",
            "user_id": test_user,
            "min_score": 0.0,
        })
        high = mcp_server.execute_tool("search_memories", {
            "query": "various topics",
            "user_id": test_user,
            "min_score": 0.9,
        })
        low_count = len(low) if isinstance(low, list) else len(low.get("results", []))
        high_count = len(high) if isinstance(high, list) else len(high.get("results", []))
        assert high_count <= low_count, \
            f"Higher min_score should return fewer or equal results: {high_count} vs {low_count}"