"""
Tests for the memory pipeline using infer=False (no LLM extraction).

These tests exercise the full CRUD pipeline — add → search → get → update →
delete — without calling the LLM for fact extraction. They DO need Qdrant +
the embedder (Ollama with nomic-embed-text), but NOT the LLM model (qwen2.5:7b).

Run inside container:
    docker compose run --rm mcp-server pytest tests/test_memory_raw.py -v

Skip conditions: tests are skipped automatically if Qdrant or Ollama is not
reachable, so they can run in mixed environments without failing.
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


# ── Skip conditions ──────────────────────────────────────────────────────────

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
    """Initialize mem0 once for all tests in this module."""
    return mcp_server.init_memory()


@pytest.fixture
def test_user():
    uid = f"rawtest_{uuid.uuid4().hex[:8]}"
    yield uid
    # Cleanup: delete all memories for this test user after the test
    try:
        mcp_server.execute_tool("delete_all_memories", {"user_id": uid})
    except Exception:
        pass


# ── Tests ────────────────────────────────────────────────────────────────────

@needs_qdrant
@needs_ollama
class TestMemoryRaw:
    """Memory pipeline tests using infer=False — no LLM, just Qdrant + embedder.

    These isolate Qdrant/embedder issues from LLM issues. If these pass but
    the TestMemoryOperations (with LLM) fail, the problem is the LLM model,
    not the storage pipeline.
    """

    def test_add_raw_and_search(self, memory, test_user):
        """Add text with infer=False, then search finds it via embedding."""
        mcp_server.execute_tool("add_raw_memory", {
            "content": "I prefer TypeScript over JavaScript and use ESLint.",
            "user_id": test_user,
            
        })
        time.sleep(1)  # Let embedding index
        result = mcp_server.execute_tool("search_memories", {
            "query": "language preference",
            "user_id": test_user,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        assert len(results) > 0, "Search should find the raw memory"
        combined = json.dumps(results).lower()
        assert "typescript" in combined or "javascript" in combined

    def test_add_raw_get_memories(self, memory, test_user):
        """Add with infer=False, then get_memories lists it."""
        mcp_server.execute_tool("add_raw_memory", {
            "content": "Deployed auth service to staging on Tuesday.",
            "user_id": test_user,
            
        })
        time.sleep(1)
        result = mcp_server.execute_tool("get_memories", {
            "user_id": test_user,
            "limit": 10,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        assert len(results) > 0, "get_memories should list the raw memory"

    def test_add_raw_get_by_id(self, memory, test_user):
        """Add with infer=False, then get_memory retrieves it by ID."""
        mcp_server.execute_tool("add_raw_memory", {
            "content": "The API rate limit is 100 requests per minute.",
            "user_id": test_user,
            
        })
        time.sleep(1)
        all_mems = mcp_server.execute_tool("get_memories", {
            "user_id": test_user,
            "limit": 10,
        })
        results = all_mems if isinstance(all_mems, list) else all_mems.get("results", [])
        assert len(results) > 0
        mem_id = results[0].get("id") or results[0].get("memory_id")
        assert mem_id, f"Memory has no ID: {results[0]}"
        single = mcp_server.execute_tool("get_memory", {"memory_id": mem_id})
        assert single is not None

    def test_add_raw_update(self, memory, test_user):
        """Add with infer=False, then update_memory overwrites content."""
        mcp_server.execute_tool("add_raw_memory", {
            "content": "Database runs on port 5432.",
            "user_id": test_user,
            
        })
        time.sleep(1)
        all_mems = mcp_server.execute_tool("get_memories", {
            "user_id": test_user,
            "limit": 10,
        })
        results = all_mems if isinstance(all_mems, list) else all_mems.get("results", [])
        assert len(results) > 0
        mem_id = results[0].get("id") or results[0].get("memory_id")
        mcp_server.execute_tool("update_memory", {
            "memory_id": mem_id,
            "content": "Database runs on port 5433.",
        })
        time.sleep(1)
        result = mcp_server.execute_tool("search_memories", {
            "query": "database port",
            "user_id": test_user,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        combined = json.dumps(results).lower()
        assert "5433" in combined, "Updated content should be searchable"

    def test_add_raw_delete(self, memory, test_user):
        """Add with infer=False, then delete_memory removes it."""
        mcp_server.execute_tool("add_raw_memory", {
            "content": "Temporary memory for deletion test.",
            "user_id": test_user,
            
        })
        time.sleep(1)
        all_mems = mcp_server.execute_tool("get_memories", {
            "user_id": test_user,
            "limit": 10,
        })
        results = all_mems if isinstance(all_mems, list) else all_mems.get("results", [])
        assert len(results) > 0
        mem_id = results[0].get("id") or results[0].get("memory_id")
        mcp_server.execute_tool("delete_memory", {"memory_id": mem_id})
        time.sleep(1)
        result = mcp_server.execute_tool("search_memories", {
            "query": "temporary deletion test",
            "user_id": test_user,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        combined = json.dumps(results).lower()
        assert "temporary memory for deletion" not in combined

    def test_add_raw_delete_all(self, memory, test_user):
        """Add multiple with infer=False, then delete_all clears them."""
        for i in range(3):
            mcp_server.execute_tool("add_raw_memory", {
                "content": f"Batch memory {i} for delete_all test.",
                "user_id": test_user,
                
            })
        time.sleep(1)
        mcp_server.execute_tool("delete_all_memories", {"user_id": test_user})
        time.sleep(1)
        result = mcp_server.execute_tool("search_memories", {
            "query": "batch memory delete_all",
            "user_id": test_user,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        assert len(results) == 0, f"Memories remain after delete_all: {len(results)}"

    def test_add_raw_with_metadata(self, memory, test_user):
        """Add with infer=False and metadata, verify it's stored."""
        mcp_server.execute_tool("add_raw_memory", {
            "content": "Important architectural decision: use event sourcing.",
            "user_id": test_user,
            
            "metadata": {"category": "architecture", "priority": "high"},
        })
        time.sleep(1)
        result = mcp_server.execute_tool("get_memories", {
            "user_id": test_user,
            "limit": 10,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        assert len(results) > 0
        combined = json.dumps(results).lower()
        assert "event sourcing" in combined or "architectural" in combined

    def test_add_raw_list_entities(self, memory, test_user):
        """Add with infer=False, then list_entities shows the user."""
        mcp_server.execute_tool("add_raw_memory", {
            "content": "Entity test for raw infer=False pipeline.",
            "user_id": test_user,
            
        })
        time.sleep(1)
        result = mcp_server.execute_tool("list_entities", {})
        assert "entities" in result
        entities = result["entities"]
        assert len(entities) > 0

    def test_add_raw_large_text_chunks(self, memory, test_user):
        """Add large text with infer=False — chunking should work without LLM."""
        # Generate text that will be chunked
        paragraphs = [f"Section {i}. " + "lorem ipsum " * 50 for i in range(20)]
        large_text = "\n\n".join(paragraphs)
        result = mcp_server.execute_tool("add_raw_memory", {
            "content": large_text,
            "user_id": test_user,
            
            "metadata": {"source": "chunk_test"},
        })
        # Should succeed and report chunks
        assert "chunks" in result or "results" in result
        time.sleep(1)
        # Verify at least one chunk is searchable
        search_result = mcp_server.execute_tool("search_memories", {
            "query": "section lorem ipsum",
            "user_id": test_user,
        })
        results = search_result if isinstance(search_result, list) else search_result.get("results", [])
        assert len(results) > 0, "Chunked text should be searchable"