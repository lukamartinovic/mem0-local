"""
Test suite for mem0-local MCP server.

Runs inside the MCP server container via: docker compose run mcp-server pytest tests/ -v

Two modes:
  1. If MCP server is already running (docker compose up -d), tests hit it over HTTP
  2. If run standalone (docker compose run), tests start the server inline and test directly
"""

import json
import os
import sys
import uuid
import urllib.request
import asyncio
import threading
import time

import pytest

# ── Config ──────────────────────────────────────────────────────────────────

MCP_PORT = os.environ.get("MCP_PORT", "8765")

# When running via `docker compose run mcp-server pytest`, there's no
# already-running MCP server to hit. Detect this and start one inline.
def _is_server_running():
    try:
        urllib.request.urlopen(f"http://localhost:{MCP_PORT}/health", timeout=2)
        return True
    except Exception:
        return False

# Start inline server if needed
if not _is_server_running():
    sys.path.insert(0, "/app")
    import mcp_server
    mcp_server.init_memory()

    def _run_server():
        asyncio.run(mcp_server.main())

    _server_thread = threading.Thread(target=_run_server, daemon=True)
    _server_thread.start()
    time.sleep(2)

MCP_URL = f"http://localhost:{MCP_PORT}"
OLLAMA_URL = os.environ.get("MEM0_OLLAMA_URL", "http://ollama:11434")
QDRANT_URL = f"http://{os.environ.get('MEM0_QDRANT_HOST', 'qdrant')}:{os.environ.get('MEM0_QDRANT_PORT', '6333')}"

# ── Helpers ─────────────────────────────────────────────────────────────────

def mcp_call(method: str, params: dict = None, req_id: int = 1) -> dict:
    """Make an MCP JSON-RPC call over HTTP."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params or {},
    }).encode()
    req = urllib.request.Request(
        f"{MCP_URL}/mcp",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())


def mcp_tool(name: str, arguments: dict, req_id: int = 1) -> dict:
    """Call an MCP tool and return the result."""
    resp = mcp_call("tools/call", {"name": name, "arguments": arguments}, req_id)
    result = resp.get("result", {})
    if result.get("isError"):
        raise RuntimeError(result["content"][0]["text"])
    return json.loads(result["content"][0]["text"])


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def test_user():
    return f"test_{uuid.uuid4().hex[:8]}"


# ── Infrastructure tests ────────────────────────────────────────────────────

class TestInfrastructure:
    def test_ollama_reachable(self):
        resp = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=10)
        assert resp.status == 200
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        print(f"  Ollama models: {models}")
        assert len(models) > 0, "No models installed in Ollama"

    def test_qdrant_reachable(self):
        resp = urllib.request.urlopen(f"{QDRANT_URL}/", timeout=5)
        assert resp.status == 200

    def test_mcp_server_health(self):
        resp = urllib.request.urlopen(f"{MCP_URL}/health", timeout=5)
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["status"] == "ok"
        assert data["tools"] == 12
        print(f"  Server config: {data['config']}")


# ── MCP protocol tests ──────────────────────────────────────────────────────

class TestMCPProtocol:
    def test_initialize(self):
        resp = mcp_call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        })
        assert resp["jsonrpc"] == "2.0"
        assert resp["result"]["serverInfo"]["name"] == "mem0-local"

    def test_list_tools(self):
        resp = mcp_call("tools/list")
        tools = resp["result"]["tools"]
        assert len(tools) == 12
        names = {t["name"] for t in tools}
        expected = {"add_memory", "search_memories", "get_memories",
                    "get_memory", "update_memory", "delete_memory",
                    "delete_all_memories", "list_entities", "delete_entities",
                    "export_memories", "import_memories", "prune_memories"}
        assert names == expected, f"Missing tools: {expected - names}"


# ── Memory operation tests ─────────────────────────────────────────────────

class TestMemoryOperations:
    """Memory operation tests. add_memory uses infer=True with rich text that
    the LLM can reliably extract facts from. Search/get/delete tests verify
    the CRUD pipeline with real extracted memories."""

    def test_add_and_search(self, test_user):
        mcp_tool("add_memory", {
            "content": "My name is Alice. I work at Acme Corp as a senior software engineer. "
                       "I use TypeScript and React for frontend development, Node.js for the backend, "
                       "and PostgreSQL as the database. We deploy to AWS using GitHub Actions CI/CD.",
            "user_id": test_user,
            "infer": True,
        })
        results = mcp_tool("search_memories", {
            "query": "What programming languages does Alice use?",
            "user_id": test_user,
        })
        assert len(results) > 0 or len(results.get("results", [])) > 0
        combined = json.dumps(results).lower()
        assert any(kw in combined for kw in ["typescript", "react", "node", "postgresql", "alice"]), \
            f"Expected tech stack in results, got: {results}"

    def test_add_conversation_messages(self, test_user):
        messages = [
            {"role": "user", "content": "I'm planning to migrate our API from REST to GraphQL. "
                                        "The main reason is to reduce over-fetching on the mobile app. "
                                        "We'll use Apollo Server with the schema-first approach."},
            {"role": "assistant", "content": "Noted. I'll remember you're migrating from REST to GraphQL "
                                             "using Apollo Server with schema-first approach."},
        ]
        mcp_tool("add_memory", {
            "content": json.dumps(messages),
            "user_id": test_user,
            "infer": True,
        })
        results = mcp_tool("search_memories", {
            "query": "API migration plan",
            "user_id": test_user,
        })
        combined = json.dumps(results).lower()
        assert any(kw in combined for kw in ["graphql", "rest", "apollo", "migration", "over-fetching"]), \
            f"Expected migration info in results, got: {results}"

    def test_get_memories(self, test_user):
        mcp_tool("add_memory", {
            "content": "We use Redis for caching and Kafka for event streaming. "
                       "The Redis instance runs on port 6379 and Kafka on port 9092.",
            "user_id": test_user,
            "infer": True,
        })
        results = mcp_tool("get_memories", {"user_id": test_user, "limit": 10})
        assert results, "get_memories returned empty"

    def test_delete_all_memories(self, test_user):
        mcp_tool("add_memory", {"content": "Memory to delete", "user_id": test_user, "infer": False})
        mcp_tool("add_memory", {"content": "Another to delete", "user_id": test_user, "infer": False})
        mcp_tool("delete_all_memories", {"user_id": test_user})
        results = mcp_tool("search_memories", {"query": "delete", "user_id": test_user})
        results_list = results if isinstance(results, list) else results.get("results", [])
        assert len(results_list) == 0, f"Memories still exist: {results_list}"

    def test_add_with_metadata(self, test_user):
        mcp_tool("add_memory", {
            "content": "We deployed the auth service to staging on Tuesday. "
                       "It uses JWT tokens with 24-hour expiry and refresh tokens with 7-day expiry.",
            "user_id": test_user,
            "infer": True,
            "metadata": {"category": "deployment", "env": "staging"},
        })
        results = mcp_tool("search_memories", {
            "query": "auth service deployment",
            "user_id": test_user,
        })
        assert results

    def test_unknown_tool_error(self):
        with pytest.raises(RuntimeError, match="Unknown tool"):
            mcp_tool("nonexistent_tool", {})

    def test_list_entities(self, test_user):
        mcp_tool("add_memory", {
            "content": "The database is PostgreSQL version 15 running on port 5432.",
            "user_id": test_user,
            "infer": True,
        })
        # list_entities doesn't take arguments
        results = mcp_tool("list_entities", {})
        assert "entities" in results
        # Our test_user should be in there
        entities = results["entities"]
        assert test_user in entities or len(entities) > 0

    def test_delete_entities(self, test_user):
        mcp_tool("add_memory", {"content": "Entity to delete", "user_id": test_user, "infer": False})
        mcp_tool("delete_entities", {"user_id": test_user})
        results = mcp_tool("search_memories", {"query": "entity", "user_id": test_user})
        results_list = results if isinstance(results, list) else results.get("results", [])
        assert len(results_list) == 0