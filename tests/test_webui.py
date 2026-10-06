"""
Web UI endpoint tests: the browser-facing API must behave exactly like the
MCP tools it mirrors.

Covers `/` (page contains working controls), `/api/import` (stores, dedupes,
validates) and `/api/memories` (the listing the UI renders). Import reuses the
same helper as the `import_memories` MCP tool, so results must agree.

Runs the real HTTP handler against a live in-process server with an in-proc
Qdrant (local file mode). Skips automatically when mem0/fastembed are absent.
"""

import asyncio
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

pytest.importorskip("mem0", reason="mem0 not installed")
pytest.importorskip("fastembed", reason="fastembed not installed")

import mcp_server  # noqa: E402

PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Boot the real handler on a scratch Qdrant with the tools wired up."""
    from qdrant_client import QdrantClient
    from mem0 import Memory

    path = str(tmp_path_factory.mktemp("qdrant_webui"))
    cfg = json.loads(json.dumps(mcp_server.CONFIG))
    cfg["vector_store"]["config"] = {
        "client": QdrantClient(path=path),
        "collection_name": "mem0",
        "embedding_model_dims": 768,
    }
    m = Memory.from_config(cfg)
    m.llm = mcp_server.NoExtractionLLM()
    mcp_server._memory = m
    mcp_server._init_status = "ready"

    async def serve():
        srv = await asyncio.start_server(mcp_server.http_handler, "127.0.0.1", PORT)
        async with srv:
            await srv.serve_forever()

    t = threading.Thread(target=lambda: asyncio.run(serve()), daemon=True)
    t.start()
    for _ in range(50):
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=2)
            break
        except Exception:
            time.sleep(0.2)
    else:
        pytest.skip("in-process server did not start")

    yield BASE


def _get(path):
    return urllib.request.urlopen(f"{BASE}{path}", timeout=15).read().decode()


def _post(path, payload):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode())


@pytest.fixture
def user():
    uid = f"webui_{uuid.uuid4().hex[:8]}"
    yield uid
    try:
        mcp_server.execute_tool("delete_all_memories", {"user_id": uid})
    except Exception:
        pass


# ── The page itself ─────────────────────────────────────────────────────────

class TestPageServesControls:
    def test_page_loads(self, server):
        html = _get("/")
        assert "<title>mem0-local</title>" in html
        # and it is served with an HTML content type, not as JSON
        resp = urllib.request.urlopen(f"{BASE}/", timeout=15)
        assert "text/html" in (resp.headers.get("Content-Type") or ""), resp.headers

    def test_page_has_export_controls(self, server):
        html = _get("/")
        assert "Export memories" in html
        assert "doExport" in html
        for fmt in ("'json'", "'csv'"):
            assert fmt in html, f"missing export format {fmt}"

    def test_page_has_import_controls(self, server):
        html = _get("/")
        assert "Import memories" in html
        assert "doImport" in html
        assert "/api/import" in html
        assert 'type="file"' in html, "import needs a file picker"
        assert "Preview only" in html, "a dry-run affordance must exist"

    def test_page_requires_entity_for_import(self, server):
        html = _get("/")
        assert 'id="import-entity"' in html

    def test_page_parses_export_cli_and_jsonl(self, server):
        html = _get("/")
        # the parser must accept the export envelope and JSON Lines, not only a bare array
        assert "data.memories" in html
        assert "JSON Lines" in html or "jsonl" in html.lower()


# ── /api/import ─────────────────────────────────────────────────────────────

class TestImportEndpoint:
    def test_imports_and_dedupes_within_batch(self, server, user):
        r = _post("/api/import", {"user_id": user, "memories": [
            {"memory": "Web UI import fact one about pipeline deployment."},
            {"memory": "Web UI import fact two about port allocation."},
            {"memory": "Web UI import fact one about pipeline deployment."},
        ]})
        assert r["imported"] == 2, r
        assert r["skipped"] == 1, r
        assert r["user_id"] == user

    def test_reimport_is_idempotent(self, server, user):
        payload = {"user_id": user, "memories": [
            {"memory": "Idempotency probe for the web import endpoint."}]}
        first = _post("/api/import", payload)
        second = _post("/api/import", payload)
        assert first["imported"] == 1
        assert second["imported"] == 0 and second["skipped"] == 1, second

    def test_metadata_is_preserved(self, server, user):
        _post("/api/import", {"user_id": user, "memories": [
            {"memory": "Metadata probe row for the web UI import.",
             "metadata": {"source": "webui", "tag": "probe"}}]})
        r = mcp_server.execute_tool("get_memories", {"user_id": user, "limit": 10})
        rows = r if isinstance(r, list) else r.get("results", [])
        found = [x for x in rows if "Metadata probe row" in (x.get("memory") or "")]
        assert found, rows
        meta = found[0].get("metadata") or {}
        assert meta.get("source") == "webui", meta

    def test_missing_user_id_rejected(self, server):
        r = _post("/api/import", {"memories": [{"memory": "x"}]})
        assert "error" in r and "user_id" in r["error"], r

    def test_empty_list_rejected(self, server, user):
        r = _post("/api/import", {"user_id": user, "memories": []})
        assert "error" in r, r

    def test_non_dict_items_are_counted_as_failures(self, server, user):
        """Same contract as the MCP tool: garbage is reported, not silently kept."""
        r = _post("/api/import", {"user_id": user, "memories": ["a bare string", {"memory": ""}]})
        assert r["failed"] == 2, r
        assert r["imported"] == 0, r
        assert r["errors"], "failures must be explained"

    def test_bad_json_body_rejected(self, server):
        req = urllib.request.Request(
            f"{BASE}/api/import", data=b"{not json",
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=15)
            raise AssertionError("invalid JSON should not succeed")
        except urllib.error.HTTPError as e:
            assert e.code == 400


# ── /api/memories (what the UI renders) ─────────────────────────────────────

class TestListingEndpoint:
    def test_imported_rows_appear_via_mcp_tool(self, server, user):
        """The UI's listing scrolls Qdrant directly (needs the real HTTP Qdrant),
        so verify the data through the tool path that shares the same store."""
        _post("/api/import", {"user_id": user, "memories": [
            {"memory": "Listing probe row stored through the web endpoint."}]})
        r = mcp_server.execute_tool("get_memories", {"user_id": user, "limit": 10})
        rows = r if isinstance(r, list) else r.get("results", [])
        assert any("Listing probe row" in (x.get("memory") or "") for x in rows), rows

    def test_listing_endpoint_responds_json(self, server):
        """Even without a remote Qdrant it must return JSON, never a 500 page."""
        body = _get("/api/memories")
        parsed = json.loads(body)
        assert "memories" in parsed and "entities" in parsed
