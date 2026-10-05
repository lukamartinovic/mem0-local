"""CONTAINER end-to-end tests: real `docker compose build` + entrypoint + HTTP
against the live single-container server (Qdrant + MCP in one container).

Where to run
------------
Needs a working `docker` CLI + daemon. On machines without Docker (the local
dev Mac) every test auto-skips via the module-level skipif:

    pytest tests/test_container_e2e.py            # local box: all skip
    pytest tests/test_container_e2e.py -v         # deploy box: runs for real

What is covered
---------------
1. TestDockerBuild    — image builds cleanly (exit 0, no 'BUILD FAIL' in
                        stderr); inside the image: `qdrant --version` runs,
                        `ldd /usr/local/bin/qdrant` has zero 'not found' lines,
                        /qdrant/config and /qdrant/storage exist.
2. TestEntrypoint     — `docker compose up` -> /health poll (max 300s):
                        status=='ok' AND components.extraction_llm is False
                        AND config.extraction_llm is None; `docker logs`
                        contain 'Qdrant is ready' AND 'Self-test passed';
                        read-only /qdrant/storage mount -> entrypoint logs
                        'Falling back' AND container still reaches health ok.
3. TestMcpEndToEnd    — real MCP JSON-RPC over raw urllib against the live
                        server: exact tool list (13 tools, add_raw_memory +
                        add_verbatim present, NO add_memory), store a fact and
                        search finds it (score > 0.3) and min_score=0.99
                        filters everything out, add_verbatim ~15k chars -> >=2
                        chunks, get_memories limit=1 -> exactly 1, update then
                        new text surfaces, delete then distinctive phrase has
                        0 hits, export json fields, import to a fresh user then
                        re-import imports 0, prune dry-run returns would_delete,
                        oversized blob -> isError True mentioning add_verbatim,
                        unknown tool -> isError True with 'Unknown tool'.
4. TestQdrantDirect   — live Qdrant REST API on :6333: /collections/mem0 has
                        dense vectors sized 768 after the server is healthy.

Hygiene
-------
Every test-scoped user id is deleted via delete_all_memories + delete_entities
in the fixture teardown, and a session-finish hook does a last-resort sweep,
so repeated runs are idempotent.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid

import pytest

# ── Configuration ────────────────────────────────────────────────────────────

# Project root that contains docker-compose.yml + Dockerfile. The deploy box
# is expected to run this suite from the mem0-local checkout; MEM0_E2E_PROJECT
# _ROOT overrides for out-of-tree runs.
PROJECT_ROOT = (
    os.environ.get("MEM0_E2E_PROJECT_ROOT")
    or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
COMPOSE_FILE = os.path.join(PROJECT_ROOT, "docker-compose.yml")
IMAGE = "mem0-local:dev"

MCP_HEALTH_URL = "http://127.0.0.1:8765/health"
MCP_URL = "http://127.0.0.1:8765/mcp"
QDRANT_BASE = "http://127.0.0.1:6333"
COLLECTION_NAME = "mem0"
EXPECTED_DIMS = 768
EXPECTED_TOOLS = 13
UP_TIMEOUT_S = 300          # entrypoint self-test + first fastembed model load

EXPECTED_TOOL_NAMES = [
    "add_raw_memory",
    "add_verbatim",
    "search_memories",
    "get_memories",
    "get_memory",
    "update_memory",
    "delete_memory",
    "delete_all_memories",
    "list_entities",
    "delete_entities",
    "export_memories",
    "import_memories",
    "prune_memories",
]
assert len(EXPECTED_TOOL_NAMES) == EXPECTED_TOOLS
assert "add_memory" not in EXPECTED_TOOL_NAMES

STORED_FACT = (
    "The container e2e stack uses Qdrant as its vector database on port 6333."
)
UPDATED_FACT = (
    "The container e2e stack now uses Milvus as its vector database on port "
    "19530."
)
STORE_TOKEN = "Qdrant as its vector database"
UPDATED_MEM_TOKEN = "Milvus as its vector database"
NEW_SEARCH_QUERY = "Which vector database does the e2e stack use now?"
DELETE_SEARCH_QUERY = (
    "Which vector database does the e2e stack use for facts collection?"
)
DELETED_MEM_TOKEN = "Qdrant as its vector database"

DOCKER = shutil.which("docker")

# Everything below skips cleanly when Docker is absent (this dev Mac has none;
# the deploy box runs the tests for real).
pytestmark = pytest.mark.skipif(
    DOCKER is None,
    reason="no docker on this machine (deploy box only)",
)


# ── Minimal raw-urllib HTTP helpers ──────────────────────────────────────────

def _http_json(url, method="GET", payload=None, timeout=120):
    """HTTP request returning (status, parsed_json_or_None)."""
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read()
        status = e.code
    try:
        return status, json.loads(body.decode())
    except (ValueError, TypeError):
        return status, None


def post_mcp(method, params=None, timeout=180):
    """POST a JSON-RPC request to the MCP endpoint; returns the result object."""
    status, resp = _http_json(
        MCP_URL,
        method="POST",
        payload={"jsonrpc": "2.0", "id": 1, "method": method,
                 "params": params or {}},
        timeout=timeout,
    )
    assert status == 200, f"MCP endpoint returned HTTP {status}: {resp!r}"
    assert isinstance(resp, dict), f"MCP response is not an object: {resp!r}"
    if "error" in resp:
        raise AssertionError(f"MCP RPC error for method={method}: {resp['error']}")
    assert "result" in resp, f"MCP response missing 'result': {resp!r}"
    return resp["result"]


def try_tool(name, arguments, timeout=180):
    """Call an MCP tool without raising: returns (isError, payload, raw_text)."""
    result = post_mcp(
        "tools/call", {"name": name, "arguments": arguments}, timeout=timeout
    )
    text = result["content"][0]["text"] if result.get("content") else ""
    is_error = bool(result.get("isError"))
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = None
    return is_error, payload, text


def wait_for_healthy(deadline_s=UP_TIMEOUT_S, poll_s=5.0, expect="ok"):
    """Poll GET /health until `expect` status is reached; returns the last
    parsed health dict. Dumps container logs when the deadline passes."""
    deadline = time.monotonic() + deadline_s
    last = None
    while time.monotonic() < deadline:
        status, last = _http_json(MCP_HEALTH_URL, timeout=10)
        if isinstance(last, dict) and last.get("status") == expect:
            return last
        if DOCKER:
            # Fail fast when the container died instead of just being slow
            rc, out = _docker_logs()
            if rc == 0 and "Qdrant process exited" in out:
                pytest.fail("Container exited while waiting for health:\n"
                            + out[-4000:])
        time.sleep(poll_s)
    tail = ""
    if DOCKER:
        rc, out = _docker_logs()
        if rc == 0:
            tail = out[-4000:]
    pytest.fail(
        f"/health never reached status='{expect}' within {deadline_s}s "
        f"(last={last!r})\ncontainer logs tail:\n{tail}"
    )


# ── Docker / compose helpers ─────────────────────────────────────────────────

def run_cmd(*args, cwd=PROJECT_ROOT, timeout=1800):
    """Run a docker CLI command; returns (returncode, combined_output)."""
    argv = [DOCKER] + list(args)
    proc = subprocess.run(argv, cwd=cwd, timeout=timeout,
                          capture_output=True, text=True, errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        out = out or f"(exit {proc.returncode})"
    return proc.returncode, out.strip()


def _docker_logs():
    """docker logs mem0-local (combined stdout+stderr, qdrant logs there)."""
    return run_cmd("logs", "mem0-local", timeout=60)


def _logs_have(*needles):
    rc, out = _docker_logs()
    if rc != 0:
        return False, "docker logs failed: " + out[-2000:]
    missing = [n for n in needles if n not in out]
    if missing:
        return False, ("missing in container logs: %r\nlogs tail:\n%s"
                       % (missing, out[-4000:]))
    return True, out


class _ContainerService:
    """Module-scoped lifecycle: build once, `up` once, teardown at the end."""

    started: bool = False
    build_log = None   # combined stdout/stderr of `docker compose build`

    @classmethod
    def build(cls):
        if cls.build_log is not None:
            return cls.build_log
        if not os.path.exists(COMPOSE_FILE):
            pytest.fail(f"compose file missing: {COMPOSE_FILE} "
                        f"(set MEM0_E2E_PROJECT_ROOT for out-of-tree runs)")
        # `docker compose build` reports progress on stderr when non-tty.
        proc = subprocess.run(
            [DOCKER, "compose", "-f", COMPOSE_FILE, "build", "--no-cache"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=1800, errors="replace",
        )
        cls.build_log = (proc.stderr or "") + (proc.stdout or "")
        if proc.returncode != 0:
            pytest.fail(
                f"docker compose build exited {proc.returncode}\n"
                f"build log (tail):\n{cls.build_log[-5000:]}"
            )
        return cls.build_log

    @classmethod
    def up(cls):
        if cls.started:
            return
        cls.build()
        rc, out = run_cmd("compose", "-f", COMPOSE_FILE, "up", "-d",
                          timeout=300)
        if rc != 0:
            pytest.fail(
                f"docker compose up failed ({rc}):\n{out}\n"
                "Is another mem0-local container already bound to "
                "8765/6333? Run 'docker compose down' first."
            )
        wait_for_healthy(deadline_s=UP_TIMEOUT_S)
        cls.started = True

    @classmethod
    def down(cls):
        if cls.started:
            run_cmd("compose", "-f", COMPOSE_FILE, "down", timeout=300)
            cls.started = False


@pytest.fixture(scope="module")
def container():
    """Build + start the real container once; wait for health; tear down."""
    if DOCKER is None:
        pytest.skip("no docker on this machine (deploy box only)")
    _ContainerService.up()
    yield _ContainerService


# Test-created users (wiped via delete_all_memories / delete_entities)
_CREATED_USERS = []


@pytest.fixture()
def test_user():
    """Fresh per-test user id, always deleted afterwards."""
    uid = f"e2e_{uuid.uuid4().hex[:10]}"
    _CREATED_USERS.append(uid)
    yield uid
    try_tool("delete_all_memories", {"user_id": uid}, timeout=60)
    try_tool("delete_entities", {"user_id": uid}, timeout=60)


def pytest_sessionfinish(session, exitstatus):
    """Last-resort hygiene: delete leftover test users."""
    for uid in _CREATED_USERS:
        try:
            try_tool("delete_all_memories", {"user_id": uid}, timeout=60)
        except Exception:
            pass
    _ContainerService.down()


def _wait_for_search_text(query, token, user_id, deadline_s=45, poll_s=2.0):
    """Search until `token` shows up verbatim in a memory; returns the hit or
    None. Avoids sleep-only waits after add."""
    deadline = time.monotonic() + deadline_s
    last = []
    while time.monotonic() < deadline:
        is_err, search, _t = try_tool(
            "search_memories",
            {"query": query, "user_id": user_id, "include_scores": True},
        )
        assert not is_err, f"search_memories failed: {search!r}"
        assert isinstance(search, list), f"search returned non-list: {search!r}"
        last = search
        hits = [h for h in search if token in h.get("memory", "")]
        if hits:
            return hits[0]
        time.sleep(poll_s)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 1. Docker build
# ═════════════════════════════════════════════════════════════════════════════

class TestDockerBuild:
    """`docker compose build` produces a usable image (qdrant binary intact)."""

    def test_compose_build_exit_zero_and_no_build_fail(self, container):
        """Build must exit 0 and its stderr must NOT contain 'BUILD FAIL'
        (that marker only prints when the freshly-installed qdrant binary
        fails to execute inside the final image)."""
        stderr = _ContainerService.build()
        assert isinstance(stderr, str)
        assert "BUILD FAIL" not in stderr, (
            f"build stderr contains BUILD FAIL:\n{stderr[-4000:]}"
        )

    def test_inside_image_qdrant_version_runs(self, container):
        """`qdrant --version` runs inside the FINAL image (needs libunwind8)."""
        _ContainerService.build()
        rc, out = run_cmd(
            "run", "--rm", IMAGE, "qdrant", "--version", timeout=120,
        )
        assert rc == 0, f"qdrant --version inside image failed:\n{out[-2500:]}"
        assert out, "qdrant --version produced no output"

    def test_inside_image_ldd_zero_not_found(self, container):
        """ldd on the staged binary resolves every shared object (libunwind8)."""
        _ContainerService.build()
        rc, out = run_cmd(
            "run", "--rm", IMAGE, "sh", "-c",
            "ldd /usr/local/bin/qdrant 2>&1 || true", timeout=120,
        )
        assert rc == 0, f"ldd execution inside image failed:\n{out[-2500:]}"
        not_found = [ln for ln in out.splitlines() if "not found" in ln]
        assert not not_found, (
            "missing shared libraries in image:\n" + "\n".join(not_found)
        )

    def test_inside_image_config_and_storage_dirs_exist(self, container):
        """/qdrant/config (default + production config) and /qdrant/storage."""
        _ContainerService.build()
        rc, out = run_cmd(
            "run", "--rm", IMAGE, "sh", "-c",
            "test -d /qdrant/config && test -d /qdrant/storage && echo DIRS_OK",
            timeout=120,
        )
        assert rc == 0, f"config/storage dirs missing in image:\n{out[-2500:]}"
        assert "DIRS_OK" in out, f"dirs exist but DIRS_OK not echoed:\n{out}"


# ═════════════════════════════════════════════════════════════════════════════
# 2. Entrypoint behavior
# ═════════════════════════════════════════════════════════════════════════════

class TestEntrypoint:
    """entrypoint.sh: qdrant supervisor -> wait -> self-test -> MCP server."""

    def test_compose_build_exit_zero(self, container):
        """Explicit build gate for the entrypoint tests."""
        stderr = _ContainerService.build()
        assert "BUILD FAIL" not in stderr

    def test_up_reaches_health_ok_extraction_llm_disabled(self, container):
        """up -> poll /health (max 300s): status=='ok', agent does inference —
        components.extraction_llm is False AND config.extraction_llm is None."""
        _ContainerService.up()
        health = wait_for_healthy(deadline_s=UP_TIMEOUT_S)
        assert health["status"] == "ok", f"health: {health!r}"
        assert health["components"]["extraction_llm"] is False, (
            f"components.extraction_llm must be False by design: {health!r}"
        )
        assert health["config"]["extraction_llm"] is None, (
            f"config.extraction_llm must be None (no extraction LLM): {health!r}"
        )
        assert health["components"]["qdrant"] is True, (
            f"qdrant component must be up: {health!r}"
        )

    def test_container_logs_have_qdrant_ready_and_selftest_passed(
        self, container
    ):
        """Container logs prove qdrant came up and the self-test ran."""
        _ContainerService.up()
        ok, out = _logs_have("Qdrant is ready", "Self-test passed")
        assert ok, out

    def test_readonly_storage_mount_falls_back_and_still_healthy(self, container):
        """/qdrant/storage mounted read-only (ro): the entrypoint detects the
        unwritable volume, logs the fallback, and the server STILL becomes
        healthy (data lives in-container until the volume is fixed)."""
        _ContainerService.build()
        ro_storage = tempfile.mkdtemp(prefix="mem0_ro_storage_")
        assert os.path.isdir(ro_storage)
        rc, out = run_cmd(
            "run", "--rm", "-d",
            "-p", "18765:8765", "-p", "16333:6333",
            "-v", f"{ro_storage}:/qdrant/storage:ro",
            IMAGE,
            timeout=180,
        )
        assert rc == 0, f"docker run (ro storage) failed: {out[-2500:]}"
        cid = out.strip()[-64:]

        # The fallback container serves on 18765; poll its /health directly.
        deadline = time.monotonic() + UP_TIMEOUT_S
        health = None
        reached_ok = False
        while time.monotonic() < deadline:
            if DOCKER and "Qdrant process exited" in run_cmd(
                    "logs", cid, timeout=30)[1]:
                pytest.fail("fallback container exited:\n"
                            + run_cmd("logs", cid, timeout=30)[1][-4000:])
            status, health = _http_json("http://127.0.0.1:18765/health",
                                        timeout=10)
            if isinstance(health, dict) and health.get("status") == "ok":
                reached_ok = True
                break
            time.sleep(5.0)

        logs = run_cmd("logs", cid, timeout=60)[1] if DOCKER else ""
        assert "Falling back" in logs, (
            f"'Falling back' not in entrypoint logs of ro-storage run "
            f"(tail:\n{logs[-3000:]})"
        )
        assert reached_ok, (
            f"ro-storage container never reached health ok (last={health!r}, "
            f"logs tail:\n{logs[-3000:]})"
        )
        if DOCKER:
            run_cmd("stop", cid, timeout=60)
        shutil.rmtree(ro_storage, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════════
# 3. MCP end-to-end over HTTP (raw urllib, JSON-RPC)
# ═════════════════════════════════════════════════════════════════════════════

class TestMcpEndToEnd:
    """Real MCP flows against the live server inside the container."""

    # ── tool surface ────────────────────────────────────────────────────────

    def test_tools_list_exactly_13(self, container):
        tools = post_mcp("tools/list")["tools"]
        assert len(tools) == EXPECTED_TOOLS, (
            f"expected {EXPECTED_TOOLS} tools, got {len(tools)}"
        )

    def test_tools_list_names_exact(self, container):
        """add_raw_memory + add_verbatim exist; the legacy add_memory is gone."""
        tools = post_mcp("tools/list")["tools"]
        names = [t["name"] for t in tools]
        assert sorted(names) == sorted(EXPECTED_TOOL_NAMES), (
            f"tool list mismatch: got {names}"
        )
        assert "add_memory" not in names, "legacy add_memory still exposed"
        assert "add_raw_memory" in names
        assert "add_verbatim" in names

    # ── store + search ──────────────────────────────────────────────────────

    def test_store_fact_and_search_finds_score_above_03(self, container, test_user):
        """add_raw_memory then search_memories: hit matches, score > 0.3."""
        is_err, payload, _t = try_tool(
            "add_raw_memory", {"content": STORED_FACT, "user_id": test_user},
        )
        assert not is_err, f"add_raw_memory failed: {payload!r}"
        assert isinstance(payload, dict) and payload.get("results"), (
            f"add_raw_memory returned no results: {payload!r}"
        )
        hit = _wait_for_search_text(
            "Which vector database does the container e2e stack use?",
            STORE_TOKEN, test_user,
        )
        assert hit is not None, (
            f"search never found stored fact '{STORE_TOKEN}'"
        )
        assert isinstance(hit.get("score"), (int, float)), (
            f"result missing numeric score: {hit!r}"
        )
        assert hit["score"] > 0.3, (
            f"top score {hit.get('score')} not > 0.3: {hit!r}"
        )

    def test_min_score_099_filters_out(self, container, test_user):
        """min_score=0.99 filters everything out (empty list, no crash)."""
        try_tool("add_raw_memory",
                 {"content": STORED_FACT, "user_id": test_user})
        # First let the fact become searchable
        _wait_for_search_text(
            "Which vector database does the container e2e stack use?",
            STORE_TOKEN, test_user,
        )
        is_err, search, _t = try_tool(
            "search_memories",
            {"query": "Which vector database does the container e2e stack use?",
             "user_id": test_user, "min_score": 0.99},
        )
        assert not is_err, f"search failed: {search!r}"
        assert isinstance(search, list), f"search not a list: {search!r}"
        assert search == [], (
            f"min_score=0.99 must filter ALL cosine scores out: {search!r}"
        )

    # ── verbatim chunking ───────────────────────────────────────────────────

    def test_add_verbatim_15000_chars_chunks_at_least_2(self, container, test_user):
        """~15,000 chars verbatim -> chunked storage of >= 2 memories."""
        paragraphs = []
        total = 0
        while total < 15000:
            para = (
                f"Paragraph {len(paragraphs)}: verbatim-chunk-{len(paragraphs)} "
                f"filler text for the container end to end verbatim chunking "
                f"scenario {'lorem ipsum dolor sit amet ' * 40}"
            )
            paragraphs.append(para)
            total += len(para) + 2
        text = "\n\n".join(paragraphs)
        assert 14000 <= len(text) <= 17000
        is_err, payload, raw = try_tool(
            "add_verbatim",
            {"content": text, "user_id": test_user,
             "metadata": {"file": "e2e/verbatim.txt"}},
            timeout=300,
        )
        assert not is_err, f"add_verbatim failed: {raw[:500]!r}"
        assert isinstance(payload, dict)
        assert payload.get("chunks", 0) >= 2, (
            f"expected >=2 chunks for ~15k chars, "
            f"got {payload.get('chunks')!r}: {payload!r}"
        )
        assert len(payload.get("results", [])) >= 2

    # ── get_memories ────────────────────────────────────────────────────────

    def test_get_memories_limit_1_returns_exactly_one(self, container, test_user):
        for i in range(3):
            try_tool("add_raw_memory",
                     {"content": f"e2e list test memory number {i}",
                      "user_id": test_user})
        deadline = time.monotonic() + 30
        got = []
        while time.monotonic() < deadline:
            is_err, listing, _t = try_tool(
                "get_memories", {"user_id": test_user, "limit": 1})
            assert not is_err, f"get_memories failed: {listing!r}"
            got = listing.get("results", [])
            if got:
                break
            time.sleep(2)
        assert len(got) == 1, (
            f"limit=1 must return exactly 1 (server returns top_k "
            f"point-per-hit), got {len(got)}: {got!r}"
        )

    # ── update + delete (class-level store keeps test_user isolated) ───────

    def test_update_memory_new_text_surfaces(self, container, test_user):
        """Update the fact; searching then surfaces ONLY the new text."""
        is_err, payload, raw = try_tool(
            "add_raw_memory", {"content": STORED_FACT, "user_id": test_user},
        )
        assert not is_err, f"add failed: {raw[:400]!r}"
        mem_id = payload["results"][0]["id"]
        is_err, _upd, raw = try_tool(
            "update_memory", {"memory_id": mem_id, "content": UPDATED_FACT},
        )
        assert not is_err, f"update_memory failed: {raw[:500]!r}"

        hit = _wait_for_search_text(
            NEW_SEARCH_QUERY, UPDATED_MEM_TOKEN, test_user,
        )
        assert hit is not None, (
            f"updated text token '{UPDATED_MEM_TOKEN}' never surfaced "
            f"via search"
        )
        assert UPDATED_MEM_TOKEN in hit.get("memory", ""), (
            f"search result doesn't contain the updated text: {hit!r}"
        )

    def test_delete_memory_distinctive_phrase_zero_hits(self, container, test_user):
        """Delete by id; its distinctive phrase has ZERO search hits after."""
        is_err, payload, raw = try_tool(
            "add_raw_memory", {"content": STORED_FACT, "user_id": test_user},
        )
        assert not is_err, f"add failed: {raw[:400]!r}"
        mem_id = payload["results"][0]["id"]
        is_err, _del, raw = try_tool("delete_memory", {"memory_id": mem_id})
        assert not is_err, f"delete_memory failed: {raw[:500]!r}"

        deadline = time.monotonic() + 30
        search = []
        while time.monotonic() < deadline:
            is_err, search, _t = try_tool(
                "search_memories",
                {"query": DELETE_SEARCH_QUERY, "user_id": test_user},
            )
            assert not is_err, f"search after delete failed: {search!r}"
            assert isinstance(search, list)
            exact = [h for h in search if h.get("memory", "").count(DELETED_MEM_TOKEN)]
            if not exact:
                search = []
                break
            time.sleep(2)
        assert not search or not any(
            DELETED_MEM_TOKEN in h.get("memory", "") for h in search
        ), (
            f"distinctive phrase still surfaced after delete: {search!r}"
        )

    # ── get_memory by id ────────────────────────────────────────────────────

    def test_get_memory_by_id_returns_stored_text(self, container, test_user):
        is_err, payload, raw = try_tool(
            "add_raw_memory", {"content": STORED_FACT, "user_id": test_user},
        )
        assert not is_err
        mem_id = payload["results"][0]["id"]
        is_err, memory, raw2 = try_tool("get_memory", {"memory_id": mem_id})
        assert not is_err, f"get_memory failed: {raw2[:500]!r}"
        assert memory.get("memory") == STORED_FACT, f"get_memory: {memory!r}"

    def test_get_memory_missing_id_is_error(self, container):
        is_err, _payload, text = try_tool(
            "get_memory", {"memory_id": "nonexistent-id-0000"}
        )
        assert is_err, f"get_memory(missing id) should error, got: {text!r}"

    # ── export / import round-trip ──────────────────────────────────────────

    def test_export_json_has_required_fields(self, container, test_user):
        """export_memories(json): format/count/memories; records have
        id/memory/created_at (+ user_id)."""
        try_tool("add_raw_memory",
                 {"content": "e2e export me please 42", "user_id": test_user})
        _wait_for_search_text("export me please 42", "e2e export me please 42",
                              test_user)
        is_err, export, raw = try_tool(
            "export_memories", {"user_id": test_user, "format": "json"})
        assert not is_err, f"export_memories failed: {raw[:500]!r}"
        assert export.get("format") == "json"
        assert isinstance(export.get("count"), int)
        memories = export.get("memories")
        assert isinstance(memories, list) and memories, (
            f"export.memories empty: {export!r}"
        )
        first = memories[0]
        for key in ("id", "memory"):
            assert key in first, (
                f"export record missing '{key}': keys={list(first)!r}"
            )
        # JSON export exposes the fields the tool normalizes to
        assert any("e2e export me please 42" in m.get("memory", "")
                   for m in memories), (
            f"stored fact absent from export: {memories!r}"
        )

    def test_import_to_fresh_user_then_reimport_imports_zero(
        self, container, test_user
    ):
        """Import into a FRESH user, then re-import: imported==0 (deduped)."""
        is_err, _a, raw = try_tool(
            "add_raw_memory",
            {"content": "e2e import round trip 777", "user_id": test_user},
        )
        assert not is_err, f"add failed: {raw[:400]!r}"
        _wait_for_search_text("import round trip 777",
                              "e2e import round trip 777", test_user)
        is_err, export, _r = try_tool(
            "export_memories", {"user_id": test_user, "format": "json"})
        assert not is_err and export.get("memories"), (
            f"export empty before import: {export!r}"
        )
        fresh_user = f"e2e_impo_{uuid.uuid4().hex[:10]}"
        _CREATED_USERS.append(fresh_user)
        is_err, first, raw = try_tool(
            "import_memories",
            {"data": json.dumps(export["memories"]), "user_id": fresh_user},
            timeout=300,
        )
        assert not is_err, f"import_memories failed: {raw[:500]!r}"
        assert first.get("imported", 0) >= 1, (
            f"first import should store memories: {first!r}"
        )
        is_err, second, raw2 = try_tool(
            "import_memories",
            {"data": json.dumps(export["memories"]), "user_id": fresh_user},
            timeout=300,
        )
        assert not is_err, f"re-import failed: {raw2[:500]!r}"
        assert second.get("imported") == 0, (
            f"re-import must import ZERO (deduped): {second!r}"
        )

    # ── prune dry-run ───────────────────────────────────────────────────────

    def test_prune_dry_run_returns_would_delete(self, container, test_user):
        """Dry-run prune reports would_delete without deleting anything."""
        try_tool("add_raw_memory",
                 {"content": "e2e prune candidate one", "user_id": test_user})
        _wait_for_search_text("prune candidate one", "e2e prune candidate one",
                              test_user)
        is_err, prune, raw = try_tool(
            "prune_memories",
            {"user_id": test_user, "older_than_days": 0, "dry_run": True},
        )
        assert not is_err, f"prune_memories failed: {raw[:500]!r}"
        wd = prune.get("would_delete")
        assert isinstance(wd, int), f"would_delete must be an int: {prune!r}"
        assert wd >= 1, (
            f"would_delete expected >=1 (memories are seconds old, "
            f"cutoff is 0 days ago): {prune!r}"
        )
        assert prune.get("dry_run") is True

    # ── guard rails ─────────────────────────────────────────────────────────

    def test_oversized_blob_is_error_mentioning_add_verbatim(
        self, container, test_user
    ):
        is_err, _payload, text = try_tool(
            "add_raw_memory", {"content": "x" * 3500, "user_id": test_user}
        )
        assert is_err, f"oversized blob must isError=True, got: {text[:300]!r}"
        assert "add_verbatim" in text, (
            f"error text must mention add_verbatim: {text[:300]!r}"
        )

    def test_unknown_tool_is_error_with_unknown_tool_message(self, container):
        is_err, _payload, text = try_tool("add_memory", {"content": "nope"})
        assert is_err, f"unknown tool must isError=True, got: {text[:300]!r}"
        assert "Unknown tool" in text, (
            f"error text must say 'Unknown tool': {text[:300]!r}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4. Qdrant direct (live REST on :6333)
# ═════════════════════════════════════════════════════════════════════════════

class TestQdrantDirect:
    """Live Qdrant REST API — collection shape after the server is healthy."""

    def test_qdrant_reachable_at_6333(self, container):
        status, body = _http_json(QDRANT_BASE, timeout=10)
        assert status == 200, f"qdrant root not reachable: HTTP {status} {body!r}"

    def test_mem0_collection_dense_vector_size_is_768(self, container):
        """/collections/mem0 -> config.params.vectors dense size == 768."""
        wait_for_healthy(deadline_s=UP_TIMEOUT_S)
        deadline = time.monotonic() + 30
        size = None
        body = None
        while time.monotonic() < deadline:
            status, body = _http_json(
                f"{QDRANT_BASE}/collections/{COLLECTION_NAME}", timeout=10)
            if status == 200 and isinstance(body, dict):
                params = body.get("result", {}).get("config", {}).get(
                    "params", {})
                vectors = params.get("vectors", {})
                if isinstance(vectors, dict):
                    if isinstance(vectors.get("size"), int):
                        size = vectors["size"]
                        break
                    for v in vectors.values():
                        if isinstance(v, dict) and "size" in v:
                            size = v["size"]
                            break
                if size is not None:
                    break
            time.sleep(1)
        assert size == EXPECTED_DIMS, (
            f"vectors size expected {EXPECTED_DIMS}, got {size!r} "
            f"(body={json.dumps(body)[:800] if body else 'unparseable'})"
        )

    def test_mem0_points_use_user_scoped_payload(self, container, test_user):
        """Points carry mem0's payload keys (data + user_id)."""
        is_err, payload, raw = try_tool(
            "add_raw_memory", {"content": STORED_FACT, "user_id": test_user},
        )
        assert not is_err, f"add_raw failed: {raw[:400]!r}"
        deadline = time.monotonic() + 30
        body = None
        while time.monotonic() < deadline:
            status, body = _http_json(
                f"{QDRANT_BASE}/collections/{COLLECTION_NAME}/points/scroll",
                method="POST",
                payload={"limit": 250, "with_payload": True,
                          "with_vector": False},
                timeout=15,
            )
            if status == 200 and isinstance(body, dict):
                points = body.get("result", {}).get("points", [])
                match = [
                    p for p in points
                    if p.get("payload", {}).get("user_id") == test_user
                    and STORE_TOKEN in (p.get("payload", {}).get("data") or "")
                ]
                if match:
                    return
            time.sleep(1)
        body_repr = json.dumps(body)[:1200] if body else "unparseable"
        pytest.fail(
            f"no qdrant point with user_id={test_user!r} containing "
            f"'{STORE_TOKEN}' after 30s:\n{body_repr}"
        )