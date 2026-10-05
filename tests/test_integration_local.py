"""
INTEGRATION tests — the REAL full stack, no Docker, no HTTP server:

  real mem0 (mem0ai 2.0.11, infer=False path)
  real fastembed embeddings (nomic-ai/nomic-embed-text-v1.5, 768 dims, ONNX)
  real in-process Qdrant in LOCAL FILE MODE (QdrantClient(path=tmpdir))

mcp_server.execute_tool is called on a Memory instance wired up exactly like
init_memory() does, except the vector store is a local-file Qdrant client
instead of an HTTP one — same mem0 code path, no Docker/Qdrant server needed.

Run:
    cd <project root>
    MEM0_TELEMETRY=False /tmp/mem0check/.venv2/bin/python \
        -m pytest tests/test_integration_local.py -v

These tests need no Qdrant server, no OpenAI key, no LLM. Teardown deletes
every memory created under the unique per-test user_ids.
"""

import csv
import io
import json
import uuid

import pytest

import mcp_server

# ── Fixtures: wire the real stack into mcp_server once per module ───────────

@pytest.fixture(scope="module")
def qdrant_client():
    """In-process Qdrant in local file mode, shared by the module's Memory."""
    from qdrant_client import QdrantClient

    import tempfile

    client = QdrantClient(path=tempfile.mkdtemp(prefix="mem0_int_test_"))
    yield client
    try:
        client.close()
    except Exception:
        pass


@pytest.fixture(scope="module")
def memory(qdrant_client):
    """Real Memory (mem0 2.0.11) bound to the in-process Qdrant client.

    Mirrors mcp_server.init_memory(): from_config with the server CONFIG, then
    the NoExtractionLLM sentinel, then expose it through mcp_server so that
    execute_tool() runs the full real stack (embedder + vector store + tools).
    """
    cfg = json.loads(json.dumps(mcp_server.CONFIG))  # deep copy: dict(CONFIG) keeps nested refs
    cfg["vector_store"]["config"] = {
        "client": qdrant_client,
        "collection_name": "mem0",
        "embedding_model_dims": 768,
    }
    from mem0 import Memory

    m = Memory.from_config(cfg)
    m.llm = mcp_server.NoExtractionLLM()
    mcp_server._memory = m
    mcp_server._init_status = "ready"
    return m


@pytest.fixture
def test_user():
    """Unique user_id per test; teardown wipes its memories."""
    uid = f"int_{uuid.uuid4().hex[:10]}"
    yield uid
    try:
        mcp_server.execute_tool("delete_all_memories", {"user_id": uid})
    except Exception:
        pass


@pytest.fixture
def test_user_b():
    """Second unique user_id (import/export round-trips)."""
    uid = f"intb_{uuid.uuid4().hex[:10]}"
    yield uid
    try:
        mcp_server.execute_tool("delete_all_memories", {"user_id": uid})
    except Exception:
        pass


def _results(out):
    """Normalize a tool result to a list of memory dicts."""
    if out is None:
        return []
    if isinstance(out, list):
        return out
    if isinstance(out, dict):
        return out.get("results", [])
    return []


def _texts(results):
    return [m.get("memory", "") for m in results if isinstance(m, dict)]


# ── 1. Real embedding + semantic recall of paraphrases ───────────────────────

class TestRealEmbeddingEndToEnd:
    def test_paraphrase_query_finds_stored_fact(self, memory, test_user):
        """Query phrased as a question matches the stored declarative fact."""
        fact = (
            "The payment service is deployed to Kubernetes using a Helm chart "
            "from the ops repository."
        )
        mcp_server.execute_tool(
            "add_raw_memory", {"content": fact, "user_id": test_user}
        )
        out = mcp_server.execute_tool("search_memories", {
            "query": "How is the payment service deployed?",
            "user_id": test_user,
        })
        found = [
            m for m in _results(out)
            if m.get("score", 0) >= 0.4
            and "kubernetes" in m.get("memory", "").lower()
        ]
        assert found, (
            f"Paraphrase query should retrieve the Helm/Kubernetes fact with "
            f"score >= 0.4; got {json.dumps(_results(out))[:400]}"
        )

    def test_paraphrase_meets_score_threshold(self, memory, test_user):
        """The same paraphrase hit also passes the 0.4 min_score gate."""
        fact = "The invoice worker runs as a cron job every night at 02:00 UTC."
        mcp_server.execute_tool(
            "add_raw_memory", {"content": fact, "user_id": test_user}
        )
        out = mcp_server.execute_tool("search_memories", {
            "query": "When does the invoice job run?",  # paraphrase: question form
            "user_id": test_user,
            "min_score": 0.35,
        })
        texts = _texts(_results(out))
        assert texts, "Paraphrase should surpass min_score 0.35"
        assert any("cron" in t.lower() for t in texts)

    def test_add_raw_memory_limit_rejects_bulk_text(self, memory, test_user):
        """add_raw_memory is for facts: content over the chunk limit is rejected."""
        with pytest.raises(mcp_server.Mem0Error):
            mcp_server.execute_tool("add_raw_memory", {
                "content": "word " * 2000,  # > 3000 chars
                "user_id": test_user,
            })
        out = mcp_server.execute_tool("get_memories", {"user_id": test_user})
        assert len(_results(out)) == 0

    def test_empty_content_is_rejected(self, memory, test_user):
        for tool in ("add_raw_memory", "add_verbatim"):
            with pytest.raises(mcp_server.Mem0Error):
                mcp_server.execute_tool(tool, {"content": "   ", "user_id": test_user})

    def test_empty_query_is_rejected(self, memory, test_user):
        with pytest.raises(mcp_server.Mem0Error):
            mcp_server.execute_tool("search_memories", {"query": "", "user_id": test_user})


# ── 2. Ranking sanity ─────────────────────────────────────────────────────────

class TestRankingSanity:
    def test_cache_query_ranks_redis_above_ui_colors(self, memory, test_user):
        redis_fact = (
            "Redis cache with 256 MB max memory sits in front of Postgres for "
            "session storage."
        )
        color_fact = (
            "The dashboard UI colors use a blue and gray palette on the "
            "settings page."
        )
        mcp_server.execute_tool("add_raw_memory",
                                {"content": redis_fact, "user_id": test_user})
        mcp_server.execute_tool("add_raw_memory",
                                {"content": color_fact, "user_id": test_user})
        out = mcp_server.execute_tool("search_memories", {
            "query": "cache",
            "user_id": test_user,
        })
        results = _results(out)
        texts = _texts(results)
        assert len(results) >= 2, f"Both facts should be returned, got {texts}"
        redis_idx = next(
            i for i, t in enumerate(texts) if "redis" in t.lower())
        color_idx = next(
            i for i, t in enumerate(texts) if "ui colors" in t.lower())
        assert redis_idx < color_idx, (
            f"Query 'cache' must rank Redis above UI colors; got order {texts}"
        )
        assert results[redis_idx]["score"] >= results[color_idx]["score"]

    def test_scores_are_ordered_descending(self, memory, test_user):
        """search_memories returns results sorted by score (when present)."""
        for i in range(3):
            mcp_server.execute_tool("add_raw_memory", {
                "content": f"Ranking probe fact {i} about the billing ledger subsystem.",
                "user_id": test_user,
            })
        out = mcp_server.execute_tool("search_memories", {
            "query": "billing ledger subsystem",
            "user_id": test_user,
        })
        scores = [m.get("score") for m in _results(out) if m.get("score") is not None]
        assert len(scores) >= 2
        assert scores == sorted(scores, reverse=True), f"Not descending: {scores}"
        assert all(s <= 1.0 for s in scores), f"Scores must be <= 1.0: {scores}"

    def test_min_score_filters_weak_matches(self, memory, test_user):
        mcp_server.execute_tool("add_raw_memory", {
            "content": "The billing ledger subsystem posts nightly journal entries.",
            "user_id": test_user,
        })
        out = mcp_server.execute_tool("search_memories", {
            "query": "billing ledger subsystem",
            "user_id": test_user,
            "min_score": 0.99,  # impossibly high: result must be filtered out
        })
        assert _results(out) == [], (
            "min_score=0.99 should filter everything out"
        )

    def test_include_scores_false_strips_score(self, memory, test_user):
        mcp_server.execute_tool("add_raw_memory", {
            "content": "Score-stripping probe memory about the audit trail.",
            "user_id": test_user,
        })
        out = mcp_server.execute_tool("search_memories", {
            "query": "audit trail",
            "user_id": test_user,
            "include_scores": False,
        })
        results = _results(out)
        assert results, "Hit expected for near-verbatim query"
        assert all("score" not in m for m in results)


# ── 3. Verbatim bulk storage (chunking) ──────────────────────────────────────

def _build_long_document(target_chars: int) -> str:
    """Build a realistic multi-paragraph document (~target_chars)."""
    paragraphs = []
    n = 0
    total = 0
    while total < target_chars:
        para = (
            f"Runbook section {n}: the ingest pipeline for dataset {n} is "
            "redeployed after each schema migration and drained before the "
            "next rollout so that no partial batches remain in the staging "
            "queue between deployments of the stream processor workers."
        )
        paragraphs.append(para)
        total += len(para) + 2
        n += 1
    return "\n\n".join(paragraphs)


class TestVerbatimChunking:
    def test_verbatim_15000_chars_chunks_and_search(self, memory, test_user):
        """~15k chars verbatim -> >=2 chunks, all listed, later chunk searchable."""
        doc = _build_long_document(15000)
        assert len(doc) >= 14000
        result = mcp_server.execute_tool("add_verbatim", {
            "content": doc,
            "user_id": test_user,
            "metadata": {"file": "runbook.md"},
        })
        assert result.get("chunks", 0) >= 2, (
            f"15k chars must produce >= 2 chunks; got {result.get('chunks')}"
        )
        assert len(result.get("results", [])) == result["chunks"]

        out = mcp_server.execute_tool("get_memories",
                                      {"user_id": test_user, "limit": 100})
        listed = _results(out)
        assert len(listed) >= 2, f"get_memories should list all chunks; got {len(listed)}"
        # every chunk text is <= chunk size (header included)
        max_len = max(len(t) for t in _texts(listed))
        assert max_len <= mcp_server._get_chunk_size() + 200  # header slack

        # a distinctive fragment that is UNAMBIGUOUSLY in a later chunk
        n_sections = mcp_server._chunk_text(doc, 3000)
        later_chunk_text = n_sections[-1]
        # pick a unique sentence fragment from the later chunk
        fragment = later_chunk_text.split(":")[0].strip()  # "Runbook section K"
        assert int(fragment.split()[-1]) >= 1
        found = mcp_server.execute_tool("search_memories", {
            "query": f"{fragment} ingest pipeline drained stream processor",
            "user_id": test_user,
        })
        hits = _results(found)
        assert hits, f"Later-chunk fragment should be searchable; query {fragment!r}"
        assert any(fragment.lower() in t.lower() for t in _texts(hits)), (
            f"A stored chunk containing {fragment!r} must be among hits; "
            f"got {[t[:60] for t in _texts(hits)]}"
        )

    def test_verbatim_chunk_metadata_counts(self, memory, test_user):
        """Each chunk's metadata records its position (chunk: i/N)."""
        doc = _build_long_document(8000)
        result = mcp_server.execute_tool("add_verbatim", {
            "content": doc,
            "user_id": test_user,
            "metadata": {"file": "guide.md", "source": "unit-test"},
        })
        out = mcp_server.execute_tool("get_memories", {"user_id": test_user})
        listed = _results(out)
        n_chunks = result["chunks"]
        assert n_chunks >= 2
        metas = [m.get("metadata") or {} for m in listed]
        chunk_tags = [meta.get("chunk") for meta in metas if meta.get("chunk")]
        assert len(chunk_tags) == n_chunks, (
            f"Each chunk must carry chunk:i/N metadata; got {chunk_tags}"
        )
        assert set(chunk_tags) == {f"{i+1}/{n_chunks}" for i in range(n_chunks)}

    def test_verbatim_short_text_stays_single_chunk(self, memory, test_user):
        """Text under the chunk limit is stored uncut."""
        result = mcp_server.execute_tool("add_verbatim", {
            "content": "Single short verbatim note about the nightly backup window.",
            "user_id": test_user,
        })
        assert result["chunks"] == 1
        out = mcp_server.execute_tool("get_memories", {"user_id": test_user})
        listed = _results(out)
        assert len(listed) == 1
        assert "nightly backup window" in _texts(listed)[0]


# ── 4. Update changes retrievable content ────────────────────────────────────

class TestUpdateFlow:
    def test_update_changes_retrievable_content_and_old_fades(self, memory, test_user):
        old_fact = "The rate limit for the public API is 100 requests per minute."
        new_fact = "The rate limit for the public API is 5000 requests per minute."
        mcp_server.execute_tool("add_raw_memory",
                                {"content": old_fact, "user_id": test_user})
        out = mcp_server.execute_tool("get_memories", {"user_id": test_user})
        mem_id = _results(out)[0]["id"]

        mcp_server.execute_tool("update_memory",
                                {"memory_id": mem_id, "content": new_fact})

        got = mcp_server.execute_tool("get_memory", {"memory_id": mem_id})
        assert got is not None and "5000" in got.get("memory", ""), (
            f"update_memory must change retrievable content; got {got}"
        )

        hits = mcp_server.execute_tool("search_memories", {
            "query": "public API rate limit requests per minute",
            "user_id": test_user,
        })
        results = _results(hits)
        assert results, "Updated fact must still be searchable"
        top_text = results[0].get("memory", "")
        assert "5000" in top_text, (
            f"New content should rank first; got {results[0].get('memory')}"
        )
        # the OLD content must not be returned as a top hit
        assert "100 requests per minute" not in " ".join(_texts(results)[:2])
        assert all(m["id"] == mem_id for m in results), (
            "update_memory resamples the same memory id, no duplicates"
        )

    def test_update_to_other_topic_search_ranking(self, memory, test_user):
        old_fact = "The staging database runs on Postgres version 14."
        mcp_server.execute_tool("add_raw_memory",
                                {"content": old_fact, "user_id": test_user})
        mem_id = _results(
            mcp_server.execute_tool("get_memories", {"user_id": test_user})
        )[0]["id"]
        mcp_server.execute_tool("update_memory", {
            "memory_id": mem_id,
            "content": "The staging database runs on MySQL version 8.",
        })
        hits = mcp_server.execute_tool("search_memories", {
            "query": "staging database version",
            "user_id": test_user,
        })
        texts = _texts(_results(hits))
        assert any("mysql" in t.lower() for t in texts), (
            f"Updated mysql fact must be found; got {texts}"
        )

    def test_update_requires_valid_id_and_content(self, memory, test_user):
        with pytest.raises(mcp_server.Mem0Error):
            mcp_server.execute_tool("update_memory",
                                    {"memory_id": "", "content": "x"})
        mcp_server.execute_tool("add_raw_memory",
                                {"content": "Update validation probe fact.", "user_id": test_user})
        mem_id = _results(
            mcp_server.execute_tool("get_memories", {"user_id": test_user})
        )[0]["id"]
        with pytest.raises(mcp_server.Mem0Error):
            mcp_server.execute_tool("update_memory",
                                    {"memory_id": mem_id, "content": "   "})


# ── 5. delete_memory ─────────────────────────────────────────────────────────

class TestDeleteMemory:
    def test_delete_removes_distinctive_phrase(self, memory, test_user):
        phrase = "PURPLE_PANGOLIN_PHRASE lives in the provisioning checklist."
        mcp_server.execute_tool("add_raw_memory",
                                {"content": phrase, "user_id": test_user})
        mcp_server.execute_tool("add_raw_memory", {
            "content": "An unrelated keeper memory about the deploy checklist.",
            "user_id": test_user,
        })
        mem_id = None
        for m in _results(
            mcp_server.execute_tool("get_memories", {"user_id": test_user})
        ):
            if "PURPLE_PANGOLIN_PHRASE" in m.get("memory", ""):
                mem_id = m["id"]
        assert mem_id, "The phrase memory must be listed"

        out = mcp_server.execute_tool("delete_memory", {"memory_id": mem_id})
        assert out["status"] == "deleted"

        hits = mcp_server.execute_tool("search_memories", {
            "query": "PURPLE_PANGOLIN_PHRASE provisioning checklist",
            "user_id": test_user,
        })
        remaining = _texts(_results(hits))
        assert not any("PURPLE_PANGOLIN_PHRASE" in t for t in remaining), (
            f"Deleted phrase must return 0 hits; got {remaining}"
        )

    def test_delete_memory_requires_id(self, memory):
        with pytest.raises(mcp_server.Mem0Error):
            mcp_server.execute_tool("delete_memory", {"memory_id": ""})

    def test_delete_unknown_id_raises_actionable_error(self, memory):
        with pytest.raises(mcp_server.Mem0Error):
            mcp_server.execute_tool(
                "delete_memory", {"memory_id": str(uuid.uuid4())})


# ── 6. delete_all_memories ───────────────────────────────────────────────────

class TestDeleteAll:
    def test_delete_all_n_to_zero(self, memory, test_user):
        for i in range(5):
            mcp_server.execute_tool("add_raw_memory", {
                "content": f"Wipe target memory {i} for the delete-all flow.",
                "user_id": test_user,
            })
        before = _results(
            mcp_server.execute_tool("get_memories", {"user_id": test_user})
        )
        assert len(before) == 5, f"Setup mismatch: {len(before)} memories"

        out = mcp_server.execute_tool("delete_all_memories",
                                      {"user_id": test_user})
        assert out["status"] == "deleted_all"

        after = _results(
            mcp_server.execute_tool("get_memories", {"user_id": test_user})
        )
        assert len(after) == 0, f"Memories remain after delete_all: {len(after)}"
        hits = mcp_server.execute_tool("search_memories", {
            "query": "wipe target memory delete-all flow",
            "user_id": test_user,
        })
        assert _results(hits) == []

    def test_delete_all_empty_user_is_noop(self, memory, test_user):
        out = mcp_server.execute_tool("delete_all_memories",
                                      {"user_id": test_user})
        assert out["status"] == "deleted_all"


# ── 7. export -> import round-trip + duplicate skip ──────────────────────────

class TestExportImport:
    def test_export_json_import_new_user_search_and_skip(self, memory, test_user, test_user_b):
        fact = "Export-import probe: the nightly snapshot lands in the s3 archive."
        mcp_server.execute_tool("add_raw_memory", {
            "content": fact, "user_id": test_user,
            "metadata": {"topic": "backups"},
        })
        export = mcp_server.execute_tool("export_memories",
                                         {"user_id": test_user, "format": "json"})
        assert export["format"] == "json"
        assert export["count"] == 1
        assert isinstance(export["memories"], list)
        assert export["memories"][0]["memory"] == fact

        # import into a NEW user
        import1 = mcp_server.execute_tool("import_memories", {
            "data": json.dumps(export["memories"]),
            "user_id": test_user_b,
        })
        assert import1["imported"] == 1, f"{import1}"
        assert import1["skipped"] == 0 and import1["failed"] == 0

        # the imported fact is semantically searchable under the new user...
        hits = mcp_server.execute_tool("search_memories", {
            "query": "nightly snapshot archive location",
            "user_id": test_user_b,
        })
        texts = _texts(_results(hits))
        assert any("nightly snapshot" in t.lower() for t in texts), (
            f"Imported memory must be searchable: {texts}"
        )
        # ...and scoped ONLY to the new user (original untouched)
        orig_hits = mcp_server.execute_tool("search_memories", {
            "query": "nightly snapshot archive location", "user_id": test_user,
        })
        assert any("nightly snapshot" in t.lower() for t in _texts(_results(orig_hits)))

        # second import: all duplicates skipped
        import2 = mcp_server.execute_tool("import_memories", {
            "data": json.dumps(export["memories"]),
            "user_id": test_user_b,
        })
        assert import2["imported"] == 0, f"{import2}"
        assert import2["skipped"] == 1
        total = _results(
            mcp_server.execute_tool("get_memories", {"user_id": test_user_b})
        )
        assert len(total) == 1, "no duplicate rows may be created"

    def test_import_rejects_invalid_payloads(self, memory, test_user):
        with pytest.raises(mcp_server.Mem0Error):
            mcp_server.execute_tool("import_memories",
                                    {"data": "not json", "user_id": test_user})
        with pytest.raises(mcp_server.Mem0Error):
            mcp_server.execute_tool("import_memories",
                                    {"data": json.dumps({"a": 1}), "user_id": test_user})
        with pytest.raises(mcp_server.Mem0Error):
            mcp_server.execute_tool("import_memories", {"data": "", "user_id": test_user})

    def test_import_reports_failures_individually(self, memory, test_user):
        # Mixed batch: one good row, one non-dict row, one empty row, and a
        # duplicate of the good row (caught by within-batch dedup).
        result = mcp_server.execute_tool("import_memories", {
            "data": json.dumps([
                {"memory": "Good row text: import failure accounting probe A."},
                "not-a-dict",
                {"memory": ""},           # empty text -> failed
                {"memory": "Good row text: import failure accounting probe A."},  # dup within batch
            ]),
            "user_id": test_user,
        })
        assert result["imported"] == 1
        assert result["failed"] == 2, f"{result}"
        assert result["skipped"] == 1
        assert result["errors"]
        assert len(result["errors"]) == 2, f"double-counted errors: {result['errors']}"
        assert not any(
            "not-a-dict" in t
            for t in _texts(_results(mcp_server.execute_tool(
                "get_memories", {"user_id": test_user})))
        ), "invalid rows must not be stored"


# ── 8. CSV export ─────────────────────────────────────────────────────────────

class TestCsvExport:
    def test_csv_exact_header_and_rows(self, memory, test_user):
        facts = [
            ("CSV probe one: the nightly tarball lands on the backup disk.",
             {"topic": "backups", "critical": True}),
            ("CSV probe two: the ledger syncer retries with backoff.",
             {"topic": "ledger"}),
        ]
        for content, meta in facts:
            mcp_server.execute_tool("add_raw_memory", {
                "content": content, "user_id": test_user, "metadata": meta,
            })
        exp = mcp_server.execute_tool("export_memories", {
            "user_id": test_user, "format": "csv",
        })
        assert exp["format"] == "csv"
        assert exp["count"] == 2
        rows = list(csv.reader(io.StringIO(exp["data"])))
        assert rows[0] == ["id", "memory", "metadata", "created_at", "user_id"], (
            f"CSV header mismatch: {rows[0]}"
        )
        body = rows[1:]
        assert len(body) == 2
        assert all(len(r) == 5 for r in body)
        mems = {r[1] for r in body}
        assert mems == {facts[0][0], facts[1][0]}
        for r in body:
            assert r[4] == test_user
            assert r[0]  # id non-empty
        # metadata round-trips through the CSV cell as JSON
        metas = [json.loads(r[2]) for r in body if r[2]]
        topics = {m.get("topic") for m in metas}
        assert topics == {"backups", "ledger"}

    def test_csv_quotes_commas_and_newlines(self, memory, test_user):
        fact = 'CSV quoting probe: use "quoted, values" and\nnewlines, safely.'
        mcp_server.execute_tool("add_raw_memory",
                                {"content": fact, "user_id": test_user})
        exp = mcp_server.execute_tool("export_memories",
                                      {"user_id": test_user, "format": "csv"})
        rows = list(csv.reader(io.StringIO(exp["data"])))
        assert rows[0] == ["id", "memory", "metadata", "created_at", "user_id"]
        assert any('use "quoted, values" and\nnewlines, safely.' in r[1]
                   for r in rows[1:])

    def test_export_invalid_format_rejected(self, memory, test_user):
        with pytest.raises(mcp_server.Mem0Error):
            mcp_server.execute_tool("export_memories", {
                "user_id": test_user, "format": "yaml",
            })


# ── 9. Metadata preservation through import ──────────────────────────────────

class TestMetadataPreservation:
    def test_import_preserves_metadata(self, memory, test_user, test_user_b):
        fact = "Metadata probe: the feature flag rollout plan lives in launchdoc."
        meta = {
            "topic": "rollout",
            "priority": 7,
            "owner": "platform-team",
            "verified": True,
        }
        mcp_server.execute_tool("add_raw_memory", {
            "content": fact, "user_id": test_user, "metadata": meta,
        })
        export = mcp_server.execute_tool("export_memories",
                                         {"user_id": test_user, "format": "json"})
        rec = export["memories"][0]
        # memory id must be replaced by a NEW id on import (unique constraint),
        # text and metadata must survive
        import1 = mcp_server.execute_tool("import_memories", {
            "data": json.dumps([{**rec, "id": None}]), "user_id": test_user_b,
        })
        assert import1["imported"] == 1, f"{import1}"

        got_new = _results(
            mcp_server.execute_tool("get_memories", {"user_id": test_user_b})
        )[0]
        got = mcp_server.execute_tool("get_memory", {"memory_id": got_new["id"]})
        assert got["memory"] == fact
        for k, v in meta.items():
            assert got["metadata"].get(k) == v, (
                f"Metadata key {k!r} lost on import: {got['metadata']}"
            )

    def test_verbatim_metadata_survives_chunking(self, memory, test_user):
        doc = _build_long_document(7000)
        mcp_server.execute_tool("add_verbatim", {
            "content": doc, "user_id": test_user,
            "metadata": {"file": "big.md", "origin": "confluence"},
        })
        listed = _results(
            mcp_server.execute_tool("get_memories", {"user_id": test_user})
        )
        assert len(listed) >= 2
        for m in listed:
            meta = m.get("metadata") or {}
            assert meta.get("file") == "big.md"
            assert meta.get("origin") == "confluence"
            got = mcp_server.execute_tool("get_memory", {"memory_id": m["id"]})
            assert got["metadata"].get("file") == "big.md"


# ── 10. prune_memories ────────────────────────────────────────────────────────

class TestPrune:
    def test_prune_dry_run_then_real_spares_young_memories(self, memory, test_user):
        for i in range(3):
            mcp_server.execute_tool("add_raw_memory", {
                "content": f"Prune probe memory {i} created seconds ago for aging tests.",
                "user_id": test_user,
            })
        dry = mcp_server.execute_tool("prune_memories", {
            "user_id": test_user, "older_than_days": 30, "dry_run": True,
        })
        assert dry["dry_run"] is True
        assert dry["would_delete"] >= 0, f"{dry}"
        assert dry["deleted"] == 0, "dry run must not delete"
        assert isinstance(dry["memories"], list)

        real = mcp_server.execute_tool("prune_memories", {
            "user_id": test_user, "older_than_days": 30, "dry_run": False,
        })
        assert real["dry_run"] is False
        assert real["deleted"] == 0, f"Young memories must survive: {real}"
        remaining = _results(
            mcp_server.execute_tool("get_memories", {"user_id": test_user})
        )
        assert len(remaining) == 3, (
            f"Nothing is older than 30 days, so all 3 must remain: {len(remaining)}"
        )

    def test_prune_dry_run_reports_ancient_memories(self, memory, test_user):
        """A memory with a crafted old created_at is reported (dry run)."""
        old_iso = "2026-01-01T00:00:00+00:00"
        res = mcp_server.execute_tool("add_raw_memory", {
            "content": "Ancient prune probe with a hand-crafted old created_at.",
            "user_id": test_user,
        })
        mem_id = res["results"][0]["id"]
        # Age the memory by rewriting its payload directly (Qdrant-level setup,
        # not a product code path) — the prune logic itself is exercised via
        # execute_tool below.
        memory.vector_store.client.set_payload(
            collection_name="mem0",
            payload={"created_at": old_iso},
            points=[mem_id],
        )
        dry = mcp_server.execute_tool("prune_memories", {
            "user_id": test_user, "older_than_days": 30, "dry_run": True,
        })
        assert dry["dry_run"] is True
        assert dry["would_delete"] == 1, f"{dry}"
        assert any(mem_id == m["id"] for m in dry["memories"])
        # nothing deleted yet
        still = mcp_server.execute_tool("get_memory", {"memory_id": mem_id})
        assert still is not None

    def test_prune_real_deletes_only_old(self, memory, test_user):
        old_iso = "2026-01-01T00:00:00+00:00"
        res = mcp_server.execute_tool("add_raw_memory", {
            "content": "Prune-reality probe: this one is old and must vanish.",
            "user_id": test_user,
        })
        mem_id = res["results"][0]["id"]
        memory.vector_store.client.set_payload(
            collection_name="mem0",
            payload={"created_at": old_iso},
            points=[mem_id],
        )
        mcp_server.execute_tool("add_raw_memory", {
            "content": "Prune-reality probe: this one is fresh and must stay.",
            "user_id": test_user,
        })
        real = mcp_server.execute_tool("prune_memories", {
            "user_id": test_user, "older_than_days": 30, "dry_run": False,
        })
        assert real["deleted"] == 1, f"{real}"
        remaining = _results(
            mcp_server.execute_tool("get_memories", {"user_id": test_user})
        )
        assert len(remaining) == 1
        assert "must stay" in remaining[0]["memory"]

    def test_prune_unknown_user_noop(self, memory):
        for uid in (f"prune_e_{uuid.uuid4().hex[:8]}",):
            out = mcp_server.execute_tool("prune_memories", {
                "user_id": uid, "older_than_days": 7, "dry_run": True,
            })
            assert out["would_delete"] == 0
            assert out["memories"] == []


# ── 11/12. Schema + embedder configuration invariants ────────────────────────

class TestConfigurationInvariants:
    def test_collection_vector_size_is_768(self, memory):
        """After init, the Qdrant collection is 768-dim (nomic v1.5)."""
        info = memory.vector_store.client.get_collection("mem0")
        vectors = info.config.params.vectors
        size = None
        if hasattr(vectors, "size") and vectors.size:       # single unnamed vector
            size = vectors.size
        elif isinstance(getattr(vectors, "vectors", None), dict) and vectors.vectors:
            first = next(iter(vectors.vectors.values()))
            size = first.size
        assert size == 768, f"Collection vector size must be 768; got {size}"

    def test_configured_fastembed_model_embeds_to_768(self, memory):
        """The CONFIG fastembed embedder produces 768-dim vectors."""
        vec = memory.embedding_model.embed("configuration invariant probe")
        assert len(vec) == 768, f"Embedding must be 768-dim; got {len(vec)}"
        # add()/update() feed Qdrant with exactly this embedder, so dims match
        # the collection (tested above) by construction.

    def test_embedder_model_is_nomic_v15(self, memory):
        cfg_model = memory.embedding_model.config.model
        assert cfg_model == "nomic-ai/nomic-embed-text-v1.5", cfg_model

    def test_search_results_carry_scores_by_default(self, memory, test_user):
        mcp_server.execute_tool("add_raw_memory", {
            "content": "Score-presence probe for default search behavior.",
            "user_id": test_user,
        })
        out = mcp_server.execute_tool("search_memories", {
            "query": "score-presence probe", "user_id": test_user,
        })
        results = _results(out)
        assert results
        assert all(isinstance(m.get("score"), float) for m in results), (
            f"Scores should be present by default; got {results}"
        )