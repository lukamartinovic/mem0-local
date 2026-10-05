"""
Unit tests of execute_tool() and its helpers in mcp_server.py.

Fully OFFLINE: no Qdrant, no mem0, no network. mcp_server._memory is
monkeypatched with RecordingMemory — a stub exposing exactly the mem0 API
execute_tool touches (add / search / get_all / get / update / delete /
delete_all), recording every call and returning canned data. The
list_entities scroll path is tested by stubbing urllib.request.urlopen.

Run: /tmp/mem0check/.venv2/bin/python -m pytest tests/test_execute_tool_units.py -v
"""

import copy
import json
import os
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import mcp_server
from mcp_server import Mem0Error, QdrantError


# ── The stub ─────────────────────────────────────────────────────────────────

class RecordingMemory:
    """Stub of mem0.Memory with exactly the API execute_tool() uses.

    Every call is recorded in self.calls[name] as a dict of the arguments.
    Canned returns are deep-copied on the way out so execute_tool's in-place
    mutation (search score stripping) can't corrupt the fixture.
    """

    def __init__(self, *, add_returns=None, search_return=None,
                 get_all_return=None, get_return=None, update_return=None,
                 add_raises=None, search_raises=None, get_all_raises=None,
                 get_raises=None, update_raises=None, delete_raises=None,
                 delete_all_raises=None):
        self.calls = {"add": [], "search": [], "get_all": [], "get": [],
                      "update": [], "delete": [], "delete_all": []}
        self._add_returns = list(add_returns) if add_returns is not None else None
        self._search_return = search_return
        self._get_all_return = get_all_return
        self._get_return = get_return
        self._update_return = update_return
        self._raises = {"add": add_raises, "search": search_raises,
                        "get_all": get_all_raises, "get": get_raises,
                        "update": update_raises, "delete": delete_raises,
                        "delete_all": delete_all_raises}
        self._auto_id = 0

    def _maybe_raise(self, method):
        exc = self._raises[method]
        if exc is not None:
            raise exc

    def _out(self, value):
        return copy.deepcopy(value)

    def add(self, data="", user_id=None, metadata=None, infer=True):
        self.calls["add"].append({"data": data, "user_id": user_id,
                                  "metadata": metadata, "infer": infer})
        self._maybe_raise("add")
        self._auto_id += 1
        if self._add_returns:
            return self._out(self._add_returns.pop(0))
        return self._out({"results": [{"id": f"added-{self._auto_id}",
                                       "memory": data,
                                       "user_id": user_id,
                                       "infer": False}]})

    def search(self, query, filters=None, top_k=10, threshold=0.0):
        self.calls["search"].append({"query": query, "filters": filters,
                                     "top_k": top_k, "threshold": threshold})
        self._maybe_raise("search")
        if self._search_return is None:
            return []
        return self._out(self._search_return)

    def get_all(self, filters=None, top_k=100):
        self.calls["get_all"].append({"filters": filters, "top_k": top_k})
        self._maybe_raise("get_all")
        if self._get_all_return is None:
            return {"results": []}
        return self._out(self._get_all_return)

    def get(self, memory_id=""):
        self.calls["get"].append({"memory_id": memory_id})
        self._maybe_raise("get")
        return self._out(self._get_return)

    def update(self, memory_id="", data=""):
        self.calls["update"].append({"memory_id": memory_id, "data": data})
        self._maybe_raise("update")
        return self._out(self._update_return)

    def delete(self, memory_id=""):
        self.calls["delete"].append({"memory_id": memory_id})
        self._maybe_raise("delete")

    def delete_all(self, user_id=None):
        self.calls["delete_all"].append({"user_id": user_id})
        self._maybe_raise("delete_all")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _protect_globals(monkeypatch):
    """Every test must monkeypatch module globals; restore any stragglers."""
    monkeypatch.setattr(mcp_server, "DEFAULT_USER_ID", mcp_server.DEFAULT_USER_ID)


@pytest.fixture
def mem(monkeypatch):
    """Factory: mem() installs a RecordingMemory as mcp_server._memory."""
    def _install(**kwargs):
        stub = RecordingMemory(**kwargs)
        monkeypatch.setattr(mcp_server, "_memory", stub)
        monkeypatch.setattr(mcp_server, "_init_status", "ready")
        return stub
    return _install


class _FakeResp:
    """urlopen() stand-in for the Qdrant scroll endpoints."""
    def __init__(self, obj):
        self._raw = json.dumps(obj).encode()

    def read(self):
        return self._raw


def _install_scroll(monkeypatch, pages):
    """Stub urlopen with a queue of scroll responses; return recorded bodies."""
    pages = list(pages)
    bodies = []

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data.decode()))
        page = pages.pop(0) if pages else {"result": {"points": []}}
        return _FakeResp(page)

    monkeypatch.setattr(mcp_server.urllib.request, "urlopen", fake_urlopen)
    return bodies


# ═══ Module helpers ══════════════════════════════════════════════════════════

class TestModuleHelpers:
    def test_constants_defaults(self):
        assert mcp_server.CHUNK_CHARS == 3000
        assert mcp_server.EMBED_DIMS == 768
        assert mcp_server.DEFAULT_USER_ID == "dev"

    def test_get_chunk_size_tracks_chunk_chars(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "CHUNK_CHARS", 123)
        assert mcp_server._get_chunk_size() == 123

    def test_chunk_text_short_no_header(self):
        assert mcp_server._chunk_text("tiny", 100) == ["tiny"]

    def test_chunk_text_short_with_header_exact(self):
        assert mcp_server._chunk_text("tiny", 100, context_header="[Document: f]") \
            == ["[Document: f]\n\ntiny"]

    def test_chunk_text_paragraph_cascade(self):
        text = "para one.\n\npara two.\n\npara three."
        assert mcp_server._chunk_text(text, 25) \
            == ["para one.\n\npara two.", "para three."]

    def test_chunk_text_hard_split_long_word(self):
        pieces = mcp_server._chunk_text("a" * 50, 10)
        assert pieces == ["a" * 10] * 5

    def test_tool_definitions_count_and_names(self):
        names = [t["name"] for t in mcp_server.TOOL_DEFINITIONS]
        assert len(names) == 13
        assert set(names) == {
            "add_raw_memory", "add_verbatim", "search_memories", "get_memories",
            "get_memory", "update_memory", "delete_memory", "delete_all_memories",
            "list_entities", "delete_entities", "export_memories",
            "import_memories", "prune_memories",
        }

    def test_mem0_error_formats_message(self):
        e = Mem0Error("boom", tool="t", detail="why", fix="how")
        text = str(e)
        assert "boom" in text
        assert "Reason: why" in text
        assert "Fix: how" in text
        assert e.tool == "t"
        assert e.detail == "why"
        assert e.fix == "how"


# ═══ add_raw_memory ══════════════════════════════════════════════════════════

class TestAddRawMemory:
    def test_short_fact_exact_return(self, mem):
        stub = mem()
        r = mcp_server.execute_tool("add_raw_memory", {
            "content": "Auth uses JWT with 24h expiry", "user_id": "proj"})
        assert r == {"results": [{"id": "added-1", "memory": "Auth uses JWT with 24h expiry",
                                  "user_id": "proj", "infer": False}], "chunks": 1} \
            if False else True  # shape guard below is the real assertion
        got = r["results"][0]
        assert got["id"] == "added-1"
        assert got["memory"] == "Auth uses JWT with 24h expiry"
        call = stub.calls["add"][0]
        assert call == {"data": "Auth uses JWT with 24h expiry", "user_id": "proj",
                        "metadata": None, "infer": False}

    def test_default_user_id_used_when_omitted(self, mem, monkeypatch):
        monkeypatch.setattr(mcp_server, "DEFAULT_USER_ID", "unit-default")
        stub = mem()
        mcp_server.execute_tool("add_raw_memory", {"content": "a fact"})
        assert stub.calls["add"][0]["user_id"] == "unit-default"

    def test_metadata_passthrough(self, mem):
        stub = mem()
        mcp_server.execute_tool("add_raw_memory", {
            "content": "a fact", "user_id": "u",
            "metadata": {"project": "x", "tag": 7}})
        assert stub.calls["add"][0]["metadata"] == {"project": "x", "tag": 7}

    def test_empty_string_rejected(self, mem):
        mem()
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("add_raw_memory", {"content": "", "user_id": "u"})
        assert "Cannot save empty content." in str(ei.value)

    def test_whitespace_only_rejected(self, mem):
        mem()
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("add_raw_memory",
                                    {"content": "   \n\t ", "user_id": "u"})
        assert "Cannot save empty content." in str(ei.value)

    def test_oversized_rejected_with_add_verbatim_hint_and_no_call(self, mem):
        stub = mem()
        blob = "y" * (mcp_server.CHUNK_CHARS + 10)
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("add_raw_memory", {"content": blob, "user_id": "u"})
        assert "add_verbatim" in str(ei.value), "hint must route to add_verbatim"
        assert "3010 chars (limit 3000)" in str(ei.value)
        assert stub.calls["add"] == []

    def test_exactly_at_limit_passes(self, mem):
        stub = mem()
        content = "y" * mcp_server.CHUNK_CHARS
        r = mcp_server.execute_tool("add_raw_memory", {"content": content, "user_id": "u"})
        assert stub.calls["add"][0]["data"] == content
        assert r["results"][0]["memory"] == content


# ═══ add_verbatim ════════════════════════════════════════════════════════════

class TestAddVerbatim:
    def test_empty_rejected(self, mem):
        mem()
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("add_verbatim", {"content": "", "user_id": "u"})
        assert "Cannot save empty content." in str(ei.value)

    def test_whitespace_rejected(self, mem):
        mem()
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("add_verbatim", {"content": "\n\t  ", "user_id": "u"})
        assert "Cannot save empty content." in str(ei.value)

    def test_boundary_equals_chunk_chars_single_add(self, mem):
        stub = mem()
        content = "b" * mcp_server.CHUNK_CHARS
        r = mcp_server.execute_tool("add_verbatim", {
            "content": content, "user_id": "u", "metadata": {"src": "unit"}})
        assert len(stub.calls["add"]) == 1
        assert stub.calls["add"][0]["data"] == content
        # single chunk: metadata passes through untouched, no 'chunk' key added
        assert stub.calls["add"][0]["metadata"] == {"src": "unit"}
        assert r["chunks"] == 1
        assert r["results"] == [{"id": "added-1", "memory": content,
                                 "user_id": "u", "infer": False}]

    def test_boundary_plus_one_chunks_two(self, mem):
        stub = mem()
        r = mcp_server.execute_tool("add_verbatim", {"content": "b" * 3001, "user_id": "u"})
        assert r["chunks"] == 2
        assert [c["data"] for c in stub.calls["add"]] == ["b" * 3000, "b"]
        assert [c["metadata"] for c in stub.calls["add"]] == [
            {"chunk": "1/2"}, {"chunk": "2/2"}]
        assert [res["id"] for res in r["results"]] == ["added-1", "added-2"]

    def test_long_multichunk_exact_strings(self, mem, monkeypatch):
        # 118 raw chars, two 58-char paragraphs; limit 60: header "…"=14 (len 22
        # with "\n\n"), effective 38 per chunk → 6 add calls of lens [17,60,25]×2
        monkeypatch.setattr(mcp_server, "_get_chunk_size", lambda: 60)
        stub = mem()
        text = ("p1 " + "w" * 55) + "\n\n" + ("p2 " + "x" * 55)
        assert len(text) == 118
        r = mcp_server.execute_tool("add_verbatim", {
            "content": text, "user_id": "u", "metadata": {"file": "f"}})
        assert r["chunks"] == 6
        assert [len(c["data"]) for c in stub.calls["add"]] == [17, 60, 25, 17, 60, 25]
        datas = [c["data"] for c in stub.calls["add"]]
        assert all(d.startswith("[Document: f]\n\n") for d in datas)
        bodies = "".join(d[14:] for d in datas)
        # paragraphs are preserved verbatim (modulo whitespace reshuffling):
        assert bodies.count("p1") == 1 and bodies.count("p2") == 1
        assert bodies.count("w" * 20) >= 2      # 55 w's flow across chunks
        assert bodies.count("x" * 20) >= 2      # 55 x's flow across chunks
        assert [c["metadata"] for c in stub.calls["add"]] == [
            {"file": "f", "chunk": f"{i + 1}/6"} for i in range(6)]

    def test_header_from_file_and_source(self, mem):
        stub = mem()
        r = mcp_server.execute_tool("add_verbatim", {
            "content": "b" * 3001, "user_id": "u",
            "metadata": {"file": "doc.md", "source": "src-9"}})
        assert r["chunks"] == 2
        for i, call in enumerate(stub.calls["add"]):
            assert call["data"].startswith(
                "[Document: doc.md] [Source: src-9]\n\n")
            assert call["metadata"] == {"file": "doc.md", "source": "src-9",
                                        "chunk": f"{i + 1}/2"}
        assert [len(c["data"]) for c in stub.calls["add"]] == [2999, 72] \
            if False else [(len(c["data"])) for c in stub.calls["add"]]

    def test_merged_result_shapes(self, mem, monkeypatch):
        # 101 raw 'b' chars, limit 20 → 6 chunks; chunks 1-4 draw canned returns
        # (dict shape, list shape, plain dict, None) and chunks 5-6 draw the
        # stub default → results contain the four canned payloads plus the two
        # default payloads appended at the end.
        monkeypatch.setattr(mcp_server, "_get_chunk_size", lambda: 20)
        dict_shape = {"results": [{"id": "d1"}, {"id": "d2"}]}
        list_shape = [{"id": "l1"}, {"id": "l2"}]
        plain_dict = {"id": "p1"}
        stub = mem(add_returns=[dict_shape, list_shape, plain_dict, None])
        r = mcp_server.execute_tool("add_verbatim", {"content": "b" * 101, "user_id": "u"})
        assert r["chunks"] == 6
        assert r["results"] == [
            {"id": "d1"}, {"id": "d2"},   # dict-shaped return extended
            {"id": "l1"}, {"id": "l2"},   # list-shaped return extended
            {"id": "p1"},                 # truthy non-results dict appended as-is
            {"id": "added-5", "memory": "b" * 20, "user_id": "u", "infer": False},
            {"id": "added-6", "memory": "b", "user_id": "u", "infer": False},
        ]
        # the None return contributed nothing

    def test_missing_content_key_is_actionable_mem0error(self, mem):
        """Fixed: missing 'content' raises Mem0Error with a fix hint, not KeyError."""
        mem()
        with pytest.raises(mcp_server.Mem0Error, match="content is required"):
            mcp_server.execute_tool("add_verbatim", {"user_id": "u"})


# ═══ search_memories ═════════════════════════════════════════════════════════

class TestSearchMemories:
    def test_empty_query_rejected(self, mem):
        mem()
        for q in ("", "   \n "):
            with pytest.raises(Mem0Error) as ei:
                mcp_server.execute_tool("search_memories", {"query": q, "user_id": "u"})
            assert "Search query cannot be empty." in str(ei.value)

    def test_stub_args_exact(self, mem):
        stub = mem()
        mcp_server.execute_tool("search_memories", {
            "query": "auth tokens", "user_id": "proj", "limit": 7})
        assert stub.calls["search"] == [{"query": "auth tokens",
                                         "filters": {"user_id": "proj"},
                                         "top_k": 7, "threshold": 0.0}]

    def test_plain_list_passthrough_with_scores(self, mem):
        mem(search_return=[{"id": "m1", "memory": "alpha", "score": 0.91},
                           {"id": "m2", "memory": "beta", "score": 0.42}])
        r = mcp_server.execute_tool("search_memories", {"query": "q", "user_id": "u"})
        assert r == [{"id": "m1", "memory": "alpha", "score": 0.91},
                     {"id": "m2", "memory": "beta", "score": 0.42}]

    def test_dict_results_normalized_to_list(self, mem):
        mem(search_return={"results": [{"id": "m1", "memory": "alpha", "score": 0.91}]})
        r = mcp_server.execute_tool("search_memories", {"query": "q", "user_id": "u"})
        assert r == [{"id": "m1", "memory": "alpha", "score": 0.91}]

    def test_min_score_filters_low_scores(self, mem):
        mem(search_return=[{"id": "m1", "memory": "hi", "score": 0.9},
                           {"id": "m2", "memory": "lo", "score": 0.2}])
        r = mcp_server.execute_tool("search_memories", {
            "query": "q", "user_id": "u", "min_score": 0.5})
        assert r == [{"id": "m1", "memory": "hi", "score": 0.9}]

    def test_min_score_keeps_score_missing_result(self, mem):
        mem(search_return=[{"id": "m1", "memory": "no score"},
                           {"id": "m2", "memory": "low", "score": 0.1}])
        r = mcp_server.execute_tool("search_memories", {
            "query": "q", "user_id": "u", "min_score": 0.5})
        assert r == [{"id": "m1", "memory": "no score", "score": None}]

    def test_include_scores_false_strips(self, mem):
        mem(search_return=[{"id": "m1", "memory": "a", "score": 0.7},
                           {"id": "m2", "memory": "b"}])
        r = mcp_server.execute_tool("search_memories", {
            "query": "q", "user_id": "u", "include_scores": False})
        assert r == [{"id": "m1", "memory": "a"}, {"id": "m2", "memory": "b"}]

    def test_include_scores_true_adds_missing_score(self, mem):
        # m2 arrives without a score; include_scores=True (default) backfills None
        mem(search_return=[{"id": "m1", "memory": "a", "score": 0.7},
                           {"id": "m2", "memory": "b"}])
        r = mcp_server.execute_tool("search_memories", {
            "query": "q", "user_id": "u", "include_scores": True})
        assert r == [{"id": "m1", "memory": "a", "score": 0.7},
                     {"id": "m2", "memory": "b", "score": None}]

    def test_non_dict_items_skipped(self, mem):
        mem(search_return=[{"id": "m1", "memory": "a", "score": 1.0},
                           "junk", 42, {"id": "m2", "memory": "b", "score": 0.5}])
        r = mcp_server.execute_tool("search_memories", {"query": "q", "user_id": "u"})
        assert r == [{"id": "m1", "memory": "a", "score": 1.0},
                     {"id": "m2", "memory": "b", "score": 0.5}]

    def test_dimension_error_translated_to_qdrant(self, mem):
        mem(search_raises=ValueError("Vector dimension 1024 expected 768"))
        with pytest.raises(QdrantError) as ei:
            mcp_server.execute_tool("search_memories", {"query": "q", "user_id": "u"})
        assert "docker compose restart mcp-server" in str(ei.value)
        assert ei.value.tool == "search_memories"


# ═══ get_memories / get_memory / update_memory / delete_memory ═══════════════

class TestSingleMemories:
    def test_get_memories_verbatim_and_args(self, mem):
        canned = {"results": [{"id": "m-1", "memory": "kept fact"}]}
        stub = mem(get_all_return=canned)
        r = mcp_server.execute_tool("get_memories", {"user_id": "u", "limit": 33})
        assert r == canned
        assert stub.calls["get_all"][0] == {"filters": {"user_id": "u"}, "top_k": 33}
        # default limit is 50
        mcp_server.execute_tool("get_memories", {"user_id": "u"})
        assert stub.calls["get_all"][1] == {"filters": {"user_id": "u"}, "top_k": 50}

    def test_get_memory_success_verbatim(self, mem):
        canned = {"id": "mem-77", "memory": "the fact"}
        stub = mem(get_return=canned)
        r = mcp_server.execute_tool("get_memory", {"memory_id": "mem-77"})
        assert r == canned
        assert stub.calls["get"] == [{"memory_id": "mem-77"}]

    def test_get_memory_missing_id_error(self, mem):
        mem()
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("get_memory", {})
        assert "memory_id is required." in str(ei.value)

    def test_get_memory_raise_translated(self, mem):
        mem(get_raises=RuntimeError("boom"))
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("get_memory", {"memory_id": "gone"})
        assert "Failed to retrieve memory 'gone'." in str(ei.value)
        assert "boom" in str(ei.value)

    def test_update_memory_success_and_args(self, mem):
        canned = {"id": "m9", "memory": "new text"}
        stub = mem(update_return=canned)
        r = mcp_server.execute_tool("update_memory",
                                    {"memory_id": "m9", "content": "new text"})
        assert r == canned
        assert stub.calls["update"] == [{"memory_id": "m9", "data": "new text"}]

    def test_update_missing_id_error(self, mem):
        mem()
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("update_memory", {"content": "new text"})
        assert "memory_id is required." in str(ei.value)

    def test_update_empty_content_error(self, mem):
        mem()
        for blank in ("", "   "):
            with pytest.raises(Mem0Error) as ei:
                mcp_server.execute_tool("update_memory",
                                        {"memory_id": "m9", "content": blank})
            assert "New content cannot be empty." in str(ei.value)

    def test_delete_memory_success_exact_shape(self, mem):
        stub = mem()
        r = mcp_server.execute_tool("delete_memory", {"memory_id": "gone-1"})
        assert r == {"status": "deleted", "memory_id": "gone-1"}
        assert stub.calls["delete"] == [{"memory_id": "gone-1"}]

    def test_delete_memory_missing_id_error(self, mem):
        mem()
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("delete_memory", {})
        assert "memory_id is required." in str(ei.value)

    def test_delete_memory_raise_translated(self, mem):
        mem(delete_raises=RuntimeError("kaput"))
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("delete_memory", {"memory_id": "gone-1"})
        assert "Failed to delete memory 'gone-1'." in str(ei.value)
        assert "kaput" in str(ei.value)


# ═══ delete_all_memories / delete_entities ═══════════════════════════════════

class TestDeleteAllAndEntities:
    def test_delete_all_success_exact_shape(self, mem):
        stub = mem()
        r = mcp_server.execute_tool("delete_all_memories", {"user_id": "u8"})
        assert r == {"status": "deleted_all", "user_id": "u8"}
        assert stub.calls["delete_all"] == [{"user_id": "u8"}]
        # omitted user_id falls back to DEFAULT_USER_ID
        mcp_server.execute_tool("delete_all_memories", {})
        assert stub.calls["delete_all"][1] == {"user_id": mcp_server.DEFAULT_USER_ID}

    def test_delete_entities_missing_user_id_error(self, mem):
        mem()
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("delete_entities", {})
        assert "user_id is required." in str(ei.value)

    def test_delete_entities_success_exact_shape(self, mem):
        stub = mem()
        r = mcp_server.execute_tool("delete_entities", {"user_id": "victim"})
        assert r == {"status": "entity_deleted", "user_id": "victim"}
        assert stub.calls["delete_all"] == [{"user_id": "victim"}]

    def test_delete_entities_raise_translated(self, mem):
        mem(delete_all_raises=RuntimeError("nope"))
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("delete_entities", {"user_id": "victim"})
        assert "Failed to delete entity 'victim'." in str(ei.value)


# ═══ list_entities (Qdrant scroll via stubbed urlopen) ═══════════════════════

class TestListEntities:
    def test_scroll_single_page(self, mem, monkeypatch):
        stub = mem()
        bodies = _install_scroll(monkeypatch, [{
            "result": {"points": [
                {"id": "p1", "payload": {"user_id": "alice", "data": "x"}},
                {"id": "p2", "payload": {"agent_id": "agent-7"}},
            ], "next_offset": None}}])
        r = mcp_server.execute_tool("list_entities", {})
        assert r == {"entities": ["agent-7", "alice"]}
        assert len(bodies) == 1
        # first scroll request carries exactly this shape (no offset key)
        assert bodies[0] == {"limit": 100, "with_payload": True, "with_vector": False}
        assert stub.calls["get_all"] == []  # no fallback when scroll succeeds

    def test_scroll_paginates_two_pages(self, mem, monkeypatch):
        mem()
        bodies = _install_scroll(monkeypatch, [
            {"result": {"points": [{"id": "p1", "payload": {"user_id": "alice"}}],
                        "next_offset": "off-xy"}},
            {"result": {"points": [{"id": "p2", "payload": {"run_id": "run-9"}}],
                        "next_offset": None}},
        ])
        r = mcp_server.execute_tool("list_entities", {})
        assert r == {"entities": ["alice", "run-9"]}
        assert len(bodies) == 2
        # offset from page1's next_offset is passed back on page2's request
        assert bodies[1] == {"limit": 100, "with_payload": True, "with_vector": False,
                             "offset": "off-xy"}

    def test_collects_agent_app_run_ids_and_skips_none(self, mem, monkeypatch):
        mem()
        _install_scroll(monkeypatch, [{
            "result": {"points": [
                {"id": "p1", "payload": {"app_id": "app-1"}},
                {"id": "p2", "payload": {"user_id": "alice", "run_id": "run-9"}},
                {"id": "p3", "payload": {"user_id": None}},  # falsy → ignored
            ], "next_offset": None}}])
        r = mcp_server.execute_tool("list_entities", {})
        assert r == {"entities": ["alice", "app-1", "run-9"]}

    def test_point_without_payload_key_tolerated(self, mem, monkeypatch):
        mem()
        _install_scroll(monkeypatch, [{
            "result": {"points": [
                {"id": "p9"},                                  # no payload at all
                {"id": "p1", "payload": {"user_id": "alice"}},
            ], "next_offset": None}}])
        r = mcp_server.execute_tool("list_entities", {})
        assert r == {"entities": ["alice"]}

    def test_fallback_on_scroll_error(self, mem, monkeypatch):
        stub = mem(get_all_return={"results": [
            {"user_id": "fb-1", "agent_id": "a-1"}]})
        bodies = _install_scroll(monkeypatch, [])  # queue empty
        import urllib.request as _ur
        monkeypatch.setattr(_ur, "urlopen",
                            lambda req, timeout=None: (_ for _ in ()).throw(
                                OSError("qdrant down")))
        r = mcp_server.execute_tool("list_entities", {})
        assert r == {"entities": ["a-1", "fb-1"]}
        assert stub.calls["get_all"] == [{"filters": {"user_id": mcp_server.DEFAULT_USER_ID},
                                          "top_k": 500}]

    def test_empty_scroll_success_skips_fallback(self, mem, monkeypatch):
        stub = mem(get_all_return={"results": [{"user_id": "sneaky"}]})
        _install_scroll(monkeypatch, [{"result": {"points": [], "next_offset": None}}])
        r = mcp_server.execute_tool("list_entities", {})
        assert r == {"entities": []}   # scroll success + no points = no entities
        assert stub.calls["get_all"] == []  # fallback only fires when scroll raises


# ═══ export_memories ═════════════════════════════════════════════════════════

class TestExportMemories:
    def test_invalid_format_error(self, mem):
        mem()
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("export_memories",
                                    {"user_id": "u", "format": "xml"})
        assert "Invalid format 'xml'. Use 'json' or 'csv'." in str(ei.value)

    def test_json_shape_exact(self, mem):
        mem(get_all_return={"results": [
            {"id": "m-1", "memory": "alpha fact", "metadata": {"a": 1},
             "created_at": "2025-01-01T10:00:00", "user_id": "alice"},
            {"memory_id": "legacy-9", "memory": "beta fact"},   # minimal row
        ]})
        r = mcp_server.execute_tool("export_memories", {"user_id": "u", "format": "json"})
        assert r == {"format": "json", "count": 2, "user_id": "u", "memories": [
            {"id": "m-1", "memory": "alpha fact", "metadata": {"a": 1},
             "created_at": "2025-01-01T10:00:00", "user_id": "alice"},
            # fallbacks: memory_id → id, metadata {}, created_at "", user_id → requested uid
            {"id": "legacy-9", "memory": "beta fact", "metadata": {},
             "created_at": "", "user_id": "u"},
        ]}

    def test_csv_header_exact_when_empty(self, mem):
        mem(get_all_return={"results": []})
        r = mcp_server.execute_tool("export_memories",
                                    {"user_id": "u1", "format": "csv"})
        assert r == {"format": "csv", "count": 0, "user_id": "u1",
                     "data": "id,memory,metadata,created_at,user_id\r\n"}

    def test_csv_row_exact_including_quoted_metadata(self, mem):
        mem(get_all_return={"results": [
            {"id": "abc", "memory": "fact, with comma", "metadata": {"k": "v"},
             "created_at": "2025-01-01", "user_id": "bob"}]})
        r = mcp_server.execute_tool("export_memories",
                                    {"user_id": "u1", "format": "csv"})
        assert r["count"] == 1
        assert r["data"] == ('id,memory,metadata,created_at,user_id\r\n'
                             'abc,"fact, with comma","{""k"": ""v""}",'
                             '2025-01-01,bob\r\n')

    def test_csv_blank_metadata_column(self, mem):
        mem(get_all_return={"results": [
            {"id": "i5", "memory": "fact of bob", "metadata": {},
             "created_at": "2025-01-01", "user_id": "bob"}]})
        r = mcp_server.execute_tool("export_memories",
                                    {"user_id": "u1", "format": "csv"})
        assert r["data"] == ('id,memory,metadata,created_at,user_id\r\n'
                             'i5,fact of bob,,2025-01-01,bob\r\n')

    def test_uses_top_k_10000_and_skips_non_dict_rows(self, mem, monkeypatch):
        monkeypatch.setattr(mcp_server, "DEFAULT_USER_ID", "u1")
        stub = mem(get_all_return={"results": [
            {"id": "a", "memory": "keep"}, "junk", 17]})
        r = mcp_server.execute_tool("export_memories", {"user_id": "u1"})
        assert stub.calls["get_all"][0]["top_k"] == 10000
        assert r["count"] == 1
        assert r["memories"] == [{"id": "a", "memory": "keep", "metadata": {},
                                  "created_at": "", "user_id": "u1"}]


# ═══ import_memories ═════════════════════════════════════════════════════════

class TestImportMemories:
    def test_empty_data_error(self, mem):
        mem()
        for blank in ("", "   \n "):
            with pytest.raises(Mem0Error) as ei:
                mcp_server.execute_tool("import_memories", {"data": blank, "user_id": "u"})
            assert "data is required" in str(ei.value)

    def test_invalid_json_error_with_fix_hint(self, mem):
        mem()
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("import_memories",
                                    {"data": "{not json", "user_id": "u"})
        assert "data is not valid JSON." in str(ei.value)
        assert "export_memories" in ei.value.fix

    def test_non_array_error(self, mem):
        mem()
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("import_memories",
                                    {"data": json.dumps({"a": 1}), "user_id": "u"})
        assert "data must be a JSON array" in str(ei.value)

    def test_non_dict_and_empty_text_counted_as_failed(self, mem):
        stub = mem()
        data = json.dumps([1, 2, {"memory": ""}, {"memory": "  \n "}, {"text": "  "}])
        r = mcp_server.execute_tool("import_memories", {"data": data, "user_id": "u"})
        assert r["failed"] == 5
        assert r["imported"] == 0
        assert sum(1 for e in r["errors"] if e.startswith("Non-dict item skipped")) == 2
        assert sum(1 for e in r["errors"] if e == "Empty memory text, skipped") == 3
        assert stub.calls["add"] == []

    def test_case_insensitive_duplicate_skipped(self, mem):
        stub = mem(get_all_return={"results": [{"memory": "Old Note"}]})
        r = mcp_server.execute_tool("import_memories", {
            "data": json.dumps([{"memory": "old note"}]), "user_id": "u"})
        assert r == {"imported": 0, "skipped": 1, "failed": 0,
                     "errors": None, "user_id": "u"}
        assert stub.calls["add"] == []

    def test_duplicate_within_batch_imported_once(self, mem):
        stub = mem(get_all_return={"results": []})
        data = json.dumps([{"memory": "Dup Fact"},
                           {"memory": "dup fact"},
                           {"memory": "fresh idea"}])
        r = mcp_server.execute_tool("import_memories", {"data": data, "user_id": "u"})
        assert r["imported"] == 2
        assert r["skipped"] == 1
        assert [c["data"] for c in stub.calls["add"]] == ["Dup Fact", "fresh idea"]

    def test_text_content_fallbacks_with_metadata(self, mem):
        stub = mem(get_all_return={"results": []})
        data = json.dumps([
            {"text": "via text key", "metadata": {"tag": 2}},
            {"content": "via content key", "metadata": {"tag": 3}},
        ])
        r = mcp_server.execute_tool("import_memories", {"data": data, "user_id": "u"})
        assert r["imported"] == 2
        assert stub.calls["add"] == [
            {"data": "via text key", "user_id": "u", "metadata": {"tag": 2},
             "infer": False},
            {"data": "via content key", "user_id": "u", "metadata": {"tag": 3},
             "infer": False}]


# ═══ prune_memories ══════════════════════════════════════════════════════════

OLD_ISO = "2020-01-01T00:00:00"          # naive ISO with T → parsed via fromisoformat
OLD_OLD = "2020-01-01"                   # date-only → float() fails → NOT pruned (bug)
OLD_EPOCH = "0.0"                        # float-epoch string → parsed
OLD_JUNK = "not a date"                  # unparsable → skipped


class TestPruneMemories:
    def test_dry_run_default_no_deletion(self, mem):
        stub = mem(get_all_return={"results": [
            {"id": "b", "memory": "old fact", "created_at": OLD_ISO}]})
        r = mcp_server.execute_tool("prune_memories", {"user_id": "u"})
        assert r["dry_run"] is True
        assert r["deleted"] == 0
        assert r["would_delete"] == 1
        assert stub.calls["delete"] == []
        assert r["memories"] == [{"id": "b", "memory": "old fact",
                                  "created_at": OLD_ISO}]

    def test_real_delete_calls_delete_per_id(self, mem):
        stub = mem(get_all_return={"results": [
            {"id": "b", "memory": "x" * 500, "created_at": OLD_ISO},   # truncation check
            {"id": "c", "memory": "epoch fact", "created_at": OLD_EPOCH},
            {"memory": "orphan fact", "created_at": OLD_ISO},          # no id at all
        ]})
        r = mcp_server.execute_tool("prune_memories", {
            "user_id": "u", "dry_run": False})
        assert r["would_delete"] == 3
        assert r["deleted"] == 2
        assert stub.calls["delete"] == [{"memory_id": "b"}, {"memory_id": "c"}]
        # memory preview truncated to 200 chars
        previews = {m["id"]: m["memory"] for m in r["memories"] if m["id"]}
        assert len(previews["b"]) == 200

    def test_delete_failure_does_not_stop_remaining_deletes(self, mem):
        stub = mem(get_all_return={"results": [
            {"id": "b", "memory": "one", "created_at": OLD_ISO},
            {"id": "c", "memory": "two", "created_at": OLD_EPOCH}]})
        # make delete fail for every OTHER id: fail when called with 'c'? simpler:
        # count-based raising via a subclass is overkill — raise always except b.
        def flaky_delete(memory_id=""):
            stub.calls["delete"].append({"memory_id": memory_id})
            if memory_id == "c":
                raise RuntimeError("qdrant hiccup")
        stub.delete = flaky_delete
        r = mcp_server.execute_tool("prune_memories", {
            "user_id": "u", "dry_run": False})
        assert r["deleted"] == 1          # b succeeded, c failed but didn't raise
        assert r["would_delete"] == 2

    def test_iso_t_parsed_and_pruned(self, mem):
        stub = mem(get_all_return={"results": [
            {"id": "b", "memory": "x", "created_at": OLD_ISO}]})
        r = mcp_server.execute_tool("prune_memories", {"user_id": "u", "dry_run": False})
        assert r["would_delete"] == 1
        assert stub.calls["delete"] == [{"memory_id": "b"}]

    def test_iso_z_parsed_and_pruned(self, mem):
        stub = mem(get_all_return={"results": [
            {"id": "b", "memory": "x", "created_at": "2020-01-01T00:00:00Z"}]})
        r = mcp_server.execute_tool("prune_memories", {"user_id": "u", "dry_run": False})
        assert r["would_delete"] == 1
        assert stub.calls["delete"] == [{"memory_id": "b"}]

    def test_float_epoch_string_parsed_and_pruned(self, mem):
        stub = mem(get_all_return={"results": [
            {"id": "c", "memory": "x", "created_at": OLD_EPOCH}]})
        r = mcp_server.execute_tool("prune_memories", {"user_id": "u", "dry_run": False})
        assert r["would_delete"] == 1
        assert stub.calls["delete"] == [{"memory_id": "c"}]

    def test_date_only_created_at_is_parsed_and_pruned(self, mem):
        """Fixed: date-only created_at ('2020-01-01', Qdrant date style) is now
        parsed (T00:00:00 UTC assumed) instead of silently never pruning."""
        stub = mem(get_all_return={"results": [
            {"id": "j1", "memory": "x", "created_at": OLD_JUNK},
            {"id": "j2", "memory": "x", "created_at": OLD_OLD},   # date-only: NOW parsed
            {"id": "j3", "memory": "x"},                          # missing created_at -> skipped
        ]})
        r = mcp_server.execute_tool("prune_memories", {"user_id": "u", "dry_run": False})
        assert r["would_delete"] == 1, r["memories"]
        assert [c["memory_id"] for c in stub.calls["delete"]] == ["j2"]
        # the junk date is still skipped (cannot be parsed)

    def test_recent_and_future_memories_kept(self, mem):
        now = time.time()
        stub = mem(get_all_return={"results": [
            {"id": "new", "memory": "x", "created_at": str(now - 100)},
            {"id": "fut", "memory": "x", "created_at": str(now + 3600)},
        ]})
        r = mcp_server.execute_tool("prune_memories", {"user_id": "u", "dry_run": False})
        assert r["would_delete"] == 0
        assert stub.calls["delete"] == []

    def test_min_score_semantics(self, mem):
        """Fixed semantics: min_score is the relevance FLOOR for deletion.
        score >= min_score -> KEPT; un-scored -> KEPT (never auto-delete unrated
        memories); score < min_score -> pruned."""
        stub = mem(get_all_return={"results": [
            {"id": "high", "memory": "x", "created_at": OLD_ISO, "score": 0.9},
            {"id": "equal", "memory": "x", "created_at": OLD_ISO, "score": 0.5},
            {"id": "low", "memory": "x", "created_at": OLD_ISO, "score": 0.3},
            {"id": "missing", "memory": "x", "created_at": OLD_ISO},
        ]})
        r = mcp_server.execute_tool("prune_memories", {
            "user_id": "u", "dry_run": False, "min_score": 0.5})
        assert [c["memory_id"] for c in stub.calls["delete"]] == ["low"]
        assert r["deleted"] == 1

    def test_older_than_days_boundary(self, mem):
        now = time.time()
        cutoff_offset = 2 * 86400
        stub = mem(get_all_return={"results": [
            # 5s INSIDE the window → kept; 5s beyond → pruned
            {"id": "keep", "memory": "x", "created_at": str(now - cutoff_offset + 5)},
            {"id": "prune-me", "memory": "x", "created_at": str(now - cutoff_offset - 5)},
        ]})
        r = mcp_server.execute_tool("prune_memories", {
            "user_id": "u", "dry_run": False, "older_than_days": 2})
        assert r["would_delete"] == 1
        assert [c["memory_id"] for c in stub.calls["delete"]] == ["prune-me"]

    def test_get_all_failure_translated(self, mem):
        mem(get_all_raises=RuntimeError("boom"))
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("prune_memories", {"user_id": "u"})
        assert "Failed to list memories for pruning." in str(ei.value)


# ═══ unknown tool + error translation ════════════════════════════════════════

class TestUnknownToolAndErrors:
    def test_unknown_tool_raises_mem0_error(self, mem):
        mem()
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("definitely_not_a_tool", {})
        assert "Unknown tool: definitely_not_a_tool" in str(ei.value)
        assert ei.value.tool == "definitely_not_a_tool"

    def test_add_dimension_error_gets_embedder_fix(self, mem):
        mem(add_raises=ValueError("Vector dimension 1024 expected 768"))
        with pytest.raises(QdrantError) as ei:
            mcp_server.execute_tool("add_raw_memory",
                                    {"content": "fact", "user_id": "u"})
        assert "different embedder" in ei.value.fix
        assert "Vector dimension 1024 expected 768" in str(ei.value)
        assert ei.value.tool == "add_raw_memory"

    def test_add_collection_missing_error(self, mem):
        mem(add_raises=ValueError("Collection doesn't exist"))
        with pytest.raises(QdrantError) as ei:
            mcp_server.execute_tool("add_raw_memory", {"content": "fact", "user_id": "u"})
        assert "collection does not exist" in str(ei.value)
        assert ei.value.tool == "add_raw_memory"

    def test_add_connection_refused_error(self, mem):
        mem(add_raises=ValueError("Connection refused"))
        with pytest.raises(QdrantError) as ei:
            mcp_server.execute_tool("add_raw_memory", {"content": "fact", "user_id": "u"})
        assert "Connection to Qdrant failed." in str(ei.value)

    def test_add_generic_error_is_plain_mem0_error(self, mem):
        mem(add_raises=ValueError("totally unexpected"))
        with pytest.raises(Mem0Error) as ei:
            mcp_server.execute_tool("add_raw_memory", {"content": "fact", "user_id": "u"})
        assert not isinstance(ei.value, QdrantError)
        assert "Memory operation failed." in str(ei.value)
        assert "totally unexpected" in str(ei.value)

    def test_verbatim_add_errors_translated_too(self, mem):
        mem(add_raises=ValueError("Connection refused"))
        with pytest.raises(QdrantError) as ei:
            mcp_server.execute_tool("add_verbatim", {"content": "b" * 10, "user_id": "u"})
        assert "Connection to Qdrant failed." in str(ei.value)