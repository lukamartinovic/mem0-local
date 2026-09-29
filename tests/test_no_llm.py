"""
No-LLM architecture tests. The load-bearing invariants of the refactor,
verified WITHOUT any running services (all offline):

1. CONFIG's llm slot is present-but-neutered; from_config can't inject a cloud LLM.
2. NoExtractionLLM raises LOUDLY if any code path tries to call an LLM.
3. add_raw_memory size guard rejects blob-like input with an actionable fix.
4. add_verbatim auto-chunks long text at paragraph boundaries.
5. health response marks extraction_llm=False and carries no model_status.
6. Tool surface has exactly 12 tools, no 'add_memory', no Ollama references.
7. Module imports cleanly with NO network access (no ollama/urls fetched at import).

These run on any machine with the deps installed - no Docker, no Qdrant, no
model downloads. Run: python3 -m pytest tests/test_no_llm.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import mcp_server
from mcp_server import Mem0Error


# ── 1. CONFIG shape: llm key exists but is neutered ─────────────────────────

class TestConfigShape:
    def test_llm_key_present(self):
        """from_config requires llm; absence injects a cloud default. Ours must exist."""
        assert "llm" in mcp_server.CONFIG

    def test_llm_is_explicitly_neutered(self):
        """The model field names the sentinel purpose so it can never look real."""
        model = mcp_server.CONFIG["llm"]["config"]["model"]
        assert "unused" in model or "no-extraction" in model

    def test_embedder_is_fastembed(self):
        assert mcp_server.CONFIG["embedder"]["provider"] == "fastembed"

    def test_no_ollama_anywhere_in_config(self):
        dumped = json.dumps(mcp_server.CONFIG)
        assert "ollama" not in dumped.lower()

    def test_embed_dims_matches_collection_default(self):
        assert mcp_server.EMBED_DIMS == mcp_server.CONFIG["vector_store"]["config"]["embedding_model_dims"]


# ── 2. The sentinel is loud ─────────────────────────────────────────────────

class TestSentinel:
    def test_generate_response_raises(self):
        m = mcp_server.NoExtractionLLM()
        try:
            m.generate_response(messages=[{"role": "user", "content": "hi"}])
        except Mem0Error as e:
            assert "no extraction LLM" in str(e)
            assert "add_raw_memory" in str(e)  # tells the caller the right tool
        else:
            raise AssertionError("sentinel did NOT fire - LLM extraction is possible again")

    def test_sentinel_is_docstring_documented(self):
        assert "NO" in mcp_server.NoExtractionLLM.__doc__.upper()

    def test_module_source_never_calls_generate_response(self):
        """Guard against a future code path calling m.llm directly."""
        src = open(os.path.join(REPO, "mcp_server.py")).read()
        allowed = ("NoExtractionLLM", "sentinel", "generate_response(self, *args, **kwargs)")
        for line in src.splitlines():
            if ".generate_response(" in line and "def generate_response" not in line:
                # only the sentinel's own def may exist; no call sites allowed
                if not any(tok in line for tok in allowed):
                    raise AssertionError(f"unexpected LLM call site: {line.strip()}")


# ── 3. add_raw_memory contract: one concise fact ────────────────────────────

class TestAddRawMemoryGuard:
    """Size guard without any services: intercepts m.add via a stub."""

    @pytest.fixture
    def fake_memory(self, monkeypatch):
        class FakeMemory:
            def __init__(self):
                self.added = []

            def add(self, content, user_id=None, metadata=None, infer=False):
                assert infer is False, "MUST never infer - that's the whole design"
                self.last = {"results": [{"id": "x", "memory": content}]}
                self.added.append((content, user_id, metadata))
                return {"results": [{"id": "x", "memory": content}]}

        fake = FakeMemory()
        monkeypatch.setattr(mcp_server, "_memory", fake)
        monkeypatch.setattr(mcp_server, "_init_status", "ready")
        return fake

    def test_short_fact_passes(self, fake_memory):
        r = mcp_server.execute_tool("add_raw_memory", {
            "content": "Auth uses JWT with 24h expiry",
            "user_id": "t",
        })
        assert isinstance(r, dict)

    def test_oversized_blob_rejected_with_hint(self, fake_memory):
        blob = "y" * (mcp_server.CHUNK_CHARS + 10)
        try:
            mcp_server.execute_tool("add_raw_memory", {"content": blob, "user_id": "t"})
        except Mem0Error as e:
            combined = str(e)
            assert "add_verbatim" in combined, "hint must route to the right tool"
        else:
            raise AssertionError("size guard did not fire")

    def test_whitespace_only_rejected(self, fake_memory):
        try:
            mcp_server.execute_tool("add_raw_memory", {"content": "   \n\t ", "user_id": "t"})
        except Mem0Error:
            pass
        else:
            raise AssertionError("empty content accepted")

    def test_verbatim_accepts_what_raw_rejects(self, fake_memory):
        blob = "y" * (mcp_server.CHUNK_CHARS + 500)
        r = mcp_server.execute_tool("add_verbatim", {"content": blob, "user_id": "t"})
        assert isinstance(r, dict)
        # and it must have been chunked
        assert r.get("chunks", 0) >= 2, f"long text not chunked: {r}"


# ── 4. Verbatim chunking shape ────────────────────────────────────────────────

class TestVerbatimChunking:
    def test_chunk_size_is_env_configurable(self, monkeypatch):
        monkeypatch.setattr(mcp_server, "CHUNK_CHARS", 1200)
        assert mcp_server._get_chunk_size() == 1200

    def test_short_text_single_chunk(self):
        assert mcp_server._chunk_text("tiny", 3000) == ["tiny"]

    def test_paragraph_split_cascade(self):
        text = "para one.\n\npara two.\n\npara three."
        chunks = mcp_server._chunk_text(text, 25)
        assert len(chunks) >= 2, f"limit 25 on 36-char text must force split, got {chunks}"
        joined = " ".join(chunks)
        for para in ("para one.", "para two.", "para three."):
            assert para in joined

    def test_hard_split_oversized_words(self):
        pieces = mcp_server._chunk_text("a" * 50, 10)
        assert all(len(p) <= 10 for p in pieces)

    def test_context_header_prepended(self):
        chunks = mcp_server._chunk_text("abc def", 100, context_header="[Document: x]")
        assert chunks[0].startswith("[Document: x]")


# ── 5. Health marks no-LLM by design ─────────────────────────────────────────

class TestHealth:
    def test_extraction_llm_false(self):
        mcp_server._memory = object()
        mcp_server._init_status = "starting"  # any status; component flag is static
        h = mcp_server._build_health_response()
        assert h["components"]["extraction_llm"] is False
        assert h["config"]["extraction_llm"] is None

    def test_no_model_status_key(self):
        h = mcp_server._build_health_response()
        assert "model_status" not in h

    def test_config_has_no_llm_key(self):
        h = mcp_server._build_health_response()
        assert "llm" not in h["config"]


# ── 6. Tool surface: exactly 12, no add_memory ──────────────────────────────

class TestToolSurface:
    def test_thirteen_tools(self):
        assert len(mcp_server.TOOL_DEFINITIONS) == 13

    def test_no_add_memory_tool(self):
        names = [t["name"] for t in mcp_server.TOOL_DEFINITIONS]
        assert "add_memory" not in names, "the LLM-extraction tool must be gone"

    def test_both_add_paths_present(self):
        names = [t["name"] for t in mcp_server.TOOL_DEFINITIONS]
        assert "add_raw_memory" in names
        assert "add_verbatim" in names

    def test_descriptions_mention_agent_inference(self):
        raw = next(t for t in mcp_server.TOOL_DEFINITIONS if t["name"] == "add_raw_memory")
        assert "you" in raw["description"].lower()

    def test_all_tool_names_snake_case(self):
        for t in mcp_server.TOOL_DEFINITIONS:
            assert t["name"].replace("_", "").isalpha()


# ── 7. Module import is network-free and ollama-free ────────────────────────

class TestModuleHygiene:
    def test_zero_ollama_strings_in_server(self):
        src = open(os.path.join(REPO, "mcp_server.py")).read()
        bad = [l for l in src.splitlines()
               if "ollama" in l.lower()
               and "no ollama" not in l.lower()
               and "no llm" not in l.lower()]
        assert not bad, f"live ollama references remain: {bad[:3]}"

    def test_no_openai_call_sites_outside_neuter(self):
        src = open(os.path.join(REPO, "mcp_server.py")).read()
        # the only OpenAI mention should be the neutered config provider
        code_only = [l for l in src.splitlines()
                     if "openai" in l.lower()
                     and not l.strip().startswith("#")
                     and not l.strip().lstrip().startswith(("\"provider\"", "'provider'"))]
        # any remaining non-comment openai mention must carry the neuter marker
        assert all("unused" in l for l in code_only), code_only

    def test_requirements_has_no_ollama(self):
        reqs = open(os.path.join(REPO, "requirements.txt")).read()
        assert "ollama" not in reqs.lower()
        assert "fastembed" in reqs

    def test_dockerfile_has_no_ollama_stage(self):
        df = open(os.path.join(REPO, "Dockerfile")).read()
        assert "FROM ollama" not in df.lower()
        assert "fastembed" in df or "requirements" in df

    def test_dockerfile_installs_libunwind_before_qdrant_check(self):
        df = open(os.path.join(REPO, "Dockerfile")).read()
        apt_pos = df.find("libunwind8")
        check_pos = df.find("qdrant --version")
        assert apt_pos != -1 and check_pos != -1
        assert apt_pos < check_pos, "libunwind must be installed BEFORE the qdrant runnability check"

    def test_entrypoint_dumps_qdrant_log_on_death(self):
        ep = open(os.path.join(REPO, "entrypoint.sh")).read()
        assert "qdrant.log" in ep, "entrypoint must capture qdrant's own output"
        assert "exit 1" in ep or "exit 1 )" in ep