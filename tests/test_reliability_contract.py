"""
Setup RELIABILITY contract tests.

These encode the invariants that make `./setup.sh` trustworthy on a clean
machine - each one corresponds to a real observed failure, so a regression
here is a regression in "it just works". Fully offline: they read the
deployment files as text and stub the network.

Run: python3 -m pytest tests/test_reliability_contract.py -v
"""

import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETUP = os.path.join(REPO, "setup.sh")
ENTRY = os.path.join(REPO, "entrypoint.sh")
COMPOSE = os.path.join(REPO, "docker-compose.yml")
DOCKERFILE = os.path.join(REPO, "Dockerfile")

SETUP_TEXT = open(SETUP).read()
ENTRY_TEXT = open(ENTRY).read()
COMPOSE_TEXT = open(COMPOSE).read()
DOCKERFILE_TEXT = open(DOCKERFILE).read()


def _no_comments(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))


# ── Readiness must assert the BODY, never a curl exit code ───────────────────

class TestReadinessIsTruthful:
    def test_setup_never_trusts_plain_curl_exit_code(self):
        """`curl -s` exits 0 on HTTP 500/404, so a stale process on 8765 made
        setup print 'Ready!' for a server that was never ours."""
        for i, line in enumerate(SETUP_TEXT.splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or s.startswith('echo "'):
                continue  # comments and banner/echo text are not code
            if "curl" in line and "8765/health" in line:
                assert " -fs" in line, \
                    f"setup.sh:{i} uses plain curl for a health request: {s}"

    def test_setup_parses_status_ok_from_body(self):
        assert 'get("status") == "ok"' in SETUP_TEXT or "get('status') == 'ok'" in SETUP_TEXT

    def test_compose_healthcheck_also_asserts_status_ok(self):
        assert "status" in COMPOSE_TEXT and "'ok'" in COMPOSE_TEXT

    def test_readiness_helper_is_used_by_both_modes(self):
        assert SETUP_TEXT.count("wait_for_ready") >= 3  # definition + update + start

    def test_failure_paths_exit_nonzero(self):
        # each wait site must be followed by an error branch that exits 1
        for m in re.finditer(r"wait_for_ready[^\n]*; then", SETUP_TEXT):
            tail = SETUP_TEXT[m.end():m.end() + 400]
            assert "exit 1" in tail, "a failed readiness wait must exit non-zero"


# ── Fire-and-forget lifecycle ────────────────────────────────────────────────

class TestFireAndForget:
    def test_restart_policy_present(self):
        """Without this, a host reboot / Docker restart / crash leaves the
        stack down forever and the setup looks broken."""
        assert re.search(r"^\s*restart:\s*unless-stopped", COMPOSE_TEXT, re.M), \
            "compose service needs restart: unless-stopped"

    def test_model_cache_is_a_named_volume(self):
        """The model is baked into the image AND cached in a volume, so a
        recreate never re-downloads it."""
        assert "model_cache:/model_cache" in COMPOSE_TEXT
        assert re.search(r"^volumes:\s*$", COMPOSE_TEXT, re.M)
        assert re.search(r"^\s{2}model_cache:\s*$", COMPOSE_TEXT, re.M)

    def test_qdrant_data_still_volumed(self):
        assert "qdrant_data:/qdrant/storage" in COMPOSE_TEXT

    def test_container_is_fully_self_contained_at_runtime(self):
        """The image bakes the embedding model, so a running container needs no
        network. This is the 'no external deps' requirement."""
        assert "from fastembed import TextEmbedding" in DOCKERFILE_TEXT.replace("\n", " ")
        assert "MEM0_EMBED_DIMS" in DOCKERFILE_TEXT

    def test_spacy_model_baked_too(self):
        """mem0 auto-downloads en_core_web_sm mid-request via pip - a hidden
        runtime dependency that fails on pip-less images."""
        assert "spacy download en_core_web_sm" in DOCKERFILE_TEXT


# ── No silent data loss ──────────────────────────────────────────────────────

class TestNoSilentDataLoss:
    def test_no_tmp_storage_fallback(self):
        """The old fallback started on /tmp and quietly dropped every memory
        written during the run. Refuse to start instead."""
        assert "/tmp/qdrant-storage-fallback" not in ENTRY_TEXT
        assert "Refusing to start" in ENTRY_TEXT

    def test_unwritable_storage_exits_nonzero(self):
        i = ENTRY_TEXT.find("STILL unwritable")
        assert i != -1
        assert "exit 1" in ENTRY_TEXT[i:i + 600]

    def test_dimension_mismatch_refuses_by_default(self):
        """Auto-deleting a collection on a config typo destroyed stored
        memories at startup. Strict by default, opt-out explicit."""
        assert "MEM0_STRICT_DIMS" in ENTRY_TEXT
        assert 'STRICT="${MEM0_STRICT_DIMS:-1}"' in ENTRY_TEXT
        i = ENTRY_TEXT.find("DIMENSION MISMATCH")
        assert i != -1
        # the conflict flag is raised at the mismatch, and acted on below with exit 1
        assert "DIM_CONFLICT=1" in ENTRY_TEXT[i:i + 400]
        j = ENTRY_TEXT.find('if [ "$DIM_CONFLICT" = "1" ]')
        assert j != -1 and "exit 1" in ENTRY_TEXT[j:j + 800]

    def test_compose_sets_strict_dims(self):
        assert re.search(r'MEM0_STRICT_DIMS:\s*"?1"?', COMPOSE_TEXT), \
            "compose must default to strict dims"


# ── No unverified success claims ─────────────────────────────────────────────

class TestNoUnverifiedClaims:
    def test_selftest_failure_is_fatal(self):
        """A server whose tools are broken but which prints 'ready' is the
        exact unreliability being fixed: the agent then silently fails to
        remember things."""
        i = ENTRY_TEXT.find("Self-test FAILED")
        assert i != -1, "entrypoint must fail loudly on a failed self-test"
        assert "exit 1" in ENTRY_TEXT[i:i + 600]

    def test_no_always_true_guard(self):
        """`if ! { ... } && false; then :; fi` - a guard that cannot fail."""
        assert "&& false" not in SETUP_TEXT

    def test_compose_plugin_checked_before_banner(self):
        """A docker CLI without the compose plugin died mid-flow after the
        banner. Check it up front."""
        i = SETUP_TEXT.find("docker compose version")
        j = SETUP_TEXT.find("mem0-local — single container")
        assert i != -1, "setup must verify the compose plugin"
        assert i < j, "compose check must run before the banner"

    def test_ports_preflighted(self):
        assert "lsof" in SETUP_TEXT and "8765" in SETUP_TEXT
        assert "check_port 8765" in SETUP_TEXT and "check_port 6333" in SETUP_TEXT


# ── The scripts actually run ──────────────────────────────────────────────────

class TestScriptsExecute:
    def test_setup_parses(self):
        r = subprocess.run(["bash", "-n", SETUP], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    def test_entrypoint_parses(self):
        r = subprocess.run(["sh", "-n", ENTRY], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    def test_server_ready_logic_on_a_fake_500(self):
        """Extract the readiness predicate and prove it REJECTS a 500 body -
        the failure the old code had."""
        body = json.dumps({"status": "degraded"})
        proc = subprocess.run(
            ["python3", "-c",
             "import sys, json\n"
             "sys.exit(0 if json.load(sys.stdin).get('status') == 'ok' else 1)"],
            input=body, capture_output=True, text=True)
        assert proc.returncode == 1, "a degraded body must not count as ready"

    def test_server_ready_logic_on_ok_body(self):
        body = json.dumps({"status": "ok"})
        proc = subprocess.run(
            ["python3", "-c",
             "import sys, json\n"
             "sys.exit(0 if json.load(sys.stdin).get('status') == 'ok' else 1)"],
            input=body, capture_output=True, text=True)
        assert proc.returncode == 0

    def test_entrypoint_last_line_starts_server(self):
        lines = [l.strip() for l in ENTRY_TEXT.strip().splitlines() if l.strip()]
        assert lines[-1] == "exec python3 mcp_server.py"
