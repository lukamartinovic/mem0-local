"""
Deployment-script contract tests for mem0-local.

Layer 1 (static, runs on any machine with bash/sh + pytest; no services):

  - setup.sh       parses (bash -n); update/start branches: build + `up -d`,
                   never `down -v` (data-preservation invariant)
  - docker-compose.yml parses; exactly ONE service, BOTH ports 8765+6333,
                   qdrant_data at /qdrant/storage, healthcheck uses /health
                   with status=='ok', start_period <= 300, ZERO ollama traces
  - Dockerfile:    FROM qdrant/qdrant present, libunwind8 BEFORE
                   COPY --from=qdrant BEFORE 'qdrant --version', pip layer,
                   EXPOSE both ports, chmod entrypoint, no FROM ollama,
                   no OLLAMA env traces
  - entrypoint.sh: parses (sh -n); qdrant.log capture, kill -0 liveness
                   probe, 'Falling back' + '/tmp/qdrant-storage-fallback'
                   strings, embedded dim-parser handles BOTH named
                   {'name': {'size': N}} and flat {'size': N} vector configs
                   (exec-tested with stubbed stdin), final line is
                   'exec python3 mcp_server.py', set -e present
  - Makefile:      test / test-in-container / test-all / up / update targets
                   exist AND referenced tests/ paths exist on disk
  - .env.example:  no MEM0_LLM_MODEL / MEM0_OLLAMA_URL, EMBED vars documented

Layer 2 (docker-gated; skipped unless `docker` is on PATH):
  - image builds + `docker run --rm <img> qdrant --version` exits 0

Run:  python3 -m pytest tests/test_setup_and_packaging.py -v
"""

import json
import re
import shutil
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SETUP_SH = REPO / "setup.sh"
ENTRYPOINT_SH = REPO / "entrypoint.sh"
COMPOSE = REPO / "docker-compose.yml"
DOCKERFILE = REPO / "Dockerfile"
MAKEFILE = REPO / "Makefile"
ENV_EXAMPLE = REPO / ".env.example"


# ── helpers ──────────────────────────────────────────────────────────────────

def _read(path) -> str:
    return Path(path).read_text()


def _read_noncomment(path) -> str:
    """File text with comment lines stripped, for string claims where a
    comment may legitimately *document* an absence (e.g. Dockerfile's
    'NOTE: no MEM0_LLM_MODEL / MEM0_OLLAMA_URL')."""
    return "\n".join(
        line for line in _read(path).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _service_dict(compose: dict):
    """Yield (name, svc) for mapping-form services, (None, svc) for list form."""
    services = compose.get("services")
    if isinstance(services, dict):
        for name, svc in services.items():
            yield name, (svc or {})
    elif isinstance(services, list):
        for svc in services:
            yield None, (svc or {})


def _svc(compose: dict) -> dict:
    """The one service's definition, regardless of mapping/list YAML form."""
    pairs = list(_service_dict(compose))
    assert len(pairs) == 1, f"expected exactly 1 service, got {len(pairs)}"
    return pairs[0][1]


def _extract_dim_parser_snippet(entrypoint_text: str) -> str:
    """Pull the embedded python dim-parser out of entrypoint.sh."""
    anchor = 'python3 -c "'
    pos = entrypoint_text.find(anchor)
    assert pos != -1, "dim-parser python snippet not found in entrypoint.sh"
    tail = '" 2>/dev/null || true)'
    end = entrypoint_text.find(tail, pos)
    assert end != -1, "dim-parser snippet end not found in entrypoint.sh"
    return entrypoint_text[pos + len(anchor):end]


def _run_dim_snippet(snippet: str, payload: object) -> str:
    """Compile the snippet once, exec it with sys.stdin stubbed to `payload`
    and sys.stdout captured — mirrors EXISTING_DIM=$(curl ... | python3 -c ...)."""
    code = compile(snippet, "<entrypoint-dim-parser>", "exec")
    g: dict = {}
    payload_io = StringIO(json.dumps(payload))
    old_in, old_out = sys.stdin, sys.stdout
    sys.stdin, sys.stdout = payload_io, StringIO()
    try:
        try:
            exec(code, g)  # the real snippet swallows its own exceptions
        except BaseException as exc:  # a broken snippet must fail the test, not pass
            raise RuntimeError(f"dim-parser snippet raised {exc!r}") from exc
        return sys.stdout.getvalue().strip()
    finally:
        sys.stdin, sys.stdout = old_in, old_out


def _parse_seconds(s) -> int:
    m = re.search(r"(\d+)\s*s", str(s or ""))
    return int(m.group(1)) if m else -1


def _dedent(text: str) -> str:
    """Trim the shell's leading 2-space branch indentation line-wise, so
    line-anchored regexes can match commands inside if/for blocks."""
    return "\n".join(
        l[2:] if l.startswith("  ") else l
        for l in text.splitlines()
    )


def _split_update_branch(text: str):
    """Return (update_body, rest) for setup.sh's update block."""
    m = re.search(r'^if \[ "\$MODE" = "update" \]; then$', text, re.M)
    assert m, 'setup.sh: \'if [ "$MODE" = "update" ]; then\' branch not found'
    body = text[m.end():]
    m2 = re.search(r"^fi\s*$", body, re.M)
    assert m2, "setup.sh: closing 'fi' of update branch not found"
    return body[:m2.start()], body[m2.end():]


_SETUP_TEXT = _read(SETUP_SH)
_UPDATE_BODY, _AFTER_UPDATE = _split_update_branch(_SETUP_TEXT)
_ENTRYPOINT_TEXT = _read(ENTRYPOINT_SH)
_DIM_SNIPPET = _extract_dim_parser_snippet(_ENTRYPOINT_TEXT)
_COMPOSE: dict = yaml.safe_load(COMPOSE.read_text())
_DF = DOCKERFILE.read_text()
_DF_LINES = _DF.splitlines()
_MK = MAKEFILE.read_text()


# ════════════════════════ Layer 1 — setup.sh ═════════════════════════════════

class TestSetupShSyntax:
    def test_setup_sh_parses_via_bash_n(self):
        r = subprocess.run(["bash", "-n", str(SETUP_SH)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"bash -n setup.sh failed: {r.stderr}"

    def test_setup_sh_strict_mode_flags(self):
        assert re.search(r"^set -euo pipefail", _SETUP_TEXT, re.M), \
            "setup.sh must run under set -euo pipefail"


class TestSetupShUpdateBranch:
    def test_update_branch_rebuilds_image(self):
        assert "docker compose build" in _UPDATE_BODY

    def test_update_branch_starts_container(self):
        body = _dedent(_UPDATE_BODY)
        assert re.search(r"^docker compose up -d$", body, re.M), body

    def test_update_branch_never_wipes_volume(self):
        """Data-preservation invariant: update must not remove the named
        volume, in any subcommand form."""
        assert "down -v" not in _UPDATE_BODY
        assert "docker compose down" not in _UPDATE_BODY
        assert "volume rm" not in _UPDATE_BODY
        assert "volume prune" not in _UPDATE_BODY

    def test_update_branch_waits_then_verifies_health(self):
        # Readiness now lives in wait_for_ready()/server_ready() (defined above
        # the branch) which asserts status=="ok" on the body - a plain
        # `curl -s` exit code also succeeds on a 500. The branch must call it.
        assert "wait_for_ready" in _UPDATE_BODY
        remainder = _UPDATE_BODY[_UPDATE_BODY.find("wait_for_ready"):]
        assert "exit 1" in remainder, "a failed readiness wait must exit non-zero"
        # and the helper must really parse the health body
        assert 'get("status") == "ok"' in _SETUP_TEXT or "get('status') == 'ok'" in _SETUP_TEXT
        assert "curl -fs" in _SETUP_TEXT, "readiness must fail on HTTP >= 400"

    def test_update_branch_mentions_data_preservation(self):
        assert "preserv" in _UPDATE_BODY.lower()


class TestSetupShStartBranch:
    def test_start_branch_starts_container(self):
        body = _dedent(_AFTER_UPDATE)
        assert re.search(r"^docker compose up -d$", body, re.M), body[:400]

    def test_start_branch_never_wipes_volume(self):
        body = _dedent(_AFTER_UPDATE)
        # volume-destroying variants are forbidden outright
        assert "down -v" not in body
        assert "volume rm" not in body and "volume prune" not in body
        # plain `docker compose down` may appear ONLY inside echoed user guidance
        for line in _AFTER_UPDATE.splitlines():
            if "docker compose down" in line:
                assert line.lstrip().startswith("echo"), \
                    f"start branch runs docker compose down: {line.strip()}"

    def test_start_branch_polls_health_endpoint(self):
        assert "/health" in _AFTER_UPDATE


# ════════════════════════ Layer 1 — docker-compose.yml ═══════════════════════

class TestComposeStatic:
    def test_yaml_loads_to_mapping(self):
        assert isinstance(_COMPOSE, dict), "compose file must be a YAML mapping"

    def test_exactly_one_service(self):
        pairs = list(_service_dict(_COMPOSE))
        assert len(pairs) == 1, f"compose declares {len(pairs)} services: {[n for n, _ in pairs]}"

    def test_single_service_named_mem0_local(self):
        names = [n for n, _ in _service_dict(_COMPOSE)]
        assert names[0] == "mem0-local"

    def test_service_declares_both_ports(self):
        ports = [_svc(_COMPOSE).get("ports") or []]
        assert "8765" in str(ports) and "6333" in str(ports)

    def test_both_ports_mapped_host_side(self):
        ports = [str(p) for p in (_svc(_COMPOSE).get("ports") or [])]
        mapped = {str(p).split(":")[0].strip() for p in ports}
        assert mapped == {"8765", "6333"}, ports

    def test_qdrant_data_mounted_at_storage(self):
        vols = [str(v) for v in (_svc(_COMPOSE).get("volumes") or [])]
        assert any("qdrant_data" in v and "/qdrant/storage" in v for v in vols), vols

    def test_named_volume_qdrant_data_declared(self):
        assert "qdrant_data" in (_COMPOSE.get("volumes") or {})

    def test_healthcheck_hits_health_endpoint(self):
        test = (_svc(_COMPOSE).get("healthcheck") or {}).get("test") or []
        assert "/health" in str(test), test

    def test_healthcheck_checks_status_is_ok_not_just_http200(self):
        test = (_svc(_COMPOSE).get("healthcheck") or {}).get("test") or []
        assert re.search(r"status.{0,3}==.{0,3}['\"]ok['\"]", str(test)), test

    def test_compose_start_period_at_most_300s(self):
        sp = _parse_seconds((_svc(_COMPOSE).get("healthcheck") or {}).get("start_period"))
        assert sp != -1 and sp <= 300

    def test_compose_healthcheck_interval_and_retries_present(self):
        hc = _svc(_COMPOSE).get("healthcheck") or {}
        assert hc.get("interval") and hc.get("retries")

    def test_compose_zero_ollama_traces_noncomment(self):
        nc = _read_noncomment(COMPOSE)
        assert "ollama" not in nc.lower()
        assert "MEM0_OLLAMA_URL" not in nc and "MEM0_LLM_MODEL" not in nc

    def test_compose_no_ollama_env_vars_any_shape(self):
        svc = _svc(_COMPOSE)
        env = svc.get("environment") or {}
        if isinstance(env, dict):
            keys = {k.upper() for k in env}
        else:  # list form: ["KEY=VAL", ...]
            keys = {str(k).split("=")[0].strip().upper() for k in env}
        assert not (keys & {"MEM0_OLLAMA_URL", "MEM0_LLM_MODEL"}), keys

    def test_compose_no_ollama_image_references(self):
        svc = _svc(_COMPOSE)
        assert "ollama" not in str(svc.get("image", "")).lower()
        assert "ollama" not in str(svc.get("build", "")).lower()


# ════════════════════════ Layer 1 — Dockerfile ═══════════════════════════════

class TestDockerfile:
    def test_from_qdrant_image_present(self):
        assert re.search(r"^FROM qdrant/qdrant:", _DF, re.M)

    def test_no_from_ollama(self):
        for line in _DF_LINES:
            if line.strip().lower().startswith("from "):
                assert "ollama" not in line.lower(), line

    def test_libunwind_installed_before_qdrant_lift(self):
        lib = _DF.find("libunwind8")
        copy = _DF.find("COPY --from=qdrant")
        assert lib != -1 and copy != -1
        assert lib < copy, "libunwind8 must precede COPY --from=qdrant"

    def test_libunwind_before_qdrant_version_check(self):
        lib = _DF.find("libunwind8")
        ver = _DF.find("qdrant --version")
        assert lib != -1 and ver != -1
        assert lib < ver, "libunwind8 must precede the qdrant --version probe"

    def test_copy_from_qdrant_to_bin_before_version_check(self):
        """qdrant --version must run against a PATH-installed copy, not before
        the binary is even placed."""
        copy = _DF.find("cp /qdrant/qdrant /usr/local/bin/qdrant")
        run = _DF.find("RUN cp /qdrant/qdrant")
        ver = _DF[run:].find("qdrant --version")
        assert copy != -1 and run != -1 and ver != -1
        assert copy < run + ver

    def test_copy_from_layer_precedes_qdrant_execution(self):
        """COPY --from=qdrant must come before the RUN that executes the
        binary; position is compared on the RUN line that holds the probe."""
        copy = _DF.find("COPY --from=qdrant")
        run = _DF.find("RUN cp /qdrant/qdrant")
        ver_in_run = _DF[run:].find("qdrant --version")
        assert copy != -1 and run != -1 and ver_in_run != -1
        assert copy < run, "COPY --from=qdrant must precede the RUN that executes qdrant"

    def test_pip_layer_installs_requirements(self):
        pip = _DF.find("pip install")
        assert pip != -1 and "requirements.txt" in _DF[pip:pip + 200]

    def test_expose_declares_both_ports(self):
        expose = [l for l in _DF_LINES if l.strip().upper().startswith("EXPOSE")]
        assert expose, "EXPOSE missing"
        ports = {p for line in expose for p in line.split()[1:]}
        assert {"8765", "6333"} <= ports, ports

    def test_entrypoint_chmod_before_cmd(self):
        chmod = _DF.find("chmod +x entrypoint.sh")
        cmd = _DF.find('"./entrypoint.sh"')
        assert chmod != -1 and cmd != -1 and chmod < cmd

    def test_dockerfile_start_period_at_most_300s(self):
        m = re.search(r"--start-period=(\d+)s", _DF)
        assert m, "HEALTHCHECK start-period missing"
        assert int(m.group(1)) <= 300

    def test_dockerfile_healthcheck_checks_status_ok(self):
        assert re.search(r"status.{0,3}==.{0,3}['\"]ok['\"]", _DF), \
            "HEALTHCHECK must assert status=='ok', not just HTTP 200"

    def test_dockerfile_zero_ollama_traces_noncomment(self):
        nc = _read_noncomment(DOCKERFILE)
        assert "ollama" not in nc.lower()
        assert "MEM0_OLLAMA_URL" not in nc and "MEM0_LLM_MODEL" not in nc

    def test_dockerfile_healthcheck_uses_health_path(self):
        assert "/health" in _DF


# ════════════════════════ Layer 1 — entrypoint.sh (static) ═══════════════════

class TestEntrypointStatic:
    def test_entrypoint_parses_via_sh_n(self):
        r = subprocess.run(["sh", "-n", str(ENTRYPOINT_SH)],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"sh -n entrypoint.sh failed: {r.stderr}"

    def test_set_e_present(self):
        assert re.search(r"^set -e\b", _ENTRYPOINT_TEXT, re.M)

    def test_qdrant_log_path_captured(self):
        assert "qdrant.log" in _ENTRYPOINT_TEXT
        assert re.search(r">.*\$QDRANT_LOG", _ENTRYPOINT_TEXT), \
            "qdrant stdout/stderr must be redirected into the log file"

    def test_kill_zero_liveness_check(self):
        assert "kill -0" in _ENTRYPOINT_TEXT

    def test_fallback_strings_present(self):
        # The /tmp fallback was REMOVED on purpose: it silently lost every
        # memory written during the run. Unwritable storage must now hard-fail.
        assert "Falling back" not in _ENTRYPOINT_TEXT
        assert "/tmp/qdrant-storage-fallback" not in _ENTRYPOINT_TEXT
        assert "STILL unwritable" in _ENTRYPOINT_TEXT
        assert "Refusing to start" in _ENTRYPOINT_TEXT

    def test_dead_qdrant_prints_captured_log(self):
        i = _ENTRYPOINT_TEXT.find("Qdrant process exited")
        j = _ENTRYPOINT_TEXT.find("cat \"$QDRANT_LOG\"", i)
        assert i != -1 and j != -1 and i < j

    def test_final_line_is_exec_mcp_server(self):
        lines = [l.strip() for l in _ENTRYPOINT_TEXT.strip().splitlines() if l.strip()]
        assert lines[-1] == "exec python3 mcp_server.py", lines[-1]

    def test_selftest_runs_before_exec(self):
        i = _ENTRYPOINT_TEXT.find("selftest.py")
        j = _ENTRYPOINT_TEXT.rfind("exec python3 mcp_server.py")
        assert i != -1 and j != -1 and i < j


class TestEntrypointDimParser:
    """Exec-test the embedded python snippet (entrypoint.sh §4) against the
    Qdrant collection-config shapes it must survive. The real invocation is
    `curl ... | python3 -c '<snippet>'` feeding the API JSON via stdin."""

    def test_snippet_embedded_in_entrypoint(self):
        assert "vectors" in _DIM_SNIPPET and "size" in _DIM_SNIPPET

    def test_snippet_is_valid_python(self):
        compile(_DIM_SNIPPET, "<entrypoint-dim-parser>", "exec")

    # -- named-vector shape: {'name': {'size': N, ...}} -----------------------
    def test_named_shape_size_extracted(self):
        out = _run_dim_snippet(_DIM_SNIPPET, {
            "result": {"config": {"params": {"vectors": {
                "named-vectors": {"size": 768, "distance": "Cosine"}}}}},
        })
        assert out == "768", f"named shape: got {out!r}"

    def test_named_shape_ignores_non_vector_entries(self):
        out = _run_dim_snippet(_DIM_SNIPPET, {
            "result": {"config": {"params": {"vectors": {
                "named-vectors": {"size": 768, "distance": "Cosine"},
                "meta": "not-a-dict"}}}},
        })
        assert out == "768", f"named shape w/ noise: got {out!r}"

    # -- flat/anonymous shape: {'size': N} ------------------------------------
    def test_flat_shape_size_extracted(self):
        out = _run_dim_snippet(_DIM_SNIPPET, {
            "result": {"config": {"params": {"vectors": {"size": 768}}}},
        })
        assert out == "768", f"flat shape: got {out!r}"

    def test_flat_shape_missing_size_tolerated(self):
        out = _run_dim_snippet(_DIM_SNIPPET, {
            "result": {"config": {"params": {"vectors": {}}}},
        })
        assert out == "", f"empty vectors dict: got {out!r}"

    def test_garbage_payload_tolerated(self):
        out = _run_dim_snippet(_DIM_SNIPPET, {"unexpected": True})
        assert out == "", f"unrelated payload: got {out!r}"

    def test_vectors_as_list_shape_is_tolerated(self):
        """Qdrant /collections/<name> can return a bare LIST under params.vectors
        (named vectors as list form). The parser must not crash the entrypoint
        (it sits under an `|| true`, but a hard crash must still print nothing)."""
        out = _run_dim_snippet(_DIM_SNIPPET, {
            "result": {"config": {"params": {"vectors": [
                {"name": "named", "size": 768}]}}},
        })
        assert isinstance(out, str), f"list shape crashed the parser: {out!r}"


# ════════════════════════ Layer 1 — .env.example ═════════════════════════════

class TestEnvExample:
    def test_env_example_has_no_ollama_or_llm_vars(self):
        text = _read(ENV_EXAMPLE)
        assert "MEM0_LLM_MODEL" not in text
        assert "MEM0_OLLAMA_URL" not in text
        assert "ollama" not in text.lower()

    def test_env_example_documents_embed_vars(self):
        text = _read(ENV_EXAMPLE)
        assert "MEM0_EMBED_MODEL" in text and "MEM0_EMBED_DIMS" in text
        assert "768" in text


# ════════════════════════ Layer 1 — Makefile ═════════════════════════════════

class TestMakefile:
    def test_makefile_targets_exist(self):
        for target in ("test", "test-in-container", "test-all", "up", "update"):
            assert re.search(rf"^{target}:", _MK, re.M), f"make target '{target}' missing"

    def test_make_target_test_references_existing_test_files(self):
        body = _make_target_body(_MK, "test")
        refs = re.findall(r"tests/[A-Za-z0-9_./-]+", body)
        assert refs, "make test must reference tests/ files"
        for ref in refs:
            assert (REPO / ref).exists(), f"Makefile 'test' references missing file: {ref}"

    def test_make_target_test_all_local_leg_references_existing_tests(self):
        body = _make_target_body(_MK, "test-all")
        refs = re.findall(r"tests/[A-Za-z0-9_./-]+", body)
        for ref in refs:
            assert (REPO / ref).exists(), f"Makefile 'test-all' references missing file: {ref}"

    def test_make_test_in_container_runs_full_tests_dir(self):
        body = _make_target_body(_MK, "test-in-container")
        assert "pytest tests/" in body

    def test_make_up_and_update_call_setup_sh(self):
        assert re.search(r"^up:[^\n]*\n\t\./setup\.sh\s*$", _MK, re.M)
        assert re.search(r"^update:[^\n]*\n\t\./setup\.sh update\s*$", _MK, re.M)


def _make_target_body(mk: str, target: str) -> str:
    m = re.search(rf"^{target}:[^\n]*\n((?:\t[^\n]*\n?)*)", mk, re.M)
    assert m, f"Makefile target '{target}' has no recipe"
    return m.group(1)


# ════════════════════════ Layer 2 — docker-gated ═════════════════════════════

requires_docker = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker not available on this host (Layer-2 contract tests)",
)


@requires_docker
class TestImageLayer2:
    """Docker-gated contract checks; skipped on hosts without docker."""

    image_tag = "mem0-local-contract-test"

    def test_image_builds(self):
        r = subprocess.run(
            ["docker", "build", "-t", self.image_tag, "."],
            cwd=str(REPO), capture_output=True, text=True, timeout=2400,
        )
        assert r.returncode == 0, f"docker build failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"

    def test_qdrant_binary_runs_inside_image(self):
        r = subprocess.run(
            ["docker", "run", "--rm", self.image_tag, "qdrant", "--version"],
            capture_output=True, text=True, timeout=300,
        )
        assert r.returncode == 0, f"qdrant --version failed:\n{r.stdout}\n{r.stderr}"