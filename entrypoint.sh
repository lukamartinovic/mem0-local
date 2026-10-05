#!/usr/bin/env sh
set -e

# Single-container supervisor: Qdrant (background) -> wait -> self-test -> MCP server.
# No Ollama anywhere: the calling agent does its own fact inference.
# If Qdrant dies, its full output lands in /tmp/qdrant.log and is printed here.

QDRANT_LOG=/tmp/qdrant.log
QDRANT_HOST="${MEM0_QDRANT_HOST:-127.0.0.1}:${MEM0_QDRANT_PORT:-6333}"

# ── 1. Storage dir writable? (named volume may exist with foreign ownership) ─
STORAGE="${QDRANT_STORAGE_PATH:-/qdrant/storage}"
mkdir -p "$STORAGE" 2>/dev/null || true
if ! touch "$STORAGE/.write_test" 2>/dev/null; then
    echo "[entrypoint] Storage '$STORAGE' is NOT writable. Trying to fix ownership..."
    chown -R "$(id -u):$(id -g)" "$STORAGE" 2>/dev/null || true
    chmod -R u+rwX "$STORAGE" 2>/dev/null || true
fi
if ! touch "$STORAGE/.write_test" 2>/dev/null; then
    echo "[entrypoint] Still unwritable (uid $(id -u)). Falling back to /tmp/qdrant-storage-fallback"
    echo "[entrypoint] WARNING: DATA IS NOT PERSISTED in fallback mode. Fix the volume:"
    echo "[entrypoint]   docker volume rm \$(docker volume ls -q | grep qdrant)  # then ./setup.sh"
    STORAGE=/tmp/qdrant-storage-fallback
    mkdir -p "$STORAGE"
fi
rm -f "$STORAGE/.write_test" 2>/dev/null || true

# ── 2. Start Qdrant in the background (output captured for diagnosis) ───────
# NOTE: the qdrant binary has NO --path CLI flag (verified in v1.13.2 main.rs:
# only bootstrap/uri/snapshot/storage_snapshot/config_path/disable_telemetry/
# stacktrace/reinit). Storage location is configured via the env var
# QDRANT__STORAGE__STORAGE_PATH (Qdrant merges env QDRANT__SECTION__KEY into
# its config, settings.rs). Using --path makes qdrant abort at arg-parse.
if [ "$STORAGE" != "/qdrant/storage" ]; then
    export QDRANT__STORAGE__STORAGE_PATH="$STORAGE"
fi
echo "[entrypoint] Starting qdrant (storage: $STORAGE)..."
qdrant > "$QDRANT_LOG" 2>&1 &
QDRANT_PID=$!

# ── 3. Wait for readiness; on death print its actual log ────────────────────
echo "[entrypoint] Waiting for Qdrant at http://${MEM0_QDRANT_HOST:-127.0.0.1}:${MEM0_QDRANT_PORT:-6333}..."
i=0
until curl -s "http://${MEM0_QDRANT_HOST:-127.0.0.1}:${MEM0_QDRANT_PORT:-6333}/" > /dev/null 2>&1; do
    if ! kill -0 "$QDRANT_PID" 2>/dev/null; then
        echo "[entrypoint] Qdrant process exited. Its output:"
        echo "────────────────────── $QDRANT_LOG ──────────────────────"
        cat "$QDRANT_LOG"
        echo "──────────────────────────────────────────────────────────"
        echo "[entrypoint] Common causes:"
        echo "[entrypoint]   'Permission denied' on storage    -> volume ownership (uid $(id -u))"
        echo "[entrypoint]   'No such file or directory' on exec -> arch mismatch (arm64 vs amd64 image?)"
        echo "[entrypoint]   'Address already in use' on the storage path -> stale qdrant holds it"
        echo "[entrypoint]   missing config/assets               -> image was built binary-only (fixed in current Dockerfile)"
        exit 1
    fi
    i=$((i+1))
    if [ "$i" -gt 30 ]; then
        echo "[entrypoint] Qdrant not reachable after 60s. Its log:"
        cat "$QDRANT_LOG"
        exit 1
    fi
    echo "[entrypoint]   waiting for Qdrant (${i}/30)..."
    sleep 2
done
echo "[entrypoint] Qdrant is ready"

# ── 4. Dimension check on existing collections ──────────────────────────────
# nomic v1.5 = 768 dims, same as the Ollama-era nomic-embed-text, so old
# collections stay compatible. Only a genuinely-different-dims collection
# built by another embedder gets deleted.
STALE_COLLECTIONS="mem0 mem0migrations"
EXPECTED_DIM="${MEM0_EMBED_DIMS:-768}"
for col in $STALE_COLLECTIONS; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${MEM0_QDRANT_HOST:-127.0.0.1}:${MEM0_QDRANT_PORT:-6333}/collections/${col}" 2>/dev/null || true)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "[entrypoint] collection '$col' exists - checking dim..."
        EXISTING_DIM=$(curl -s "http://${MEM0_QDRANT_HOST:-127.0.0.1}:${MEM0_QDRANT_PORT:-6333}/collections/${col}" 2>/dev/null | \
            python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    vecs = data.get('result', {}).get('config', {}).get('params', {}).get('vectors', {})
    if isinstance(vecs, dict):
        for _k, v in vecs.items():
            if isinstance(v, dict) and 'size' in v:
                print(v['size'])
                break
        if not any(isinstance(v, dict) for v in vecs.values()) and 'size' in vecs:
            print(vecs['size'])
except Exception:
    pass
" 2>/dev/null || true)
        if [ -n "$EXISTING_DIM" ] && [ "$EXISTING_DIM" != "$EXPECTED_DIM" ]; then
            echo "[entrypoint] WARNING: '$col' has ${EXISTING_DIM}-dim vectors, embedder needs ${EXPECTED_DIM}. Deleting stale collection..."
            curl -s -X DELETE "http://${MEM0_QDRANT_HOST:-127.0.0.1}:${MEM0_QDRANT_PORT:-6333}/collections/${col}" > /dev/null
            echo "[entrypoint] deleted stale collection '$col'"
        else
            echo "[entrypoint] collection '$col' dims OK (${EXISTING_DIM:-unknown}) - existing memories preserved"
        fi
    fi
done

# ── 5. Self-test (all 12 tools, real Qdrant + real fastembed) ───────────────
echo "[entrypoint] Running self-test..."
if python3 selftest.py; then
    echo "[entrypoint] Self-test passed"
    touch /app/.downloaded 2>/dev/null || true  # model cache warm - setup.sh next-start hint
else
    echo "[entrypoint] WARNING: self-test had failures (see above). Server will still start."
    echo "[entrypoint]   Failing tools will return errors to your IDE agent."
fi

echo "[entrypoint] Starting MCP server..."
exec python3 mcp_server.py