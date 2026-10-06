#!/usr/bin/env sh
set -e

# Single-container supervisor: Qdrant (background) -> wait -> self-test -> MCP server.
# No Ollama anywhere: the calling agent does its own fact inference.
# If Qdrant dies, its full output lands in /tmp/qdrant.log and is printed here.

QDRANT_LOG=/tmp/qdrant.log
QDRANT_URL="http://${MEM0_QDRANT_HOST:-127.0.0.1}:${MEM0_QDRANT_PORT:-6333}"
MODEL_CACHE="${HF_HOME:-/model_cache}"

# ── 1. Storage dir writable? (named volume may exist with foreign ownership) ─
STORAGE="${QDRANT_STORAGE_PATH:-/qdrant/storage}"
mkdir -p "$STORAGE" 2>/dev/null || true
if ! touch "$STORAGE/.write_test" 2>/dev/null; then
    echo "[entrypoint] Storage '$STORAGE' is NOT writable. Trying to fix ownership..."
    chown -R "$(id -u):$(id -g)" "$STORAGE" 2>/dev/null || true
    chmod -R u+rwX "$STORAGE" 2>/dev/null || true
fi
if ! touch "$STORAGE/.write_test" 2>/dev/null; then
    echo "[entrypoint] Storage '$STORAGE' is STILL unwritable as uid $(id -u)."
    echo "[entrypoint] Refusing to start: a fallback to /tmp would silently lose"
    echo "[entrypoint] every memory written in this run. Fix the volume instead:"
    echo "[entrypoint]   docker compose down"
    echo "[entrypoint]   docker volume rm \$(docker volume ls -q | grep qdrant_data)"
    echo "[entrypoint]   ./setup.sh          # recreates the volume with correct ownership"
    exit 1
fi
rm -f "$STORAGE/.write_test" 2>/dev/null || true

# ── 2. Start Qdrant in the background (output captured for diagnosis) ───────
# NOTE: the qdrant binary has NO --path CLI flag (verified in v1.13.2 main.rs:
# only bootstrap/uri/snapshot/storage_snapshot/config_path/disable_telemetry/
# stacktrace/reinit). Storage location is configured via the env var
# QDRANT__STORAGE__STORAGE_PATH (Qdrant merges env QDRANT__SECTION__KEY into
# its config, settings.rs). Passing --path makes qdrant abort at arg-parse.
if [ "$STORAGE" != "/qdrant/storage" ]; then
    export QDRANT__STORAGE__STORAGE_PATH="$STORAGE"
fi
echo "[entrypoint] Starting qdrant (storage: $STORAGE)..."
qdrant > "$QDRANT_LOG" 2>&1 &
QDRANT_PID=$!

# ── 3. Wait for readiness; on death print its actual log ────────────────────
echo "[entrypoint] Waiting for Qdrant at $QDRANT_URL..."
i=0
until curl -s "$QDRANT_URL/" > /dev/null 2>&1; do
    if ! kill -0 "$QDRANT_PID" 2>/dev/null; then
        echo "[entrypoint] Qdrant process exited. Its output:"
        echo "────────────────────── $QDRANT_LOG ──────────────────────"
        cat "$QDRANT_LOG"
        echo "──────────────────────────────────────────────────────────"
        echo "[entrypoint] Common causes:"
        echo "[entrypoint]   'Permission denied' on storage       -> volume ownership (uid $(id -u))"
        echo "[entrypoint]   'No such file or directory' on exec  -> arch mismatch (arm64 vs amd64 image?)"
        echo "[entrypoint]   'Address already in use'             -> a stale process holds port 6333"
        echo "[entrypoint]   'unexpected argument'                -> a CLI flag qdrant does not support"
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
# collections stay compatible and are left untouched. A GENUINE mismatch means
# the user changed embedders: that is a destructive change, so refuse to start
# (MEM0_STRICT_DIMS=0 opts into the old auto-delete behaviour).
STALE_COLLECTIONS="mem0 mem0migrations"
EXPECTED_DIM="${MEM0_EMBED_DIMS:-768}"
STRICT="${MEM0_STRICT_DIMS:-1}"
DIM_CONFLICT=0
for col in $STALE_COLLECTIONS; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$QDRANT_URL/collections/${col}" 2>/dev/null || true)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "[entrypoint] collection '$col' exists - checking dim..."
        EXISTING_DIM=$(curl -s "$QDRANT_URL/collections/${col}" 2>/dev/null | \
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
            if [ "$STRICT" = "1" ]; then
                echo "[entrypoint] ❌ DIMENSION MISMATCH on '$col': stored ${EXISTING_DIM} dims, embedder produces ${EXPECTED_DIM}."
                DIM_CONFLICT=1
            else
                echo "[entrypoint] WARNING: '$col' has ${EXISTING_DIM}-dim vectors, embedder needs ${EXPECTED_DIM}. Deleting (MEM0_STRICT_DIMS=0)..."
                curl -s -X DELETE "$QDRANT_URL/collections/${col}" > /dev/null
                echo "[entrypoint] deleted stale collection '$col'"
            fi
        else
            echo "[entrypoint] collection '$col' dims OK (${EXISTING_DIM:-unknown}) - existing memories preserved"
        fi
    fi
done

if [ "$DIM_CONFLICT" = "1" ]; then
    echo "[entrypoint] Refusing to start: continuing would either corrupt searches or"
    echo "[entrypoint] require deleting your stored memories. Choose one:"
    echo "[entrypoint]   a) restore the previous embedder: unset MEM0_EMBED_DIMS / MEM0_EMBED_MODEL in .env"
    echo "[entrypoint]   b) keep the new embedder and WIPE the old collection deliberately:"
    echo "[entrypoint]        curl -X DELETE $QDRANT_URL/collections/mem0"
    echo "[entrypoint]   c) accept automatic deletion on every start: set MEM0_STRICT_DIMS=0"
    exit 1
fi

# ── 5. Self-test (all tools, real Qdrant + real fastembed) ──────────────────
# Fatal on failure: a server that starts with broken tools while printing
# "ready" is exactly the unreliable-setup failure mode this guards against.
echo "[entrypoint] Running self-test..."
if python3 selftest.py; then
    echo "[entrypoint] Self-test passed"
    mkdir -p "$MODEL_CACHE" 2>/dev/null || true
    touch "$MODEL_CACHE/.ready" 2>/dev/null || true   # read by setup.sh to skip the download warning
else
    echo "[entrypoint] ❌ Self-test FAILED - not starting the MCP server."
    echo "[entrypoint]    A server whose tools error out is worse than no server,"
    echo "[entrypoint]    because the calling agent will silently fail to remember things."
    echo "[entrypoint]    Logs above show which tool failed. Common causes:"
    echo "[entrypoint]      - embedding model could not be downloaded (no network on first run)"
    echo "[entrypoint]      - Qdrant storage unwritable"
    exit 1
fi

echo "[entrypoint] Starting MCP server..."
exec python3 mcp_server.py