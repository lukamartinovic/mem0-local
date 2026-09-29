#!/usr/bin/env sh
set -e

# Single-container supervisor: Qdrant (background) -> wait -> self-test -> MCP server.
# No Ollama anywhere: the calling agent does its own fact inference.

# ── 1. Start Qdrant in the background ────────────────────────────────────────
echo "[entrypoint] Starting Qdrant (storage: /qdrant/storage)..."
qdrant --path /qdrant/storage &
QDRANT_PID=$!

# ── 2. Wait for Qdrant ──────────────────────────────────────────────────────
echo "[entrypoint] Waiting for Qdrant at http://${MEM0_QDRANT_HOST:-127.0.0.1}:${MEM0_QDRANT_PORT:-6333}..."
until curl -s "http://${MEM0_QDRANT_HOST:-127.0.0.1}:${MEM0_QDRANT_PORT:-6333}/" > /dev/null 2>&1; do
    if ! kill -0 "$QDRANT_PID" 2>/dev/null; then
        echo "[entrypoint] ❌ Qdrant process died — check volume permissions on /qdrant/storage"
        exit 1
    fi
    echo "[entrypoint]   Qdrant not ready, retrying in 2s..."
    sleep 2
done
echo "[entrypoint] ✅ Qdrant is ready"

# ── 3. Delete stale Qdrant collections with wrong embedding dimensions ──────
# nomic v1.5 = 768 dims. Existing collections from the Ollama nomic-embed-text
# era are also 768-dim, so data SURVIVES this migration. Only a collection
# built with a genuinely different embedder (different dims) gets deleted.
STALE_COLLECTIONS="mem0 mem0migrations"
EXPECTED_DIM="${MEM0_EMBED_DIMS:-768}"
for col in $STALE_COLLECTIONS; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${MEM0_QDRANT_HOST:-127.0.0.1}:${MEM0_QDRANT_PORT:-6333}/collections/${col}" 2>/dev/null || true)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "[entrypoint] Found existing collection '${col}' — checking dimensions..."
        EXISTING_DIM=$(curl -s "http://${MEM0_QDRANT_HOST:-127.0.0.1}:${MEM0_QDRANT_PORT:-6333}/collections/${col}" 2>/dev/null | \
            python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    vectors = data.get('result', {}).get('config', {}).get('params', {}).get('vectors', {})
    if isinstance(vectors, dict):
        for key, val in vectors.items():
            if isinstance(val, dict) and 'size' in val:
                print(val['size'])
                break
        if not any(isinstance(v, dict) for v in vectors.values()):
            if 'size' in vectors:
                print(vectors['size'])
except Exception:
    pass
" 2>/dev/null || true)

        if [ -n "$EXISTING_DIM" ] && [ "$EXISTING_DIM" != "$EXPECTED_DIM" ]; then
            echo "[entrypoint] ⚠️  Collection '${col}' has ${EXISTING_DIM}-dim vectors but embedder uses ${EXPECTED_DIM}. Deleting stale collection..."
            curl -s -X DELETE "http://${MEM0_QDRANT_HOST:-127.0.0.1}:${MEM0_QDRANT_PORT:-6333}/collections/${col}" > /dev/null
            echo "[entrypoint] ✅ Deleted stale collection '${col}'"
        else
            echo "[entrypoint] ✅ Collection '${col}' dimensions OK (${EXISTING_DIM:-unknown}) — existing memories preserved"
        fi
    fi
done

# ── 4. Self-test (all tools, real Qdrant, real fastembed) ───────────────────
echo "[entrypoint] Running self-test..."
if python3 selftest.py; then
    echo "[entrypoint] ✅ Self-test passed"
else
    echo "[entrypoint] ⚠️  Self-test had failures (see above). Server will still start."
    echo "[entrypoint]    The failing tools will return errors to your IDE agent."
fi

echo "[entrypoint] Starting MCP server..."
exec python3 mcp_server.py