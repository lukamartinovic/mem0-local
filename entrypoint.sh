#!/usr/bin/env sh
set -e

# Wait for Ollama to be ready
echo "[entrypoint] Waiting for Ollama at ${MEM0_OLLAMA_URL}..."
until curl -s "${MEM0_OLLAMA_URL}/api/tags" > /dev/null 2>&1; do
    echo "[entrypoint]   Ollama not ready, retrying in 2s..."
    sleep 2
done
echo "[entrypoint] ✅ Ollama is ready"

# Pull models if not already present
LLM_MODEL="${MEM0_LLM_MODEL:-qwen2.5:7b}"
EMBED_MODEL="${MEM0_EMBED_MODEL:-nomic-embed-text}"

# Check if LLM model is available, pull if not
if ! curl -s "${MEM0_OLLAMA_URL}/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = [m['name'] for m in data.get('models', [])]
sys.exit(0 if any('${LLM_MODEL}' in m for m in models) else 1)
" 2>/dev/null; then
    echo "[entrypoint] Pulling LLM model: ${LLM_MODEL} (first run — this takes a while)..."
    curl -s "${MEM0_OLLAMA_URL}/api/pull" -d "{\"name\": \"${LLM_MODEL}\"}" > /dev/null
    echo "[entrypoint] ✅ LLM model pulled"
else
    echo "[entrypoint] ✅ LLM model already available: ${LLM_MODEL}"
fi

# Check if embed model is available, pull if not
if ! curl -s "${MEM0_OLLAMA_URL}/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = [m['name'] for m in data.get('models', [])]
sys.exit(0 if any('${EMBED_MODEL}' in m for m in models) else 1)
" 2>/dev/null; then
    echo "[entrypoint] Pulling embedder model: ${EMBED_MODEL}..."
    curl -s "${MEM0_OLLAMA_URL}/api/pull" -d "{\"name\": \"${EMBED_MODEL}\"}" > /dev/null
    echo "[entrypoint] ✅ Embedder model pulled"
else
    echo "[entrypoint] ✅ Embedder model already available: ${EMBED_MODEL}"
fi

# Wait for Qdrant
echo "[entrypoint] Waiting for Qdrant at http://${MEM0_QDRANT_HOST}:${MEM0_QDRANT_PORT}..."
until curl -s "http://${MEM0_QDRANT_HOST}:${MEM0_QDRANT_PORT}/" > /dev/null 2>&1; do
    echo "[entrypoint]   Qdrant not ready, retrying in 2s..."
    sleep 2
done
echo "[entrypoint] ✅ Qdrant is ready"

# Delete stale Qdrant collections that might have wrong embedding dimensions
# This prevents "Vector dimension error: expected dim: X, got Y" when switching embedder models
STALE_COLLECTIONS="mem0 mem0migrations"
EXPECTED_DIM="${MEM0_EMBED_DIMS:-768}"
for col in $STALE_COLLECTIONS; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${MEM0_QDRANT_HOST}:${MEM0_QDRANT_PORT}/collections/${col}" 2>/dev/null || true)
    if [ "$HTTP_CODE" = "200" ]; then
        echo "[entrypoint] Found existing collection '${col}' — checking dimensions..."
        # Get the collection's vector size
        EXISTING_DIM=$(curl -s "http://${MEM0_QDRANT_HOST}:${MEM0_QDRANT_PORT}/collections/${col}" 2>/dev/null | \
            python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    vectors = data.get('result', {}).get('config', {}).get('params', {}).get('vectors', {})
    if isinstance(vectors, dict):
        # Named vectors: {name: {size: N, ...}}
        for key, val in vectors.items():
            if isinstance(val, dict) and 'size' in val:
                print(val['size'])
                break
        # Flat config: {size: N, distance: ...}
        if not any(isinstance(v, dict) for v in vectors.values()):
            if 'size' in vectors:
                print(vectors['size'])
except Exception:
    pass
" 2>/dev/null || true)

        # nomic-embed-text = 768 dims. If the existing collection has a different
        # dimension, delete it so mem0 recreates it with the correct size.
        if [ -n "$EXISTING_DIM" ] && [ "$EXISTING_DIM" != "$EXPECTED_DIM" ]; then
            echo "[entrypoint] ⚠️  Collection '${col}' has ${EXISTING_DIM}-dim vectors but embedder uses ${EXPECTED_DIM}. Deleting stale collection..."
            curl -s -X DELETE "http://${MEM0_QDRANT_HOST}:${MEM0_QDRANT_PORT}/collections/${col}" > /dev/null
            echo "[entrypoint] ✅ Deleted stale collection '${col}'"
        else
            echo "[entrypoint] ✅ Collection '${col}' dimensions OK (${EXISTING_DIM:-unknown})"
        fi
    fi
done

echo "[entrypoint] Running self-test (exercises all 9 MCP tools)..."
if python3 selftest.py; then
    echo "[entrypoint] ✅ Self-test passed — all tools verified"
else
    echo "[entrypoint] ⚠️  Self-test had failures (see above). Server will still start."
    echo "[entrypoint]    The failing tools will return errors to your IDE agent."
fi

echo "[entrypoint] Starting MCP server..."
exec python3 mcp_server.py