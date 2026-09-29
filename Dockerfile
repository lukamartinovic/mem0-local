# mem0-local — single container: Qdrant + MCP server, no Ollama anywhere.
# Fact inference is the calling agent's job; embeddings run locally via
# fastembed (ONNX, CPU) with nomic-ai/nomic-embed-text-v1.5 (768 dims).

# ── Stage 1: lift the pinned Qdrant binary (same version as before) ────────
FROM qdrant/qdrant:v1.13.2 AS qdrant

# ── Stage 2: the server ─────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Qdrant binary + its config dir + dashboard assets from the pinned image
# (binary alone is not enough: it expects ./config/default.yaml next to CWD)
COPY --from=qdrant /qdrant/ /qdrant/
RUN cp /qdrant/qdrant /usr/local/bin/qdrant && chmod +x /usr/local/bin/qdrant \
    && mkdir -p /qdrant/storage /qdrant/config \
    && qdrant --version \
    || (echo "BUILD FAIL: qdrant binary not runnable - see ldd below"; ldd /qdrant/qdrant; exit 1)

# Install dependencies first (layer caching)
# libunwind8: the qdrant binary links it (libunwind-ptrace/-aarch64), and the
# python:3.12-slim base lacks it (surfaced by ldd in the build-time check below)
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends curl libunwind8 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Copy the MCP server
COPY mcp_server.py selftest.py entrypoint.sh pytest.ini conftest.py ./
COPY tests/ tests/
RUN chmod +x entrypoint.sh
# Clear any Python bytecode cache so updated .py files are always used
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
    find /app -name "*.pyc" -delete 2>/dev/null; true

# Default config — override via environment in docker-compose.yml
# NOTE: no MEM0_LLM_MODEL / MEM0_OLLAMA_URL — there is no extraction LLM.
ENV QDRANT__SERVICE__HTTP_PORT=6333 \
    MEM0_QDRANT_HOST=127.0.0.1 \
    MEM0_QDRANT_PORT=6333 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8765 \
    MEM0_TELEMETRY=False \
    MEM0_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5 \
    MEM0_EMBED_DIMS=768 \
    MEM0_DEFAULT_USER_ID=dev

EXPOSE 8765 6333

# Health check — the server exposes /health with status field
# Checks for status=="ok" (not just HTTP 200) so the container is marked
# unhealthy while mem0/fastembed are still initializing.
HEALTHCHECK --interval=10s --timeout=10s --retries=5 --start-period=180s \
    CMD python3 -c "import urllib.request,json; d=json.loads(urllib.request.urlopen('http://localhost:8765/health',timeout=10).read()); exit(0 if d.get('status')=='ok' else 1)" || exit 1

CMD ["./entrypoint.sh"]