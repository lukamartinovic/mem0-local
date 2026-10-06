# mem0-local — single container: Qdrant + MCP server, no Ollama anywhere.
# Fact inference is the calling agent's job; embeddings run locally via
# fastembed (ONNX, CPU) with nomic-ai/nomic-embed-text-v1.5 (768 dims).
#
# Fully self-contained: the embedding model is baked in at build time, so the
# running container needs NO network access at all.

# ── Stage 1: the pinned Qdrant image (binary + config + assets) ─────────────
FROM qdrant/qdrant:v1.13.2 AS qdrant

# ── Stage 2: the server ──────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Runtime config — no MEM0_LLM_MODEL / MEM0_OLLAMA_URL: there is no extraction LLM.
# Declared BEFORE the model bake so the bake uses the same values the server will.
ENV QDRANT__SERVICE__HTTP_PORT=6333 \
    MEM0_QDRANT_HOST=127.0.0.1 \
    MEM0_QDRANT_PORT=6333 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8765 \
    MEM0_TELEMETRY=False \
    MEM0_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5 \
    MEM0_EMBED_DIMS=768 \
    MEM0_DEFAULT_USER_ID=dev \
    HF_HOME=/model_cache

# libunwind8 FIRST (before any qdrant run): the qdrant binary needs
# libunwind-ptrace/-aarch64, absent from the python-slim base. Getting this
# order wrong was a real build failure: the check ran before the install.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libunwind8 \
    && rm -rf /var/lib/apt/lists/*

# Qdrant binary + config + dashboard assets lifted from the pinned image.
COPY --from=qdrant /qdrant/ /qdrant/

# Verify the binary RUNS here, at build time. Split from the copy so a copy
# failure and a "binary cannot load libraries" failure give distinct messages.
RUN cp /qdrant/qdrant /usr/local/bin/qdrant && chmod +x /usr/local/bin/qdrant \
    && mkdir -p /qdrant/storage /qdrant/config
RUN qdrant --version >/dev/null \
    || { echo "BUILD FAIL: qdrant binary cannot run - missing libraries?"; ldd /qdrant/qdrant; exit 1; }

# Python dependencies (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake every runtime model INTO the image: the container then needs ZERO
# network access at runtime.
#  1. the fastembed ONNX embedder (~160 MB) - downloaded at build time instead
#     of on first container start, where it was an invisible hang that read as
#     "the setup is unreliable";
#  2. mem0's spaCy lemmatization model, which mem0 otherwise auto-downloads
#     mid-request via pip (that path fails outright on images without pip).
RUN python3 -c "\
from fastembed import TextEmbedding; \
import os; \
m = TextEmbedding(model_name=os.environ['MEM0_EMBED_MODEL']); \
v = list(m.embed(['warmup']))[0]; \
dims = int(os.environ['MEM0_EMBED_DIMS']); \
assert len(v) == dims, f'model dims {len(v)} != MEM0_EMBED_DIMS {dims}'; \
print(f'baked embedding model OK: {len(v)} dims')"
RUN python3 -m spacy download en_core_web_sm \
    && python3 -c "import spacy; spacy.load('en_core_web_sm'); print('baked spaCy model OK')"

# Application code
COPY mcp_server.py selftest.py entrypoint.sh pytest.ini conftest.py ./
COPY tests/ tests/
RUN chmod +x entrypoint.sh
# Clear any Python bytecode cache so updated .py files are always used
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
    find /app -name "*.pyc" -delete 2>/dev/null; true

EXPOSE 8765 6333

# Health check — status=="ok" (not just HTTP 200) so the container is marked
# unhealthy while mem0/Qdrant are still initializing. The compose service
# defines the same check; keep the two in sync.
HEALTHCHECK --interval=10s --timeout=10s --retries=5 --start-period=180s \
    CMD python3 -c "import urllib.request,json; d=json.loads(urllib.request.urlopen('http://localhost:8765/health',timeout=10).read()); exit(0 if d.get('status')=='ok' else 1)" || exit 1

CMD ["./entrypoint.sh"]