FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Copy the MCP server
COPY mcp_server.py selftest.py entrypoint.sh pytest.ini conftest.py ./
COPY tests/ tests/
RUN chmod +x entrypoint.sh
# Clear any Python bytecode cache so updated .py files are always used
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
    find /app -name "*.pyc" -delete 2>/dev/null; true

# Default config — override via environment in docker-compose.yml
ENV MEM0_OLLAMA_URL=http://host.docker.internal:11434 \
    MEM0_QDRANT_HOST=qdrant \
    MEM0_QDRANT_PORT=6333 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8765 \
    MEM0_TELEMETRY=False \
    MEM0_LLM_MODEL=qwen2.5:7b \
    MEM0_EMBED_DIMS=768

EXPOSE 8765

# Health check — the server exposes /health with status field
# Checks for status=="ok" (not just HTTP 200) so the container is marked
# unhealthy while mem0 is still initializing.
HEALTHCHECK --interval=10s --timeout=10s --retries=5 --start-period=600s \
    CMD python3 -c "import urllib.request,json; d=json.loads(urllib.request.urlopen('http://localhost:8765/health',timeout=10).read()); exit(0 if d.get('status')=='ok' else 1)" || exit 1

CMD ["./entrypoint.sh"]