# mem0-local

Self-hosted [Mem0](https://mem0.ai) memory layer for AI coding agents, running **entirely on your machine** — no cloud API calls, no data leaving your laptop.

## The one thing to understand

**This server has NO extraction LLM.** Fact inference is the calling agent's responsibility: your agent is smart, so it extracts facts itself and stores each one as its own concise memory. All storage paths are LLM-free and fully local (fastembed ONNX embeddings + Qdrant vectors inside a single container).

## Target behavior

The system gives your AI coding agent **persistent memory across sessions**. Without it, every new session starts from zero — the agent re-reads your codebase for context. With it:

- **Before answering:** the agent calls `search_memories` for relevant past context (decisions, patterns, preferences)
- **After completing tasks:** the agent stores what it learned — one `add_raw_memory` call per concise fact
- **Next session:** yesterday's work recalled instantly instead of re-reading everything

Memory is **semantic** (natural-language search with relevance scores) and **private** — everything runs locally.

**One command:**

```bash
./setup.sh
```

Builds and starts ONE container (Qdrant + the MCP server together), waits for health, prints the success banner. Point your IDE at `http://localhost:8765/mcp`.

## Quick start

```bash
# 1. Run setup (build + start + self-test)
./setup.sh

# 2. Add to your IDE config (one-time, table below)
#    URL: http://localhost:8765/mcp

# 3. System prompt for your agent (REQUIRED - see "Memory protocol" below)
```

> The first container start downloads the embedding model (~160 MB, once). No models to pull manually, no Ollama, no GPU needed.

### Other commands

```bash
./setup.sh          # build + start the single container
./setup.sh update   # rebuild image with latest code, preserve all data
docker compose down # stop (data preserved)
make health         # health check pretty-printed
make logs           # tail container logs
```

## Architecture

```
Your IDE agent (does its own FACT INFERENCE)
  :  add_raw_memory("one concise fact")      add_verbatim("bulk text")
  ↓ HTTP (MCP JSON-RPC)
mem0-local container (:8765)
  ├─ MCP server (13 tools, health endpoint, web UI at /)
  ├─ fastembed (ONNX, CPU) -> nomic-ai/nomic-embed-text-v1.5, 768 dims
  └─ Qdrant v1.13.2 (same container, :6333)
```

| Component | Where | Why |
|---|---|---|
| **MCP server** | Docker container | Portable, reproducible |
| **Qdrant v1.13.2** | Same container (binary lifted from the pinned image) | One image, one `docker compose up` |
| **fastembed** | Same container (ONNX on CPU) | Local embeddings, no GPU, no Ollama |
| **Extraction LLM** | **Nowhere** — your agent does it | Simpler, faster (instant), one less model to run |

Existing memories from the Ollama era survive the migration: both `nomic-embed-text` (Ollama) and `nomic-ai/nomic-embed-text-v1.5` (fastembed) are 768-dim, so the Qdrant collection is compatible as-is.

## Configure your IDE

```json
{
  "mcpServers": {
    "mem0-local": {
      "type": "http",
      "url": "http://localhost:8765/mcp"
    }
  }
}
```

| IDE | Config file |
|---|---|
| Cursor | `.cursor/mcp.json` |
| Claude Code | `.mcp.json` |
| Codex | `~/.codex/config.toml` (TOML format) |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` |
| VS Code | `~/.vscode/mcp.json` |
| OpenCode | `~/.opencode/config.json` |

Restart your IDE after adding the config.

## Web UI (:8765/)

Open **http://localhost:8765/** in a browser to browse and manage memories.
Reachable from other machines on your LAN at `http://<host-ip>:8765/`.

| Control | What it does |
|---|---|
| Filter box / entity dropdown | Client-side filtering over all memories; counts update live |
| **Export** panel | Download **JSON** or **CSV**, or copy JSON to the clipboard. Optionally scoped to one entity. Same payload shape as the `export_memories` MCP tool. |
| **Import** panel | Upload a file or paste JSON, targeted at a chosen entity. Accepts a UI/CLI export envelope, a bare JSON array, **JSON Lines**, or the CSV this UI exports. "Preview only" counts what would be imported without writing. |

Import runs through the same helper as the `import_memories` MCP tool, so
duplicates (same text, same entity) are skipped and failures are reported
per item - the UI cannot write anything the agent could not have written.

Exports are interchangeable between the UI, the CLI (`make export` / `make import`)
and your agent, so a browser download can be restored with one command and vice versa.

## Available MCP tools (12)

### Storing

| Tool | Contract |
|---|---|
| `add_raw_memory` | Store ONE concise self-extracted fact (10-30 words, one fact per call). REJECTED with a actionable hint if the content looks like raw text rather than a fact (> chunk size). |
| `add_verbatim` | Store raw text as-is: bulk doc imports, backups, reference material. Long text auto-chunks at paragraph boundaries with `[Document: path]` context headers. |

### Reading

| Tool | Description |
|---|---|
| `search_memories` | Semantic search with relevance scores and `min_score` filtering |
| `get_memories` | List memories with filters and pagination |
| `get_memory` | Retrieve a single memory by ID |

### Writing other than add

| Tool | Description |
|---|---|
| `update_memory` | Overwrite a memory's text by ID (LLM-free) |
| `delete_memory` | Delete a single memory by ID |
| `delete_all_memories` | Delete all memories for a user |
| `list_entities` | List distinct user/agent/app IDs stored |
| `delete_entities` | Delete a user/agent entity and all its memories |

### Backup & maintenance

| Tool | Description |
|---|---|
| `export_memories` | Export all memories for a user as JSON or CSV |
| `import_memories` | Import memories from JSON (skips exact duplicates) |
| `prune_memories` | Delete memories older than N days (dry-run by default) |

### Tool details

#### `add_raw_memory`

```json
{
  "content": "Auth service uses JWT with 24-hour token expiry",
  "user_id": "myproject",
  "metadata": {"category": "architecture"}
}
```

- `content` (required): ONE concise self-contained fact, 10-30 words. Oversized content (> `MEM0_CHUNK_CHARS`) is rejected with a hint to split into facts or use `add_verbatim`.
- `user_id` (optional, default `dev`): Project/user identifier for scoping
- `metadata` (optional): Key-value pairs attached to the memory

#### `add_verbatim`

```json
{
  "content": "Long document text...",
  "user_id": "myproject",
  "metadata": {"file": "docs/deploy.md", "source": "docs_import"}
}
```

Auto-chunked at paragraph boundaries (-> line -> word), each chunk labeled `chunk: n/m` in metadata. Default chunk size 3000 chars (`MEM0_CHUNK_CHARS`).

#### `search_memories`

```json
{
  "query": "database choice",
  "user_id": "myproject",
  "limit": 10,
  "min_score": 0.5,
  "include_scores": true
}
```

Results each carry a `score` field (cosine similarity 0.0-1.0) when `include_scores` is true.

#### `export_memories` / `import_memories` / `prune_memories`

Same shapes as before (JSON/CSV export, duplicate-skipping import, dry-run-default prune). See `Makefile export` / `make import` for CLI one-liners.

## Memory protocol for your agent

Since the server does no fact extraction, add this to your IDE's custom instructions (`.cursorrules`, `CLAUDE.md`, etc.):

```markdown
## Memory Protocol

- Before answering, call `search_memories` for relevant past context
  (min_score 0.5 filters noise).
- After completing a task or learning something durable, store it:
  extract the facts YOURSELF, then call `add_raw_memory` once per fact.
  One concise self-contained fact per call (10-30 words, active voice,
  include names/versions/ports, preserve rationale with the choice).
- Do NOT paste raw conversation into `add_raw_memory` - it will be
  rejected as too long. Raw text goes through `add_verbatim`.
- Use `user_id` = project/team name for scoping.
- Periodically `prune_memories` (older_than_days=30, dry_run=true first).
```

### Fact extraction rules for `add_raw_memory`

1. **One fact per call.** Each call stores a single self-contained fact.
2. **Be concise.** 10-30 words. Focused sentences embed better than paragraphs.
3. **Be specific.** Include names, versions, ports, concrete values.
   - GOOD: "PostgreSQL 15 runs on port 5432 with pgvector extension"
   - BAD: "They use some database"
4. **Preserve decisions and rationale.** Store the choice AND the reason.
5. **Active voice.** "Auth uses JWT" not "JWT is used by auth".
6. **Skip filler.** No greetings, agreement, opinions. Technical facts only.
7. **Oversized input is rejected** - that's the signal you're pasting text, not stating facts. Split it or use `add_verbatim`.

## Configuration

Override via `.env` (see `.env.example`; no LLM vars exist anymore):

| Variable | Default | Description |
|---|---|---|
| `MEM0_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | fastembed model (768 dims) |
| `MEM0_EMBED_DIMS` | `768` | Must match the stored Qdrant collection |
| `MEM0_CHUNK_CHARS` | `3000` | Verbatim chunk size (chars) |
| `MEM0_QDRANT_HOST` | `127.0.0.1` | Qdrant in the same container - leave default |
| `MEM0_QDRANT_PORT` | `6333` | Qdrant port |
| `MEM0_DEFAULT_USER_ID` | `dev` | Default user_id in tool calls |
| `MEM0_TELEMETRY` | `False` | Disable mem0 phone-home telemetry |

Switching embedder requires wiping Qdrant (dimensions change). Switching nothing else is configurable.

## Health monitoring

```bash
curl http://localhost:8765/health | python3 -m json.tool
```

```json
{
  "status": "ok",
  "init_status": "ready",
  "server": "mem0-local",
  "tools": 12,
  "components": {
    "qdrant": true,
    "mem0": true,
    "extraction_llm": false
  },
  "config": {
    "extraction_llm": null,
    "embedder": "nomic-ai/nomic-embed-text-v1.5",
    "embed_dims": 768,
    "vector_store": "qdrant"
  }
}
```

| Field | Values | Meaning |
|---|---|---|
| `status` | `ok`, `starting`, `degraded` | Overall health |
| `init_status` | `starting`, `initializing`, `ready`, `error` | mem0 init state |
| `components.qdrant` | bool | Qdrant API reachable |
| `components.extraction_llm` | always `false` | By design |

The Docker healthcheck waits for `status: "ok"` (start period 180s; the fastembed model loads in ~10-60s, no LLM load time).

## Backup and restore

`./setup.sh update` rebuilds the image with the latest code preserving the Qdrant volume (all memories). Export/import via the tools above, or:

```bash
make export USER=myproject > backup.json
make import FILE=backup.json USER=myproject
```

Between machines: export on A, copy the JSON, import on B.

## Testing

```bash
# Pure Python unit tests (no Docker needed - run on any machine)
python3 -m pytest tests/test_chunking.py -v

# Everything else runs inside the container (real Qdrant + fastembed)
docker compose run --rm mem0-local pytest tests/ -v
```

| Test file | Needs |
|---|---|
| `test_chunking.py` | Pure Python |
| `test_dimension_mismatch.py` | Qdrant only |
| `test_memory_raw.py` | Qdrant + embedder |
| `test_mem0_local.py` | Full stack (HTTP) |
| `test_new_features.py` | Qdrant + embedder |

The entrypoint runs a 12-tool self-test on every start; failures don't block startup but are visible in logs.

## File structure

```
mem0-local/
├── setup.sh             # build + start single container (update mode preserves data)
├── docker-compose.yml   # ONE service: mem0-local (Qdrant + server)
├── Dockerfile           # multistage: pinned Qdrant binary + python:3.12-slim
├── entrypoint.sh        # Qdrant background -> self-test -> server
├── mcp_server.py        # MCP server - 13 tools, NO LLM, NoExtractionLLM sentinel
├── selftest.py          # 12/12 tool verification (runs in entrypoint)
├── import_docs.py       # Import markdown docs verbatim
├── requirements.txt     # mem0ai[nlp], qdrant-client, fastembed
├── Makefile             # make up / logs / health / export / import
├── .env.example         # embedder config only - no LLM vars
└── tests/
    ├── test_chunking.py           # pure Python
    ├── test_dimension_mismatch.py # Qdrant only
    ├── test_memory_raw.py         # CRUD, no LLM
    ├── test_mem0_local.py         # full stack HTTP
    └── test_new_features.py       # export/import/prune/scores
```

## Troubleshooting

**Container unhealthy** — `curl http://localhost:8765/health`. `init_status: initializing` = fastembed still loading (wait). Qdrant down = check `docker compose logs mem0-local`.

**First start slow** — the fastembed model downloads from HuggingFace (~160 MB, once). Later starts are fast.

**"Connection failed" in IDE** — `curl http://localhost:8765/health`

**"Dimension mismatch"** — `MEM0_EMBED_DIMS` must match the stored collection (768 for nomic v1.5). Only wipe the collection if you deliberately changed embedders:

```bash
curl -X DELETE http://localhost:6333/collections/mem0
./setup.sh
```

**Qdrant process died** — volume permissions on `/qdrant/storage`; check `docker compose logs mem0-local`.

**Want a fresh start** — `docker compose down -v` deletes containers AND all memories.