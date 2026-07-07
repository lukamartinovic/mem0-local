# mem0-local

Self-hosted [Mem0](https://mem0.ai) memory layer for AI coding agents, running **entirely on your machine** — no cloud API calls, no data leaving your laptop.

## Target behavior

The system gives your AI coding agent **persistent memory across sessions**. Without it, every new session starts from zero — the agent has to re-read your codebase to understand context. With it:

- **Before answering:** the agent searches stored memories for relevant past context (decisions, patterns, preferences)
- **After completing tasks:** the agent saves what it learned (files changed, decisions made, bugs fixed)
- **Next session:** the agent recalls yesterday's work instantly instead of re-reading everything

The memory is **semantic** — the agent searches with natural language and gets back the most relevant facts, not keyword matches. It's **intelligent** — when storing new info, the LLM extracts key facts and deduplicates against existing memories. And it's **private** — everything runs locally, no data leaves your machine.

**One command:**

```bash
./setup.sh
```

That's it. It installs Ollama if needed, starts it with Metal GPU acceleration, pulls the models, brings up Qdrant + the MCP server in Docker, and waits until everything is healthy. When it prints the success banner, point your IDE at `http://localhost:8765/mcp`.

## Quick start

```bash
# 1. Run setup (does everything)
./setup.sh

# 2. Add to your IDE config (one-time, see below)
#    URL: http://localhost:8765/mcp

# 3. Run tests to verify
./setup.sh test
```

### Other commands

```bash
./setup.sh          # start (native Ollama + Docker containers)
./setup.sh test     # run tests inside the container
./setup.sh update   # rebuild Docker image with latest code, preserve all data
./setup.sh clean    # stop everything, delete all data
```

Or via Makefile:

```bash
make up        # same as ./setup.sh
make test      # run tests
make health    # health check
make logs      # tail server logs
make clean     # stop + delete everything
make export USER=dev    # export memories to JSON
make import FILE=backup.json USER=dev  # import memories from JSON
make shell     # shell into the MCP server container
```

## How it works

```
Your IDE (Cursor / Claude Code / Codex / etc.)
  ↓ HTTP (MCP JSON-RPC)
MCP server (Docker, :8765)
  ↓ Python (mem0ai library)
  ↓         ↓
Ollama    Qdrant (Docker, :6333)
(host, native, Metal GPU)
```

| Component | Where it runs | Why |
|---|---|---|
| **Ollama** | Host (native) or external | Metal GPU acceleration — 10x faster than Docker emulation |
| **Qdrant** | Docker container | State isolation, persistent volume |
| **MCP server** | Docker container | Portable, reproducible, no bare metal deps |

In native mode (default), Ollama runs directly on your machine with full Metal GPU access. The MCP server in Docker reaches it via `host.docker.internal:11434`.

## Configure your IDE

Add this to your IDE's MCP config file:

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

## Available MCP tools

### Core memory operations

| Tool | Description |
|---|---|
| `add_memory` | Save text or conversation history to memory (with LLM fact extraction) |
| `search_memories` | Semantic search with relevance scores and min_score filtering |
| `get_memories` | List memories with filters and pagination |
| `get_memory` | Retrieve a single memory by ID |
| `update_memory` | Overwrite a memory's text by ID |
| `delete_memory` | Delete a single memory by ID |
| `delete_all_memories` | Delete all memories for a user |
| `list_entities` | List distinct user/agent/app IDs in memory |
| `delete_entities` | Delete a user/agent entity and all its memories |

### Backup & maintenance

| Tool | Description |
|---|---|
| `export_memories` | Export all memories for a user as JSON or CSV |
| `import_memories` | Import memories from JSON (skips duplicates) |
| `prune_memories` | Delete memories older than N days (dry-run by default) |

### Tool details

#### `add_memory`

```json
{
  "content": "We decided to use PostgreSQL 15 with pgvector for embeddings.",
  "user_id": "myproject",
  "metadata": {"category": "architecture"},
  "infer": true
}
```

- `content` (required): Text or JSON array of `{role, content}` conversation messages
- `user_id` (optional, default: `dev`): Project/user identifier
- `metadata` (optional): Key-value pairs attached to the memory
- `infer` (optional, default: `true`): If `true`, LLM extracts facts. If `false`, stores raw text (no LLM needed — useful for testing or when LLM is unavailable)

Large text is automatically chunked at paragraph → line → word boundaries based on the model's context size.

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

- `query` (required): Natural language search
- `user_id` (optional, default: `dev`)
- `limit` (optional, default: `10`): Max results
- `min_score` (optional, default: `0.0`): Filter out results below this cosine similarity score (0.0–1.0)
- `include_scores` (optional, default: `true`): Include relevance score in each result

Results include a `score` field (cosine similarity, 0.0–1.0) when `include_scores` is true.

#### `export_memories`

```json
{
  "user_id": "myproject",
  "format": "json"
}
```

- `user_id` (optional, default: `dev`)
- `format` (optional, default: `json`): `"json"` or `"csv"`

Exports all memories for the user. Each memory includes: `id`, `memory` (text), `metadata`, `created_at`, `user_id`.

Returns the exported data directly in the MCP response. Use `make export USER=myproject > backup.json` from the CLI.

#### `import_memories`

```json
{
  "data": "[{\"memory\": \"We use Redis for caching\", \"user_id\": \"myproject\"}]",
  "user_id": "myproject"
}
```

- `data` (required): JSON string of memory array (as exported by `export_memories`)
- `user_id` (optional, default: `dev`): Default user_id for memories that don't specify one

Skips duplicates (memories with identical text already exist for the user). Returns: `{ "imported": N, "skipped": N, "failed": N }`.

#### `prune_memories`

```json
{
  "user_id": "myproject",
  "older_than_days": 30,
  "dry_run": true
}
```

- `user_id` (optional, default: `dev`)
- `older_than_days` (optional, default: `30`): Delete memories older than this many days
- `dry_run` (optional, default: `true`): If `true`, only report what would be deleted without actually deleting

Returns: `{ "would_delete": N, "deleted": N, "dry_run": bool, "memories": [...] }`.

**Safe by default** — `dry_run` is `true` unless explicitly set to `false`.

## Configuration

Override defaults via `.env` file:

```bash
cp .env.example .env
# Edit .env, then restart:
./setup.sh
```

| Variable | Default | Description |
|---|---|---|
| `MEM0_LLM_MODEL` | `qwen2.5:7b` | Ollama LLM model (must support tool calling) |
| `MEM0_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `MEM0_EMBED_DIMS` | `768` | Embedding dimensions (must match embedder model) |
| `MEM0_DEFAULT_USER_ID` | `dev` | Default user_id in tool calls |
| `MEM0_OLLAMA_URL` | `host.docker.internal:11434` | Ollama URL — set to use external provider |
| `MEM0_LLM_TEMPERATURE` | `0.1` | LLM temperature for fact extraction |
| `MEM0_LLM_MAX_TOKENS` | `2000` | Max tokens for LLM extraction response |
| `MEM0_TELEMETRY` | `False` | Disable mem0 phone-home telemetry |

### Using an external Ollama provider

For cases where you can't run models locally (limited RAM, company VPN, shared GPU server), point mem0-local at a remote Ollama instance:

```bash
cp .env.example .env
# Edit .env:
echo "MEM0_OLLAMA_URL=https://ollama.internal.company.com" >> .env
echo "MEM0_LLM_MODEL=qwen2.5:7b" >> .env  # must be pre-pulled on remote

# Start — setup.sh detects external URL and skips local Ollama
./setup.sh
```

In external mode:
- Qdrant still runs locally in Docker (vectors stay on your machine)
- Only LLM + embedding calls go to the remote Ollama
- Models must be pre-pulled on the remote instance (`ollama pull qwen2.5:7b`)
- `setup.sh` verifies connectivity and model availability before starting
- If the VPN drops, memory operations fail gracefully with actionable error messages

### Switching to a bigger model

```bash
# Pull it (setup.sh can also do this automatically)
ollama pull qwen3.5:9b

# Update .env
echo "MEM0_LLM_MODEL=qwen3.5:9b" > .env

# Restart
./setup.sh
```

Model must support tool calling — check [ollama.com/search?c=tools](https://ollama.com/search?c=tools).

## Health monitoring

The `/health` endpoint provides real-time component status:

```bash
curl http://localhost:8765/health | python3 -m json.tool
```

Response:

```json
{
  "status": "ok",
  "init_status": "ready",
  "server": "mem0-local",
  "tools": 12,
  "components": {
    "ollama": true,
    "qdrant": true,
    "mem0": true
  },
  "model_status": "loaded",
  "config": {
    "llm": "qwen2.5:7b",
    "embedder": "nomic-embed-text",
    "vector_store": "qdrant"
  }
}
```

| Field | Values | Meaning |
|---|---|---|
| `status` | `ok`, `starting`, `degraded` | Overall health |
| `init_status` | `starting`, `initializing`, `ready`, `error` | mem0 initialization state |
| `components.ollama` | `true`/`false` | Ollama API reachable |
| `components.qdrant` | `true`/`false` | Qdrant API reachable |
| `components.mem0` | `true`/`false` | mem0 library initialized |
| `model_status` | `loaded`, `unloaded`, `unreachable`, `unknown` | LLM model responsive (cached 30s) |

The server starts accepting HTTP requests immediately — mem0 initialization runs in a background thread. During startup, `/health` returns `status: "starting"` and `init_status: "initializing"`. The Docker healthcheck waits for `status: "ok"` before marking the container healthy.

The LLM model ping is cached for 30 seconds to avoid excessive calls. It sends a minimal 1-token generation request to verify the model is loaded and responsive, not just that Ollama is running.

## Backup and restore

### Update without losing data

When you pull new code or modify `mcp_server.py`, use `update` instead of `clean`:

```bash
./setup.sh update
```

This rebuilds the Docker image with the latest code while preserving the Qdrant volume (all memories). Contrast with `./setup.sh clean` which deletes everything.

### Export memories

```bash
# Export all memories for a user (JSON)
make export USER=myproject > backup.json

# Or via MCP tool call
curl -X POST http://localhost:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"export_memories",
       "arguments":{"user_id":"myproject","format":"json"}}}'
```

### Import memories

```bash
# Import from a backup file
curl -X POST http://localhost:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"import_memories",
       "arguments":{"data":"'"$(cat backup.json | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')"'",
                    "user_id":"myproject"}}}'
```

Duplicates are automatically skipped (same memory text for the same user_id).

### Transfer between machines

1. Export on machine A: `make export USER=dev > backup.json`
2. Copy `backup.json` to machine B
3. Import on machine B: use `import_memories` MCP tool

## Memory maintenance

### Prune stale memories

Stale memories (e.g., "we use version 2.3" when you're on 3.1) are worse than no memories. Prune them:

```bash
# Dry run first (safe — default)
curl -X POST http://localhost:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"prune_memories",
       "arguments":{"user_id":"dev","older_than_days":30,"dry_run":true}}}'

# If the results look correct, actually delete:
curl -X POST http://localhost:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"prune_memories",
       "arguments":{"user_id":"dev","older_than_days":30,"dry_run":false}}}'
```

### Search with relevance filtering

Filter out low-relevance results to avoid noise:

```bash
# Only return results with score >= 0.7
curl -X POST http://localhost:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"search_memories",
       "arguments":{"query":"database choice","user_id":"dev",
                    "min_score":0.7,"include_scores":true}}}'
```

Each result includes a `score` field (0.0–1.0) indicating cosine similarity to the query.

## Testing

```bash
# Full test suite (runs inside the container, needs models loaded)
./setup.sh test

# Pure Python unit tests (no Docker needed — run on any machine)
python3 -m pytest tests/test_chunking.py -v

# Infrastructure checks (fast — no model calls)
docker compose run --rm mcp-server pytest tests/ -v -k TestInfrastructure

# MCP protocol tests
docker compose run --rm mcp-server pytest tests/ -v -k TestMCPProtocol

# Memory operations with LLM (needs models loaded)
docker compose run --rm mcp-server pytest tests/ -v -k TestMemoryOperations

# Memory operations without LLM (infer=False, needs only Qdrant + embedder)
docker compose run --rm mcp-server pytest tests/ -v -k TestMemoryRaw

# Dimension mismatch tests (needs only Qdrant)
docker compose run --rm mcp-server pytest tests/ -v -k TestDimensionMismatch

# New feature tests (export, import, prune, search scores)
docker compose run --rm mcp-server pytest tests/ -v -k TestExportImport
docker compose run --rm mcp-server pytest tests/ -v -k TestPruneMemories
docker compose run --rm mcp-server pytest tests/ -v -k TestSearchWithScores
```

### Test layers

| Test file | Tests | Needs | Run on 8GB Air? |
|---|---|---|---|
| `test_chunking.py` | 25 | Pure Python | ✅ Direct |
| `test_dimension_mismatch.py` | 7 | Qdrant only | ✅ Via Docker |
| `test_memory_raw.py` | 10 | Qdrant + embedder | ✅ Via Docker |
| `test_mem0_local.py` | 12 | Qdrant + Ollama | Via Docker |
| `test_new_features.py` | 20 | Qdrant + Ollama | Via Docker |

The chunking tests are pure Python and can run without Docker on any machine. They cover the `_chunk_text()` function which was the source of two real bugs (text with no paragraph/line boundaries wasn't split correctly).

## System prompt for auto-memory

Since local MCP doesn't have lifecycle hooks, add this to your IDE's custom instructions (`.cursorrules`, `CLAUDE.md`, etc.):

```markdown
## Memory Protocol

- Before answering any question, call `search_memories` with your query to
  check for relevant past context.
- After completing a task or learning something new, call `add_memory` to
  store: decisions made, files modified, user preferences, patterns established.
- Use `user_id` = "dev" unless in a project context, then use the project name.
- Use `min_score` = 0.5 to filter low-relevance search results.
- Periodically run `prune_memories` with `older_than_days` = 30 to clean up stale memories.
```

## File structure

```
mem0-local/
├── setup.sh             # One-command setup (local + external Ollama modes)
├── docker-compose.yml   # Qdrant + MCP server (healthcheck checks status=ok)
├── Dockerfile            # MCP server image (healthcheck checks status=ok)
├── entrypoint.sh         # Waits for Ollama + Qdrant, runs selftest, starts server
├── mcp_server.py         # MCP server — 12 tools, HTTP JSON-RPC
├── selftest.py           # 12/12 tool verification (runs in entrypoint)
├── import_docs.py        # Import markdown docs into memory
├── requirements.txt      # Python dependencies
├── Makefile              # make up / test / health / logs / export / clean
├── .env.example          # Override model, user_id, Ollama URL
├── .dockerignore
├── pytest.ini
├── conftest.py            # Pytest config (import path for local + container)
├── README.md
└── tests/
    ├── test_chunking.py          # 25 tests: chunking, conversation detection (pure Python)
    ├── test_dimension_mismatch.py # 7 tests: Qdrant collection dim fix (Qdrant only)
    ├── test_memory_raw.py        # 10 tests: CRUD pipeline with infer=False (Qdrant + embedder)
    ├── test_mem0_local.py        # 12 tests: infra + protocol + memory ops (full stack)
    └── test_new_features.py      # 20 tests: export, import, prune, search scores, health
```

## Troubleshooting

**Container shows "unhealthy"** — Check what the health endpoint reports:
```bash
curl http://localhost:8765/health | python3 -m json.tool
```
If `init_status` is `initializing`, the model is still loading — wait. If `model_status` is `unreachable`, Ollama isn't running.

**First run is slow** — Ollama downloads models (2-5 GB). `setup.sh` shows progress. The Docker healthcheck has a 600s start period to accommodate this.

**"Connection failed" in IDE** — `curl http://localhost:8765/health`

**"Model not found"** — `ollama pull <model_name>` then restart: `./setup.sh`

**"Dimension mismatch"** — You changed the embedding model. Delete the collection:
```bash
curl -X DELETE http://localhost:6333/collections/mem0
./setup.sh
```

**Ollama not starting** — `ollama serve` and check for errors. Common: port 11434 already in use (`lsof -iTCP:11434`).

**External Ollama unreachable** — If using `MEM0_OLLAMA_URL` with a remote provider:
```bash
# Check connectivity
curl ${MEM0_OLLAMA_URL}/api/tags

# Common issues:
#   • VPN not connected
#   • Firewall blocking the port
#   • Models not pulled on the remote instance
```

**LLM returns empty results** — The model may be too small for JSON extraction. Try:
```bash
ollama pull qwen2.5:7b  # or larger
echo "MEM0_LLM_MODEL=qwen2.5:7b" >> .env
./setup.sh
```

Or store without LLM extraction: pass `infer: false` in `add_memory` calls.

**Want to start fresh** — `./setup.sh clean` deletes everything (containers, volumes, memories).