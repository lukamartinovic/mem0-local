# mem0-local

Self-hosted [Mem0](https://mem0.ai) memory layer for AI coding agents, running **entirely on your machine** — no cloud API calls, no data leaving your laptop.

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
./setup.sh clean    # stop everything, delete all data
```

Or via Makefile:

```bash
make up        # same as ./setup.sh
make test      # run tests
make health    # health check
make logs      # tail server logs
make clean     # stop + delete everything
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
| **Ollama** | Host (native) | Metal GPU acceleration — 10x faster than Docker emulation |
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

| Tool | Description |
|---|---|
| `add_memory` | Save text or conversation history to memory |
| `search_memories` | Semantic search across stored memories |
| `get_memories` | List memories with filters and pagination |
| `get_memory` | Retrieve a single memory by ID |
| `update_memory` | Overwrite a memory's text by ID |
| `delete_memory` | Delete a single memory by ID |
| `delete_all_memories` | Delete all memories for a user |
| `list_entities` | List distinct user/agent/app IDs in memory |
| `delete_entities` | Delete a user/agent entity and all its memories |

## Configuration

Override defaults via `.env` file:

```bash
cp .env.example .env
# Edit .env, then restart:
./setup.sh
```

| Variable | Default | Description |
|---|---|---|
| `MEM0_LLM_MODEL` | `qwen2.5:3b` | Ollama LLM model (must support tool calling) |
| `MEM0_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `MEM0_DEFAULT_USER_ID` | `dev` | Default user_id in tool calls |

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

## Testing

```bash
# Full test suite (runs inside the container)
./setup.sh test

# Just infrastructure checks (fast — no model calls)
docker compose run --rm mcp-server pytest tests/ -v -k TestInfrastructure

# Just MCP protocol tests
docker compose run --rm mcp-server pytest tests/ -v -k TestMCPProtocol

# Just memory operations (needs models loaded)
docker compose run --rm mcp-server pytest tests/ -v -k TestMemoryOperations
```

## System prompt for auto-memory

Since local MCP doesn't have lifecycle hooks, add this to your IDE's custom instructions (`.cursorrules`, `CLAUDE.md`, etc.):

```markdown
## Memory Protocol

- Before answering any question, call `search_memories` with your query to
  check for relevant past context.
- After completing a task or learning something new, call `add_memory` to
  store: decisions made, files modified, user preferences, patterns established.
- Use `user_id` = "dev" unless in a project context, then use the project name.
```

## File structure

```
mem0-local/
├── setup.sh             # One-command setup (installs Ollama, pulls models, starts containers)
├── docker-compose.yml   # Qdrant + MCP server (+ optional Ollama in Docker mode)
├── Dockerfile            # MCP server image
├── entrypoint.sh         # Waits for Ollama + Qdrant, starts MCP server
├── mcp_server.py         # MCP server — 9 tools, HTTP JSON-RPC
├── requirements.txt      # Python dependencies
├── Makefile              # make up / test / health / logs / clean
├── .env.example          # Override model, user_id
├── .dockerignore
├── pytest.ini
├── README.md
└── tests/
    └── test_mem0_local.py  # 12 tests: infra + protocol + memory operations
```

## Troubleshooting

**First run is slow** — Ollama downloads models (2-5 GB). `setup.sh` shows progress.

**"Connection failed" in IDE** — `curl http://localhost:8765/health`

**"Model not found"** — `ollama pull <model_name>` then restart: `./setup.sh`

**"Dimension mismatch"** — You changed the embedding model. Delete the collection:
```bash
curl -X DELETE http://localhost:6333/collections/mem0
./setup.sh
```

**Ollama not starting** — `ollama serve` and check for errors. Common: port 11434 already in use (`lsof -iTCP:11434`).

**Want to start fresh** — `./setup.sh clean` deletes everything (containers, volumes, memories).