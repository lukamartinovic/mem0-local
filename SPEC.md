# mem0-local - Current Specification

Version: 1.1.0 (`SERVER_VERSION` in `mcp_server.py`)
Repository: github.com/lukamartinovic/mem0-local @ `43d8c6b`
Status: running in production, 272 tests passing

## 1. Purpose and scope

mem0-local is a self-hosted persistent memory server for AI coding agents (Claude Code, Cursor, Codex, IBM Bob). It gives stateless IDE agents a durable store for facts, decisions, and document excerpts that survives across sessions, projects, and agent tools.

Scope, stated precisely: mem0-local stores text, embeds it locally, and retrieves it by semantic similarity. It is a store, not an analyst. There is deliberately **no extraction LLM inside the server** - the calling agent performs fact inference itself and submits concise facts via `add_raw_memory`, or raw text via `add_verbatim`. Nothing inside the container filters, summarises, or interprets content.

Everything runs in one Docker container. No cloud dependency, no telemetry, no accounts. Zero network access is required at runtime.

## 2. Who it is for

- Individual developers running one or more AI coding agents who want one shared memory across tools and across projects.
- Small teams running a shared instance on a VPN or LAN.
- Anyone whose data must never leave the machine: memory content, embeddings, and the vector index all stay local.

The operational model assumes a technical operator comfortable running `docker compose`. The trust model assumes loopback or LAN; there is no authentication (see Security model).

## 3. Architecture

One Docker container packages the MCP server, Qdrant, and the embedding stack. Both models are baked into the image at build time, so no download happens at runtime.

```
+-------------------- host ---------------------------+
|                                                     |
|  AI agents (Claude Code, Cursor, Codex, Bob)        |
|     |                                               |
|     | MCP JSON-RPC 2.0   HTTP/GET                   |
|     v                                               |
|  +------------------- Docker container ------------+|
|  |  mcp_server.py (asyncio HTTP, :8765)            ||
|  |    |-- POST /mcp           (MCP tools)          ||
|  |    |-- GET  /              (web UI)             ||
|  |    |-- GET  /health        (health contract)    ||
|  |    |-- GET  /api/memories                       ||
|  |    |-- POST /api/import                         ||
|  |    |                                            ||
|  |    +-- mem0 2.0.11 (infer=False; LLM neutered)  ||
|  |         |                                       ||
|  |         +--> fastembed (ONNX, CPU)              ||
|  |              nomic-ai/nomic-embed-text-v1.5     ||
|  |              768 dims, baked into image         ||
|  |    |                                            ||
|  |    +--> Qdrant v1.13.2 (:6333)                  ||
|  |         vector store, volume: qdrant_data       ||
|  +-------------------------------------------------+|
+-----------------------------------------------------+

Volumes: qdrant_data (vectors)  model_cache (HF cache)
```

Key architectural facts:

- **Single container.** One compose service named `mem0-local`. Ports 8765 (MCP + web UI) and 6333 (Qdrant).
- **Embeddings in-container.** fastembed (ONNX, CPU) with `nomic-ai/nomic-embed-text-v1.5`, 768 dimensions, baked into the image at build time. The legacy Ollama tag `nomic-embed-text` is aliased to the same model at config load, so collections built before the migration remain compatible.
- **No LLM hop.** mem0 2.0.11 (`mem0ai[nlp]==2.0.11`) is pinned; all calls use `infer=False`. mem0's required `llm` config key is present but neutered, because mem0 injects a default OpenAI LLM when the key is absent. Extraction is the calling agent's job.
- **spaCy guard.** mem0 lemmatizes on `add()` and `search()` for BM25 metadata; when `en_core_web_sm` is missing, mem0's download path calls `sys.exit(1)`, and `SystemExit` inherits `BaseException`, killing the server process. The model is baked into the image, and a guard neuters mem0's download hook so a missing model degrades to raw text instead of exiting.
- **Supervised startup.** `entrypoint.sh` runs: storage guard -> Qdrant -> dimension check -> self-test -> exec server. Each stage is fail-fast (see Reliability).

Dependencies: mem0 2.0.11 pinned; `qdrant-client >=1.12,<1.14`; `fastembed >=0.3.1`; Qdrant image pinned to `v1.13.2` (`:latest` jumped to v1.18.x, incompatible with the client pin).

## 4. Data model and namespacing

A stored memory record carries:

| Field | Notes |
|---|---|
| `id` | assigned by mem0/Qdrant |
| `memory` | the text (one fact, or one chunk of verbatim text) |
| `metadata` | free-form dict; `add_verbatim` adds `chunk: "i/n"`; context-header fields `file`/`source` are read from metadata |
| `created_at` | ISO-T, date-only, unix epoch, trailing `Z`, or tz-naive (treated as UTC) - the parser accepts all of these |
| `user_id` | the namespace |

**Namespacing is by `user_id` only.** All tools accept `user_id`, defaulting to `MEM0_DEFAULT_USER_ID` (`"dev"`). There is no project, repo, or agent dimension in the data model beyond what callers put in `metadata`. Cross-agent sharing is achieved by pointing several agents at the same server with an agreed `user_id`; isolation between contexts is achieved by using different `user_id` values.

## 5. MCP tool reference

All 13 tools are exposed over `POST /mcp`. `user_id` defaults to `$MEM0_DEFAULT_USER_ID` on every tool that accepts it. `memory_id` values come from store responses.

| # | Tool | Required params | Optional params | Returns / notable behaviour |
|---|---|---|---|---|
| 1 | `add_raw_memory` | `content` | `user_id`, `metadata` | Stores ONE concise fact via mem0 `add(infer=False)`. Rejects content longer than `MEM0_CHUNK_CHARS` with an error plus a fix hint routing the caller to `add_verbatim`. Not a bulk path. |
| 2 | `add_verbatim` | `content` | `user_id`, `metadata` | Stores raw text of any length. Auto-chunks with a cascade: paragraph -> line -> word -> character boundary, chunk size `MEM0_CHUNK_CHARS` (default 3000). Adds `chunk: "i/n"` metadata and prepends a `[Document: X] [Source: Y]` context header built from `metadata.file`/`metadata.source`. |
| 3 | `search_memories` | `query` | `user_id`, `limit`, `min_score`, `include_scores` | Returns a LIST. Passes `threshold=0.0` to mem0 so all candidates come back; `min_score` acts as a floor on results. `include_scores=False` strips the score field from results. |
| 4 | `get_memories` | none | `user_id`, `limit`, `page` | Paginated listing of one user's memories. |
| 5 | `get_memory` | `memory_id` | - | Fetches a single memory by id. |
| 6 | `update_memory` | `memory_id`, `content` | - | Replaces a memory's text. mem0's `update()` is LLM-free. |
| 7 | `delete_memory` | `memory_id` | - | Deletes one memory. |
| 8 | `delete_all_memories` | none | `user_id` | Deletes every memory in the user's namespace. |
| 9 | `list_entities` | none | - | Lists known `user_id` namespaces. Tolerates present-but-null payload keys in the Qdrant scroll. |
| 10 | `delete_entities` | `user_id` | - | Deletes an entity namespace. |
| 11 | `export_memories` | none | `user_id`, `format` | `format` is `json` or `csv`. JSON returns `{format, count, user_id, memories[]}` with records `{id, memory, metadata, created_at, user_id}`. CSV returns `{format, count, user_id, data}` with header `id,memory,metadata,created_at,user_id`. Entity-scoped only. |
| 12 | `import_memories` | `data` | `user_id` | Imports records, skipping duplicates by case-insensitive exact text match within the user. Returns `{imported, skipped, failed, errors, user_id}`. The HTTP `/api/import` route uses the same helper. |
| 13 | `prune_memories` | none | `user_id`, `older_than_days`, `dry_run`, `min_score` | Deletes memories older than `older_than_days`. **`dry_run` defaults to TRUE** - deletion is opt-in. `min_score` acts as a deletion floor: memories scoring `>= min_score` are kept, and **un-scored memories are always kept** (never delete what could not be evaluated). `created_at` parsing handles ISO-T, date-only, unix epoch, trailing `Z`, and tz-naive (treated as UTC); unparsable values are skipped, never treated as ancient. |

Error behaviour shared across tools: missing required arguments raise a typed `Mem0Error` with an actionable `Fix:` line - never a bare `KeyError` escaping into the transport's generic handler.

## 6. HTTP API reference

One asyncio HTTP server on `MCP_PORT` (default 8765) serves all routes:

| Method | Path | Behaviour |
|---|---|---|
| GET | `/` | Web UI: browse and filter memories, export as JSON, CSV, or clipboard, import from file or pasted text with preview. |
| GET | `/health` | Health contract (next section). |
| GET | `/api/memories` | `{memories[], entities[]}` via a direct Qdrant scroll (not through mem0). |
| POST | `/api/import` | Body `{user_id, memories[]}`; returns imported/skipped/failed counts. Uses the same helper as the `import_memories` MCP tool. |
| POST | `/mcp` | MCP JSON-RPC 2.0: `initialize`, `tools/list`, `tools/call`, `ping`, `notifications/initialized`. |

## 7. Health and observability contract

`GET /health` returns: `status`, `init_status`, `init_error`, `components{qdrant, mem0, extraction_llm}`, `server`, `version`, `tools`, and `config{extraction_llm, lemmatizer, embedder, embed_dims, vector_store}`.

Status semantics:

- `status == "ok"` **only when** `init_status == "ready"` AND Qdrant is reachable AND mem0 is initialised.
- `"starting"` while initialisation is in progress, `"degraded"` on error.
- The Docker healthcheck asserts `status == "ok"` - not merely HTTP 200. Health never reports a curl-exit-code success.
- `version` identifies the running build; `init_error` carries the actual exception on failure, so a stuck `starting` is diagnosable from outside. Initialisation is retryable: guards check the status, not just the object.

Known observability gaps are listed under Known limitations.

## 8. Configuration reference

All configuration is environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MEM0_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | fastembed model id. The legacy Ollama tag `nomic-embed-text` is aliased to it at load. |
| `MEM0_EMBED_DIMS` | `768` | Embedding dimension; must agree with the Qdrant collection. |
| `MEM0_CHUNK_CHARS` | `3000` | Chunk size for `add_verbatim`; oversized-content threshold for `add_raw_memory`. |
| `MEM0_QDRANT_HOST` | `127.0.0.1` | Qdrant host. |
| `MEM0_QDRANT_PORT` | `6333` | Qdrant port. |
| `MEM0_DEFAULT_USER_ID` | `dev` | Default namespace for all tools. |
| `MCP_HOST` | `0.0.0.0` | HTTP bind address. |
| `MCP_PORT` | `8765` | HTTP port (MCP + web UI). |
| `MEM0_STRICT_DIMS` | `1` | When `1`, a dimension mismatch refuses to start. Set `0` to override. |
| `MEM0_TELEMETRY` | `False` | Telemetry is off. |
| `HF_HOME` | `/model_cache` | HuggingFace cache path, backed by the `model_cache` volume. |

## 9. Deployment and lifecycle

Files: `setup.sh`, `docker-compose.yml`, `Dockerfile` (multistage: Qdrant image + `python:3.12-slim`, `libunwind8` installed before the Qdrant binary check), `entrypoint.sh`, `mcp_server.py` (77 KB), `selftest.py`, `import_docs.py` (bulk markdown import, verbatim), `Makefile`.

- **Start:** `./setup.sh` (or `make up`). One command; prompts are idempotent and fire only on first run.
- **Update:** `./setup.sh update` (or `make update`). Rebuilds the image and preserves Qdrant data, which lives in the `qdrant_data` volume, so memories survive container rebuilds.
- **Reboot behaviour:** compose service runs with `restart: unless-stopped`; the container starts unattended with Docker. Survives reboots fire-and-forget.
- **Other lifecycle targets:** `make down`, `make logs`, `make health`, `make shell`, `make export`, `make import`, `make import-docs`.

## 10. Reliability and failure policy

The tests enforce these guarantees:

- **No silent data loss.** Unwritable storage refuses to start; there is no `/tmp` fallback.
- **Dimension safety.** A vector-dimension mismatch refuses to start unless `MEM0_STRICT_DIMS=0`.
- **Failed self-test is fatal.** `selftest.py` exercises 12 tools at startup; the server does not exec until it passes.
- **Honest health.** Health asserts the payload status, never an HTTP-200-only success.
- **No legacy model names at runtime.** The embedder is always a valid fastembed repo id.
- **Actionable errors.** `Mem0Error` and `QdrantError` produce messages with a `Fix:` line.
- **Destructive tools default to safe.** `prune_memories` dry-runs unless told otherwise, and never deletes memories it could not score.

## 11. Testing strategy

272 passed, 76 skipped (skips require Docker or the live stack). Tiers:

| Suite | Tests | Tier |
|---|---|---|
| test_chunking | 13 | offline |
| test_no_llm | 32 | offline (asserts the no-LLM invariant) |
| test_execute_tool_units | 83 | offline units |
| test_integration_local | 36 | local stack |
| test_setup_and_packaging | 62 | packaging |
| test_reliability_contract | 34 | failure policy above |
| test_webui | 14 | web UI |
| test_container_e2e | 26 | Docker-gated |
| test_mem0_local | 13 | offline (Qdrant file mode) |
| test_memory_raw | 9 | offline |
| test_new_features | 19 | mixed |
| test_dimension_mismatch | 7 | failure policy |

`make test` runs the suite inside the container; `make test-host` runs the offline tiers on the host. `make test-full` runs everything.

## 12. Security model and its honest limits

The model is loopback/LAN trust. Data never leaves the machine: no cloud APIs, no telemetry, no accounts. That is the security perimeter, and it is enforced structurally (no network calls at runtime), not by policy.

Honest limits:

- **No authentication on any endpoint.** Anyone who can reach port 8765 can read, write, and delete every memory. Deployment on a shared LAN means trusting everyone on it.
- **No multi-tenant isolation** beyond the `user_id` namespace. Namespaces are conventions, not boundaries - any caller can read or delete any `user_id`.
- **Bind address is `0.0.0.0` by default**, so the port is reachable from the network unless firewalled.

## 13. Known limitations

- No memory TTL/expiry exposed through the tools (mem0 supports `expiration_date` internally; it is not wired up).
- Prune is age-based only; no per-fact lifetime.
- No access/usage tracking (no `last_used_at`); no signal about whether a memory was ever useful.
- `/health` has no memory count or disk usage.
- Web UI listing cannot filter by time; export is entity-scoped only.
- No auth on any endpoint.
- Single-user orientation.
- Dedup is the caller's responsibility: with no extraction LLM, the server stores what it is told; callers should search before adding (~0.85 similarity is a working duplicate threshold).

## 14. Non-goals

- Extraction, summarisation, or any inference inside the server - permanently out of scope.
- Being an analyst: no ranking of "importance", no consolidation, no dreaming.
- Cloud sync, hosted tiers, accounts, telemetry.
- Project-scoped namespacing in the data model (callers use `user_id` and `metadata`).