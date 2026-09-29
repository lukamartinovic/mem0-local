# mem0-local — 10-Minute Team Demo

> Open `architecture.html` in a browser as the visual guide. Follow the sections left-to-right.

---

## 1. The Problem (1.5 min)

**Every AI coding IDE is stateless.** Claude Code, Cursor, Codex, Bob — each session starts from zero.

- You ask the agent something -> it helps -> you close the laptop
- Next session: same question, same re-reading of the codebase, same re-discovery of decisions made yesterday
- Context windows are finite. Long projects = re-explaining architecture every session
- Cloud memory APIs (Mem0 hosted, OpenAI memory) exist but **send your codebase to a third party**

> **The problem:** Stateless agents waste time and tokens re-learning context, and cloud memory solutions leak proprietary code.

---

## 2. The Solution (2 min)

**A self-hosted memory layer that runs entirely on your machine.** No cloud, no data leaving the laptop.

**KEY POINT: the server has NO extraction LLM.** The calling agent is the smart one - it does fact inference itself. The server is pure storage: fastembed (ONNX) embeddings + Qdrant vectors, in ONE Docker container.

**Point at the architecture diagram - trace the flow:**

1. **AI IDE agent** (left box) -> does fact inference itself, calls `add_raw_memory("one concise fact")`
2. **MCP server** (center) -> 12 tools, health endpoint, web UI at `:8765/`
3. **fastembed** (ONNX in-container) -> nomic v1.5, 768 dims, CPU
4. **Qdrant v1.13.2** (same container) -> vector database, semantic search

**One command to start:** `./setup.sh` - builds the single container, runs the self-test, waits for health.

**For the team:** Point your IDE at `http://localhost:8765/mcp` and you have persistent memory. That's it.

---

## 3. Use Cases (3 min)

### Use case 1: Cross-session project memory

> "Agent, we decided to use PostgreSQL 15 with pgvector for embeddings last week."

**Before:** Agent has no idea - starts from zero.
**After:** Agent calls `search_memories("database choice")` -> gets back "Chose PostgreSQL 15 with pgvector for embeddings" -> picks up where you left off.

### Use case 2: Team knowledge sharing on VPN

- Memory is scoped by `user_id` - use `"project-alpha"` or `"team-backend"`
- One server on the VPN, everyone points their IDE at it
- When dev A stores "Auth service uses JWT with 24h expiry", dev B's agent finds it next session
- Export/import for transfer between machines

### Use case 3: Bulk knowledge import

```bash
python3 import_docs.py /path/to/docs --user-id myproject
```

- Chunks markdown docs -> stores VERBATIM (no LLM anywhere)
- Feed it architecture docs, ADRs, runbooks - semantic search finds them next session
- For curated "agent wisdom", the AGENT extracts facts itself via `add_raw_memory`

### Use case 4: Stale memory cleanup

```bash
# Dry run first (safe)
prune_memories(older_than_days=30, dry_run=true)
# Then actually delete
prune_memories(older_than_days=30, dry_run=false)
```

Old decisions ("we use version 2.3" when you're on 3.1) are **worse** than no memory. Prune keeps the DB clean.

---

## 4. Technical Details (3 min)

### Architecture decisions and why

| Decision | Why |
|---|---|
| **NO extraction LLM** | The agent is the one with the powerful model. Extraction added 30-60s latency, a model to run, and a whole failure mode class (truncation, thinking tokens, JSON breakage). Removed all of it. |
| **Single container** | Qdrant binary lifted from the pinned image into the server image. One `docker compose up`, one log stream, one restart command. |
| **fastembed (ONNX)** | Local embeddings on CPU, no Ollama, no GPU, no model server. nomic v1.5 = 768 dims = same as the old collection, so data survives the migration. |
| **`NoExtractionLLM` sentinel** | mem0 requires an `llm` config key (injects default OpenAI if absent). The sentinel makes any accidental LLM call fail loudly instead of silently calling a cloud API. |
| **Qdrant pinned to v1.13.2** | `:latest` jumped to v1.18, broke the client library. Pin prevents silent breakage. |
| **Size guard on `add_raw_memory`** | Oversized content is rejected with a "split into facts or use add_verbatim" hint - keeps memories atomic and embeddings precise. |

### Tool surface

| Tool | Contract |
|---|---|
| `add_raw_memory` | ONE concise self-contained fact (10-30 words); oversized rejected with hint |
| `add_verbatim` | Raw text verbatim, auto-chunked at paragraph boundaries |
| search/get/update/delete/export/import/prune | Unchanged, all LLM-free |

### Deployment

```
./setup.sh          -> build + start (first start: ~160MB embed model download, once)
./setup.sh update   -> rebuild image, preserve all Qdrant data
```

- ONE container (Qdrant + MCP server), port 8765 (MCP) + 6333 (Qdrant)
- Data in Docker volume `qdrant_data` - survives container rebuilds
- Health endpoint: `curl localhost:8765/health` -> `"extraction_llm": false` by design

### Privacy

- **Zero** cloud calls for storage. No telemetry (mem0's phone-home disabled via env var).
- Embeddings and vector store run on your machine.
- The agent's own LLM (whatever powers the IDE) never sees stored memories except through explicit `search_memories` results.

---

## Quick Q&A Anticipated

**"Why remove the extraction LLM?"** -> It was the slowest, most fragile part (truncation, thinking models, JSON parsing, context windows) for work the calling agent already does better. Storage is instant now.

**"Performance?"** -> Storage is instant (embedding ~10ms). Search ~100ms. No model warm-up, no extraction wait.

**"What runs locally?"** -> Everything: embeddings (fastembed ONNX), vectors (Qdrant). Zero cloud APIs.

**"How many memories can it hold?"** -> Qdrant handles millions of vectors. Storage is no longer the bottleneck - there IS no extraction bottleneck.

**"Can multiple agents share one server?"** -> Yes - scope by `user_id`. One server, N projects/agents.

**"Migration from the Ollama version?"** -> Drop-in. Both nomic models are 768-dim, so existing Qdrant collections work as-is.

---

## Demo Flow Checklist

- [ ] Open `architecture.html` in browser
- [ ] 1. Problem - stateless IDEs waste time (1.5 min)
- [ ] 2. Solution - trace the diagram, emphasize NO LLM (2 min)
- [ ] 3. Use cases - pick 2-3 to show (3 min)
- [ ] 4. Technical - decision table + tool surface (3 min)
- [ ] 1 min buffer for questions