#!/usr/bin/env python3
"""
mem0-local MCP server — HTTP transport.

Runs inside a Docker container alongside Ollama and Qdrant.
Exposes 9 memory tools via MCP over HTTP (Streamable HTTP transport).

Environment variables (all have sensible defaults for Docker Compose):
    MEM0_LLM_MODEL          default: qwen2.5:7b
    MEM0_EMBED_MODEL        default: nomic-embed-text
    MEM0_OLLAMA_URL          default: http://host.docker.internal:11434
    MEM0_QDRANT_HOST         default: qdrant
    MEM0_QDRANT_PORT         default: 6333
    MEM0_LLM_TEMPERATURE     default: 0.1
    MEM0_LLM_MAX_TOKENS      default: 2000
    MEM0_DEFAULT_USER_ID     default: dev
    MCP_HOST                 default: 0.0.0.0
    MCP_PORT                 default: 8765
"""

import asyncio
import json
import os
import sys
from typing import Any

# ── Configuration ────────────────────────────────────────────────────────────

def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default

OLLAMA_URL = _env("MEM0_OLLAMA_URL", "http://host.docker.internal:11434")
QDRANT_HOST = _env("MEM0_QDRANT_HOST", "qdrant")
QDRANT_PORT = _env_int("MEM0_QDRANT_PORT", 6333)
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": _env("MEM0_LLM_MODEL", "qwen2.5:7b"),
            "ollama_base_url": OLLAMA_URL,
            "temperature": _env_float("MEM0_LLM_TEMPERATURE", 0.1),
            "max_tokens": _env_int("MEM0_LLM_MAX_TOKENS", 2000),
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": _env("MEM0_EMBED_MODEL", "nomic-embed-text"),
            "ollama_base_url": OLLAMA_URL,
            "embedding_dims": _env_int("MEM0_EMBED_DIMS", 768),
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": _env("MEM0_QDRANT_HOST", "qdrant"),
            "port": _env_int("MEM0_QDRANT_PORT", 6333),
            "embedding_model_dims": _env_int("MEM0_EMBED_DIMS", 768),
        },
    },
}

DEFAULT_USER_ID = _env("MEM0_DEFAULT_USER_ID", "dev")
MCP_HOST = _env("MCP_HOST", "0.0.0.0")
MCP_PORT = _env_int("MCP_PORT", 8765)

# ── Lazy initialization ──────────────────────────────────────────────────────

_memory: Any = None

def get_memory() -> Any:
    global _memory
    if _memory is None:
        from mem0 import Memory
        _memory = Memory.from_config(CONFIG)
    return _memory


def _chunk_text(text: str, max_chars: int = 3000, context_header: str = "") -> list[str]:
    """Split text into chunks at paragraph boundaries, each under max_chars.
    If context_header is provided, it's prepended to each chunk so the LLM
    knows what document the chunk belongs to."""
    header_len = len(context_header) + 2 if context_header else 0  # +2 for \n\n
    if len(text) + header_len <= max_chars:
        return [text] if not context_header else [context_header + "\n\n" + text]

    header_len = len(context_header) + 2 if context_header else 0  # +2 for \n\n
    effective_max = max_chars - header_len

    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > effective_max:
            if current:
                chunks.append(current)
            if len(para) > effective_max:
                for line in para.split("\n"):
                    if len(current) + len(line) + 1 > effective_max:
                        if current:
                            chunks.append(current)
                        current = line
                    else:
                        current = (current + "\n" + line) if current else line
            else:
                current = para
        else:
            current = (current + "\n\n" + para) if current else para
    if current:
        chunks.append(current)

    # Prepend context header to each chunk
    if context_header:
        chunks = [context_header + "\n\n" + chunk for chunk in chunks]

    return chunks


# Model-specific chunk sizes (chars). Larger models handle longer context for JSON extraction.
# mem0's extraction prompt is ~2K chars, so chunk + prompt must fit in the model's effective context.
_CHUNK_SIZES = {
    "qwen2.5:3b": 1500,
    "qwen2.5:7b": 3000,
    "qwen3.5:9b": 4000,
    "gemma4:12b": 5000,
    "qwen3.5:27b": 8000,
}

def _get_chunk_size() -> int:
    """Get the chunk size (in chars) for the configured LLM model."""
    model = CONFIG["llm"]["config"]["model"]
    # Exact match first
    if model in _CHUNK_SIZES:
        return _CHUNK_SIZES[model]
    # Prefix match: find the most specific (longest) matching key
    best_match = None
    best_len = 0
    for key, size in _CHUNK_SIZES.items():
        if model.startswith(key) and len(key) > best_len:
            best_match = size
            best_len = len(key)
    return best_match if best_match is not None else 3000

def init_memory() -> Any:
    """Eagerly initialize Memory — creates Qdrant collections on startup.
    Also checks for and fixes dimension mismatches before init."""
    global _memory
    if _memory is None:
        import urllib.request
        import urllib.error
        import json as _json

        embed_dims = CONFIG["vector_store"]["config"]["embedding_model_dims"]
        qdrant_host = CONFIG["vector_store"]["config"]["host"]
        qdrant_port = CONFIG["vector_store"]["config"]["port"]
        qdrant_base = f"http://{qdrant_host}:{qdrant_port}"

        # Check existing collections for dimension mismatches
        # If a collection exists with wrong dims, delete it so mem0 recreates correctly
        for col_name in ("mem0", "mem0migrations"):
            try:
                resp = urllib.request.urlopen(f"{qdrant_base}/collections/{col_name}", timeout=5)
                data = _json.loads(resp.read())
                vectors = data.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
                existing_dim = None
                if isinstance(vectors, dict):
                    # Named vectors: {name: {size: N, ...}}
                    for key, val in vectors.items():
                        if isinstance(val, dict) and "size" in val:
                            existing_dim = val["size"]
                            break
                    # Flat config: {size: N, distance: ...}
                    if existing_dim is None and "size" in vectors:
                        existing_dim = vectors["size"]
                if existing_dim is not None and existing_dim != embed_dims:
                    print(f"[mcp_server] ⚠️  Collection '{col_name}' has {existing_dim} dims, "
                          f"embedder needs {embed_dims}. Deleting...", flush=True)
                    req = urllib.request.Request(
                        f"{qdrant_base}/collections/{col_name}",
                        method="DELETE",
                    )
                    urllib.request.urlopen(req, timeout=10)
                    print(f"[mcp_server] ✅ Deleted stale collection '{col_name}'", flush=True)
                elif existing_dim is not None:
                    print(f"[mcp_server] ✅ Collection '{col_name}' dims OK ({existing_dim})", flush=True)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    pass  # Collection doesn't exist yet — that's fine
            except Exception:
                pass  # Qdrant not ready yet — mem0 will handle it

        from mem0 import Memory
        print("[mcp_server] Initializing mem0 (creates Qdrant collections)...", flush=True)
        print(f"[mcp_server] CONFIG: {json.dumps(CONFIG, default=str, indent=2)}", flush=True)
        print(f"[mcp_server] MEM0_TELEMETRY env: {os.environ.get('MEM0_TELEMETRY', 'NOT SET')}", flush=True)
        _memory = Memory.from_config(CONFIG)
        print("[mcp_server] ✅ mem0 initialized", flush=True)

        # Verify the collection was created with correct dims
        # If not, delete it and recreate by calling create_col directly
        try:
            resp = urllib.request.urlopen(f"{qdrant_base}/collections/mem0", timeout=5)
            data = _json.loads(resp.read())
            vectors = data.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
            actual_dim = None
            if isinstance(vectors, dict):
                for key, val in vectors.items():
                    if isinstance(val, dict) and "size" in val:
                        actual_dim = val["size"]
                        break
                if actual_dim is None and "size" in vectors:
                    actual_dim = vectors["size"]
            if actual_dim is not None:
                if actual_dim == embed_dims:
                    print(f"[mcp_server] ✅ Qdrant collection 'mem0' has correct dims: {actual_dim}", flush=True)
                else:
                    print(f"[mcp_server] ❌ Qdrant collection 'mem0' has {actual_dim} dims, "
                          f"expected {embed_dims}. Deleting and recreating...", flush=True)
                    # Delete the wrong collection
                    req = urllib.request.Request(f"{qdrant_base}/collections/mem0", method="DELETE")
                    urllib.request.urlopen(req, timeout=10)
                    # Force mem0 to recreate it with correct dims
                    _memory.vector_store.create_col(embed_dims, False)
                    # Verify again
                    resp2 = urllib.request.urlopen(f"{qdrant_base}/collections/mem0", timeout=5)
                    data2 = _json.loads(resp2.read())
                    vectors2 = data2.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
                    new_dim = None
                    if isinstance(vectors2, dict):
                        for key, val in vectors2.items():
                            if isinstance(val, dict) and "size" in val:
                                new_dim = val["size"]
                                break
                    if new_dim == embed_dims:
                        print(f"[mcp_server] ✅ Fixed! Collection 'mem0' now has {new_dim} dims", flush=True)
                    else:
                        print(f"[mcp_server] ❌ Still wrong: {new_dim} dims. Writes will fail!", flush=True)
        except Exception as e:
            print(f"[mcp_server] Could not verify Qdrant collection: {e}", flush=True)

    return _memory

# ── Tool definitions ────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "add_memory",
        "description": "Save text or conversation history to persistent memory. "
                       "Use this to remember facts, decisions, user preferences, "
                       "code patterns, or anything worth recalling later.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The text or conversation to remember."},
                "user_id": {"type": "string", "description": "User identifier", "default": DEFAULT_USER_ID},
                "metadata": {"type": "object", "description": "Optional metadata to attach"},
                "infer": {"type": "boolean", "default": True,
                          "description": "If true (default), use LLM to extract facts. If false, store raw text directly."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "search_memories",
        "description": "Semantic search across stored memories. Returns the most relevant memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "user_id": {"type": "string", "default": DEFAULT_USER_ID},
                "limit": {"type": "integer", "default": 10, "description": "Max results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_memories",
        "description": "List memories with optional filters and pagination.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "default": DEFAULT_USER_ID},
                "limit": {"type": "integer", "default": 50},
                "page": {"type": "integer", "default": 1},
            },
        },
    },
    {
        "name": "get_memory",
        "description": "Retrieve a single memory by its ID.",
        "inputSchema": {
            "type": "object",
            "properties": {"memory_id": {"type": "string"}},
            "required": ["memory_id"],
        },
    },
    {
        "name": "update_memory",
        "description": "Overwrite the text of an existing memory by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string"},
                "content": {"type": "string", "description": "New text for the memory"},
            },
            "required": ["memory_id", "content"],
        },
    },
    {
        "name": "delete_memory",
        "description": "Delete a single memory by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {"memory_id": {"type": "string"}},
            "required": ["memory_id"],
        },
    },
    {
        "name": "delete_all_memories",
        "description": "Delete ALL memories for a given user. Use with caution.",
        "inputSchema": {
            "type": "object",
            "properties": {"user_id": {"type": "string", "default": DEFAULT_USER_ID}},
        },
    },
    {
        "name": "list_entities",
        "description": "List distinct user_id/agent_id/app_id values stored in memory.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "delete_entities",
        "description": "Delete a user/agent/app entity and all its memories.",
        "inputSchema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
]

# ── Error handling ──────────────────────────────────────────────────────────

class Mem0Error(Exception):
    """Base error for all mem0-local tool failures. Produces user-friendly
    messages with actionable fixes, not raw tracebacks."""
    def __init__(self, message: str, *, tool: str = "", detail: str = "", fix: str = ""):
        self.tool = tool
        self.detail = detail
        self.fix = fix
        # Build a clean multi-line message
        parts = [message]
        if detail:
            parts.append(f"  Reason: {detail}")
        if fix:
            parts.append(f"  Fix: {fix}")
        super().__init__("\n".join(parts))


class LLMExtractionError(Mem0Error):
    """LLM failed to extract facts — the silent failure case."""
    pass


class QdrantError(Mem0Error):
    """Qdrant operation failed (dimension mismatch, connection, etc.)."""
    pass


class OllamaError(Mem0Error):
    """Ollama is unreachable or returned an error."""
    pass


def _check_ollama():
    """Verify Ollama is reachable. Raises OllamaError if not."""
    import urllib.request
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
    except Exception as e:
        raise OllamaError(
            "Cannot reach Ollama. Memory operations require a running Ollama instance.",
            detail=f"{OLLAMA_URL}/api/tags — {e}",
            fix="Start Ollama: 'ollama serve' (or run ./setup.sh)",
        )


def _check_qdrant():
    """Verify Qdrant is reachable. Raises QdrantError if not."""
    import urllib.request
    try:
        urllib.request.urlopen(QDRANT_URL, timeout=5)
    except Exception as e:
        raise QdrantError(
            "Cannot reach Qdrant. Memory storage requires a running Qdrant instance.",
            detail=f"{QDRANT_URL} — {e}",
            fix="Start Qdrant: 'docker compose up -d qdrant'",
        )


def _diagnose_llm_failure(m, model_name: str) -> str:
    """Run a diagnostic LLM call to find out WHY extraction returned empty.
    Returns a human-readable reason string."""
    try:
        test = m.llm.generate_response(
            messages=[
                {"role": "system", "content": "Output valid JSON only."},
                {"role": "user", "content": 'Return: {"test": true}'},
            ],
            response_format={"type": "json_object"},
        )
        if not test or not test.strip():
            return "LLM returned an empty response — the model may be overloaded, out of memory, or too small"
        if "test" not in test and "true" not in test:
            return f"LLM returned non-JSON output instead of structured data: {test[:300]}"
        return "LLM responded correctly to a simple test, but failed on the actual extraction prompt (input may be too complex or too long for this model)"
    except Exception as e:
        return f"LLM diagnostic call failed: {e}"


class LLMResponseLogger:
    """Wraps an LLM instance and logs every response to stderr.
    Used to capture what the LLM actually returns during extraction,
    so when mem0 silently returns empty, we can see why in the logs."""

    def __init__(self, llm):
        self._llm = llm
        self.last_response = None

    def generate_response(self, *args, **kwargs):
        response = self._llm.generate_response(*args, **kwargs)
        self.last_response = response
        # Log a truncated version for debugging
        if response:
            preview = response[:200].replace("\n", "\\n")
            print(f"[llm] Response ({len(response)} chars): {preview}...", file=sys.stderr)
        else:
            print(f"[llm] Response was empty/null", file=sys.stderr)
        return response

    def __getattr__(self, name):
        return getattr(self._llm, name)


# ── Tool execution ──────────────────────────────────────────────────────────

def execute_tool(name: str, arguments: dict) -> dict:
    m = get_memory()
    uid = arguments.get("user_id", DEFAULT_USER_ID)

    # ── add_memory ──────────────────────────────────────────────────────────
    if name == "add_memory":
        content = arguments["content"]
        if not content or not content.strip():
            raise Mem0Error("Cannot save empty content.", tool=name,
                            fix="Provide text or a conversation to remember.")

        infer = arguments.get("infer", True)
        metadata = arguments.get("metadata")
        model_name = CONFIG["llm"]["config"]["model"]
        MAX_CHUNK_CHARS = _get_chunk_size()
        all_results = []
        chunk_errors = []

        # Detect conversation messages (JSON array of {role, content} dicts)
        is_conversation = False
        try:
            parsed_json = json.loads(content)
            if isinstance(parsed_json, list) and all(isinstance(msg, dict) and "role" in msg for msg in parsed_json):
                is_conversation = True
        except (json.JSONDecodeError, TypeError):
            pass

        def _do_add(text, user_id, meta, use_infer):
            """Call m.add() with LLM response logging for diagnostics."""
            # Wrap the LLM to capture responses
            logger = LLMResponseLogger(m.llm)
            m.llm = logger

            try:
                result = m.add(text, user_id=user_id, metadata=meta, infer=use_infer)

                if use_infer:
                    r = result.get("results", []) if isinstance(result, dict) else result
                    if not r:
                        # Build diagnostic from captured LLM response
                        actual = logger.last_response
                        if actual is None:
                            reason = "LLM was never called — mem0 may have skipped extraction (input too short or unrecognized format)"
                        elif not actual or not actual.strip():
                            reason = "LLM returned an empty response"
                        elif "memory" not in actual:
                            reason = f"LLM response did not contain 'memory' key. Response (first 500 chars): {actual[:500]}"
                        else:
                            reason = f"LLM returned JSON with 'memory' key but it was empty or unparseable. Response (first 500 chars): {actual[:500]}"
                        chunk_errors.append(reason)

                return result

            except Exception as e:
                err_str = str(e)
                # Classify known errors
                if "Vector dimension" in err_str:
                    raise QdrantError(
                        "Vector dimension mismatch in Qdrant.",
                        tool=name, detail=err_str,
                        fix="Delete stale collection and restart: "
                            "'curl -X DELETE http://localhost:6333/collections/mem0 && "
                            "docker compose restart mcp-server'",
                    )
                elif "Collection" in err_str and "doesn't exist" in err_str:
                    raise QdrantError(
                        "Qdrant collection does not exist.",
                        tool=name, detail=err_str,
                        fix="Restart the MCP server so it creates the collection: "
                            "'docker compose restart mcp-server'",
                    )
                elif "Connection" in err_str or "refused" in err_str:
                    raise OllamaError(
                        "Connection to Ollama failed during memory extraction.",
                        tool=name, detail=err_str,
                        fix="Check Ollama is running: 'ollama serve'",
                    )
                chunk_errors.append(err_str)
                return {"results": [], "error": err_str}
            finally:
                # Restore original LLM
                m.llm = logger._llm

        if is_conversation or len(content) <= MAX_CHUNK_CHARS:
            # Single call — no chunking needed
            if is_conversation:
                result = _do_add(parsed_json, uid, metadata, infer)
            else:
                result = _do_add(content, uid, metadata, infer)
            all_results.append(result)
        else:
            # Chunk large text at paragraph boundaries with context header
            header_parts = []
            if metadata and metadata.get("file"):
                header_parts.append(f"[Document: {metadata['file']}]")
            if metadata and metadata.get("source"):
                header_parts.append(f"[Source: {metadata['source']}]")
            context_header = " ".join(header_parts) if header_parts else ""
            chunks = _chunk_text(content, MAX_CHUNK_CHARS, context_header=context_header)
            for ci, chunk in enumerate(chunks):
                chunk_meta = dict(metadata) if metadata else {}
                chunk_meta["chunk"] = f"{ci+1}/{len(chunks)}"
                result = _do_add(chunk, uid, chunk_meta, infer)
                all_results.append(result)

        # Merge results
        merged = {"results": [], "chunks": len(all_results)}
        for r in all_results:
            if isinstance(r, dict) and "results" in r:
                merged["results"].extend(r["results"])
            elif isinstance(r, list):
                merged["results"].extend(r)
            elif r:
                merged["results"].append(r)

        # Check for silent failures (only when LLM extraction was expected)
        if infer:
            total = len(merged.get("results", []))
            if total == 0:
                reason = "; ".join(chunk_errors) if chunk_errors else "No facts were extracted from the input"
                raise LLMExtractionError(
                    f"Memory could not be saved. The LLM ({model_name}) did not extract any facts.",
                    tool=name,
                    detail=f"{reason}\n  Input: {len(content)} chars, {len(all_results)} chunk(s) "
                           f"(chunk size: {MAX_CHUNK_CHARS} chars for {model_name})",
                    fix=f"(1) Use a larger model: 'ollama pull qwen3.5:9b' then update .env\n"
                        f"    (2) Check Ollama: 'ollama serve' (runs on host, not in Docker)\n"
                        f"    (3) Store without extraction: pass infer=false",
                )
        return merged

    # ── search_memories ─────────────────────────────────────────────────────
    elif name == "search_memories":
        query = arguments.get("query", "")
        if not query or not query.strip():
            raise Mem0Error("Search query cannot be empty.", tool=name)
        try:
            return m.search(query, filters={"user_id": uid},
                            limit=arguments.get("limit", 10))
        except Exception as e:
            err_str = str(e)
            if "dimension" in err_str.lower():
                raise QdrantError("Vector dimension mismatch during search.",
                                  tool=name, detail=err_str,
                                  fix="Restart MCP server: 'docker compose restart mcp-server'")
            raise Mem0Error("Search failed.", tool=name, detail=err_str)

    # ── get_memories ─────────────────────────────────────────────────────────
    elif name == "get_memories":
        try:
            return m.get_all(filters={"user_id": uid},
                             limit=arguments.get("limit", 50),
                             page=arguments.get("page", 1))
        except Exception as e:
            raise Mem0Error("Failed to list memories.", tool=name, detail=str(e))

    # ── get_memory ──────────────────────────────────────────────────────────
    elif name == "get_memory":
        memory_id = arguments.get("memory_id", "")
        if not memory_id:
            raise Mem0Error("memory_id is required.", tool=name)
        try:
            return m.get(memory_id)
        except Exception as e:
            raise Mem0Error(f"Failed to retrieve memory '{memory_id}'.",
                            tool=name, detail=str(e))

    # ── update_memory ───────────────────────────────────────────────────────
    elif name == "update_memory":
        memory_id = arguments.get("memory_id", "")
        new_content = arguments.get("content", "")
        if not memory_id:
            raise Mem0Error("memory_id is required.", tool=name)
        if not new_content or not new_content.strip():
            raise Mem0Error("New content cannot be empty.", tool=name)
        try:
            return m.update(memory_id, new_content)
        except Exception as e:
            raise Mem0Error(f"Failed to update memory '{memory_id}'.",
                            tool=name, detail=str(e))

    # ── delete_memory ───────────────────────────────────────────────────────
    elif name == "delete_memory":
        memory_id = arguments.get("memory_id", "")
        if not memory_id:
            raise Mem0Error("memory_id is required.", tool=name)
        try:
            m.delete(memory_id)
            return {"status": "deleted", "memory_id": memory_id}
        except Exception as e:
            raise Mem0Error(f"Failed to delete memory '{memory_id}'.",
                            tool=name, detail=str(e))

    # ── delete_all_memories ─────────────────────────────────────────────────
    elif name == "delete_all_memories":
        try:
            m.delete_all(user_id=uid)
            return {"status": "deleted_all", "user_id": uid}
        except Exception as e:
            raise Mem0Error(f"Failed to delete all memories for user '{uid}'.",
                            tool=name, detail=str(e))

    # ── list_entities ───────────────────────────────────────────────────────
    elif name == "list_entities":
        # mem0 2.0.11 requires filters in get_all()
        # We can't list ALL entities without a filter, so we use the default user_id
        # and also try the known user_id patterns from metadata
        entities = set()
        # Try the default user_id first
        for filter_val in [uid]:
            try:
                all_mems = m.get_all(filters={"user_id": filter_val}, limit=500)
                results = all_mems if isinstance(all_mems, list) else all_mems.get("results", [])
                for mem in results:
                    for key in ("user_id", "agent_id", "app_id", "run_id"):
                        if key in (mem or {}):
                            entities.add(mem[key])
            except Exception as e:
                print(f"[list_entities] Warning: {e}", file=sys.stderr)
        return {"entities": sorted(entities)}

    # ── delete_entities ─────────────────────────────────────────────────────
    elif name == "delete_entities":
        target_user = arguments.get("user_id", "")
        if not target_user:
            raise Mem0Error("user_id is required.", tool=name)
        try:
            m.delete_all(user_id=target_user)
            return {"status": "entity_deleted", "user_id": target_user}
        except Exception as e:
            raise Mem0Error(f"Failed to delete entity '{target_user}'.",
                            tool=name, detail=str(e))

    else:
        raise Mem0Error(f"Unknown tool: {name}", tool=name)


# ── MCP HTTP server (manual JSON-RPC over HTTP) ─────────────────────────────

async def http_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle a single HTTP connection. Each connection runs in its own
    asyncio task so the health endpoint stays responsive while LLM calls
    are in progress."""
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=10)
        if not request_line:
            writer.close()
            await writer.wait_closed()
            return

        parts = request_line.decode().strip().split()
        if len(parts) < 2:
            writer.close()
            await writer.wait_closed()
            return

        method, path = parts[0], parts[1]

        # Read headers
        headers = {}
        while True:
            line = await reader.readline()
            if line == b"\r\n" or line == b"\n" or not line:
                break
            try:
                key, val = line.decode().strip().split(":", 1)
                headers[key.lower()] = val.strip()
            except ValueError:
                pass

        # Read body if present
        body = b""
        content_length = int(headers.get("content-length", "0"))
        if content_length > 0:
            body = await asyncio.wait_for(reader.readexactly(content_length), timeout=30)

        # Health check — respond immediately, even if LLM calls are in progress
        if method == "GET" and path == "/health":
            response = json.dumps({
                "status": "ok",
                "server": "mem0-local",
                "tools": len(TOOL_DEFINITIONS),
                "config": {
                    "llm": CONFIG["llm"]["config"]["model"],
                    "embedder": CONFIG["embedder"]["config"]["model"],
                    "vector_store": "qdrant",
                },
            }).encode()
            header = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(response)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode()
            writer.write(header + response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return

        # MCP endpoint
        if method == "POST" and path == "/mcp":
            try:
                request = json.loads(body)
            except json.JSONDecodeError:
                response = json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }).encode()
            else:
                # Run MCP request handler in a thread so it doesn't block the event loop
                # (m.add() does synchronous LLM calls that can take 30+ seconds)
                response_obj = await asyncio.to_thread(handle_mcp_request_sync, request)
                if response_obj is None:
                    # Notification — no response body
                    response = b""
                    header = (
                        f"HTTP/1.1 202 Accepted\r\n"
                        f"Content-Length: 0\r\n"
                        f"Connection: close\r\n\r\n"
                    ).encode()
                    writer.write(header)
                    await writer.drain()
                    writer.close()
                    await writer.wait_closed()
                    return
                response = json.dumps(response_obj).encode()

            header = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(response)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode()
            writer.write(header + response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return

        # 404
        response = b'{"error": "Not found"}'
        header = (
            f"HTTP/1.1 404 Not Found\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(response)}\r\n"
            f"Connection: close\r\n\r\n"
        ).encode()
        writer.write(header + response)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    except asyncio.TimeoutError:
        # Client didn't send data in time — close connection
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
    except Exception as e:
        print(f"Handler error: {e}", file=sys.stderr)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def handle_mcp_request_sync(request: dict) -> dict | None:
    """Synchronous MCP request handler — called from a worker thread
    so long LLM calls don't block the async event loop."""
    return _handle_mcp_request(request)


def _handle_mcp_request(request: dict) -> dict | None:
    """Sync version of the MCP request handler. Runs in a thread so it
    doesn't block the event loop during long LLM calls."""
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mem0-local", "version": "1.0.0"},
            },
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOL_DEFINITIONS},
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        try:
            result = execute_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, default=str, indent=2),
                    }],
                },
            }
        except Mem0Error as e:
            print(f"[tool:{tool_name}] {e}", file=sys.stderr)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": str(e),
                    }],
                    "isError": True,
                },
            }
        except Exception as e:
            import traceback
            print(f"[tool:{tool_name}] UNEXPECTED: {e}\n{traceback.format_exc()}", file=sys.stderr)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": f"Unexpected error in {tool_name}: {e}\n\n"
                                f"This is a bug. Check server logs: docker compose logs mcp-server",
                    }],
                    "isError": True,
                },
            }

    elif method == "notifications/initialized":
        return None

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }


async def main():
    # Eagerly initialize mem0 — creates Qdrant collections before accepting requests
    # This prevents "Collection doesn't exist" errors on the first tool call
    init_memory()

    server = await asyncio.start_server(http_handler, MCP_HOST, MCP_PORT)
    print(f"mem0-local MCP server listening on {MCP_HOST}:{MCP_PORT}", flush=True)
    print(f"  LLM:      {CONFIG['llm']['config']['model']} @ {OLLAMA_URL}", flush=True)
    print(f"  Embedder: {CONFIG['embedder']['config']['model']} @ {OLLAMA_URL}", flush=True)
    print(f"  Vector:   Qdrant @ {QDRANT_URL}", flush=True)
    print(f"  Tools:    {len(TOOL_DEFINITIONS)}", flush=True)
    print(f"  Health:   GET  http://{MCP_HOST}:{MCP_PORT}/health", flush=True)
    print(f"  MCP:      POST http://{MCP_HOST}:{MCP_PORT}/mcp", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())