#!/usr/bin/env python3
"""
mem0-local MCP server — HTTP transport.

Runs inside a Docker container alongside Ollama and Qdrant.
Exposes 12 memory tools via MCP over HTTP (Streamable HTTP transport).

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
import csv
import io
import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error
from typing import Any, Tuple, List, Optional

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


# ── LLM setup ───────────────────────────────────────────────────────────────
# We subclass mem0's OllamaLLM to pass num_ctx (context window size) to Ollama.
# Without this, Ollama defaults to 2048-4096 context and truncates the
# ~8000-token extraction prompt, causing silent extraction failures.
# Configurable via MEM0_LLM_CONTEXT_LENGTH env var (default: 32768).

_LLM_MODEL = _env("MEM0_LLM_MODEL", "qwen2.5:7b")
_LLM_MAX_TOKENS = _env_int("MEM0_LLM_MAX_TOKENS", 4000)
_LLM_NUM_CTX = _env_int("MEM0_LLM_CONTEXT_LENGTH", 32768)


class ContextAwareOllamaLLM:
    """Wrapper around mem0's OllamaLLM that:
    1. Injects num_ctx into the Ollama options dict (mem0 doesn't expose it)
    2. Logs LLM responses to stderr for debugging extraction failures

    Without num_ctx, Ollama defaults to 2048-4096 context and truncates the
    ~8000-token extraction prompt, causing silent extraction failures.
    """

    def __init__(self, original_llm):
        self._llm = original_llm
        self.last_response = None

    def generate_response(self, *args, **kwargs):
        original_chat = self._llm.client.chat

        def _chat_with_ctx(**params):
            if "options" not in params:
                params["options"] = {}
            params["options"]["num_ctx"] = _LLM_NUM_CTX
            # Disable thinking mode — thinking models (qwen3.5) produce
            # reasoning tokens before the JSON, which consumes the output
            # budget and breaks mem0's JSON parser.
            params["think"] = False
            return original_chat(**params)

        self._llm.client.chat = _chat_with_ctx
        try:
            response = self._llm.generate_response(*args, **kwargs)
        finally:
            self._llm.client.chat = original_chat

        # Log the response for debugging
        self.last_response = response
        if response:
            preview = response[:200].replace("\n", "\\n")
            print(f"[llm] Response ({len(response)} chars): {preview}...", file=sys.stderr)
        else:
            print(f"[llm] Response was empty/null", file=sys.stderr)

        return response

    def __getattr__(self, name):
        return getattr(self._llm, name)

_DEFAULT_CUSTOM_INSTRUCTIONS = (
    "You are extracting memories for a software engineering knowledge base. "
    "Focus ONLY on technical facts relevant to software development:\n"
    "- Architecture decisions and their rationale (e.g., 'Chose PostgreSQL for ACID compliance requirements')\n"
    "- Technology choices and trade-offs (e.g., 'Using Redis for session caching, TTL 30min')\n"
    "- API contracts and data models (e.g., 'Auth API returns JWT with 24h expiry')\n"
    "- Infrastructure and deployment details (e.g., 'Deploys to AWS ECS via GitHub Actions')\n"
    "- Code patterns and conventions (e.g., 'All API endpoints use snake_case, versioned with /v1/ prefix')\n"
    "- Error patterns and fixes (e.g., 'Memory leak in UserService caused by unclosed DB connections')\n"
    "- Performance characteristics (e.g., 'Search latency under 10ms for up to 1M vectors')\n"
    "- Project structure and dependencies (e.g., 'Backend is Node.js, frontend is React 18 with TypeScript')\n\n"
    "DO NOT extract:\n"
    "- Personal preferences unrelated to engineering (food, movies, hobbies)\n"
    "- Greetings, conversational filler, or acknowledgments\n"
    "- Information from the few-shot examples in the system prompt (those are illustrative, not real)\n"
    "- Meta-information about the conversation itself\n\n"
    "Each memory should be a self-contained technical fact, 10-40 words, "
    "understandable without context from the original conversation."
)

CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": _LLM_MODEL,
            "ollama_base_url": OLLAMA_URL,
            "temperature": _env_float("MEM0_LLM_TEMPERATURE", 0.1),
            "max_tokens": _LLM_MAX_TOKENS,
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
    "custom_instructions": _env("MEM0_CUSTOM_INSTRUCTIONS", _DEFAULT_CUSTOM_INSTRUCTIONS),
}

DEFAULT_USER_ID = _env("MEM0_DEFAULT_USER_ID", "dev")
MCP_HOST = _env("MCP_HOST", "0.0.0.0")
MCP_PORT = _env_int("MCP_PORT", 8765)

# ── Lazy initialization ──────────────────────────────────────────────────────

# Initialization lifecycle for the health endpoint.
# "starting" → "initializing" → "ready" or "error"
_init_status = "starting"
_init_error: Optional[str] = None

# Cache for the LLM model status ping (Feature 5).
# Updated by _ping_llm_model() at most every 30 seconds.
_LLM_PING_CACHE_TTL = 30.0
_llm_ping_cache: dict = {
    "status": None,        # "loaded" | "unloaded" | "unreachable"
    "checked_at": 0.0,     # monotonic timestamp of last check
}
_llm_ping_lock = threading.Lock()

_memory: Any = None

def get_memory() -> Any:
    """Get the mem0 Memory instance. If not initialized, delegate to init_memory()
    which handles dimension checks, collection creation, and verification."""
    if _memory is None:
        init_memory()
    return _memory

def _chunk_text(text: str, max_chars: int = 3000, context_header: str = "") -> List[str]:
    """Split text into chunks at paragraph boundaries, each under max_chars.
    If context_header is provided, it's prepended to each chunk so the LLM
    knows what document the chunk belongs to.

    Splits at \\n\\n (paragraphs) → \\n (lines) → word boundaries, so even
    text with no structure (single long line) is correctly chunked."""
    header_len = len(context_header) + 2 if context_header else 0  # +2 for \n\n
    if len(text) + header_len <= max_chars:
        return [text] if not context_header else [context_header + "\n\n" + text]

    effective_max = max_chars - header_len

    def _hard_split(s: str, limit: int) -> List[str]:
        """Split a string with no newlines at word boundaries, each under limit.
        If a single word exceeds limit, splits at character boundaries."""
        if len(s) <= limit:
            return [s]
        words = s.split(" ")
        chunks = []
        current = ""
        for word in words:
            # If the word itself exceeds the limit, hard-split it at char boundaries
            if len(word) > limit:
                if current:
                    chunks.append(current)
                    current = ""
                for i in range(0, len(word), limit):
                    chunks.append(word[i:i + limit])
            elif len(current) + len(word) + 1 > limit and current:
                chunks.append(current)
                current = word
            else:
                current = (current + " " + word) if current else word
        if current:
            chunks.append(current)
        return chunks

    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > effective_max:
            if current:
                chunks.append(current)
            if len(para) > effective_max:
                # Split paragraph by lines first
                for line in para.split("\n"):
                    if len(line) > effective_max:
                        # Line itself is too long — hard-split at word boundaries
                        for piece in _hard_split(line, effective_max):
                            chunks.append(piece)
                    else:
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


# Chunk size based on the configured context window.
# The extraction prompt is ~6000 tokens. chunk + prompt must fit in num_ctx.
# chunk_chars = (num_ctx - 6000) * 4 chars/token, clamped to 1000–16000.
_CHUNK_SIZES = {
    "qwen2.5:3b": 1500,
    "qwen2.5:7b": 3000,
    "qwen3.5:9b": 4000,
    "gemma4:12b": 5000,
    "qwen3.5:27b": 8000,
}


def _get_chunk_size() -> int:
    """Get the chunk size (in chars) for the configured LLM model.

    Computed from MEM0_LLM_CONTEXT_LENGTH: (context - 6000) * 4, clamped.
    Falls back to the hardcoded _CHUNK_SIZES table for the model.
    """
    # Compute from context length: reserve 6000 tokens for prompt + output
    chunk_tokens = max(1000, _LLM_NUM_CTX - 6000)
    computed = max(1000, min(chunk_tokens * 4, 16000))
    return computed


def _detect_conversation(content: str) -> Tuple[bool, Optional[List]]:
    """Detect if content is a JSON array of {role, content} messages.

    Returns (is_conversation, parsed_messages) — if not a conversation,
    returns (False, None). This is extracted from add_memory so it can
    be tested independently of the LLM/Qdrant pipeline.
    """
    if not content or not isinstance(content, str):
        return False, None
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and len(parsed) > 0 and \
                all(isinstance(msg, dict) and "role" in msg for msg in parsed):
            return True, parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return False, None


def _create_collection_rest(qdrant_base: str, collection_name: str, dims: int):
    """Create a Qdrant collection via the REST API as a fallback when
    mem0's create_col() fails (e.g. sparse_vectors_config unsupported)."""
    payload = json.dumps({
        "vectors": {"size": dims, "distance": "Cosine"},
    }).encode()
    req = urllib.request.Request(
        f"{qdrant_base}/collections/{collection_name}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    urllib.request.urlopen(req, timeout=10)


def _check_and_fix_collection_dims(qdrant_base: str, embed_dims: int) -> List[str]:
    """Check Qdrant collections for dimension mismatches and delete stale ones.

    Returns a list of action strings for logging (e.g., "mem0 OK (768)",
    "mem0 Deleted (was 1024)"). If Qdrant is not reachable, returns empty
    list — this is not an error, mem0 will handle collection creation.

    Extracted from init_memory() so it can be tested with just Qdrant,
    without needing to initialize the full mem0 stack.
    """
    actions: List[str] = []
    for col_name in ("mem0", "mem0migrations"):
        try:
            resp = urllib.request.urlopen(
                f"{qdrant_base}/collections/{col_name}", timeout=5)
            data = json.loads(resp.read())
            vectors = (
                data.get("result", {})
                .get("config", {})
                .get("params", {})
                .get("vectors", {})
            )
            existing_dim = None
            if isinstance(vectors, dict):
                for key, val in vectors.items():
                    if isinstance(val, dict) and "size" in val:
                        existing_dim = val["size"]
                        break
                if existing_dim is None and "size" in vectors:
                    existing_dim = vectors["size"]
            if existing_dim is not None and existing_dim != embed_dims:
                print(f"[mcp_server] ⚠️  Collection '{col_name}' has "
                      f"{existing_dim} dims, embedder needs {embed_dims}. "
                      f"Deleting...", flush=True)
                req = urllib.request.Request(
                    f"{qdrant_base}/collections/{col_name}",
                    method="DELETE",
                )
                urllib.request.urlopen(req, timeout=10)
                print(f"[mcp_server] ✅ Deleted stale collection "
                      f"'{col_name}'", flush=True)
                actions.append(f"{col_name} Deleted (was {existing_dim})")
            elif existing_dim is not None:
                print(f"[mcp_server] ✅ Collection '{col_name}' dims OK "
                      f"({existing_dim})", flush=True)
                actions.append(f"{col_name} OK ({existing_dim})")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                pass  # Collection doesn't exist yet — that's fine
        except Exception:
            pass  # Qdrant not ready yet — mem0 will handle it
    return actions

def init_memory() -> Any:
    """Eagerly initialize Memory — creates Qdrant collections on startup.
    Also checks for and fixes dimension mismatches before init.

    Updates the module-level _init_status: "initializing" → "ready" or "error".
    Safe to call from a background thread."""
    global _memory, _init_status, _init_error
    if _memory is None:
        _init_status = "initializing"
        try:
            embed_dims = CONFIG["vector_store"]["config"]["embedding_model_dims"]
            qdrant_host = CONFIG["vector_store"]["config"]["host"]
            qdrant_port = CONFIG["vector_store"]["config"]["port"]
            qdrant_base = f"http://{qdrant_host}:{qdrant_port}"

            # Check existing collections for dimension mismatches (extracted for testability)
            _check_and_fix_collection_dims(qdrant_base, embed_dims)

            from mem0 import Memory
            print("[mcp_server] Initializing mem0 (creates Qdrant collections)...", flush=True)
            print(f"[mcp_server] CONFIG: {json.dumps(CONFIG, default=str, indent=2)}", flush=True)
            print(f"[mcp_server] MEM0_TELEMETRY env: {os.environ.get('MEM0_TELEMETRY', 'NOT SET')}", flush=True)
            _memory = Memory.from_config(CONFIG)
            # Wrap the LLM to inject num_ctx into every Ollama call
            _memory.llm = ContextAwareOllamaLLM(_memory.llm)
            print("[mcp_server] ✅ mem0 initialized", flush=True)

            # Verify the collection was actually created. mem0 doesn't always
            # create it eagerly — if the collection doesn't exist, force-create it.
            try:
                resp = urllib.request.urlopen(
                    f"{qdrant_base}/collections/mem0", timeout=5)
                data = json.loads(resp.read())
                vectors = (
                    data.get("result", {})
                    .get("config", {})
                    .get("params", {})
                    .get("vectors", {})
                )
                actual_dim = None
                if isinstance(vectors, dict):
                    for key, val in vectors.items():
                        if isinstance(val, dict) and "size" in val:
                            actual_dim = val["size"]
                            break
                    if actual_dim is None and "size" in vectors:
                        actual_dim = vectors["size"]
                if actual_dim is not None and actual_dim == embed_dims:
                    print(f"[mcp_server] ✅ Qdrant collection 'mem0' has correct dims: {actual_dim}", flush=True)
                else:
                    print(f"[mcp_server] ⚠️  Collection 'mem0' missing or wrong dims "
                          f"(got {actual_dim}, expected {embed_dims}). Force-creating...", flush=True)
                    _memory.vector_store.create_col(embed_dims, False)
                    print(f"[mcp_server] ✅ Collection 'mem0' created with {embed_dims} dims", flush=True)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    # Collection doesn't exist — mem0 didn't create it. Force-create.
                    print("[mcp_server] ⚠️  Collection 'mem0' not found after init. Force-creating...", flush=True)
                    try:
                        _memory.vector_store.create_col(embed_dims, False)
                    except Exception as ce:
                        print(f"[mcp_server] create_col failed: {ce}. Trying direct REST API...", flush=True)
                        _create_collection_rest(qdrant_base, "mem0", embed_dims)
                    print(f"[mcp_server] ✅ Collection 'mem0' created with {embed_dims} dims", flush=True)
                else:
                    print(f"[mcp_server] Could not verify Qdrant collection: {e}", flush=True)
            except Exception as e:
                print(f"[mcp_server] Could not verify Qdrant collection: {e}", flush=True)

            _init_status = "ready"
            _init_error = None
        except Exception as e:
            _init_status = "error"
            _init_error = str(e)
            print(f"[mcp_server] ❌ init_memory failed: {e}", flush=True)
            # Re-raise so callers (e.g. selftest) see the failure
            raise

    return _memory

# ── Tool definitions ────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "add_memory",
        "description": "Save text or conversation history to persistent memory with LLM fact extraction. "
                       "Large content is automatically chunked based on the model's context window. "
                       "Use this to remember facts, decisions, user preferences, "
                       "code patterns, or anything worth recalling later.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The text or conversation to remember."},
                "user_id": {"type": "string", "description": "User identifier", "default": DEFAULT_USER_ID},
                "metadata": {"type": "object", "description": "Optional metadata to attach"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "add_raw_memory",
        "description": "Store text directly as a memory WITHOUT LLM fact extraction. "
                       "Use for: (1) bulk imports/backups, (2) agent-managed extraction — you extract "
                       "facts yourself and store each as a concise 10-30 word sentence (one fact per call, "
                       "include specific names/versions/ports, preserve rationale, use active voice). "
                       "Set user_id to the project/team name for scoping.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The text to store as-is. "
                    "For agent-managed extraction, store one concise fact per call."},
                "user_id": {"type": "string", "description": "User identifier", "default": DEFAULT_USER_ID},
                "metadata": {"type": "object", "description": "Optional metadata to attach"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "search_memories",
        "description": "Semantic search across stored memories. Returns the most relevant memories, "
                       "optionally filtered by relevance score.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "user_id": {"type": "string", "default": DEFAULT_USER_ID},
                "limit": {"type": "integer", "default": 10, "description": "Max results"},
                "min_score": {
                    "type": "number", "default": 0.0,
                    "description": "Minimum relevance score (0.0–1.0). Results below this are filtered out. "
                                   "Set to 0.0 (default) to include all results.",
                },
                "include_scores": {
                    "type": "boolean", "default": True,
                    "description": "If true (default), include a 'score' field in each result with "
                                   "the cosine similarity from Qdrant.",
                },
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
    {
        "name": "export_memories",
        "description": "Export all memories for a user as JSON or CSV. "
                       "Returns the exported data directly in the response text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "default": DEFAULT_USER_ID},
                "format": {
                    "type": "string", "enum": ["json", "csv"], "default": "json",
                    "description": "Output format: 'json' (default) or 'csv'",
                },
            },
        },
    },
    {
        "name": "import_memories",
        "description": "Import memories from a JSON array (as exported by export_memories). "
                       "Skips duplicates by checking if memory text already exists for the user.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "JSON string of memory array. Each item should have 'memory' (text) "
                                   "and optionally 'metadata', 'user_id'.",
                },
                "user_id": {"type": "string", "default": DEFAULT_USER_ID},
            },
            "required": ["data"],
        },
    },
    {
        "name": "prune_memories",
        "description": "Delete memories older than a specified number of days. "
                       "By default runs in dry-run mode (reports only, no deletion).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "default": DEFAULT_USER_ID},
                "older_than_days": {
                    "type": "integer", "default": 30,
                    "description": "Delete memories older than this many days",
                },
                "dry_run": {
                    "type": "boolean", "default": True,
                    "description": "If true (default), only report what would be deleted without deleting",
                },
                "min_score": {
                    "type": "number",
                    "description": "Optional: only prune memories with a relevance score below this threshold",
                },
            },
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
    try:
        urllib.request.urlopen(QDRANT_URL, timeout=5)
    except Exception as e:
        raise QdrantError(
            "Cannot reach Qdrant. Memory storage requires a running Qdrant instance.",
            detail=f"{QDRANT_URL} — {e}",
            fix="Start Qdrant: 'docker compose up -d qdrant'",
        )


def _ollama_reachable() -> bool:
    """Check if Ollama is reachable (non-raising). For health endpoint."""
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _qdrant_reachable() -> bool:
    """Check if Qdrant is reachable (non-raising). For health endpoint."""
    try:
        urllib.request.urlopen(QDRANT_URL, timeout=3)
        return True
    except Exception:
        return False


def _mem0_ready() -> bool:
    """Check if mem0 is initialized and ready (non-raising). For health endpoint."""
    return _memory is not None and _init_status == "ready"


def _build_health_response() -> dict:
    """Build the health response dict. Used by the /health HTTP endpoint
    and directly by tests. This is extracted so tests can check health
    without making HTTP requests."""
    ollama_ok = _ollama_reachable()
    qdrant_ok = _qdrant_reachable()
    mem0_ok = _mem0_ready()

    if _init_status == "ready" and ollama_ok and qdrant_ok and mem0_ok:
        status = "ok"
    elif _init_status in ("starting", "initializing"):
        status = "starting"
    else:
        status = "degraded"

    if ollama_ok:
        model_status = _ping_llm_model()
    else:
        model_status = "unreachable"

    return {
        "status": status,
        "init_status": _init_status,
        "components": {
            "ollama": ollama_ok,
            "qdrant": qdrant_ok,
            "mem0": mem0_ok,
        },
        "model_status": model_status,
        "server": "mem0-local",
        "tools": len(TOOL_DEFINITIONS),
        "config": {
            "llm": CONFIG["llm"]["config"]["model"],
            "embedder": CONFIG["embedder"]["config"]["model"],
            "vector_store": "qdrant",
        },
    }


def _ping_llm_model() -> str:
    """Lightweight LLM ping to check if the configured model is loaded and responsive.

    Sends a minimal request to Ollama /api/generate with a tiny prompt ("hi")
    and 1 token max. Cached for 30 seconds to avoid pinging on every health check.

    Returns:
        "loaded"     — model responded successfully
        "unloaded"   — model not found (needs ollama pull)
        "unreachable"— Ollama is down or request timed out
    """

    with _llm_ping_lock:
        now = time.monotonic()
        cached_status = _llm_ping_cache.get("status")
        cached_at = _llm_ping_cache.get("checked_at", 0.0)
        if cached_status is not None and (now - cached_at) < _LLM_PING_CACHE_TTL:
            return cached_status

        model = CONFIG["llm"]["config"]["model"]
        try:
            req_body = json.dumps({
                "model": model,
                "prompt": "hi",
                "stream": False,
                "options": {"num_predict": 1},
            }).encode()
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            status = "loaded"
        except urllib.error.HTTPError as e:
            # 404 = model not found
            if e.code == 404:
                status = "unloaded"
            else:
                # Other HTTP errors — treat as unreachable
                status = "unreachable"
        except Exception:
            status = "unreachable"

        _llm_ping_cache["status"] = status
        _llm_ping_cache["checked_at"] = now
        return status


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

        infer = True  # add_memory always uses LLM extraction
        metadata = arguments.get("metadata")
        model_name = CONFIG["llm"]["config"]["model"]
        MAX_CHUNK_CHARS = _get_chunk_size()
        all_results = []
        chunk_errors = []

        # Detect conversation messages (JSON array of {role, content} dicts)
        is_conversation, parsed_json = _detect_conversation(content)

        def _do_add(text, user_id, meta, use_infer):
            """Call m.add() with LLM response logging for diagnostics."""
            try:
                result = m.add(text, user_id=user_id, metadata=meta, infer=use_infer)

                if use_infer:
                    r = result.get("results", []) if isinstance(result, dict) else result
                    if not r:
                        # Build diagnostic from captured LLM response
                        actual = m.llm.last_response if hasattr(m.llm, 'last_response') else None
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

        if is_conversation or len(content) <= MAX_CHUNK_CHARS:
            # Single call — no chunking needed
            # Pass the original string content for both cases. mem0's add()
            # expects a string; for conversations, it detects the JSON format
            # internally and processes the messages.
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
                    fix=f"(1) Check model supports JSON output: 'ollama run {model_name} \"Respond with JSON: {{\\\"facts\\\": []}}\"'\n"
                        f"    (2) Check Ollama context window: 'ollama show {model_name}' (look for num_ctx)\n"
                        f"    (3) Use add_raw_memory instead to store without LLM extraction",
                )
        return merged

    # ── add_raw_memory ──────────────────────────────────────────────────────
    elif name == "add_raw_memory":
        content = arguments["content"]
        if not content or not content.strip():
            raise Mem0Error("Cannot save empty content.", tool=name,
                            fix="Provide text to store.")
        metadata = arguments.get("metadata")
        try:
            result = m.add(content, user_id=uid, metadata=metadata, infer=False)
            return result
        except Exception as e:
            raise Mem0Error("Failed to store raw memory.", tool=name, detail=str(e))

    # ── search_memories ─────────────────────────────────────────────────────
    elif name == "search_memories":
        query = arguments.get("query", "")
        if not query or not query.strip():
            raise Mem0Error("Search query cannot be empty.", tool=name)
        min_score = arguments.get("min_score", 0.0)
        include_scores = arguments.get("include_scores", True)
        try:
            raw_results = m.search(query, filters={"user_id": uid},
                                   top_k=arguments.get("limit", 10),
                                   threshold=0.0)  # v3 defaults to 0.1, override to get all results
            # Normalize: mem0 may return a list or a dict with "results"
            results_list = raw_results if isinstance(raw_results, list) else raw_results.get("results", [])

            # Enrich each result with a score field and filter by min_score
            enriched: List[dict] = []
            for mem in results_list:
                if not isinstance(mem, dict):
                    continue
                # mem0 search results may contain a "score" field from Qdrant
                score = mem.get("score")
                if score is None:
                    score = None

                # Apply min_score filter
                if min_score and score is not None and score < min_score:
                    continue

                if include_scores:
                    if "score" not in mem:
                        mem["score"] = score
                    enriched.append(mem)
                else:
                    # Strip score if include_scores is false
                    mem.pop("score", None)
                    enriched.append(mem)

            return enriched
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
                             top_k=arguments.get("limit", 50),
                             )
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
        # Query Qdrant directly for all distinct user_id values in the collection.
        # mem0's get_all() requires a user_id filter, so we can't list all entities
        # through the library API. Instead, scroll the Qdrant collection directly.
        entities = set()
        try:
            # Use Qdrant scroll API to get all points with their payloads
            qdrant_host = CONFIG["vector_store"]["config"]["host"]
            qdrant_port = CONFIG["vector_store"]["config"]["port"]
            qdrant_base = f"http://{qdrant_host}:{qdrant_port}"

            # Scroll through all points in the mem0 collection
            offset = None
            while True:
                url = f"{qdrant_base}/collections/mem0/points/scroll"
                payload = json.dumps({
                    "limit": 100,
                    "with_payload": True,
                    "with_vector": False,
                    **({"offset": offset} if offset else {}),
                }).encode()
                req = urllib.request.Request(url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST")
                resp = urllib.request.urlopen(req, timeout=10)
                data = json.loads(resp.read())
                points = data.get("result", {}).get("points", [])
                if not points:
                    break
                for point in points:
                    payload = point.get("payload", {})
                    for key in ("user_id", "agent_id", "app_id", "run_id"):
                        val = payload.get(key)
                        if val:
                            entities.add(val)
                offset = data.get("result", {}).get("next_offset")
                if offset is None:
                    break
        except Exception as e:
            print(f"[list_entities] Warning: {e}", file=sys.stderr)
            # Fallback: try the default user_id via mem0 API
            try:
                all_mems = m.get_all(filters={"user_id": uid}, top_k=500)
                results = all_mems if isinstance(all_mems, list) else all_mems.get("results", [])
                for mem in results:
                    for key in ("user_id", "agent_id", "app_id", "run_id"):
                        if key in (mem or {}):
                            entities.add(mem[key])
            except Exception:
                pass
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

    # ── export_memories ────────────────────────────────────────────────────
    elif name == "export_memories":
        fmt = arguments.get("format", "json")
        if fmt not in ("json", "csv"):
            raise Mem0Error(f"Invalid format '{fmt}'. Use 'json' or 'csv'.", tool=name)
        try:
            all_mems = m.get_all(filters={"user_id": uid}, top_k=10000)
            results_list = all_mems if isinstance(all_mems, list) else all_mems.get("results", [])

            # Normalize each memory to a clean export record
            export_records: List[dict] = []
            for mem in results_list:
                if not isinstance(mem, dict):
                    continue
                export_records.append({
                    "id": mem.get("id") or mem.get("memory_id", ""),
                    "memory": mem.get("memory", ""),
                    "metadata": mem.get("metadata", {}),
                    "created_at": mem.get("created_at", ""),
                    "user_id": mem.get("user_id", uid),
                })

            if fmt == "json":
                return {"format": "json", "count": len(export_records),
                        "user_id": uid, "memories": export_records}
            else:
                # CSV output
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=["id", "memory", "metadata", "created_at", "user_id"])
                writer.writeheader()
                for rec in export_records:
                    writer.writerow({
                        "id": rec["id"],
                        "memory": rec["memory"],
                        "metadata": json.dumps(rec["metadata"]) if rec["metadata"] else "",
                        "created_at": rec["created_at"],
                        "user_id": rec["user_id"],
                    })
                return {"format": "csv", "count": len(export_records),
                        "user_id": uid, "data": output.getvalue()}
        except Exception as e:
            raise Mem0Error(f"Failed to export memories for user '{uid}'.",
                            tool=name, detail=str(e))

    # ── import_memories ─────────────────────────────────────────────────────
    elif name == "import_memories":
        data_str = arguments.get("data", "")
        if not data_str or not data_str.strip():
            raise Mem0Error("data is required (JSON string of memory array).", tool=name)

        try:
            parsed = json.loads(data_str)
        except (json.JSONDecodeError, TypeError) as e:
            raise Mem0Error("data is not valid JSON.", tool=name, detail=str(e),
                            fix="Provide a JSON array exported by export_memories.")

        if not isinstance(parsed, list):
            raise Mem0Error("data must be a JSON array of memory objects.", tool=name)

        imported = 0
        skipped = 0
        failed = 0
        errors: List[str] = []

        # Fetch existing memory texts for duplicate detection
        try:
            existing_mems = m.get_all(filters={"user_id": uid}, top_k=10000)
            existing_list = existing_mems if isinstance(existing_mems, list) else existing_mems.get("results", [])
            existing_texts = set()
            for mem in existing_list:
                if isinstance(mem, dict):
                    text = mem.get("memory", "")
                    if text:
                        existing_texts.add(text.strip().lower())
        except Exception:
            existing_texts = set()

        for item in parsed:
            if not isinstance(item, dict):
                failed += 1
                errors.append(f"Non-dict item skipped: {type(item).__name__}")
                continue
            mem_text = item.get("memory", "") or item.get("text", "") or item.get("content", "")
            if not mem_text or not mem_text.strip():
                failed += 1
                errors.append("Empty memory text, skipped")
                continue

            # Skip duplicates (case-insensitive comparison)
            if mem_text.strip().lower() in existing_texts:
                skipped += 1
                continue

            try:
                mem_metadata = item.get("metadata")
                m.add(mem_text, user_id=uid, metadata=mem_metadata, infer=False)
                existing_texts.add(mem_text.strip().lower())
                imported += 1
            except Exception as e:
                failed += 1
                errors.append(f"Failed to import: {e}")

        return {
            "imported": imported,
            "skipped": skipped,
            "failed": failed,
            "errors": errors if errors else None,
            "user_id": uid,
        }

    # ── prune_memories ──────────────────────────────────────────────────────
    elif name == "prune_memories":
        older_than_days = arguments.get("older_than_days", 30)
        dry_run = arguments.get("dry_run", True)
        min_score = arguments.get("min_score")

        try:
            all_mems = m.get_all(filters={"user_id": uid}, top_k=10000)
            results_list = all_mems if isinstance(all_mems, list) else all_mems.get("results", [])
        except Exception as e:
            raise Mem0Error("Failed to list memories for pruning.", tool=name, detail=str(e))

        cutoff = time.time() - (older_than_days * 86400)
        to_prune: List[dict] = []

        for mem in results_list:
            if not isinstance(mem, dict):
                continue

            # Determine the memory's age from created_at
            created_at = mem.get("created_at", "")
            mem_age = None
            if created_at:
                # Try ISO 8601 parsing
                try:
                    import datetime as _dt
                    if "T" in str(created_at):
                        mem_age = _dt.datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).timestamp()
                    else:
                        # Try as a unix timestamp or date string
                        mem_age = float(created_at)
                except (ValueError, TypeError):
                    pass

            if mem_age is not None and mem_age < cutoff:
                # Apply optional min_score filter
                if min_score is not None:
                    score = mem.get("score")
                    if score is not None and score > min_score:
                        continue

                to_prune.append({
                    "id": mem.get("id") or mem.get("memory_id", ""),
                    "memory": mem.get("memory", "")[:200],
                    "created_at": created_at,
                })

        deleted_count = 0
        if not dry_run:
            for item in to_prune:
                mem_id = item.get("id")
                if not mem_id:
                    continue
                try:
                    m.delete(mem_id)
                    deleted_count += 1
                except Exception as e:
                    print(f"[prune_memories] Failed to delete {mem_id}: {e}", file=sys.stderr)

        return {
            "would_delete": len(to_prune),
            "deleted": deleted_count,
            "dry_run": dry_run,
            "memories": to_prune,
        }

    else:
        raise Mem0Error(f"Unknown tool: {name}", tool=name)


# ── MCP HTTP server (manual JSON-RPC over HTTP) ─────────────────────────────

def _scroll_all_memories() -> dict:
    """Scroll the Qdrant mem0 collection directly to get all memories with
    their payloads. Used by the web UI. Returns {memories: [...], entities: [...]}.
    """
    memories: List[dict] = []
    entities = set()

    qdrant_host = CONFIG["vector_store"]["config"]["host"]
    qdrant_port = CONFIG["vector_store"]["config"]["port"]
    qdrant_base = f"http://{qdrant_host}:{qdrant_port}"

    try:
        offset = None
        while True:
            url = f"{qdrant_base}/collections/mem0/points/scroll"
            payload = json.dumps({
                "limit": 250,
                "with_payload": True,
                "with_vector": False,
                **({"offset": offset} if offset else {}),
            }).encode()
            req = urllib.request.Request(url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            points = data.get("result", {}).get("points", [])
            if not points:
                break
            for point in points:
                p = point.get("payload", {})
                # Extract the memory text and metadata
                mem = {
                    "id": point.get("id", ""),
                    "memory": p.get("data", p.get("memory", p.get("text", ""))),
                    "user_id": p.get("user_id", ""),
                    "created_at": p.get("created_at", ""),
                    "metadata": {k: v for k, v in p.items()
                                 if k not in ("data", "memory", "text", "user_id",
                                              "created_at", "embedding", "text_lemmatized")},
                }
                memories.append(mem)
                if p.get("user_id"):
                    entities.add(p["user_id"])
            offset = data.get("result", {}).get("next_offset")
            if offset is None:
                break
    except Exception as e:
        print(f"[web_ui] Failed to scroll memories: {e}", file=sys.stderr)

    return {"memories": memories, "entities": sorted(entities)}


def _render_web_ui() -> str:
    """Render a simple HTML page for browsing memories in the database."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mem0-local</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #58a6ff; margin-bottom: 16px; font-size: 1.4rem; }
.stats { display: flex; gap: 16px; margin-bottom: 20px; }
.stat { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; }
.stat-value { font-size: 1.5rem; font-weight: bold; color: #58a6ff; }
.stat-label { font-size: 0.8rem; color: #8b949e; }
.filter-bar { margin-bottom: 16px; }
input, select { background: #161b22; border: 1px solid #30363d; color: #c9d1d9; padding: 8px 12px; border-radius: 6px; font-size: 0.9rem; }
input[type=text] { width: 300px; }
.memories { display: flex; flex-direction: column; gap: 8px; }
.mem-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
.mem-text { font-size: 0.95rem; line-height: 1.5; margin-bottom: 8px; }
.mem-meta { display: flex; gap: 12px; font-size: 0.75rem; color: #8b949e; }
.mem-meta span { background: #21262d; padding: 2px 8px; border-radius: 4px; }
.empty { color: #8b949e; text-align: center; padding: 40px; }
.error { color: #f85149; }
a { color: #58a6ff; }
.refresh { float: right; }
</style>
</head>
<body>
<h1>mem0-local <button class="refresh" onclick="load()">Refresh</button></h1>
<div class="stats">
  <div class="stat"><div class="stat-value" id="count">—</div><div class="stat-label">memories</div></div>
  <div class="stat"><div class="stat-value" id="entities">—</div><div class="stat-label">entities</div></div>
</div>
<div class="filter-bar">
  <input type="text" id="search" placeholder="Filter memories..." oninput="filter()">
  <select id="entity-filter" onchange="filter()"><option value="">All entities</option></select>
</div>
<div class="memories" id="list"><div class="empty">Loading...</div></div>
<script>
let allMems = [];
async function load() {
  try {
    const r = await fetch('/api/memories');
    const d = await r.json();
    allMems = d.memories || [];
    document.getElementById('count').textContent = allMems.length;
    document.getElementById('entities').textContent = (d.entities || []).length;
    const sel = document.getElementById('entity-filter');
    sel.innerHTML = '<option value="">All entities</option>';
    (d.entities || []).forEach(e => {
      const opt = document.createElement('option');
      opt.value = e; opt.textContent = e; sel.appendChild(opt);
    });
    render(allMems);
  } catch(e) {
    document.getElementById('list').innerHTML = '<div class="error">Failed to load: ' + e + '</div>';
  }
}
function filter() {
  const q = document.getElementById('search').value.toLowerCase();
  const ent = document.getElementById('entity-filter').value;
  let filtered = allMems;
  if (q) filtered = filtered.filter(m => JSON.stringify(m).toLowerCase().includes(q));
  if (ent) filtered = filtered.filter(m => (m.user_id || '') === ent);
  render(filtered);
}
function render(mems) {
  const el = document.getElementById('list');
  if (!mems.length) { el.innerHTML = '<div class="empty">No memories found</div>'; return; }
  el.innerHTML = mems.map(m => {
    const text = (m.memory || m.memory_text || m.text || '').replace(/</g, '&lt;');
    const uid = m.user_id || '—';
    const id = (m.id || m.memory_id || '').substring(0, 12);
    const created = m.created_at ? new Date(m.created_at).toLocaleString() : '—';
    const score = m.score != null ? m.score.toFixed(3) : '';
    let meta = `<span>${uid}</span><span>${id}</span><span>${created}</span>`;
    if (score) meta += `<span>score: ${score}</span>`;
    return `<div class="mem-card"><div class="mem-text">${text}</div><div class="mem-meta">${meta}</div></div>`;
  }).join('');
}
load();
</script>
</body>
</html>"""


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
            health = _build_health_response()
            response = json.dumps(health).encode()
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

        # Web UI — simple HTML page for browsing memories
        if method == "GET" and path == "/":
            html = _render_web_ui()
            response = html.encode()
            header = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: text/html; charset=utf-8\r\n"
                f"Content-Length: {len(response)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode()
            writer.write(header + response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            return

        # API for web UI — list all memories via direct Qdrant scroll
        if method == "GET" and path == "/api/memories":
            try:
                response_obj = await asyncio.to_thread(_scroll_all_memories)
                response = json.dumps(response_obj, default=str).encode()
                header = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(response)}\r\n"
                    f"Connection: close\r\n\r\n"
                ).encode()
                writer.write(header + response)
            except Exception as e:
                response = json.dumps({"error": str(e)}).encode()
                header = (
                    f"HTTP/1.1 500 Internal Server Error\r\n"
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


def handle_mcp_request_sync(request: dict) -> Optional[dict]:
    """Synchronous MCP request handler — called from a worker thread
    so long LLM calls don't block the async event loop."""
    return _handle_mcp_request(request)


def _handle_mcp_request(request: dict) -> Optional[dict]:
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
    # Initialize mem0 synchronously — creates Qdrant collections before
    # accepting requests. This blocks the server from starting until the
    # collection exists, preventing "Collection doesn't exist" 404 errors.
    # The Docker healthcheck has start_period=600s to accommodate this.
    init_memory()

    server = await asyncio.start_server(http_handler, MCP_HOST, MCP_PORT)
    print(f"mem0-local MCP server listening on {MCP_HOST}:{MCP_PORT}", flush=True)
    print(f"  LLM:      {CONFIG['llm']['config']['model']} @ {OLLAMA_URL} (ctx: {_LLM_NUM_CTX}, max_tokens: {_LLM_MAX_TOKENS})", flush=True)
    print(f"  Embedder: {CONFIG['embedder']['config']['model']} @ {OLLAMA_URL}", flush=True)
    print(f"  Vector:   Qdrant @ {QDRANT_URL}", flush=True)
    print(f"  Tools:    {len(TOOL_DEFINITIONS)}", flush=True)
    print(f"  Health:   GET  http://{MCP_HOST}:{MCP_PORT}/health", flush=True)
    print(f"  MCP:      POST http://{MCP_HOST}:{MCP_PORT}/mcp", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())