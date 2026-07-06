#!/usr/bin/env python3
"""
Import markdown files into mem0-local memory.

Usage:
    python3 import_docs.py /path/to/repo/docs --user-id myproject
    python3 import_docs.py /path/to/repo1/docs --user-id repo1
    python3 import_docs.py /path/to/repo/docs --user-id myproject --dry-run

Each .md file is chunked (model-aware) and sent to mem0 for LLM fact extraction.
Errors are reported with actionable messages, not raw tracebacks.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

MCP_URL = "http://localhost:8765/mcp"
HEALTH_URL = "http://localhost:8765/health"

# ── Colors ──────────────────────────────────────────────────────────────────
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
NC = "\033[0m"

# ── Model-specific chunk sizes ──────────────────────────────────────────────
MODEL_CHUNK_SIZES = {
    "qwen2.5:3b": 1500,
    "qwen2.5:7b": 3000,
    "qwen3.5:9b": 4000,
    "gemma4:12b": 5000,
    "qwen3.5:27b": 8000,
}

def get_chunk_size(model: str) -> int:
    """Get the chunk size for a given model."""
    # Exact match first
    if model in MODEL_CHUNK_SIZES:
        return MODEL_CHUNK_SIZES[model]
    # Prefix match: find the most specific (longest) matching key
    best_match = None
    best_len = 0
    for key, size in MODEL_CHUNK_SIZES.items():
        if model.startswith(key) and len(key) > best_len:
            best_match = size
            best_len = len(key)
    return best_match if best_match is not None else 3000


# ── HTTP client ──────────────────────────────────────────────────────────────

class DocImportError(Exception):
    """User-friendly import error with actionable fix."""
    pass


def check_server():
    """Verify the MCP server is running and healthy."""
    try:
        resp = urllib.request.urlopen(HEALTH_URL, timeout=5)
        data = json.loads(resp.read())
        if data.get("status") != "ok":
            raise DocImportError(
                f"MCP server unhealthy: {data}\n"
                f"  Fix: Restart with 'docker compose restart mcp-server'"
            )
        return data
    except urllib.error.URLError:
        raise DocImportError(
            "Cannot reach mem0-local server at localhost:8765.\n"
            "  The server is not running.\n"
            "  Fix: Start it with './setup.sh' or 'docker compose up -d'"
        )
    except DocImportError:
        raise
    except Exception as e:
        raise DocImportError(
            f"Unexpected error checking server health: {e}\n"
            f"  Fix: Check Docker: 'docker compose ps' and 'docker compose logs mcp-server'"
        )


def mcp_call(method: str, params: dict, req_id: int = 1) -> dict:
    """Make an MCP JSON-RPC call. Raises DocImportError on failure."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params,
    }).encode()
    req = urllib.request.Request(
        MCP_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
        # Check for JSON-RPC error response
        if "error" in data:
            error_msg = data["error"].get("message", "Unknown error")
            raise DocImportError(
                f"Server returned an error: {error_msg}\n"
                f"  Fix: Check server logs: 'docker compose logs mcp-server'"
            )
        return data
    except urllib.error.HTTPError as e:
        raise DocImportError(
            f"Server returned HTTP {e.code}.\n"
            f"  Fix: Check server logs: 'docker compose logs mcp-server'"
        )
    except urllib.error.URLError as e:
        raise DocImportError(
            f"Connection failed: {e.reason}\n"
            f"  Fix: Server may have crashed. Restart: 'docker compose restart mcp-server'"
        )


def add_memory(content: str, user_id: str, metadata: dict, req_id: int) -> bool:
    """Send a memory to the MCP server. Returns True on success, False on failure.
    Prints the error message (not traceback) on failure."""
    try:
        resp = mcp_call("tools/call", {
            "name": "add_memory",
            "arguments": {
                "content": content,
                "user_id": user_id,
                "metadata": metadata,
            },
        }, req_id)
        result = resp.get("result", {})
        if result.get("isError"):
            error_text = result.get("content", [{}])[0].get("text", "Unknown error")
            print(f"\n    {RED}✗{NC} {error_text[:300]}")
            return False
        return True
    except DocImportError as e:
        print(f"\n    {RED}✗{NC} {e}")
        return False
    except Exception as e:
        print(f"\n    {RED}✗{NC} Unexpected error: {e}")
        return False


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 3000, context_header: str = "") -> list[str]:
    """Split text into chunks at paragraph boundaries, each under max_chars.
    If context_header is provided, it's prepended to each chunk."""
    header_len = len(context_header) + 2 if context_header else 0
    if len(text) + header_len <= max_chars:
        return [text] if not context_header else [context_header + "\n\n" + text]

    header_len = len(context_header) + 2 if context_header else 0
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

    if context_header:
        chunks = [context_header + "\n\n" + chunk for chunk in chunks]

    return chunks


# ── File discovery ───────────────────────────────────────────────────────────

def find_md_files(directory: str) -> list[Path]:
    path = Path(directory)
    if not path.exists():
        print(f"{RED}Error:{NC} '{directory}' does not exist")
        sys.exit(1)
    if not path.is_dir():
        print(f"{RED}Error:{NC} '{directory}' is not a directory")
        sys.exit(1)
    files = sorted(path.rglob("*.md"))
    skip = {"node_modules", ".git", "vendor", "__pycache__", ".venv", "dist", "build"}
    files = [f for f in files if not any(s in f.parts for s in skip)]
    return files


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Import markdown docs into mem0-local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 import_docs.py ./docs --user-id myproject
  python3 import_docs.py ./repo1/docs --user-id dev --llm-model qwen2.5:7b
  python3 import_docs.py ./docs --user-id dev --dry-run
        """,
    )
    parser.add_argument("directory", help="Path to directory containing .md files")
    parser.add_argument("--user-id", required=True,
                        help="User/project ID for memories (e.g. repo name)")
    parser.add_argument("--llm-model", default="qwen2.5:7b",
                        help="LLM model in use (determines chunk size). Default: qwen2.5:7b")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without importing")
    args = parser.parse_args()

    # Pre-flight: check server
    try:
        health = check_server()
    except DocImportError as e:
        print(f"{RED}Error:{NC} {e}")
        sys.exit(1)

    # Auto-detect model from server if not explicitly set
    server_model = health.get("config", {}).get("llm", "")
    if server_model and args.llm_model == "qwen2.5:7b" and server_model != "qwen2.5:7b":
        args.llm_model = server_model
        print(f"{YELLOW}Auto-detected model from server:{NC} {server_model}")

    model_name = server_model or args.llm_model
    print(f"Server: {GREEN}healthy{NC} (LLM: {model_name})")

    # Find files
    files = find_md_files(args.directory)
    if len(files) == 0:
        print(f"{YELLOW}No markdown files found in {args.directory}{NC}")
        sys.exit(0)

    print(f"Found {len(files)} markdown files in {args.directory}")
    print(f"User ID: {args.user_id}")
    print(f"Chunk size: {get_chunk_size(args.llm_model):,} chars (model: {args.llm_model})")
    print()

    if args.dry_run:
        for f in files:
            size = f.stat().st_size
            chunks = chunk_text(f.read_text(encoding="utf-8", errors="replace"),
                               get_chunk_size(args.llm_model))
            chunk_info = f" → {len(chunks)} chunks" if len(chunks) > 1 else ""
            print(f"  {f.relative_to(args.directory)}  ({size:,} bytes){chunk_info}")
        print(f"\n{YELLOW}Dry run{NC} — {len(files)} files would be imported.")
        return

    success = 0
    failed = 0
    skipped = 0

    for i, f in enumerate(files):
        rel_path = str(f.relative_to(args.directory))
        content = f.read_text(encoding="utf-8", errors="replace")

        # Skip very small files
        if len(content.strip()) < 50:
            print(f"  [{i+1}/{len(files)}] {YELLOW}⏭{NC}  {rel_path} (too short, skipping)")
            skipped += 1
            continue

        metadata = {
            "source": "docs_import",
            "file": rel_path,
            "repo": args.user_id,
        }

        chunk_size = get_chunk_size(args.llm_model)
        context_header = f"[Document: {rel_path}][Source: docs_import]"
        chunks = chunk_text(content, max_chars=chunk_size, context_header=context_header)

        if len(chunks) == 1:
            print(f"  [{i+1}/{len(files)}] → {rel_path}  ({len(content):,} chars)", end="", flush=True)
            # Send the chunk (which includes context header), not raw content
            if add_memory(chunks[0], args.user_id, metadata, req_id=i + 100):
                print(f"  {GREEN}✅{NC}")
                success += 1
            else:
                failed += 1
        else:
            print(f"  [{i+1}/{len(files)}] → {rel_path}  ({len(content):,} chars, {len(chunks)} chunks)")
            all_ok = True
            for ci, chunk in enumerate(chunks):
                chunk_meta = {**metadata, "chunk": f"{ci+1}/{len(chunks)}"}
                print(f"    chunk {ci+1}/{len(chunks)} ({len(chunk):,} chars)", end="", flush=True)
                if add_memory(chunk, args.user_id, chunk_meta, req_id=i * 100 + ci + 100):
                    print(f"  {GREEN}✅{NC}")
                else:
                    print(f"  {RED}✗{NC}")
                    all_ok = False
            if all_ok:
                success += 1
            else:
                failed += 1

        time.sleep(0.5)

    # Summary
    print()
    total = success + failed + skipped
    print(f"{GREEN}Done:{NC} {success} imported, {RED}{failed} failed{NC}, {YELLOW}{skipped} skipped{NC}, {total} total files")
    if failed > 0:
        print()
        print(f"{YELLOW}Some files failed to import.{NC} Common causes:")
        print(f"  • LLM too small for JSON extraction → try: ollama pull qwen2.5:7b")
        print(f"  • Ollama not running → try: ollama serve")
        print(f"  • Qdrant dimension mismatch → try: docker compose restart mcp-server")
    print()
    print(f"Search your docs:")
    print(f"  Ask your IDE agent: 'Search memories for ...' with user_id={args.user_id}")
    print(f"  Or: curl -X POST http://localhost:8765/mcp \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",")
    print(f"        \"params\":{{\"name\":\"search_memories\",")
    print(f"        \"arguments\":{{\"query\":\"your search\",\"user_id\":\"{args.user_id}\"}}}}}}'")


if __name__ == "__main__":
    main()