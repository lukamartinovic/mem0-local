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
from typing import List

# Reuse chunking logic from mcp_server — single source of truth.
# import_docs.py runs on the host (not in Docker), so mcp_server must be importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcp_server
_chunk_text = mcp_server._chunk_text
_get_chunk_size = mcp_server._get_chunk_size

MCP_URL = "http://localhost:8765/mcp"
HEALTH_URL = "http://localhost:8765/health"

# ── Colors ──────────────────────────────────────────────────────────────────
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
NC = "\033[0m"


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


def add_memory(content: str, user_id: str, metadata: dict, req_id: int,
               max_retries: int = 2) -> bool:
    """Send a memory to the MCP server. Returns True on success, False on failure.
    Retries on transient failures (timeouts, connection errors) with backoff.
    Prints the error message (not traceback) on failure."""
    last_error = ""
    for attempt in range(max_retries + 1):
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
                # Don't retry on LLM extraction errors — retrying won't help
                if "did not extract" in error_text or "LLM" in error_text:
                    print(f"\n    {RED}✗{NC} {error_text[:300]}")
                    return False
                last_error = error_text[:300]
                if attempt < max_retries:
                    wait = (attempt + 1) * 5
                    print(f"\n    {YELLOW}↻{NC} Retry {attempt+1}/{max_retries} in {wait}s...", end="", flush=True)
                    time.sleep(wait)
                    continue
                print(f"\n    {RED}✗{NC} {error_text[:300]}")
                return False
            return True
        except DocImportError as e:
            last_error = str(e)
            if attempt < max_retries:
                wait = (attempt + 1) * 5
                print(f"\n    {YELLOW}↻{NC} Retry {attempt+1}/{max_retries} in {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            print(f"\n    {RED}✗{NC} {e}")
            return False
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                wait = (attempt + 1) * 5
                print(f"\n    {YELLOW}↻{NC} Retry {attempt+1}/{max_retries} in {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            print(f"\n    {RED}✗{NC} Unexpected error: {e}")
            return False
    print(f"\n    {RED}✗{NC} Failed after {max_retries} retries: {last_error[:200]}")
    return False


# ── File discovery ───────────────────────────────────────────────────────────

def find_md_files(directory: str) -> List[Path]:
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
    print(f"Chunk size: {_get_chunk_size():,} chars (model: {args.llm_model})")
    print()

    if args.dry_run:
        for f in files:
            size = f.stat().st_size
            chunks = _chunk_text(f.read_text(encoding="utf-8", errors="replace"),
                               _get_chunk_size())
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

        chunk_size = _get_chunk_size()
        context_header = f"[Document: {rel_path}][Source: docs_import]"
        chunks = _chunk_text(content, max_chars=chunk_size, context_header=context_header)

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