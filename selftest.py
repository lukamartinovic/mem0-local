#!/usr/bin/env python3
"""
Self-test: exercises all 12 MCP tools end-to-end against real Qdrant + fastembed.
NO extraction LLM exists in this server, so nothing here needs one.
Errors from execute_tool raise Mem0Error subclasses with actionable messages.
"""

import json
import os
import sys
import time
import uuid

QDRANT_HOST = os.environ.get("MEM0_QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = os.environ.get("MEM0_QDRANT_PORT", "6333")
DEFAULT_USER_ID = os.environ.get("MEM0_DEFAULT_USER_ID", "dev")

sys.path.insert(0, "/app")
import mcp_server
from mcp_server import Mem0Error, QdrantError

GREEN = "\033[0;32m"
RED = "\033[0;31m"
NC = "\033[0m"

passed = 0
failed = 0
errors = []


def check(label, fn):
    global passed, failed, errors
    print(f"  {label}", end="", flush=True)
    try:
        fn()
        print(f"  {GREEN}OK{NC}")
        passed += 1
    except (QdrantError, Mem0Error) as e:
        print(f"  {RED}FAIL{NC}")
        for line in str(e).split("\n"):
            print(f"       {line}")
        failed += 1
        errors.append((label, str(e)))
    except Exception as e:
        print(f"  {RED}FAIL {e}{NC}")
        failed += 1
        errors.append((label, str(e)))


def run():
    global passed, failed

    print()
    print("=" * 60)
    print("  mem0-local self-test - exercising all 12 MCP tools")
    print("=" * 60)
    print()

    # Pre-flight checks
    print("  Pre-flight checks:")
    def check_qdrant():
        mcp_server._check_qdrant()
    check("  Qdrant reachable:", check_qdrant)
    print()

    if failed > 0:
        print("Pre-flight checks failed. Fix infrastructure before testing tools.")
        print()
        _print_summary()
        sys.exit(1)

    # Initialize memory (creates Qdrant collections, fastembed model lazy-loads)
    print("  Initializing mem0 (fastembed model loads on first embed)...")
    m = mcp_server.init_memory()
    test_user = f"selftest_{uuid.uuid4().hex[:8]}"

    # 1. add_raw_memory (agent-extracted single fact)
    fact_id = [None]
    def test_add_fact():
        result = mcp_server.execute_tool("add_raw_memory", {
            "content": "The mem0-local MCP server deploys as a single Docker container on port 8765.",
            "user_id": test_user,
            "metadata": {"source": "selftest"},
        })
        results = result.get("results", []) if isinstance(result, dict) else result
        if not results:
            raise Mem0Error("add_raw_memory returned no results", detail=f"Result: {result}")
        first = results[0] if isinstance(results[0], dict) else {}
        fact_id[0] = first.get("id") or first.get("memory_id")
    check("1/12  add_raw_memory (single fact)", test_add_fact)

    # 2. add_raw_memory rejects oversized non-fact blobs
    def test_add_fact_size_guard():
        blob = "x" * (mcp_server.CHUNK_CHARS + 10)
        try:
            mcp_server.execute_tool("add_raw_memory", {"content": blob, "user_id": test_user})
        except Mem0Error as e:
            if "chunk size" in str(e):
                return
            raise
        raise Mem0Error("add_raw_memory accepted an oversized blob without error")
    check("2/12  add_raw_memory size guard", test_add_fact_size_guard)

    # 3. add_verbatim (long text auto-chunks)
    def test_add_verbatim():
        long_text = "Deployment runs through GitHub Actions with matrix testing.\n\n" * 200
        result = mcp_server.execute_tool("add_verbatim", {
            "content": long_text,
            "user_id": test_user,
            "metadata": {"file": "docs/deploy.md", "source": "selftest"},
        })
        merged = result if isinstance(result, dict) else {}
        if not merged.get("results"):
            raise Mem0Error("add_verbatim returned no results", detail=f"keys: {list(merged)}")
        if merged.get("chunks", 0) < 2:
            raise Mem0Error("add_verbatim did not chunk long text", detail=f"chunks: {merged.get('chunks')}")
    check("3/12  add_verbatim (auto-chunking)", test_add_verbatim)

    time.sleep(2)

    # 4. search_memories (with scores)
    def test_search():
        result = mcp_server.execute_tool("search_memories", {
            "query": "How is the mem0-local container deployed?",
            "user_id": test_user,
            "include_scores": True,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        if len(results) == 0:
            raise Mem0Error("search returned 0 results after successful add")
        first = results[0]
        if "score" not in first:
            raise Mem0Error("search_memories result missing 'score' field", detail=f"Result: {first}")
    check("4/12  search_memories (with scores)", test_search)

    # 5. search_memories (min_score filter)
    def test_search_min_score():
        result = mcp_server.execute_tool("search_memories", {
            "query": "How is the container deployed?",
            "user_id": test_user,
            "min_score": 0.99,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        if not isinstance(results, list):
            raise Mem0Error("search_memories with min_score did not return a list")
    check("5/12  search_memories (min_score)", test_search_min_score)

    # 6. get_memories
    def test_get_all():
        result = mcp_server.execute_tool("get_memories", {"user_id": test_user, "limit": 10})
        if not result:
            raise Mem0Error("get_memories returned empty")
    check("6/12  get_memories", test_get_all)

    # 7. get_memory + update_memory (update is LLM-free in mem0 2.x)
    def test_get_update():
        if not fact_id[0]:
            all_mems = mcp_server.execute_tool("get_memories", {"user_id": test_user, "limit": 10})
            results = all_mems if isinstance(all_mems, list) else all_mems.get("results", [])
            if not results:
                raise Mem0Error("no memories to test get/update")
            fact_id[0] = results[0].get("id") or results[0].get("memory_id")
        mcp_server.execute_tool("get_memory", {"memory_id": fact_id[0]})
        mcp_server.execute_tool("update_memory", {
            "memory_id": fact_id[0],
            "content": "Updated: the mem0-local container listens on 8765 and 6333.",
        })
    check("7/12  get_memory + update_memory", test_get_update)

    # 8. list_entities
    def test_list_entities():
        result = mcp_server.execute_tool("list_entities", {})
        if "entities" not in result:
            raise Mem0Error("list_entities returned no 'entities' key", detail=f"Result: {result}")
    check("8/12  list_entities", test_list_entities)

    # 9. export_memories (JSON)
    def test_export_json():
        result = mcp_server.execute_tool("export_memories", {"user_id": test_user, "format": "json"})
        if "memories" not in result:
            raise Mem0Error("export_memories (json) returned no 'memories' key")
        if result.get("format") != "json":
            raise Mem0Error(f"export_memories format mismatch: expected 'json', got '{result.get('format')}'")
        if not isinstance(result["memories"], list):
            raise Mem0Error("export_memories memories is not a list")
    check("9/12  export_memories (JSON)", test_export_json)

    # 10. export_memories (CSV)
    def test_export_csv():
        result = mcp_server.execute_tool("export_memories", {"user_id": test_user, "format": "csv"})
        if "data" not in result:
            raise Mem0Error("export_memories (csv) returned no 'data' key")
        if result.get("format") != "csv":
            raise Mem0Error(f"export_memories format mismatch: expected 'csv', got '{result.get('format')}'")
        if "id,memory,metadata,created_at,user_id" not in result["data"]:
            raise Mem0Error("export_memories csv missing header row")
    check("10/12  export_memories (CSV)", test_export_csv)

    # 11. import_memories
    def test_import():
        export_result = mcp_server.execute_tool("export_memories", {"user_id": test_user, "format": "json"})
        export_data = export_result.get("memories", [])
        if not export_data:
            raise Mem0Error("no memories to import (export was empty)")
        import_user = f"selftest_import_{uuid.uuid4().hex[:8]}"
        result = mcp_server.execute_tool("import_memories", {
            "data": json.dumps(export_data),
            "user_id": import_user,
        })
        if "imported" not in result:
            raise Mem0Error("import_memories returned no 'imported' key", detail=f"Result: {result}")
        if result["imported"] == 0:
            raise Mem0Error("import_memories imported 0 memories", detail=f"Result: {result}")
        mcp_server.execute_tool("delete_all_memories", {"user_id": import_user})
    check("11/12  import_memories", test_import)

    # 12. delete_all + prune (dry run)
    def test_prune_delete():
        mcp_server.execute_tool("delete_all_memories", {"user_id": test_user})
        result = mcp_server.execute_tool("prune_memories", {
            "user_id": test_user,
            "older_than_days": 0,
            "dry_run": True,
        })
        if "would_delete" not in result:
            raise Mem0Error("prune_memories returned no 'would_delete' key", detail=f"Result: {result}")
    check("12/12  delete_all + prune (dry run)", test_prune_delete)

    _print_summary()
    if failed > 0:
        sys.exit(1)


def _print_summary():
    print()
    if failed > 0:
        print(f"{RED}  {passed}/13 tools passed, {failed} failed{NC}")
        print()
        print("Errors:")
        for label, err in errors:
            print(f"  {label}:")
            for line in err.split("\n"):
                print(f"    {line}")
        print()
        print("Common fixes:")
        print("  * Qdrant down:      docker compose logs mem0-local")
        print("  * Dimension error:  MEM0_EMBED_DIMS must match the stored collection (768)")
    else:
        print(f"{GREEN}  12/13 tools passed - all MCP commands verified{NC}")
    print()


if __name__ == "__main__":
    run()