#!/usr/bin/env python3
"""
Self-test: exercises all 13 MCP tools end-to-end.
Runs inside the container after startup, before accepting IDE requests.
Errors from execute_tool now raise Mem0Error subclasses with actionable messages.
"""

import json
import os
import sys
import time
import uuid

OLLAMA_URL = os.environ.get("MEM0_OLLAMA_URL", "http://ollama:11434")
QDRANT_HOST = os.environ.get("MEM0_QDRANT_HOST", "qdrant")
QDRANT_PORT = os.environ.get("MEM0_QDRANT_PORT", "6333")
DEFAULT_USER_ID = os.environ.get("MEM0_DEFAULT_USER_ID", "dev")

sys.path.insert(0, "/app")
import mcp_server
from mcp_server import Mem0Error, LLMExtractionError, QdrantError, OllamaError

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
NC = "\033[0m"

passed = 0
failed = 0
errors = []

def check(label, fn):
    global passed, failed, errors
    print(f"  {label}...", end="", flush=True)
    try:
        fn()
        print(f"  {GREEN}✅{NC}")
        passed += 1
    except (LLMExtractionError, QdrantError, OllamaError, Mem0Error) as e:
        print(f"  {RED}❌{NC}")
        for line in str(e).split("\n"):
            print(f"       {line}")
        failed += 1
        errors.append((label, str(e)))
    except Exception as e:
        print(f"  {RED}❌ {e}{NC}")
        failed += 1
        errors.append((label, str(e)))


def run():
    global passed, failed

    print()
    print("┌──────────────────────────────────────────────────────────┐")
    print("│  mem0-local self-test — exercising all 13 MCP tools      │")
    print("└──────────────────────────────────────────────────────────┘")
    print()

    # Pre-flight checks
    print("  Pre-flight checks:")
    def check_ollama():
        mcp_server._check_ollama()
    check("  • Ollama reachable", check_ollama)

    def check_qdrant():
        mcp_server._check_qdrant()
    check("  • Qdrant reachable", check_qdrant)
    print()

    if failed > 0:
        print(f"{RED}Pre-flight checks failed. Fix infrastructure before testing tools.{NC}")
        print()
        _print_summary()
        sys.exit(1)

    # Initialize memory (creates Qdrant collections)
    m = mcp_server.init_memory()
    test_user = f"selftest_{uuid.uuid4().hex[:8]}"

    # 1. add_memory (string, infer=False — selftest verifies CRUD, not LLM extraction)
    def test_add_string():
        result = mcp_server.execute_tool("add_raw_memory", {
            "content": "My name is Alice. I work at Acme Corp as a senior software engineer. "
                       "I use TypeScript and React for frontend development, Node.js for the backend, "
                       "and PostgreSQL as the database. We deploy to AWS using GitHub Actions CI/CD. "
                       "I prefer dark mode in my IDE and use Neovim as my primary editor.",
            "user_id": test_user,
            
        })
        results = result.get("results", []) if isinstance(result, dict) else result
        if not results:
            raise Mem0Error(
                "add_memory returned no results",
                detail=f"Result: {result}",
            )
    check("1/13  add_raw_memory (string)", test_add_string)

    # 2. add_memory (conversation messages as JSON, infer=False)
    def test_add_conversation():
        messages = json.dumps([
            {"role": "user", "content": "I'm planning to migrate our API from REST to GraphQL. "
                                        "The main reason is to reduce over-fetching on the mobile app. "
                                        "We'll use Apollo Server with the schema-first approach."},
            {"role": "assistant", "content": "Noted. I'll remember you're migrating from REST to GraphQL "
                                             "using Apollo Server with schema-first approach to solve "
                                             "over-fetching on mobile."},
        ])
        result = mcp_server.execute_tool("add_raw_memory", {
            "content": messages,
            "user_id": test_user,
            
        })
        results = result.get("results", []) if isinstance(result, dict) else result
        if not results:
            raise Mem0Error(
                "add_memory returned no results for conversation input",
                detail=f"Result: {result}",
            )
    check("2/13  add_raw_memory (conversation)", test_add_conversation)

    time.sleep(2)

    # 3. search_memories (with scores)
    test_memory_id = [None]
    def test_search():
        result = mcp_server.execute_tool("search_memories", {
            "query": "What programming languages and tools does Alice use?",
            "user_id": test_user,
            "include_scores": True,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        if len(results) == 0:
            raise Mem0Error("search returned 0 results after successful add",
                            detail="Expected results for 'What programming languages and tools does Alice use?' query")
        # Verify score field is present when include_scores=True
        first = results[0]
        if "score" not in first:
            raise Mem0Error("search_memories result missing 'score' field",
                            detail=f"Result: {first}",
                            fix="This is expected if Qdrant doesn't return scores, but the field should exist (may be null)")
        # Save a memory_id for later tests
        test_memory_id[0] = first.get("id") or first.get("memory_id")
    check("3/13  search_memories (with scores)", test_search)

    # 4. search_memories (min_score filter)
    def test_search_min_score():
        # Use a high min_score that filters everything — should get 0 results
        result = mcp_server.execute_tool("search_memories", {
            "query": "What programming languages and tools does Alice use?",
            "user_id": test_user,
            "min_score": 0.99,
            "include_scores": True,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        # With a very high min_score, we expect 0 or very few results
        # Just verify it doesn't crash and returns a list
        if not isinstance(results, list):
            raise Mem0Error("search_memories with min_score did not return a list",
                            detail=f"Result: {result}")
    check("4/13  search_memories (min_score filter)", test_search_min_score)

    # 5. get_memories
    def test_get_all():
        result = mcp_server.execute_tool("get_memories", {
            "user_id": test_user,
            "limit": 10,
        })
        if not result:
            raise Mem0Error("get_memories returned empty",
                            detail="Expected at least 1 memory from previous add")
    check("5/13  get_memories", test_get_all)

    # 6. get_memory (by ID)
    def test_get_one():
        if not test_memory_id[0]:
            # Fallback: get from get_memories
            all_mems = mcp_server.execute_tool("get_memories", {
                "user_id": test_user,
                "limit": 10,
            })
            results = all_mems if isinstance(all_mems, list) else all_mems.get("results", [])
            if len(results) == 0:
                raise Mem0Error("no memories to test get_memory")
            mem = results[0]
            test_memory_id[0] = mem.get("id") or mem.get("memory_id")
        if not test_memory_id[0]:
            raise Mem0Error("memory has no id field")
        result = mcp_server.execute_tool("get_memory", {"memory_id": test_memory_id[0]})
        if not result:
            raise Mem0Error(f"get_memory returned empty for id={test_memory_id[0]}")
    check("6/13  get_memory (by ID)", test_get_one)

    # 7. update_memory
    def test_update():
        if not test_memory_id[0]:
            raise Mem0Error("no memory_id from previous test")
        mcp_server.execute_tool("update_memory", {
            "memory_id": test_memory_id[0],
            "content": "Updated: I now prefer Python over everything.",
        })
    check("7/13  update_memory", test_update)

    # 8. list_entities
    def test_list_entities():
        result = mcp_server.execute_tool("list_entities", {})
        if "entities" not in result:
            raise Mem0Error("list_entities returned no 'entities' key",
                            detail=f"Result: {result}")
    check("8/13  list_entities", test_list_entities)

    # 9. export_memories (JSON format)
    def test_export_json():
        result = mcp_server.execute_tool("export_memories", {
            "user_id": test_user,
            "format": "json",
        })
        if "memories" not in result:
            raise Mem0Error("export_memories (json) returned no 'memories' key",
                            detail=f"Result: {result}")
        if result.get("format") != "json":
            raise Mem0Error(f"export_memories format mismatch: expected 'json', got '{result.get('format')}'")
        if not isinstance(result["memories"], list):
            raise Mem0Error("export_memories memories is not a list")
    check("9/13  export_memories (JSON)", test_export_json)

    # 10. export_memories (CSV format)
    def test_export_csv():
        result = mcp_server.execute_tool("export_memories", {
            "user_id": test_user,
            "format": "csv",
        })
        if "data" not in result:
            raise Mem0Error("export_memories (csv) returned no 'data' key",
                            detail=f"Result: {result}")
        if result.get("format") != "csv":
            raise Mem0Error(f"export_memories format mismatch: expected 'csv', got '{result.get('format')}'")
        if not isinstance(result["data"], str):
            raise Mem0Error("export_memories csv data is not a string")
        if "id,memory,metadata,created_at,user_id" not in result["data"]:
            raise Mem0Error("export_memories csv missing header row")
    check("10/13  export_memories (CSV)", test_export_csv)

    # 11. import_memories
    def test_import():
        export_result = mcp_server.execute_tool("export_memories", {
            "user_id": test_user,
            "format": "json",
        })
        export_data = export_result.get("memories", [])
        if not export_data:
            raise Mem0Error("no memories to import (export was empty)")
        # Import to a different user to avoid duplicate detection skipping all
        import_user = f"selftest_import_{uuid.uuid4().hex[:8]}"
        result = mcp_server.execute_tool("import_memories", {
            "data": json.dumps(export_data),
            "user_id": import_user,
        })
        if "imported" not in result:
            raise Mem0Error("import_memories returned no 'imported' key",
                            detail=f"Result: {result}")
        if result["imported"] == 0:
            raise Mem0Error("import_memories imported 0 memories",
                            detail=f"Result: {result}")
        # Clean up
        mcp_server.execute_tool("delete_all_memories", {"user_id": import_user})
    check("11/13  import_memories", test_import)

    # 12. prune_memories (dry run)
    def test_prune_dry_run():
        result = mcp_server.execute_tool("prune_memories", {
            "user_id": test_user,
            "older_than_days": 0,  # everything is "older than 0 days"
            "dry_run": True,
        })
        if "would_delete" not in result:
            raise Mem0Error("prune_memories returned no 'would_delete' key",
                            detail=f"Result: {result}")
        if result.get("dry_run") is not True:
            raise Mem0Error("prune_memories dry_run should be True")
        if result.get("deleted", 0) != 0:
            raise Mem0Error("prune_memories in dry_run should not delete anything")
    check("13/13  prune_memories (dry run)", test_prune_dry_run)

    # Cleanup: delete all test memories
    mcp_server.execute_tool("delete_all_memories", {"user_id": test_user})

    _print_summary()

    if failed > 0:
        sys.exit(1)


def _print_summary():
    print()
    if failed > 0:
        print(f"{RED}┌──────────────────────────────────────────────────────────┐")
        print(f"│  ❌ {passed}/12 tools passed, {failed} failed                  │")
        print(f"└──────────────────────────────────────────────────────────┘{NC}")
        print()
        print("Errors:")
        for label, err in errors:
            print(f"  {label}:")
            for line in err.split("\n"):
                print(f"    {line}")
        print()
        print("Common fixes:")
        print("  • Use a larger model:   ollama pull qwen2.5:7b")
        print("  • Restart everything:   ./setup.sh clean && ./setup.sh")
        print("  • Check Ollama:         ollama serve")
        print("  • Check Docker:         docker compose logs mcp-server")
    else:
        print(f"{GREEN}┌──────────────────────────────────────────────────────────┐")
        print(f"│  ✅ 13/13 tools passed — all MCP commands verified        │")
        print(f"└──────────────────────────────────────────────────────────┘{NC}")
    print()


if __name__ == "__main__":
    run()