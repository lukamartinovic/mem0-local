#!/usr/bin/env python3
"""
Self-test: exercises all 9 MCP tools end-to-end.
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
    print("│  mem0-local self-test — exercising all 9 MCP tools       │")
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

    # 1. add_memory (string)
    def test_add_string():
        result = mcp_server.execute_tool("add_memory", {
            "content": "I prefer TypeScript over JavaScript and use ESLint.",
            "user_id": test_user,
        })
        results = result.get("results", []) if isinstance(result, dict) else result
        if not results:
            raise LLMExtractionError(
                "add_memory returned no results",
                detail=f"Result: {result}",
                fix="Check Ollama model can produce JSON output",
            )
    check("1/9  add_memory (string)", test_add_string)

    # 2. add_memory (conversation messages as JSON)
    def test_add_conversation():
        messages = json.dumps([
            {"role": "user", "content": "I love sci-fi movies but hate thrillers."},
            {"role": "assistant", "content": "Noted — will recommend sci-fi, avoid thrillers."},
        ])
        result = mcp_server.execute_tool("add_memory", {
            "content": messages,
            "user_id": test_user,
        })
        results = result.get("results", []) if isinstance(result, dict) else result
        if not results:
            raise LLMExtractionError(
                "add_memory returned no results for conversation input",
                detail=f"Result: {result}",
                fix="Check Ollama model can produce JSON output",
            )
    check("2/9  add_memory (conversation)", test_add_conversation)

    time.sleep(2)

    # 3. search_memories
    def test_search():
        result = mcp_server.execute_tool("search_memories", {
            "query": "language preference",
            "user_id": test_user,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        if len(results) == 0:
            raise Mem0Error("search returned 0 results after successful add",
                            detail="Expected results for 'language preference' query")
    check("3/9  search_memories", test_search)

    # 4. get_memories
    def test_get_all():
        result = mcp_server.execute_tool("get_memories", {
            "user_id": test_user,
            "limit": 10,
        })
        if not result:
            raise Mem0Error("get_memories returned empty",
                            detail="Expected at least 1 memory from previous add")
    check("4/9  get_memories", test_get_all)

    # 5. get_memory (by ID)
    test_memory_id = [None]
    def test_get_one():
        all_mems = mcp_server.execute_tool("get_memories", {
            "user_id": test_user,
            "limit": 10,
        })
        results = all_mems if isinstance(all_mems, list) else all_mems.get("results", [])
        if len(results) == 0:
            raise Mem0Error("no memories to test get_memory")
        mem = results[0]
        mem_id = mem.get("id") or mem.get("memory_id")
        if not mem_id:
            raise Mem0Error("memory has no id field",
                            detail=f"Memory object: {mem}")
        test_memory_id[0] = mem_id
        result = mcp_server.execute_tool("get_memory", {"memory_id": mem_id})
        if not result:
            raise Mem0Error(f"get_memory returned empty for id={mem_id}")
    check("5/9  get_memory (by ID)", test_get_one)

    # 6. update_memory
    def test_update():
        if not test_memory_id[0]:
            raise Mem0Error("no memory_id from previous test")
        mcp_server.execute_tool("update_memory", {
            "memory_id": test_memory_id[0],
            "content": "Updated: I now prefer Python over everything.",
        })
    check("6/9  update_memory", test_update)

    # 7. list_entities
    def test_list_entities():
        result = mcp_server.execute_tool("list_entities", {})
        if "entities" not in result:
            raise Mem0Error("list_entities returned no 'entities' key",
                            detail=f"Result: {result}")
    check("7/9  list_entities", test_list_entities)

    # 8. delete_memory
    def test_delete_one():
        if not test_memory_id[0]:
            raise Mem0Error("no memory_id from previous test")
        mcp_server.execute_tool("delete_memory", {
            "memory_id": test_memory_id[0],
        })
    check("8/9  delete_memory", test_delete_one)

    # 9. delete_all_memories
    def test_delete_all():
        mcp_server.execute_tool("delete_all_memories", {"user_id": test_user})
        result = mcp_server.execute_tool("search_memories", {
            "query": "anything",
            "user_id": test_user,
        })
        results = result if isinstance(result, list) else result.get("results", [])
        if len(results) > 0:
            raise Mem0Error(f"memories still exist after delete_all ({len(results)} found)")
    check("9/9  delete_all_memories", test_delete_all)

    _print_summary()

    if failed > 0:
        sys.exit(1)


def _print_summary():
    print()
    if failed > 0:
        print(f"{RED}┌──────────────────────────────────────────────────────────┐")
        print(f"│  ❌ {passed}/9 tools passed, {failed} failed                   │")
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
        print(f"│  ✅ 9/9 tools passed — all MCP commands verified          │")
        print(f"└──────────────────────────────────────────────────────────┘{NC}")
    print()


if __name__ == "__main__":
    run()