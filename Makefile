.PHONY: up down logs health update export import import-docs import-docs-dry

up:       ## Build + start the single container (Qdrant + MCP server, no Ollama)
	./setup.sh

update:   ## Rebuild image with latest code, preserve all data
	./setup.sh update

down:     ## Stop the container (keeps data)
	docker compose down

logs:     ## Tail container logs
	docker compose logs -f mem0-local

health:   ## Check server health
	curl http://localhost:8765/health | python3 -m json.tool

shell:    ## Shell into the container
	docker compose exec mem0-local bash

PY = $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
OFFLINE_TESTS = tests/test_chunking.py tests/test_no_llm.py tests/test_execute_tool_units.py \
                tests/test_setup_and_packaging.py tests/test_reliability_contract.py

venv:     ## Create/bootstraps the local .venv used by `make test` (needs uv)
	@if [ ! -x .venv/bin/python ]; then \
	  echo "Creating .venv (python 3.11) and installing dependencies..."; \
	  if command -v uv >/dev/null 2>&1; then \
	    uv venv .venv --python 3.11 >/dev/null && \
	    uv pip install --python .venv/bin/python -q -r requirements.txt && \
	    echo "✓ .venv ready"; \
	  else \
	    echo "✗ uv not found. Install it: brew install uv"; \
	    echo "  (or skip the host venv entirely: make test-in-container)"; \
	    exit 1; \
	  fi; \
	fi

test: venv     ## Fast LOCAL tests: pure-Python, no Docker/services (seconds)
	@$(PY) -c "import pytest" 2>/dev/null || { \
	  echo "pytest missing for $(PY). Install it with one of:"; \
	  echo "  uv pip install --python .venv/bin/python pytest   (repo venv)"; \
	  echo "  python3 -m pip install --user pytest              (system python)"; \
	  echo "  make test-in-container                            (no host python needed)"; \
	  exit 1; }
	$(PY) -m pytest $(OFFLINE_TESTS) -v

test-in-container: ## Full suite inside the container - NO host python needed
	docker compose run --rm mem0-local pytest tests/ -v

test-all: ## Unit locally (all offline tiers) + full suite in container
	$(PY) -m pytest $(OFFLINE_TESTS) -q \
	  && docker compose run --rm mem0-local pytest tests/ -q

export:   ## Export memories to JSON (usage: make export USER=dev)
	@USER_ID=$$(echo "$(USER)" | sed 's/^$$/dev/'); \
	curl -s -X POST http://localhost:8765/mcp \
	  -H 'Content-Type: application/json' \
	  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"export_memories","arguments":{"user_id":"'"$$USER_ID"'","format":"json"}}}' \
	| python3 -c "import sys,json; r=json.load(sys.stdin); print(r['result']['content'][0]['text'])" 2>/dev/null \
	|| echo 'Server not running. Start with: make up'

import:   ## Import memories from JSON file (usage: make import FILE=backup.json USER=dev)
	@if [ -z "$(FILE)" ]; then echo 'Usage: make import FILE=backup.json USER=dev'; exit 1; fi
	@USER_ID=$$(echo "$(USER)" | sed 's/^$$/dev/'); \
	DATA=$$(cat $(FILE) | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))"); \
	curl -s -X POST http://localhost:8765/mcp \
	  -H 'Content-Type: application/json' \
	  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"import_memories","arguments":{"data":"'"$$DATA"'","user_id":"'"$$USER_ID"'"}}}' \
	| python3 -c "import sys,json; r=json.load(sys.stdin); print(r['result']['content'][0]['text'])" 2>/dev/null \
	|| echo 'Server not running. Start with: make up'

import-docs: ## Import markdown docs (usage: make import-docs DOCS=/path/to/docs USER=myproject)
	python3 import_docs.py $(DOCS) --user-id $(USER)

import-docs-dry: ## Preview what would be imported (usage: make import-docs-dry DOCS=/path USER=myproject)
	python3 import_docs.py $(DOCS) --user-id $(USER) --dry-run