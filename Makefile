.PHONY: up down logs health update venv test test-container test-all test-full

up:       ## Build + start the single container (Qdrant + MCP server)
	./setup.sh

update:   ## Rebuild image with latest code, preserve all data
	./setup.sh update

down:     ## Stop the container (keeps data)
	docker compose down

logs:     ## Tail container logs
	docker compose logs -f mem0-local

health:   ## Check server health
	curl -fs http://localhost:8765/health | python3 -m json.tool

shell:    ## Shell into the container
	docker compose exec mem0-local bash

# ── Tests ────────────────────────────────────────────────────────────────────
# Everything runs INSIDE the image: same python, same deps, same baked models,
# no host venv to install. That is the whole point - `make test` behaves
# identically on a fresh clone, a CI runner, and your laptop.

test test-container:  ## Run the full test suite inside the container (builds if needed)
	docker compose build -q mem0-local
	docker compose run --rm --entrypoint python3 mem0-local -m pytest tests/ -q

test-full: ## Same, with per-test output
	docker compose run --rm --entrypoint python3 mem0-local -m pytest tests/ -v

venv:     ## OPTIONAL host venv for fast iteration without Docker (needs uv)
	@if [ ! -x .venv/bin/python ]; then \
	  if command -v uv >/dev/null 2>&1; then \
	    echo "Creating .venv and installing dependencies (optional path)..."; \
	    uv venv .venv --python 3.11 >/dev/null && \
	    uv pip install --python .venv/bin/python -q -r requirements.txt && \
	    echo "✓ .venv ready"; \
	  else \
	    echo "✗ uv not found (brew install uv). Not needed: use 'make test'."; \
	    exit 1; \
	  fi; \
	fi

PY = $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
OFFLINE_TESTS = tests/test_chunking.py tests/test_no_llm.py tests/test_execute_tool_units.py \
                tests/test_setup_and_packaging.py tests/test_reliability_contract.py

test-host: venv ## Offline-only tests on the HOST (fast, no Docker; needs .venv)
	@$(PY) -c "import pytest" 2>/dev/null || { \
	  echo "pytest missing for $(PY) - run 'make venv' or just use 'make test'"; exit 1; }
	$(PY) -m pytest $(OFFLINE_TESTS) -q

test-all: test ## Local (container) suite, then the same suite again for good measure
	@echo ""
	@echo "Suite ran in-container. For host-only iteration: make test-host"

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