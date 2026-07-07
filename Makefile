.PHONY: up down logs test health clean docker export import import-docs import-docs-dry shell pull-model

up:       ## Start everything (native Ollama + Docker containers) — one command
	./setup.sh

docker:   ## All-in-Docker mode (Ollama in container, CPU emulation) — NOT IMPLEMENTED YET
	@echo "This mode is not implemented. Ollama always runs on the host."
	@echo "Use: ./setup.sh"

down:     ## Stop containers (keeps data)
	docker compose down

logs:     ## Tail MCP server logs
	docker compose logs -f mcp-server

test:    ## Run test suite inside container
	./setup.sh test

health:   ## Check server health
	curl http://localhost:8765/health | python3 -m json.tool

clean:    ## Stop everything, delete all data + models
	./setup.sh clean

shell:    ## Shell into the MCP server container
	docker compose exec mcp-server bash

pull-model: ## Pull a new Ollama model (usage: make pull-model MODEL=qwen3.5:9b)
	ollama pull $(MODEL)
	echo "Update .env: MEM0_LLM_MODEL=$(MODEL)"
	docker compose restart mcp-server

export:   ## Export memories to JSON (usage: make export [USER=dev] [FORMAT=json])
	@USER_ID=$${USER:-dev}; FORMAT=$${FORMAT:-json}; \
	curl -s -X POST http://localhost:8765/mcp \
	  -H 'Content-Type: application/json' \
	  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"export_memories\",\"arguments\":{\"user_id\":\"$$USER_ID\",\"format\":\"$$FORMAT\"}}}" \
	| python3 -c "import sys,json; r=json.load(sys.stdin); print(r['result']['content'][0]['text'])" 2>/dev/null \
	|| echo 'Server not running. Start with: make up'

import:    ## Import memories from JSON file (usage: make import FILE=backup.json [USER=dev])
	@if [ -z "$$FILE" ]; then echo 'Usage: make import FILE=backup.json [USER=dev]'; exit 1; fi
	@USER_ID=$${USER:-dev}; \
	DATA=$$(cat "$$FILE" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))"); \
	curl -s -X POST http://localhost:8765/mcp \
	  -H 'Content-Type: application/json' \
	  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"import_memories\",\"arguments\":{\"data\":$$DATA,\"user_id\":\"$$USER_ID\"}}}" \
	| python3 -c "import sys,json; r=json.load(sys.stdin); print(r['result']['content'][0]['text'])" 2>/dev/null \
	|| echo 'Server not running. Start with: make up'

import-docs: ## Import markdown docs (usage: make import-docs DOCS=/path/to/docs USER=myproject)
	python3 import_docs.py $(DOCS) --user-id $(USER)

import-docs-dry: ## Preview what would be imported (usage: make import-docs-dry DOCS=/path USER=myproject)
	python3 import_docs.py $(DOCS) --user-id $(USER) --dry-run