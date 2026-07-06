.PHONY: up down logs test health clean docker

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

import:    ## Import markdown docs (usage: make import DOCS=/path/to/docs USER=myproject)
	python3 import_docs.py $(DOCS) --user-id $(USER)

import-dry: ## Preview what would be imported (usage: make import-dry DOCS=/path USER=myproject)
	python3 import_docs.py $(DOCS) --user-id $(USER) --dry-run