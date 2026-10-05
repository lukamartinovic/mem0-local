#!/usr/bin/env bash
set -euo pipefail

# mem0-local — one-command setup
#
#   ./setup.sh          → build + start the single container
#   ./setup.sh update   → rebuild image with latest code, preserve all data
#
# No Ollama anywhere: fact inference is the calling agent's job; embeddings
# run in-container via fastembed. Qdrant runs inside the same container.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-start}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
info()  { echo -e "${GREEN}✓${NC} $1"; }
warn()  { echo -e "${YELLOW}⚠${NC} $1"; }
err()   { echo -e "${RED}✗${NC} $1"; }

# ── Check container engine (docker or podman-as-docker) ─────────────────────
PODMAN_BIN="$(command -v podman 2>/dev/null || true)"
DOCKER_BIN="$(command -v docker 2>/dev/null || true)"
if [ -z "$DOCKER_BIN" ] && [ -n "$PODMAN_BIN" ]; then
  docker() { podman "$@"; }        # shim `docker` → `podman` for this script
fi
if ! { [ -n "$DOCKER_BIN" ] || [ -n "$PODMAN_BIN" ]; } && false; then :; fi

if command -v docker >/dev/null 2>&1 && docker info &>/dev/null; then
  info "Docker daemon is running"
elif command -v podman >/dev/null 2>&1 && podman info &>/dev/null; then
  info "Podman is running (docker-compatible)"
else
  err "Docker not installed or daemon not running."
  echo "  Install Docker Desktop: https://desktop.docker.com/mac/main/arm64/Docker.dmg"
  echo "  Or use podman:          brew install podman docker-compose"
  exit 1
fi

# ── Update mode ──────────────────────────────────────────────────────────────
# Rebuilds the image with the latest code while preserving the Qdrant volume.
if [ "$MODE" = "update" ]; then
  echo "Updating mem0-local (preserving all data)..."
  warn "Rebuilding Docker image (this picks up code changes)..."
  docker compose build
  info "Image rebuilt"
  warn "Restarting container..."
  docker compose up -d
  info "Container started"
  echo ""
  warn "Waiting for MCP server to be ready..."
  for i in $(seq 1 60); do
    if curl -s http://localhost:8765/health >/dev/null 2>&1; then
      break
    fi
    sleep 2
    echo -n "."
  done
  echo ""
  if curl -s http://localhost:8765/health >/dev/null 2>&1; then
    info "Ready! Update complete - all memories preserved."
    curl -s http://localhost:8765/health | python3 -m json.tool 2>/dev/null
  else
    err "MCP server didn't become healthy in time."
    echo "  Check logs: docker compose logs mem0-local"
    exit 1
  fi
  exit 0
fi

# ── Start ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  mem0-local — single container, no Ollama        ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
info "Extraction LLM: none - the calling agent does its own fact inference"
info "Embeddings: fastembed (ONNX, in-container, 768 dims)"

# First-run guidance: the model downloads on first embed call, not at build.
if [ ! -f .downloaded ]; then
  warn "First container start will download the embedding model (~160 MB, one time)."
else
  info "Embedding model cached - starting fast"
fi

docker compose up -d
info "Container started (Qdrant + MCP server in one container)"

echo ""
warn "Waiting for MCP server to be ready (first start downloads the embed model)..."
for i in $(seq 1 120); do
  if curl -s http://localhost:8765/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
  echo -n "."
done
echo ""

if curl -s http://localhost:8765/health >/dev/null 2>&1; then
  echo ""
  info "Ready! Health check passed."
  echo ""
  echo "┌──────────────────────────────────────────────────────────┐"
  echo "│  mem0-local is live!                                     │"
  echo "│                                                          │"
  echo "│  MCP endpoint:  http://localhost:8765/mcp                │"
  echo "│  Health check:  curl http://localhost:8765/health        │"
  echo "│  Qdrant dash:   http://localhost:6333/dashboard          │"
  echo "│                                                          │"
  echo "│  ADD TO YOUR AGENT'S SYSTEM PROMPT:                      │"
  echo "│  This server has NO extraction LLM. When storing         │"
  echo "│  memories, extract facts yourself and call               │"
  echo "│  add_raw_memory once per concise fact (10-30 words).     │"
  echo "│  Use add_verbatim only for bulk document imports.        │"
  echo "│                                                          │"
  echo "│  Import docs:  python3 import_docs.py /path/to/docs      │"
  echo "│  Update code:  ./setup.sh update                         │"
  echo "│  Stop all:     docker compose down (data preserved)      │"
  echo "└──────────────────────────────────────────────────────────┘"
  echo ""
  curl -s http://localhost:8765/health | python3 -m json.tool 2>/dev/null
else
  err "mem0-local didn't become healthy in time."
  echo ""
  echo "  Check logs with: docker compose logs mem0-local"
  exit 1
fi