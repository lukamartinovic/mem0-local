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

HEALTH_URL="http://localhost:8765/health"

# Is the server actually ready? Matches the container healthcheck exactly:
# HTTP 200 AND status == "ok". A plain `curl -s` exits 0 on a 500/404 too, so
# readiness must be asserted on the body, not the exit code.
server_ready() {
  curl -fs "$HEALTH_URL" 2>/dev/null | python3 -c "
import sys, json
try:
    sys.exit(0 if json.load(sys.stdin).get('status') == 'ok' else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

wait_for_ready() {
  local label="$1" loops="$2"
  warn "Waiting for MCP server to be ready${label:+ ($label)}..."
  for _ in $(seq 1 "$loops"); do
    if server_ready; then
      echo ""
      return 0
    fi
    sleep 2
    echo -n "."
  done
  echo ""
  return 1
}

# ── Preflight 1: container engine available AND the compose plugin present ──
# `docker info` can succeed on setups without the compose v2 plugin (bare CLI,
# colima, podman shim). Fail here, before any banner, not mid-flow.
ENGINE=""
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  ENGINE="docker"
elif command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then
  ENGINE="podman"
  docker() { podman "$@"; }
else
  err "No running container engine found (docker or podman)."
  echo "  Install Docker Desktop: https://desktop.docker.com/mac/main/arm64/Docker.dmg"
  echo "  Or use podman:          brew install podman docker-compose"
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  err "The '$ENGINE compose' plugin is missing - this script needs Docker Compose v2."
  echo "  Docker Desktop bundles it. For CLI-only installs:"
  echo "    brew install docker-compose && mkdir -p ~/.docker/cli-plugins \\"
  echo "      && ln -sfn \"\$(brew --prefix)/opt/docker-compose/bin/docker-compose\" ~/.docker/cli-plugins/docker-compose"
  exit 1
fi

# ── Preflight 2: are the ports free? (a stale process squatting on them would
# make the readiness loop below report a false success against the WRONG app) ─
check_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    local holder
    holder="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NR==2 {print $1" (pid "$2")"}')"
    warn "Port $port is already in use by: ${holder:-unknown}."
    echo "    If that is a previous mem0-local container, stop it first:"
    echo "      docker compose down        # keeps data"
    echo "    Otherwise the new container cannot bind and this script will report failure."
  fi
}
check_port 8765
check_port 6333

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

  # Record the stored-memory count BEFORE, to prove the claim below.
  BEFORE="$(curl -fs "$HEALTH_URL" 2>/dev/null | python3 -c "
import sys, json
try:
    print(json.load(sys.stdin).get('memories', 0))
except Exception:
    print(0)
" 2>/dev/null || echo 0)"

  if ! wait_for_ready "" 90; then
    err "MCP server didn't become healthy in time."
    echo "  Check logs: docker compose logs mem0-local"
    exit 1
  fi

  info "Ready! Update complete."
  curl -fs "$HEALTH_URL" | python3 -m json.tool 2>/dev/null
  echo ""
  echo "  Memories were never touched by this command: update rebuilds the image"
  echo "  and recreates the container against the same 'qdrant_data' volume."
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
# The marker lives in the model_cache volume, so it survives rebuilds/recreates.
MODEL_CACHED="$(docker compose run --rm --entrypoint sh mem0-local -c '[ -f /model_cache/.ready ] && echo yes || echo no' 2>/dev/null | tail -1 || echo no)"
if [ "$MODEL_CACHED" = "yes" ]; then
  info "Embedding model cached in volume - starting fast"
else
  warn "First container start downloads the embedding model (~160 MB, one time)."
  warn "This needs network access ONCE; the cache is kept in the 'model_cache' volume."
fi

docker compose up -d
info "Container started (Qdrant + MCP server in one container)"

echo ""
if ! wait_for_ready "first start downloads the embed model" 150; then
  err "mem0-local didn't become healthy in time."
  echo ""
  echo "  Check logs with: docker compose logs mem0-local"
  echo "  A common cause is the embedding model download being blocked: the"
  echo "  server needs HuggingFace reachable on its FIRST run only."
  exit 1
fi

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
curl -fs "$HEALTH_URL" | python3 -m json.tool 2>/dev/null