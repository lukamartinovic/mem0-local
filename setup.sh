#!/usr/bin/env bash
set -euo pipefail

# mem0-local — one-command setup
#
#   ./setup.sh         → ensure Ollama running on host, pick model, pull models, start Qdrant + MCP in Docker
#   ./setup.sh test    → run tests inside the container
#   ./setup.sh update  → rebuild Docker image with latest code, preserve all data
#   ./setup.sh clean   → stop everything, delete all data

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-start}"  # start | test | update | clean

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'
info()  { echo -e "${GREEN}✓${NC} $1"; }
warn()  { echo -e "${YELLOW}⚠${NC} $1"; }
err()   { echo -e "${RED}✗${NC} $1"; }

# ── Clean ────────────────────────────────────────────────────────────────────
if [ "$MODE" = "clean" ]; then
  echo "Stopping everything and deleting data..."
  docker compose down -v 2>/dev/null || true
  info "Clean. All containers stopped, all data deleted."
  exit 0
fi

# ── Update ───────────────────────────────────────────────────────────────────
# Rebuilds the Docker image with the latest code while preserving the Qdrant
# volume (all memories). Use this after pulling code updates or changing
# mcp_server.py. Does NOT delete data — contrast with ./setup.sh clean.
if [ "$MODE" = "update" ]; then
  echo "Updating mem0-local (preserving all data)..."
  echo ""
  # Stop containers but keep volumes
  docker compose down 2>/dev/null || true
  info "Containers stopped (volumes preserved)"
  # Rebuild the MCP server image with latest code
  warn "Rebuilding Docker image (this picks up code changes)..."
  docker compose build mcp-server
  info "Image rebuilt"
  # Start everything back up
  # Load .env for any config changes
  if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs) 2>/dev/null || true
  fi
  docker compose up -d
  info "Containers started"
  # Wait for health
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
    info "Ready! Update complete — all memories preserved."
    curl -s http://localhost:8765/health | python3 -m json.tool 2>/dev/null
  else
    err "MCP server didn't become healthy in time."
    echo "  Check logs: docker compose logs mcp-server"
    exit 1
  fi
  exit 0
fi

# ── Test ─────────────────────────────────────────────────────────────────────
if [ "$MODE" = "test" ]; then
  echo "Running tests inside container..."
  # Rebuild image first to pick up any code changes since last build.
  # Without this, tests run against stale code in the cached Docker image.
  warn "Rebuilding Docker image to pick up latest code..."
  docker compose build mcp-server 2>&1 | tail -3
  # Run via entrypoint.sh which: waits for Ollama+Qdrant, pulls models,
  # checks dimensions, runs selftest (creates collections), then runs pytest.
  docker compose run --rm -e RUN_TESTS=1 mcp-server ./entrypoint.sh
  exit $?
fi

# ── Check Docker ────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  err "Docker not installed."
  echo "  Install Docker Desktop:"
  echo "    macOS:  https://desktop.docker.com/mac/main/arm64/Docker.dmg"
  echo "    Linux:  curl -fsSL https://get.docker.com | sh"
  exit 1
fi
if ! docker info &>/dev/null; then
  err "Docker daemon not running."
  echo "  Fix: Start Docker Desktop (macOS) or 'sudo systemctl start docker' (Linux)"
  exit 1
fi
info "Docker is running"

# ── Load .env ────────────────────────────────────────────────────────────────
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs) 2>/dev/null || true
fi
EMBED_MODEL="${MEM0_EMBED_MODEL:-nomic-embed-text}"

# ── External Ollama mode ─────────────────────────────────────────────────────
# If MEM0_OLLAMA_URL is set to something other than localhost/host.docker.internal,
# we use it as an external provider (e.g., Ollama inside a company VPN).
# In this mode, we skip all local Ollama install/start/model-pull logic.
EXTERNAL_OLLAMA_URL="${MEM0_OLLAMA_URL:-}"

is_external_ollama() {
  # External if URL is set AND not pointing at localhost or host.docker.internal
  if [ -n "$EXTERNAL_OLLAMA_URL" ]; then
    case "$EXTERNAL_OLLAMA_URL" in
      *localhost*|*host.docker.internal*|*127.0.0.1*) return 1 ;;
      *) return 0 ;;
    esac
  fi
  return 1
}

if is_external_ollama; then
  echo ""
  echo "╔══════════════════════════════════════════════════╗"
  echo "║  mem0-local — external Ollama + Docker containers ║"
  echo "╚══════════════════════════════════════════════════╝"
  echo ""
  info "External Ollama provider: ${CYAN}${EXTERNAL_OLLAMA_URL}${NC}"
  echo ""

  # Verify the external Ollama is reachable
  if curl -s "${EXTERNAL_OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    info "External Ollama is reachable"
    # Show available models
    EXTERNAL_MODELS=$(curl -s "${EXTERNAL_OLLAMA_URL}/api/tags" 2>/dev/null | \
      python3 -c "import sys,json; models=[m['name'] for m in json.load(sys.stdin).get('models',[])]; print(', '.join(models) if models else '(none)')" 2>/dev/null)
    if [ -n "$EXTERNAL_MODELS" ]; then
      echo "  Available models: ${EXTERNAL_MODELS}"
    fi
  else
    warn "Cannot reach external Ollama at ${EXTERNAL_OLLAMA_URL}"
    echo "  The server will still start, but memory operations will fail until"
    echo "  Ollama is reachable. If you're on a VPN, make sure you're connected."
    echo ""
    echo "  To check connectivity:"
    echo "    curl ${EXTERNAL_OLLAMA_URL}/api/tags"
  fi
  echo ""

  # Validate that the configured models are available on the external provider
  LLM_MODEL="${MEM0_LLM_MODEL:-qwen2.5:7b}"

  # Check if LLM model is available on external Ollama
  if curl -s "${EXTERNAL_OLLAMA_URL}/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = [m['name'] for m in data.get('models', [])]
sys.exit(0 if any('${LLM_MODEL}' in m for m in models) else 1)
" 2>/dev/null; then
    info "LLM model available: ${LLM_MODEL}"
  else
    warn "LLM model '${LLM_MODEL}' not found on external Ollama."
    echo "  Available models will be listed on startup, or check:"
    echo "    curl ${EXTERNAL_OLLAMA_URL}/api/tags"
    echo "  To use a different model, update .env: MEM0_LLM_MODEL=<model>"
  fi

  # Check if embed model is available
  if curl -s "${EXTERNAL_OLLAMA_URL}/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = [m['name'] for m in data.get('models', [])]
sys.exit(0 if any('${EMBED_MODEL}' in m for m in models) else 1)
" 2>/dev/null; then
    info "Embedder model available: ${EMBED_MODEL}"
  else
    warn "Embedder model '${EMBED_MODEL}' not found on external Ollama."
    echo "  To use a different embedder, update .env: MEM0_EMBED_MODEL=<model>"
  fi

  # ── Start Qdrant + MCP server in Docker (pointing at external Ollama) ───────
  # Don't override MEM0_OLLAMA_URL — let docker-compose use the .env value
  export MEM0_OLLAMA_URL="${EXTERNAL_OLLAMA_URL}"
  docker compose up -d
  info "Containers started (Qdrant + MCP server, pointing at external Ollama)"

  # ── Wait for health check ──────────────────────────────────────────────────
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
    echo ""
    info "Ready! Health check passed."
    echo ""
    echo "┌──────────────────────────────────────────────────────────┐"
    echo "│  mem0-local is live!                                     │"
    echo "│                                                          │"
    echo "│  MCP endpoint:  http://localhost:8765/mcp                │"
    echo "│  Health check:  curl http://localhost:8765/health        │"
    echo "│  Ollama:        ${EXTERNAL_OLLAMA_URL}                    "
    echo "│  Model:         $LLM_MODEL                                "
    echo "│                                                          │"
    echo "│  Add to your IDE's MCP config:                           │"
    echo '│  {"mcpServers":{"mem0-local":{                          │'
    echo '│    "type":"http",                                        │'
    echo '│    "url":"http://localhost:8765/mcp"                     │'
    echo '  }}}                                                      │'
    echo "│                                                          │"
    echo "│  Run tests:   ./setup.sh test                            │"
    echo "│  Stop all:    ./setup.sh clean                           │"
    echo "└──────────────────────────────────────────────────────────┘"
    echo ""
    curl -s http://localhost:8765/health | python3 -m json.tool 2>/dev/null
  else
    err "MCP server didn't become healthy in time."
    echo ""
    echo "  This could be:"
    echo "    • External Ollama not reachable (VPN disconnected?)"
    echo "    • Models not available on external Ollama"
    echo "    • Docker issue: 'docker compose ps' and 'docker compose logs mcp-server'"
    echo ""
    echo "  Check: curl ${EXTERNAL_OLLAMA_URL}/api/tags"
    echo "  Logs:   docker compose logs mcp-server"
    exit 1
  fi
  exit 0
fi

# ── Local Ollama mode (default) ─────────────────────────────────────────────
# If we get here, MEM0_OLLAMA_URL is not set or points at localhost.
# Use the original local Ollama flow.

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  mem0-local — native Ollama + Docker containers  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

if ! command -v ollama &>/dev/null; then
  warn "Ollama not installed. Installing now..."
  if command -v brew &>/dev/null; then
    brew install ollama
  elif command -v curl &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    err "Could not install Ollama. Install manually from https://ollama.com"
    exit 1
  fi
  info "Ollama installed"
else
  info "Ollama already installed ($(ollama --version 2>/dev/null | head -1))"
fi

# ── Start Ollama if not running ─────────────────────────────────────────────
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  info "Ollama already running"
else
  warn "Starting Ollama..."
  ollama serve > /dev/null 2>&1 &
  for i in $(seq 1 15); do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    info "Ollama started"
  else
    err "Ollama failed to start."
    echo "  Fix: Run 'ollama serve' manually to see the error."
    echo "  Common issues:"
    echo "    • Port 11434 in use: 'lsof -iTCP:11434' and kill the process"
    echo "    • Ollama not in PATH: 'which ollama'"
    exit 1
  fi
fi

# ── Detect available RAM ───────────────────────────────────────────────────
detect_ram_gb() {
  local ram_gb
  # macOS
  if command -v sysctl &>/dev/null && sysctl -n hw.memsize >/dev/null 2>&1; then
    ram_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
  # Linux
  elif [ -f /proc/meminfo ]; then
    local kb
    kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    ram_gb=$(( kb / 1048576 ))
  else
    ram_gb=8
  fi
  echo "$ram_gb"
}

RAM_GB=$(detect_ram_gb)

# Detect CPU model for display
CPU_INFO="Unknown"
if command -v sysctl &>/dev/null; then
  CPU_INFO=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
fi

echo "Detected: ${CYAN}${RAM_GB}GB${NC} unified memory (${CPU_INFO})"
echo ""

# ── Model selection ─────────────────────────────────────────────────────────
# Models that support tool calling (required by mem0 for JSON extraction)
# Format: "name:size_gb description min_ram_gb"

MODELS=(
  "qwen2.5:3b|1.9|Fastest, basic JSON extraction|8"
  "qwen2.5:7b|4.7|Reliable JSON extraction (recommended)|16"
  "qwen3.5:9b|5.5|Better quality, still fast|32"
  "gemma4:12b|8.0|High quality extraction|32"
  "qwen3.5:27b|16.0|Best quality, slower|64"
)

DEFAULT_MODEL="qwen2.5:7b"

echo "Select an LLM model for memory extraction:"
echo ""

idx=1
for model_entry in "${MODELS[@]}"; do
  IFS='|' read -r name size desc min_ram <<< "$model_entry"
  marker=""
  if [ "$RAM_GB" -lt "$min_ram" ]; then
    marker="${RED}⚠ needs ${min_ram}GB${NC}"
  else
    marker="${GREEN}✓ ${min_ram}GB+ RAM${NC}"
  fi
  if [ "$name" = "$DEFAULT_MODEL" ]; then
    echo -e "  ${CYAN}${idx}.${NC} ${name}  ${size}GB   ${desc}  ${marker} ${YELLOW}(default)${NC}"
  else
    echo -e "  ${CYAN}${idx}.${NC} ${name}  ${size}GB   ${desc}  ${marker}"
  fi
  idx=$((idx + 1))
done

echo ""
read -rp "Choice [1-5] (default 2): " model_choice

case "${model_choice:-2}" in
  1) LLM_MODEL="qwen2.5:3b" ;;
  2) LLM_MODEL="qwen2.5:7b" ;;
  3) LLM_MODEL="qwen3.5:9b" ;;
  4) LLM_MODEL="gemma4:12b" ;;
  5) LLM_MODEL="qwen3.5:27b" ;;
  *) LLM_MODEL="qwen2.5:7b" ;;
esac

info "Selected: ${CYAN}${LLM_MODEL}${NC}"

# Save to .env (preserve existing customizations)
if [ ! -f .env ]; then
  echo "MEM0_LLM_MODEL=${LLM_MODEL}" > .env
  echo "MEM0_EMBED_MODEL=${EMBED_MODEL}" >> .env
  echo "MEM0_DEFAULT_USER_ID=dev" >> .env
else
  # Update only the model line, preserve everything else
  if grep -q "^MEM0_LLM_MODEL=" .env; then
    sed -i.bak "s/^MEM0_LLM_MODEL=.*/MEM0_LLM_MODEL=${LLM_MODEL}/" .env
    rm -f .env.bak
  else
    echo "MEM0_LLM_MODEL=${LLM_MODEL}" >> .env
  fi
fi
info "Config saved to .env"

# ── Pull models if not present ──────────────────────────────────────────────
check_model() {
  local model="$1"
  curl -s http://localhost:11434/api/tags 2>/dev/null | \
    python3 -c "import sys,json; print(1 if any('$model' in m['name'] for m in json.load(sys.stdin).get('models',[])) else 0)" 2>/dev/null
}

if [ "$(check_model "$LLM_MODEL")" = "1" ]; then
  info "LLM model ready: $LLM_MODEL"
else
  warn "Pulling $LLM_MODEL (first time only, may take a while)..."
  ollama pull "$LLM_MODEL"
  info "LLM model pulled: $LLM_MODEL"
fi

if [ "$(check_model "$EMBED_MODEL")" = "1" ]; then
  info "Embedder model ready: $EMBED_MODEL"
else
  warn "Pulling $EMBED_MODEL..."
  ollama pull "$EMBED_MODEL"
  info "Embedder model pulled: $EMBED_MODEL"
fi

# ── Start Qdrant + MCP server in Docker ──────────────────────────────────────
export MEM0_OLLAMA_URL="http://host.docker.internal:11434"
docker compose up -d
info "Containers started (Qdrant + MCP server, pointing at host Ollama)"

# ── Wait for health check ────────────────────────────────────────────────────
echo ""
warn "Waiting for MCP server to be ready (first request loads the model, ~30-60s)..."
for i in $(seq 1 60); do
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
  echo "│  Model:         $LLM_MODEL                                "
  echo "│                                                          │"
  echo "│  Add to your IDE's MCP config:                           │"
  echo '│  {"mcpServers":{"mem0-local":{                          │'
  echo '│    "type":"http",                                        │'
  echo '│    "url":"http://localhost:8765/mcp"                     │'
  echo '│  }}}                                                      │'
  echo "│                                                          │"
  echo "│  Import docs:  python3 import_docs.py /path/to/docs --user-id dev"
  echo "│  Run tests:   ./setup.sh test                            │"
  echo "│  Stop all:    ./setup.sh clean                           │"
  echo "└──────────────────────────────────────────────────────────┘"
  echo ""
  curl -s http://localhost:8765/health | python3 -m json.tool 2>/dev/null
else
  err "MCP server didn't become healthy in time."
  echo ""
  echo "  This could be:"
  echo "    • Model still loading (first run pulls ~4GB, can take 10+ min)"
  echo "    • Ollama crashed: 'ollama serve' and check for errors"
  echo "    • Docker issue: 'docker compose ps' and 'docker compose logs mcp-server'"
  echo ""
  echo "  Check logs with: docker compose logs mcp-server"
  exit 1
fi