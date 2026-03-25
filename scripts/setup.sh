#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# tokagotchi MLX — Setup Script
#
# Installs the project, prepares the Apple Silicon inference path,
# builds the arena image when Docker is available, creates seed data,
# and initializes the git experiment repo.
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ------------------------------------------------------------------
# 1. Check Python version (>=3.11)
# ------------------------------------------------------------------
info "Checking Python version..."
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    error "Python not found. Please install Python 3.11+ and set PYTHON env var."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    error "Python 3.11+ required, found $PY_VERSION"
    exit 1
fi
info "Python $PY_VERSION — OK"

# ------------------------------------------------------------------
# 2. Install the project with the appropriate extras
# ------------------------------------------------------------------
info "Installing tokagotchi dependencies..."
"$PYTHON" -m pip install --upgrade pip
if [ "$(uname -s)" = "Darwin" ]; then
    "$PYTHON" -m pip install -e ".[mlx]"
else
    "$PYTHON" -m pip install -e ".[training]"
fi
info "Package installation complete"

# ------------------------------------------------------------------
# 3. Prepare the default local model
# ------------------------------------------------------------------
MODEL_ID="mlx-community/Qwen3-14B-4bit"
MODEL_DIR="./models/mlx-community-Qwen3-14B-4bit"

if [ "$(uname -s)" = "Darwin" ]; then
    info "Default MLX model: $MODEL_ID"
    if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
        info "Local MLX model cache already present at $MODEL_DIR"
    else
        warn "No local cache found for $MODEL_ID."
        warn "The model will be downloaded automatically the first time mlx_lm.server starts."
        warn "To prefetch manually: huggingface-cli download $MODEL_ID --local-dir $MODEL_DIR"
    fi
else
    warn "Non-Darwin host detected. This branch defaults to MLX serving on Apple Silicon."
    warn "You can still override config/model.provider if you want to use Ollama or vLLM."
fi

# ------------------------------------------------------------------
# 4. Build Docker arena image
# ------------------------------------------------------------------
info "Building Docker arena image..."
if command -v docker &>/dev/null; then
    if [ -f "docker/Dockerfile" ]; then
        docker build -t qwen-arena:latest -f docker/Dockerfile docker/
        info "Docker image qwen-arena:latest built"
    else
        warn "docker/Dockerfile not found — skipping Docker build"
    fi
else
    warn "Docker not found — skipping arena image build"
fi

# ------------------------------------------------------------------
# 5. Create initial seed data
# ------------------------------------------------------------------
info "Setting up seed data..."

# Ensure data directories exist
mkdir -p data/prompts data/curriculum data/checkpoints data/traces data/eval_results

# Copy seed files if not already present
if [ ! -f "data/prompts/seed.json" ]; then
    warn "data/prompts/seed.json not found — please create it manually"
fi
if [ ! -f "data/curriculum/seed_tasks.json" ]; then
    warn "data/curriculum/seed_tasks.json not found — please create it manually"
fi

# ------------------------------------------------------------------
# 6. Initialize git experiment repo
# ------------------------------------------------------------------
info "Initializing git experiment repository..."
if [ ! -d ".git" ]; then
    git init
    git add -A
    git commit -m "Initial project setup"
    info "Git repository initialized"
else
    info "Git repository already exists — skipping init"
fi

# ------------------------------------------------------------------
# 7. Verify setup
# ------------------------------------------------------------------
info "Running setup verification..."

"$PYTHON" -c "
import sys
checks = []

# Check core imports
try:
    from src.config import load_config
    checks.append(('config loading', True))
except Exception as e:
    checks.append(('config loading', False))

try:
    from src.models import TaskSpec, PromptGenome, Trajectory
    checks.append(('data models', True))
except Exception as e:
    checks.append(('data models', False))

try:
    from src.loop1_gepa import GEPAEngine
    checks.append(('loop1 (GEPA)', True))
except Exception as e:
    checks.append(('loop1 (GEPA)', False))

try:
    from src.loop2_distill import TraceCollector, SFTLauncher
    checks.append(('loop2 (distill)', True))
except Exception as e:
    checks.append(('loop2 (distill)', False))

try:
    from src.loop3_rl import RLRunner
    checks.append(('loop3 (RL)', True))
except Exception as e:
    checks.append(('loop3 (RL)', False))

try:
    from src.orchestrator.master_loop import MasterLoop
    checks.append(('master loop', True))
except Exception as e:
    checks.append(('master loop', False))

try:
    from src.infra.eval_harness import EvalHarness
    checks.append(('eval harness', True))
except Exception as e:
    checks.append(('eval harness', False))

all_ok = True
for name, ok in checks:
    status = 'OK' if ok else 'FAIL'
    print(f'  {name:.<30s} {status}')
    if not ok:
        all_ok = False

if not all_ok:
    print('\nSome checks failed. Review the output above.')
    sys.exit(1)
print('\nAll checks passed.')
"

echo ""
info "=========================================="
info " Setup complete!"
info "=========================================="
info ""
info " Quick start:"
info "   python scripts/smoke_test.py"
info "   python scripts/run_loop1.py --config config/ --iterations 10"
info "   python scripts/run_all.py   --config config/"
info ""
