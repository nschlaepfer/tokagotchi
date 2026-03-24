#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Qwen Self-Improve — Setup Script
#
# Checks prerequisites, installs the project, downloads the
# base model, builds the Docker arena image, creates seed data,
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
# 2. Install the project with all extras
# ------------------------------------------------------------------
info "Installing qwen-self-improve with all extras..."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e ".[training,dev]"
info "Package installation complete"

# ------------------------------------------------------------------
# 3. Download Qwen 3.5 27B (AWQ quantised)
# ------------------------------------------------------------------
MODEL_ID="Qwen/Qwen3.5-27B-AWQ"
MODEL_DIR="./models/Qwen3.5-27B-AWQ"

if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    info "Model already downloaded at $MODEL_DIR — skipping"
else
    info "Downloading $MODEL_ID ..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"
    else
        warn "huggingface-cli not found. Install with: pip install huggingface-hub[cli]"
        warn "Then run: huggingface-cli download $MODEL_ID --local-dir $MODEL_DIR"
    fi
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
info "   python scripts/run_loop1.py --config config/ --iterations 10"
info "   python scripts/run_all.py   --config config/"
info ""
