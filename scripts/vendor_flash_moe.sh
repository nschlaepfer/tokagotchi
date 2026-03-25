#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/vendor/flash-moe"
UPSTREAM_URL="${FLASH_MOE_UPSTREAM_URL:-https://github.com/Anemll/flash-moe}"
UPSTREAM_BRANCH="${FLASH_MOE_UPSTREAM_BRANCH:-iOS-App}"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

echo "Cloning ${UPSTREAM_URL} (${UPSTREAM_BRANCH})..."
git clone --depth 1 --branch "${UPSTREAM_BRANCH}" "${UPSTREAM_URL}" "${tmp_dir}/flash-moe"
upstream_commit="$(git -C "${tmp_dir}/flash-moe" rev-parse HEAD)"

mkdir -p "${ROOT_DIR}/vendor"
rm -rf "${VENDOR_DIR}"
rsync -a \
  --exclude '.git' \
  --exclude 'AGENTS.md' \
  --exclude 'CLAUDE.md' \
  --exclude 'HANDOFF.md' \
  "${tmp_dir}/flash-moe/" "${VENDOR_DIR}/"

cat > "${VENDOR_DIR}/UPSTREAM_COMMIT" <<EOF
${upstream_commit}
EOF

echo "Vendored flash-moe at ${upstream_commit} into ${VENDOR_DIR}"
