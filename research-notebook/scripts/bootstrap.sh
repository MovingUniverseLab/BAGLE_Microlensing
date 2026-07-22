#!/usr/bin/env bash
# Bootstrap research-notebook on this machine (laptop or Berkeley host).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="${HOME}/.config/research_notebook"

mkdir -p "$CONFIG_DIR"

echo "==> Installing Python package (editable)"
if command -v mamba >/dev/null 2>&1; then
  # Prefer astro env when available.
  if mamba env list 2>/dev/null | grep -qE '^\s*astro\s'; then
    mamba run -n astro pip install -e "$REPO_ROOT"
  else
    pip install -e "$REPO_ROOT"
  fi
else
  pip install -e "$REPO_ROOT"
fi

echo "==> Installing Cursor skill + rule"
bash "$REPO_ROOT/scripts/install_cursor_hooks.sh"

echo "==> Writing OAuth setup docs"
research-notebook setup-oauth-docs || python -m research_notebook setup-oauth-docs

echo "==> Status"
research-notebook status || true

cat <<EOF

Next steps:
  1. Place OAuth Desktop client JSON at:
       $CONFIG_DIR/client_secret.json
  2. research-notebook auth
  3. research-notebook init
  4. research-notebook link

On a second machine: copy client_secret.json, token.json, and config.yaml
from this host after init, then re-run this bootstrap (skip auth if token copied).
EOF
