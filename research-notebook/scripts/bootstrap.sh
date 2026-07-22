#!/usr/bin/env bash
# Bootstrap research-notebook on this machine (laptop or Berkeley host).
set -euo pipefail

# Prefer standalone repo; fall back to bagle workspace copy.
if [[ -d "${HOME}/code/python/research-notebook/src/research_notebook" ]]; then
  REPO_ROOT="${HOME}/code/python/research-notebook"
elif [[ -d "$(cd "$(dirname "$0")/.." && pwd)/src/research_notebook" ]]; then
  REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
else
  echo "Cannot find research-notebook package" >&2
  exit 1
fi

CONFIG_DIR="${HOME}/.config/research_notebook"
SKILL_SRC="${REPO_ROOT}/.cursor/skills/research-notebook"
# Workspace copies may omit .cursor/; use scripts/SKILL.md fallback.
SKILL_FALLBACK="${REPO_ROOT}/scripts/SKILL.md"
SKILL_DST="${HOME}/.cursor/skills/research-notebook"
RULE_SRC="${REPO_ROOT}/scripts/research-notebook.mdc"
RULE_DST="${HOME}/.cursor/rules/research-notebook.mdc"

mkdir -p "$CONFIG_DIR" "${HOME}/.cursor/skills" "${HOME}/.cursor/rules"

echo "==> Installing Python package (editable) into astro if available"
if [[ -x /opt/miniforge3/envs/astro/bin/pip ]]; then
  /opt/miniforge3/envs/astro/bin/pip install -e "$REPO_ROOT"
elif command -v mamba >/dev/null 2>&1; then
  mamba run -n astro pip install -e "$REPO_ROOT"
else
  pip install -e "$REPO_ROOT"
fi

echo "==> Installing Cursor skill + rule"
mkdir -p "$SKILL_DST"
if [[ -f "${SKILL_SRC}/SKILL.md" ]]; then
  cp "${SKILL_SRC}/SKILL.md" "${SKILL_DST}/SKILL.md"
elif [[ -f "$SKILL_FALLBACK" ]]; then
  cp "$SKILL_FALLBACK" "${SKILL_DST}/SKILL.md"
else
  echo "WARN: SKILL.md not found in repo; keeping existing skill if any"
fi
cp "$RULE_SRC" "$RULE_DST"
echo "Installed skill: $SKILL_DST"
echo "Installed rule:  $RULE_DST"

RN_BIN="$(command -v research-notebook || true)"
if [[ -z "$RN_BIN" && -x /opt/miniforge3/envs/astro/bin/research-notebook ]]; then
  RN_BIN=/opt/miniforge3/envs/astro/bin/research-notebook
fi

echo "==> Writing OAuth setup docs + status"
"$RN_BIN" setup-oauth-docs || true
"$RN_BIN" status || true

cat <<EOF

Next steps:
  1. Place OAuth Desktop client JSON at:
       $CONFIG_DIR/client_secret.json
  2. research-notebook auth
  3. research-notebook init
  4. research-notebook link

On a second machine (e.g. dragon):
  - rsync/copy this repo
  - copy client_secret.json, token.json, config.yaml from this host
  - re-run this bootstrap script
EOF
