#!/usr/bin/env bash
# Install Cursor skill + always-apply user rule for research-notebook.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKILL_SRC="$REPO_ROOT/.cursor/skills/research-notebook"
SKILL_DST="${HOME}/.cursor/skills/research-notebook"
RULE_DST="${HOME}/.cursor/rules/research-notebook.mdc"

mkdir -p "${HOME}/.cursor/skills" "${HOME}/.cursor/rules"

rm -rf "$SKILL_DST"
ln -s "$SKILL_SRC" "$SKILL_DST"
echo "Linked skill: $SKILL_DST -> $SKILL_SRC"

cp "$REPO_ROOT/scripts/research-notebook.mdc" "$RULE_DST"
echo "Installed user rule: $RULE_DST"

echo "Done. Open a new Cursor chat and try:"
echo "  add these results to my research notebook"
