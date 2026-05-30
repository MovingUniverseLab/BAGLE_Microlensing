#!/usr/bin/env python3
"""Regenerate docs/jax_migration_dashboard.md from jax_migration_status.json."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bagle.jax.migration_tasks import FAMILY_ORDER, family_summary, generate_tasks  # noqa: E402

DOCS = REPO / "docs"
STATUS_JSON = DOCS / "jax_migration_status.json"
DASHBOARD_MD = DOCS / "jax_migration_dashboard.md"


def _overall(row: dict) -> str:
    if not row["applicable"]:
        return "n/a"
    if (
        row["jax_forward"] == "jax_only"
        and row["parity"] == "pass"
        and row["grad"] == "pass"
    ):
        return "done"
    if row["parity"] == "fail" or row["grad"] == "fail":
        return "blocked"
    return "pending"


def main() -> int:
    if not STATUS_JSON.is_file():
        print(f"Missing {STATUS_JSON}; run generate_jax_migration_tasks.py first")
        return 1

    data = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    rows = [r for r in data.get("tasks", []) if r.get("applicable")]
    done_ids = {
        f"{r['class_name']}::{r['method_name']}"
        for r in rows
        if _overall(r) == "done"
    }

    import bagle.model_jax as model_jax

    all_tasks = generate_tasks(model_jax)
    fam = family_summary(all_tasks, done_ids)
    total_done = len(done_ids)
    total = sum(1 for t in all_tasks if t.applicable)

    lines = [
        "# JAX migration dashboard",
        "",
        f"Updated: {date.today().isoformat()} | Tasks: {total_done}/{total} done",
        "",
        "| Class | Method | JAX forward | Parity | Grad | Overall |",
        "|-------|--------|-------------|--------|------|---------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_name']} | {row['method_name']} | {row['jax_forward']} | "
            f"{row['parity']} | {row['grad']} | {_overall(row)} |"
        )
    lines.append("")
    for family in FAMILY_ORDER:
        d, n = fam.get(family, (0, 0))
        if n:
            lines.append(f"**{family} methods:** {d}/{n} done")
    lines.extend(
        [
            "",
            f"**Total methods:** {total_done}/{total} done",
            "",
            "See also [`jax_migration_tasks.md`](jax_migration_tasks.md).",
            "",
        ]
    )
    DASHBOARD_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {DASHBOARD_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
