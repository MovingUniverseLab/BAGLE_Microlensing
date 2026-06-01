#!/usr/bin/env python3
"""Generate docs/jax_migration_tasks.md from concrete model classes.

Method inventory and per-class applicability come from
``bagle.jax.migration_tasks`` (``ALL_FORWARD_METHODS``, ``applicable_methods``).
Regenerate after changing that module or adding PSPL extended methods.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bagle.jax.migration_tasks import MigrationTask, generate_tasks  # noqa: E402

DOCS = REPO / "docs"
TASKS_MD = DOCS / "jax_migration_tasks.md"
STATUS_JSON = DOCS / "jax_migration_status.json"


def _load_prev() -> dict[tuple[str, str], dict]:
    if not STATUS_JSON.is_file():
        return {}
    data = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    return {(t["class_name"], t["method_name"]): t for t in data.get("tasks", [])}


def _task_row(task: MigrationTask, prev: dict) -> dict:
    key = (task.class_name, task.method_name)
    old = prev.get(key, {})
    return {
        "class_name": task.class_name,
        "method_name": task.method_name,
        "family": task.family,
        "applicable": task.applicable,
        "jax_forward": old.get("jax_forward", "numpy_fallback"),
        "parity": old.get("parity", "not_run"),
        "grad": old.get("grad", "not_run"),
        "parity_enabled": bool(old.get("parity_enabled", False)),
    }


def _step_done(row: dict, step: str) -> bool:
    if step == "jax_forward":
        return row["jax_forward"] == "jax_only"
    if step == "parity":
        return row["parity"] == "pass"
    return row["grad"] == "pass"


def write_tasks_md(rows: list[dict]) -> None:
    applicable = [r for r in rows if r["applicable"]]
    lines = [
        "# JAX migration task queue",
        "",
        f"Updated: {date.today().isoformat()} | Applicable tasks: {len(applicable)}",
        "",
    ]
    current_fam = None
    for row in rows:
        if not row["applicable"]:
            continue
        if row["family"] != current_fam:
            current_fam = row["family"]
            n = sum(1 for r in applicable if r["family"] == current_fam)
            lines.extend(["", f"## {current_fam} ({n} tasks)", ""])
        tid = f"{row['class_name']}::{row['method_name']}"
        for step in ("jax_forward", "parity", "grad"):
            mark = "x" if _step_done(row, step) else " "
            lines.append(f"- [{mark}] {tid}::{step}")
    TASKS_MD.parent.mkdir(parents=True, exist_ok=True)
    TASKS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import bagle.model_jax as model_jax

    prev = _load_prev()
    tasks = generate_tasks(model_jax)
    rows = [_task_row(t, prev) for t in tasks]
    summary = {
        "total_applicable": sum(1 for r in rows if r["applicable"]),
        "total_done": sum(
            1
            for r in rows
            if r["applicable"]
            and r["jax_forward"] == "jax_only"
            and r["parity"] == "pass"
            and r["grad"] == "pass"
        ),
    }
    STATUS_JSON.write_text(
        json.dumps({"tasks": rows, "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_tasks_md(rows)
    print(f"Wrote {TASKS_MD} ({summary['total_applicable']} applicable tasks)")
    print(f"Wrote {STATUS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
