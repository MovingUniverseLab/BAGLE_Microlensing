"""Shared helpers for ``docs/jax_migration_status.json`` updates."""
from __future__ import annotations

GRAD_TERMINAL = frozenset(("pass", "skip"))


def grad_complete(row: dict) -> bool:
    """Return True when grad verification is finished (pass or documented skip)."""
    return row.get("grad") in GRAD_TERMINAL


def migration_done(row: dict) -> bool:
    """Parity-complete row with grad pass or skip."""
    return bool(
        row.get("applicable")
        and row.get("jax_forward") == "jax_only"
        and row.get("parity") == "pass"
        and grad_complete(row)
    )


def grad_pass_done(row: dict) -> bool:
    """Row with verified grad (``grad == pass`` only)."""
    return bool(
        row.get("applicable")
        and row.get("jax_forward") == "jax_only"
        and row.get("parity") == "pass"
        and row.get("grad") == "pass"
    )


def summarize_tasks(tasks: list[dict]) -> dict:
    """Count grad / migration states over applicable tasks."""
    applicable = [r for r in tasks if r.get("applicable")]
    grad_counts: dict[str, int] = {}
    for row in applicable:
        g = row.get("grad", "not_run")
        grad_counts[g] = grad_counts.get(g, 0) + 1
    return {
        "total_applicable": len(applicable),
        "total_done": sum(1 for r in applicable if grad_pass_done(r)),
        "total_closed": sum(1 for r in applicable if migration_done(r)),
        "grad_pass": grad_counts.get("pass", 0),
        "grad_skip": grad_counts.get("skip", 0),
        "grad_not_run": grad_counts.get("not_run", 0),
        "grad_fail": grad_counts.get("fail", 0),
    }


def overall_status(row: dict) -> str:
    """Dashboard overall column: done | skipped | blocked | pending | n/a."""
    if not row.get("applicable"):
        return "n/a"
    if row.get("parity") == "fail" or row.get("grad") == "fail":
        return "blocked"
    if (
        row.get("jax_forward") == "jax_only"
        and row.get("parity") == "pass"
        and row.get("grad") == "pass"
    ):
        return "done"
    if (
        row.get("jax_forward") == "jax_only"
        and row.get("parity") == "pass"
        and row.get("grad") == "skip"
    ):
        return "skipped"
    return "pending"
