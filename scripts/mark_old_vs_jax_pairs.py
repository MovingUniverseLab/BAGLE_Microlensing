#!/usr/bin/env python3
"""Mark old-vs-jax harness pairs in ``jax_migration_status.json``."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

from jax_migration_status_utils import summarize_tasks  # noqa: E402

STATUS_JSON = REPO / "docs" / "jax_migration_status.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairs_fn",
        help="Fixture function in model_old_vs_jax_fixtures, e.g. psbl_gp_param1_pairs",
    )
    parser.add_argument(
        "--grad",
        choices=("pass", "not_run", "skip"),
        default="not_run",
        help="Grad status to record (default: not_run)",
    )
    args = parser.parse_args()

    mod = importlib.import_module("model_old_vs_jax_fixtures")
    pairs_fn = getattr(mod, args.pairs_fn)
    pairs = set(pairs_fn())

    data = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    marked = 0
    for row in data["tasks"]:
        key = (row["class_name"], row["method_name"])
        if key in pairs:
            row["jax_forward"] = "jax_only"
            row["parity"] = "pass"
            row["grad"] = args.grad
            row["parity_enabled"] = True
            marked += 1
    applicable = [r for r in data["tasks"] if r["applicable"]]
    data["summary"] = summarize_tasks(data["tasks"])
    STATUS_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        f"marked {marked} pairs from {args.pairs_fn} "
        f"(grad={args.grad}); total_done={data['summary']['total_done']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
