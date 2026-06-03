#!/usr/bin/env python3
"""Mark multiple old-vs-jax pair fixture functions in jax_migration_status.json."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

STATUS_JSON = REPO / "docs" / "jax_migration_status.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairs_fns",
        nargs="+",
        help="Fixture functions in model_old_vs_jax_fixtures",
    )
    parser.add_argument(
        "--grad",
        choices=("pass", "not_run"),
        default="not_run",
    )
    args = parser.parse_args()

    mod = importlib.import_module("model_old_vs_jax_fixtures")
    pairs: set[tuple[str, str]] = set()
    for name in args.pairs_fns:
        pairs |= set(getattr(mod, name)())

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
    data["summary"] = {
        "total_applicable": len(applicable),
        "total_done": sum(
            1
            for r in applicable
            if r["jax_forward"] == "jax_only"
            and r["parity"] == "pass"
            and r["grad"] == "pass"
        ),
    }
    STATUS_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        f"marked {marked} rows from {len(args.pairs_fns)} fns "
        f"({len(pairs)} pairs); total_done={data['summary']['total_done']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
