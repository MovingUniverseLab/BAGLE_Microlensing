#!/usr/bin/env python3
"""Mark PSBL PhotAstrom seed pairs as jax_only + parity pass (grad not_run)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from model_old_vs_jax_fixtures import psbl_photastrom_first_pairs  # noqa: E402

STATUS_JSON = REPO / "docs" / "jax_migration_status.json"


def main() -> int:
    pairs = set(psbl_photastrom_first_pairs())
    data = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    marked = 0
    for row in data["tasks"]:
        key = (row["class_name"], row["method_name"])
        if key in pairs:
            row["jax_forward"] = "jax_only"
            row["parity"] = "pass"
            row["grad"] = "not_run"
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
    print(f"marked {marked} PSBL PhotAstrom parity pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
