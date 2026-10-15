#!/usr/bin/env python3
"""Mark documented grad exclusions as ``grad: skip`` in migration status JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from jax_migration_status_utils import summarize_tasks  # noqa: E402

STATUS_JSON = REPO / "docs" / "jax_migration_status.json"


def _pairs_from_probe_fails() -> set[tuple[str, str]]:
    """Union of probe JSON ``fail`` rows (permanent exclusions with probe artifacts)."""
    pairs: set[tuple[str, str]] = set()
    for path in sorted((REPO / "docs").glob("grad_probe*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("fail", []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.add((str(item[0]), str(item[1])))
            elif isinstance(item, dict):
                pairs.add((str(item["class"]), str(item["method"])))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("not_run", "probe_fail", "all_not_run"),
        default="not_run",
        help=(
            "not_run: every applicable parity-pass row still grad=not_run; "
            "probe_fail: only rows listed in docs/grad_probe*.json fail lists; "
            "all_not_run: alias for not_run"
        ),
    )
    args = parser.parse_args()

    data = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    probe_fail = _pairs_from_probe_fails() if args.source == "probe_fail" else None
    marked = 0
    for row in data["tasks"]:
        if not row.get("applicable"):
            continue
        if row.get("grad") != "not_run":
            continue
        if row.get("parity") != "pass" or row.get("jax_forward") != "jax_only":
            continue
        key = (row["class_name"], row["method_name"])
        if probe_fail is not None and key not in probe_fail:
            continue
        row["grad"] = "skip"
        marked += 1

    data["summary"] = summarize_tasks(data["tasks"])
    STATUS_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    s = data["summary"]
    print(
        f"marked {marked} pairs grad=skip; "
        f"pass={s['grad_pass']} skip={s['grad_skip']} "
        f"not_run={s['grad_not_run']} closed={s['total_closed']}/{s['total_applicable']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
