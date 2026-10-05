#!/usr/bin/env bash
# Full narrow-prior PSPL + PSBL suite: all solvers, JAX on/off, grads on/off.
# Each backend runs in its own process so a segfault (e.g. jaxns/TFP) cannot
# abort the remainder of the suite.
set -euo pipefail
ROOT="/Users/jlu/code/python/bagle"
OUT="${ROOT}/tests/psbl_sampler_compare/runs_narrow_full"
LOG="${OUT}/suite.log"
export MPLCONFIGDIR="${ROOT}/tests/psbl_sampler_compare/.mplconfig"
mkdir -p "${OUT}" "${MPLCONFIGDIR}"
cd "${ROOT}/tests/psbl_sampler_compare"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

BACKENDS=(
  multinest
  multinest_host
  numpyro_nuts_grad
  numpyro_sa_nograd
  numpyro_smc_nuts
  pymc_smc
  pymc_smc_nojax
  jaxns_grad
  jaxns_nograd
)

COMMON=(
  --prior-mode narrow
  --model-cadence 10
  --resume
  --mnest-live 100
  --mnest-max-iter 15000
  --mnest-tol 0.5
  --nuts-draws 800
  --nuts-tune 400
  --nuts-chains 2
  --sa-draws 2500
  --sa-tune 1000
  --smc-particles 28
  --smc-temperatures 5
  --smc-nuts-tune 25
  --pymc-smc-draws 400
  --pymc-smc-chains 2
  --jaxns-live 80
  --jaxns-max-samples 40000
  --jaxns-dlogz 0.5
  --jaxns-posterior 800
)

run_model() {
  local model="$1"
  local scenario="$2"
  local outdir="${OUT}/${model}"
  echo "======== ${model} ${scenario} ========"
  for be in "${BACKENDS[@]}"; do
    echo "----- backend ${be} -----"
    set +e
    python -u run_comparison.py \
      --outdir "${outdir}" \
      --model "${model}" \
      --scenario "${scenario}" \
      --only "${be}" \
      "${COMMON[@]}"
    local rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
      echo "WARNING: backend ${be} exited with code ${rc}"
      python -u - <<PY
import json
from pathlib import Path
run_dir = Path("${outdir}") / "${scenario}" / "${be}"
run_dir.mkdir(parents=True, exist_ok=True)
result = run_dir / "result.json"
if not result.exists() or json.loads(result.read_text()).get("status") != "ok":
    rec = {
        "label": "${be}",
        "status": "failed",
        "runtime_sec": None,
        "error": "process exited with code ${rc} (possible segfault / import crash)",
        "model": "${model}",
        "scenario": "${scenario}",
        "prior_mode": "narrow",
    }
    result.write_text(json.dumps(rec, indent=2))
    print("Wrote failure record", result)
PY
    fi
  done
  # Rebuild scenario HTML from all per-backend result.json files.
  python -u - <<PY
import json
from pathlib import Path
from report import write_html_report
from run_comparison import BACKEND_META

outdir = Path("${outdir}") / "${scenario}"
backends = """${BACKENDS[*]}""".split()
results = []
for be in backends:
    path = outdir / be / "result.json"
    if not path.exists():
        continue
    rec = json.loads(path.read_text())
    rec.setdefault("model", "${model}")
    rec.setdefault("scenario", "${scenario}")
    rec.setdefault("prior_mode", "narrow")
    rec.update(BACKEND_META.get(be, {}))
    results.append(rec)
(outdir / "all_results.json").write_text(json.dumps(results, indent=2, default=float))
write_html_report(
    results,
    outdir / "comparison_report.html",
    title="${model} narrow-prior comparison — ${scenario}",
    subtitle="All solvers · JAX on/off · grads on/off",
)
print("Wrote", outdir / "comparison_report.html")
PY
}

{
  run_model pspl fake_data1
  run_model psbl bulge_q0p5

  echo "======== Suite summary HTML ========"
  python -u - <<'PY'
import json
from pathlib import Path
from report import write_suite_summary_html
from run_comparison import BACKEND_META

root = Path("runs_narrow_full")
suite = []
for model, scen in [("pspl", "fake_data1"), ("psbl", "bulge_q0p5")]:
    path = root / model / scen / "all_results.json"
    if not path.exists():
        # Fall back to per-backend result.json files.
        results = []
        for be_dir in sorted((root / model / scen).glob("*/result.json")):
            rec = json.loads(be_dir.read_text())
            be = be_dir.parent.name
            rec.setdefault("model", model)
            rec.setdefault("scenario", scen)
            rec.setdefault("prior_mode", "narrow")
            rec.update(BACKEND_META.get(be, {}))
            results.append(rec)
        if results:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(results, indent=2, default=float))
    if path.exists():
        suite.extend(json.loads(path.read_text()))

out = write_suite_summary_html(
    suite,
    root / "suite_summary.html",
    title="Narrow-prior full solver suite — PSPL & PSBL",
)
print("Wrote", out.resolve())

rows = []
for model, scen in [("pspl", "fake_data1"), ("psbl", "bulge_q0p5")]:
    path = root / model / scen / "all_results.json"
    if not path.exists():
        continue
    results = json.loads(path.read_text())
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    rows.append(
        "<tr><td>{}</td><td>{}</td><td>{}/{}</td>"
        "<td><a href='{}/{}/comparison_report.html'>scenario</a></td></tr>"
        .format(model, scen, n_ok, len(results), model, scen)
    )
(root / "index.html").write_text(
    """<!doctype html><html><head><meta charset='utf-8'>
<title>Narrow full suite</title></head><body>
<h1>Narrow-prior full solver suite</h1>
<p><a href='suite_summary.html'><b>Open consolidated summary</b></a>
(timing, lnL, logZ, bias)</p>
<table border='0' cellpadding='8'>
<tr><th>model</th><th>scenario</th><th>ok</th><th>report</th></tr>
"""
    + "".join(rows)
    + "</table></body></html>"
)
print("Wrote", (root / "index.html").resolve())
PY
} 2>&1 | tee -a "${LOG}"

echo "Log: ${LOG}"
echo "Summary: ${OUT}/suite_summary.html"
