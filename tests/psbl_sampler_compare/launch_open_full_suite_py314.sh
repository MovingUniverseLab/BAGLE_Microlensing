#!/usr/bin/env bash
# Full open (wide) prior PSPL + PSBL suite in conda env py314.
# Each backend runs in its own process so a segfault cannot abort the suite.
set -euo pipefail
ROOT="/Users/jlu/code/python/bagle"
OUT="${ROOT}/tests/psbl_sampler_compare/runs_open_full_py314"
LOG="${OUT}/suite.log"
export MPLCONFIGDIR="${ROOT}/tests/psbl_sampler_compare/.mplconfig"
mkdir -p "${OUT}" "${MPLCONFIGDIR}"

# Prefer py314 (working jaxns/TFP stack) over astro.
# Avoid `set -u` breaking conda deactivate hooks (CONDA_BACKUP_* unbound).
set +u
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /Users/jlu/.conda/envs/py314
set -u
cd "${ROOT}/tests/psbl_sampler_compare"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PATH="/Users/jlu/.conda/envs/py314/bin:${PATH}"

echo "Using Python: $(which python)"
python -c 'import sys; print(sys.version)'

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

# More generous knobs for open (~40× narrow) prior volumes.
COMMON=(
  --prior-mode open
  --model-cadence 10
  --resume
  --mnest-live 200
  --mnest-max-iter 50000
  --mnest-tol 0.5
  --nuts-draws 1000
  --nuts-tune 1000
  --nuts-chains 2
  --sa-draws 6000
  --sa-tune 2500
  --smc-particles 40
  --smc-temperatures 8
  --smc-nuts-tune 40
  --pymc-smc-draws 500
  --pymc-smc-chains 2
  --jaxns-live 150
  --jaxns-max-samples 80000
  --jaxns-dlogz 0.5
  --jaxns-posterior 1500
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
        "error": "process exited with code ${rc} (possible segfault / timeout / import crash)",
        "model": "${model}",
        "scenario": "${scenario}",
        "prior_mode": "open",
        "env": "py314",
    }
    result.write_text(json.dumps(rec, indent=2))
    print("Wrote failure record", result)
PY
    fi
  done
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
    rec.setdefault("prior_mode", "open")
    rec["env"] = "py314"
    rec.update(BACKEND_META.get(be, {}))
    results.append(rec)
(outdir / "all_results.json").write_text(json.dumps(results, indent=2, default=float))
write_html_report(
    results,
    outdir / "comparison_report.html",
    title="${model} open-prior comparison (py314) — ${scenario}",
    subtitle="Wide priors · all solvers · JAX on/off · grads on/off · conda py314",
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

root = Path("runs_open_full_py314")
suite = []
for model, scen in [("pspl", "fake_data1"), ("psbl", "bulge_q0p5")]:
    path = root / model / scen / "all_results.json"
    if not path.exists():
        results = []
        for be_dir in sorted((root / model / scen).glob("*/result.json")):
            rec = json.loads(be_dir.read_text())
            be = be_dir.parent.name
            rec.setdefault("model", model)
            rec.setdefault("scenario", scen)
            rec.setdefault("prior_mode", "open")
            rec["env"] = "py314"
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
    title="Open-prior full solver suite (py314) — PSPL & PSBL",
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
<title>Open full suite (py314)</title></head><body>
<h1>Open-prior (wide) full solver suite (conda py314)</h1>
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
} 2>&1 | tee "${LOG}"

echo "Log: ${LOG}"
echo "Summary: ${OUT}/suite_summary.html"
