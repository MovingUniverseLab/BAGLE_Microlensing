# PSBL sampler comparison tutorial

Manual runner for comparing **MultiNest**, **NumPyro NUTS**, **NumPyro SA**
(no grads), and **jaxns** (± `gradient_guided`) on fake PSBL photometry +
astrometry.

## Where the code lives

| Path | Role |
|------|------|
| [`psbl_sampler_compare/run_comparison.py`](psbl_sampler_compare/run_comparison.py) | CLI entry point — builds fake data, applies priors, runs each backend, writes HTML |
| [`psbl_sampler_compare/report.py`](psbl_sampler_compare/report.py) | Self-contained HTML report (runtimes, lnL, logZ, traces, model/data plots) |
| [`psbl_sampler_compare/launch_open_suite.sh`](psbl_sampler_compare/launch_open_suite.sh) | Optional detached launcher for the full open-prior suite |

Backends (labels for `--only`):

- `multinest` — PyMultiNest with JAX lnL (`MicrolensSolverJaxLike` in `bagle.model_fitter_jax`)
- `numpyro_nuts_grad` — NumPyro NUTS (`use_jax_grad=True`)
- `numpyro_sa_nograd` — NumPyro SA (gradient-free MCMC)
- `jaxns_grad` — jaxns with `gradient_guided=True`
- `jaxns_nograd` — jaxns with `gradient_guided=False`

Injected scenarios (`--scenario`):

- `bulge_q0p5` — bulge-like, \(q=0.5\), \(\mathrm{sep}\approx\theta_E\), \(\alpha=90^\circ\)
- `close_unequal` — close unequal binary (\(q=0.1\)), \(\alpha=35^\circ\)
- `wide_near_equal` — wider near-equal binary (\(q\approx0.9\)), \(\alpha=160^\circ\)

## Environment

Use a conda env with JAX, NumPyro, PyMultiNest, and (for jaxns) `jaxns` +
TensorFlow Probability. The open/narrow comparison suites were run in
**`py314`**:

```bash
zsh -lic 'source /opt/miniforge3/etc/profile.d/conda.sh && conda activate /Users/jlu/.conda/envs/py314 && cd /Users/jlu/code/python/bagle'
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

Matplotlib needs a writable config dir if the home cache is restricted:

```bash
export MPLCONFIGDIR="${PWD}/tests/psbl_sampler_compare/.mplconfig"
mkdir -p "$MPLCONFIGDIR"
```

## Quick start

From the repo root (with `PYTHONPATH=src` and the conda env active):

```bash
cd tests/psbl_sampler_compare

# Narrow priors around truth, one scenario, all backends
python -u run_comparison.py \
  --outdir runs_narrow \
  --prior-mode narrow \
  --scenario bulge_q0p5 \
  --model-cadence 10

# Open priors (~40× narrow half-widths, α ∈ [0, 360)), all three scenarios
python -u run_comparison.py \
  --outdir runs_open \
  --prior-mode open \
  --scenario all \
  --model-cadence 10
```

Open the report(s):

- Single scenario: `runs_*/<scenario>/comparison_report.html`
- Multi-scenario index: `runs_*/index.html`

Existing completed outputs (for reference):

- Narrow: `tests/psbl_sampler_compare/runs/comparison_report.html`
- Open: `tests/psbl_sampler_compare/runs_open/index.html`

## Useful options

```bash
python -u run_comparison.py --help
```

| Flag | Meaning |
|------|---------|
| `--prior-mode {narrow,open}` | Prior width (default: `open`) |
| `--scenario NAME\|a,b\|all` | Which injected binaries to run |
| `--only multinest,numpyro_nuts_grad` | Subset of backends |
| `--model-cadence 10` | Oversample model curves through gaps (days) |
| `--outdir PATH` | Output root (per-scenario subdirs when multiple scenarios) |
| `--resume` | Skip backends whose `result.json` already has `status=ok` |
| `--verbose` | Sampler progress / MultiNest verbose |
| `--fitter-plots` | Also dump heavy `plot_model_and_data` PNGs |
| `--mnest-live`, `--mnest-max-iter`, `--mnest-tol` | MultiNest knobs |
| `--nuts-draws`, `--nuts-tune`, `--nuts-chains` | NUTS knobs |
| `--sa-draws`, `--sa-tune` | SA knobs |
| `--jaxns-live`, `--jaxns-max-samples`, `--jaxns-dlogz` | jaxns knobs |

### Smoke test (MultiNest only, one scenario)

```bash
python -u run_comparison.py \
  --outdir runs_smoke \
  --prior-mode narrow \
  --scenario bulge_q0p5 \
  --only multinest \
  --mnest-live 100 \
  --mnest-max-iter 5000 \
  --model-cadence 10
```

### Resume a interrupted open suite

```bash
python -u run_comparison.py \
  --outdir runs_open \
  --prior-mode open \
  --scenario all \
  --model-cadence 10 \
  --resume
```

Failed backends (e.g. timed-out NUTS) are re-run; successful ones are skipped.

### Detached full open suite

```bash
./launch_open_suite.sh
# log: runs_open/run.log
```

## What each run writes

Under `--outdir` / `<scenario>/`:

```
fake_data.pkl
comparison_report.html
all_results.json
multinest/          # result.json, trace.png, model_data.png, …
numpyro_nuts_grad/
numpyro_sa_nograd/
jaxns_grad/
jaxns_nograd/
```

Plus MultiNest / NumPyro basename files (`multinest_*.dat`, `nuts_*.fits`, …).

Model-vs-data plots oversample the model at `--model-cadence` (default 10 d)
so seasonal gaps show the full light curve and on-sky track.

## Prior modes (short)

- **narrow** — tight uniforms around the injected truth (easy nested sampling;
  good for checking backend agreement).
- **open** — much wider windows (~40× those half-widths; `alpha` fully open on
  \([0,360)\)). Still centered on truth so a 17-D phot+astrom search remains
  tractable. Fully sky-wide physical priors are usually too hard for this
  high-SNR fake data.

## Tips

- Prefer `python -u` so MultiNest / sampler logs flush promptly.
- Open-prior NUTS can stall; the runner applies a soft SIGALRM timeout
  (~20 min) for open NUTS and (~2 h) for open jaxns.
- jaxns needs optional deps (`jaxns`, TensorFlow Probability). If imports fail,
  run with `--only multinest,numpyro_nuts_grad,numpyro_sa_nograd`.
- Long open-prior suites (3 scenarios × 5 backends) can take many hours,
  mostly from jaxns.
