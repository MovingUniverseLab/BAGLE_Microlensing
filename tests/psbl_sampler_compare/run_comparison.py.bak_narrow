#!/usr/bin/env python
"""
PSBL Phot+Astrom sampler comparison: MultiNest, NumPyro NUTS/SA, jaxns ± grads.

Produces JSON result records and an HTML report under this directory.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Package imports expect PYTHONPATH=src.
from bagle import fake_data
from bagle import model_fitter_jax as model_fitter
from bagle import model_jax as model

from report import write_html_report


HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = HERE / "runs"


class MicrolensSolverJaxLike(model_fitter.MicrolensSolver):
    """MultiNest solver that evaluates the explicit JAX likelihood."""

    def LogLikelihood(self, cube, ndim=None, n_params=None):
        """Evaluate JAX lnL for PyMultiNest.

        Parameters
        ----------
        cube : array_like
            Current parameter vector (unit-cube transformed).
        ndim, n_params : int or None
            Unused PyMultiNest signature arguments.

        Returns
        -------
        lnL : float
            Joint photometry + astrometry log-likelihood.
        """
        return self.evaluate_loglik_jax(cube)


def _scalar(val, idx=0):
    """Pull a scalar from a nested list/array truth entry."""
    arr = np.asarray(val).ravel()
    return float(arr[idx])


def _truth_dict(p_in, names):
    """Map fitter parameter names onto fake-data truth values."""
    out = {}
    for name in names:
        if name in p_in:
            out[name] = _scalar(p_in[name])
            continue
        base = "".join(c for c in name if not c.isdigit())
        digits = "".join(c for c in name if c.isdigit())
        idx = int(digits) - 1 if digits else 0
        out[name] = _scalar(p_in[base], idx)
    return out


def apply_narrow_priors(fitter, p_in, half_width=0.05, stats_pkg="scipy"):
    """Set tight uniform priors around the injected truth.

    Parameters
    ----------
    fitter : MicrolensSolver
        Configured solver whose ``priors`` dict is updated in place.
    p_in : dict
        Injected fake-data parameters.
    half_width : float, optional
        Absolute half-width for most parameters. Relative widths are used
        for ``t0`` / ``dL`` / ``dS`` / ``mag_src``.
    stats_pkg : {'scipy', 'numpyro'}, optional
        Prior backend matching the solver.

    Returns
    -------
    None
    """
    truth = _truth_dict(p_in, fitter.fitter_param_names)
    make = model_fitter.make_gen

    # Absolute half-widths; override a few scales below.
    widths = {name: half_width for name in fitter.fitter_param_names}
    widths["t0"] = 0.5
    widths["dL"] = 5.0
    widths["dS"] = 10.0
    widths["mag_src1"] = 0.05
    widths["b_sff1"] = 0.05
    widths["dmag_Lp_Ls1"] = 0.5
    widths["alpha"] = 1.0
    widths["sep"] = 0.2
    widths["mLp"] = 0.1
    widths["mLs"] = 0.1
    widths["beta"] = 0.1
    widths["muL_E"] = 0.1
    widths["muL_N"] = 0.1
    widths["muS_E"] = 0.1
    widths["muS_N"] = 0.1
    widths["xS0_E"] = 1e-4
    widths["xS0_N"] = 1e-4

    for name in fitter.fitter_param_names:
        lo = truth[name] - widths[name]
        hi = truth[name] + widths[name]
        if name.startswith("mL") and lo <= 0:
            lo = 1e-4
        if name.startswith("dL") or name.startswith("dS"):
            lo = max(lo, 1.0)
        if name.startswith("sep") and lo <= 0:
            lo = 1e-3
        fitter.priors[name] = make(name, lo, hi, stats_pkg=stats_pkg)

    return None


def _best_params(fitter, def_best="maxl"):
    """Return a flat best-fit parameter dict."""
    best = fitter.get_best_fit(def_best=def_best)
    if isinstance(best, tuple):
        best = best[0]
    return {k: float(best[k]) for k in fitter.fitter_param_names}


def _logz(fitter):
    """Extract log-evidence when available."""
    logz = getattr(fitter, "_logZ", np.nan)
    if np.isfinite(logz):
        return float(logz)
    try:
        smy = fitter.load_mnest_summary()
        if "logZ" in smy.colnames:
            return float(smy["logZ"][0])
    except Exception:
        pass
    return float("nan")


def _max_lnL(fitter, best):
    """Evaluate host and JAX lnL at the best-fit point."""
    host = float(fitter.log_likely(best))
    jax_lnL = float(fitter.evaluate_loglik_jax(best))
    return host, jax_lnL


def _save_trace_png(fitter, out_png, n_params=8):
    """Write a compact posterior-trace / 1D histogram figure."""
    tab = fitter.load_mnest_results()
    names = list(fitter.fitter_param_names)[:n_params]
    n = len(names)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, name in enumerate(names):
        ax = axes[i]
        vals = np.asarray(tab[name], dtype=float)
        ax.plot(vals, lw=0.6, color="C0", alpha=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("sample")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return None


def _save_model_data_png(fitter, best, truth_model, out_png):
    """Photometry + astrometry data vs best-fit model panels."""
    mod = fitter.get_model(best)
    data = fitter.data

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Photometry.
    t = data["t_phot1"]
    mag = data["mag1"]
    err = data["mag_err1"]
    t_mod = np.linspace(t.min(), t.max(), 800)
    axes[0, 0].errorbar(t, mag, yerr=err, fmt=".", ms=2, alpha=0.5, color="k",
                        label="data")
    axes[0, 0].plot(t_mod, mod.get_photometry(t_mod), color="C1", lw=1.5,
                    label="best fit")
    axes[0, 0].plot(t_mod, truth_model.get_photometry(t_mod), color="C0",
                    lw=1.0, ls="--", label="truth")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylabel("mag")
    axes[0, 0].set_title("Photometry")
    axes[0, 0].legend(fontsize=8)

    # Phot residuals.
    mag_best = np.asarray(mod.get_photometry(t))
    axes[1, 0].errorbar(t, mag - mag_best, yerr=err, fmt=".", ms=2, alpha=0.5,
                        color="k")
    axes[1, 0].axhline(0.0, color="C1", lw=1)
    axes[1, 0].set_xlabel("MJD")
    axes[1, 0].set_ylabel("residual (mag)")

    # Astrometry on-sky.
    ta = data["t_ast1"]
    xe, ye = data["xpos1"], data["ypos1"]
    xe_err, ye_err = data["xpos_err1"], data["ypos_err1"]
    pos_b = np.asarray(mod.get_astrometry(ta))
    pos_t = np.asarray(truth_model.get_astrometry(ta))
    axes[0, 1].errorbar(xe * 1e3, ye * 1e3, xerr=xe_err * 1e3, yerr=ye_err * 1e3,
                        fmt=".", ms=3, alpha=0.6, color="k", label="data")
    axes[0, 1].plot(pos_b[:, 0] * 1e3, pos_b[:, 1] * 1e3, color="C1", lw=1.5,
                    label="best fit")
    axes[0, 1].plot(pos_t[:, 0] * 1e3, pos_t[:, 1] * 1e3, color="C0", lw=1.0,
                    ls="--", label="truth")
    axes[0, 1].set_xlabel("East (mas)")
    axes[0, 1].set_ylabel("North (mas)")
    axes[0, 1].set_title("Astrometry")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].invert_xaxis()

    # Astrom residuals vs time.
    axes[1, 1].errorbar(ta, (xe - pos_b[:, 0]) * 1e3, yerr=xe_err * 1e3,
                        fmt=".", ms=3, alpha=0.6, color="C3", label="E")
    axes[1, 1].errorbar(ta, (ye - pos_b[:, 1]) * 1e3, yerr=ye_err * 1e3,
                        fmt=".", ms=3, alpha=0.6, color="C0", label="N")
    axes[1, 1].axhline(0.0, color="0.5", lw=1)
    axes[1, 1].set_xlabel("MJD")
    axes[1, 1].set_ylabel("residual (mas)")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return None


def run_one(label, factory, outdir, data, p_in, truth_model, resume=False):
    """Run one sampler configuration and return a result record.

    Parameters
    ----------
    label : str
        Short run name used in filenames.
    factory : callable
        Zero-arg callable returning a configured fitter.
    outdir : Path
        Output directory for this comparison.
    data, p_in : dict
        Fake data and injected parameters.
    truth_model : bagle model
        Injected-truth model instance for plots.
    resume : bool, optional
        Pass through to MultiNest when supported.

    Returns
    -------
    record : dict
        Summary metrics, paths, and best-fit parameters.
    """
    print(f"\n===== Starting {label} =====", flush=True)
    run_dir = outdir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "label": label,
        "status": "running",
        "runtime_sec": None,
        "error": None,
    }

    try:
        fitter = factory()
        t0 = time.time()
        fitter.solve()
        runtime = time.time() - t0
        print(f"===== Finished {label} in {runtime:.1f}s =====", flush=True)

        best = _best_params(fitter, def_best="maxl")
        host_lnL, jax_lnL = _max_lnL(fitter, best)
        logz = _logz(fitter)
        truth = _truth_dict(p_in, fitter.fitter_param_names)

        # Bias = best - truth for each parameter.
        bias = {k: best[k] - truth[k] for k in best}

        trace_png = run_dir / "trace.png"
        model_png = run_dir / "model_data.png"
        _save_trace_png(fitter, trace_png)
        _save_model_data_png(fitter, best, truth_model, model_png)

        # Optional fitter-native diagnostic plots.
        try:
            fitter.plot_model_and_data(
                fitter.get_model(best), input_model=truth_model, N_traces=30
            )
        except Exception as exc:
            print(f"plot_model_and_data warning ({label}): {exc}", flush=True)

        # Persist raw posterior table when possible.
        try:
            tab = fitter.load_mnest_results()
            tab.write(run_dir / "posterior.fits", overwrite=True)
        except Exception as exc:
            print(f"posterior write warning ({label}): {exc}", flush=True)

        record.update(
            {
                "status": "ok",
                "runtime_sec": float(runtime),
                "logZ": logz,
                "host_lnL": host_lnL,
                "jax_lnL": jax_lnL,
                "best": best,
                "truth": truth,
                "bias": bias,
                "trace_png": str(trace_png),
                "model_png": str(model_png),
                "param_names": list(fitter.fitter_param_names),
                "outputfiles_basename": fitter.outputfiles_basename,
            }
        )
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{exc}\n{traceback.format_exc()}"
        print(f"===== FAILED {label}: {exc} =====", flush=True)

    with open(run_dir / "result.json", "w") as f:
        json.dump(record, f, indent=2, default=float)

    return record


def build_factories(data, p_in, outdir, args):
    """Construct sampler factory callables for the comparison suite."""
    model_class = model.PSBL_PhotAstrom_Par_Param1
    base = str(outdir) + os.sep

    def multinest():
        fitter = MicrolensSolverJaxLike(
            data,
            model_class,
            n_live_points=args.mnest_live,
            max_iter=args.mnest_max_iter,
            evidence_tolerance=args.mnest_tol,
            sampling_efficiency=0.8,
            outputfiles_basename=base + "multinest_",
            dump_callback=None,
            verbose=False,
            resume=args.resume,
        )
        apply_narrow_priors(fitter, p_in, stats_pkg="scipy")
        return fitter

    def nuts():
        fitter = model_fitter.MicrolensSolverNumPyro(
            data,
            model_class,
            sampler="nuts",
            use_jax_grad=True,
            draws=args.nuts_draws,
            tune=args.nuts_tune,
            chains=args.nuts_chains,
            target_accept=0.9,
            random_seed=0,
            outputfiles_basename=base + "nuts_",
            verbose=False,
        )
        apply_narrow_priors(fitter, p_in, stats_pkg="numpyro")
        return fitter

    def sa():
        fitter = model_fitter.MicrolensSolverNumPyro(
            data,
            model_class,
            sampler="sa",
            use_jax_grad=False,
            draws=args.sa_draws,
            tune=args.sa_tune,
            chains=1,
            random_seed=1,
            outputfiles_basename=base + "sa_",
            verbose=False,
        )
        apply_narrow_priors(fitter, p_in, stats_pkg="numpyro")
        return fitter

    def jaxns_grad():
        fitter = model_fitter.MicrolensSolverNumPyro(
            data,
            model_class,
            sampler="jaxns",
            use_jax_grad=True,
            gradient_guided=True,
            n_live_points=args.jaxns_live,
            max_samples=args.jaxns_max_samples,
            dlogz=args.jaxns_dlogz,
            posterior_samples=args.jaxns_posterior,
            random_seed=2,
            outputfiles_basename=base + "jaxns_grad_",
            verbose=False,
        )
        apply_narrow_priors(fitter, p_in, stats_pkg="numpyro")
        return fitter

    def jaxns_nograd():
        fitter = model_fitter.MicrolensSolverNumPyro(
            data,
            model_class,
            sampler="jaxns",
            use_jax_grad=True,
            gradient_guided=False,
            n_live_points=args.jaxns_live,
            max_samples=args.jaxns_max_samples,
            dlogz=args.jaxns_dlogz,
            posterior_samples=args.jaxns_posterior,
            random_seed=3,
            outputfiles_basename=base + "jaxns_nograd_",
            verbose=False,
        )
        apply_narrow_priors(fitter, p_in, stats_pkg="numpyro")
        return fitter

    factories = [
        ("multinest", multinest),
        ("numpyro_nuts_grad", nuts),
        ("numpyro_sa_nograd", sa),
        ("jaxns_grad", jaxns_grad),
        ("jaxns_nograd", jaxns_nograd),
    ]
    if args.only:
        keep = set(args.only.split(","))
        factories = [f for f in factories if f[0] in keep]
    return factories


def parse_args(argv=None):
    """Parse CLI arguments for the comparison runner."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated subset of run labels")
    p.add_argument("--mnest-live", type=int, default=100)
    p.add_argument("--mnest-max-iter", type=int, default=4000)
    p.add_argument("--mnest-tol", type=float, default=0.5)
    p.add_argument("--nuts-draws", type=int, default=800)
    p.add_argument("--nuts-tune", type=int, default=400)
    p.add_argument("--nuts-chains", type=int, default=2)
    p.add_argument("--sa-draws", type=int, default=1200)
    p.add_argument("--sa-tune", type=int, default=600)
    p.add_argument("--jaxns-live", type=int, default=100)
    p.add_argument("--jaxns-max-samples", type=int, default=30000)
    p.add_argument("--jaxns-dlogz", type=float, default=0.5)
    p.add_argument("--jaxns-posterior", type=int, default=1000)
    return p.parse_args(argv)


def main(argv=None):
    """Run the full PSBL sampler comparison and write the HTML report."""
    args = parse_args(argv)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    data, p_in, _, _ = fake_data.fake_data_PSBL(parallax=True, animate=False)

    with open(outdir / "fake_data.pkl", "wb") as f:
        pickle.dump({"data": data, "p_in": p_in}, f)

    truth_model = model.PSBL_PhotAstrom_Par_Param1(
        p_in["mLp"], p_in["mLs"], p_in["t0"], p_in["xS0_E"], p_in["xS0_N"],
        p_in["beta"], p_in["muL_E"], p_in["muL_N"], p_in["muS_E"], p_in["muS_N"],
        p_in["dL"], p_in["dS"], p_in["sep"], p_in["alpha"],
        p_in["b_sff"], p_in["mag_src"], p_in["dmag_Lp_Ls"],
        raL=data["raL"], decL=data["decL"], root_tol=1e-8,
    )

    factories = build_factories(data, p_in, outdir, args)
    results = []
    for label, factory in factories:
        record = run_one(
            label, factory, outdir, data, p_in, truth_model, resume=args.resume
        )
        results.append(record)
        # Incremental report after each backend.
        write_html_report(results, outdir / "comparison_report.html")

    with open(outdir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nReport: {outdir / 'comparison_report.html'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
