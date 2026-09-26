#!/usr/bin/env python
"""
Phot+Astrom sampler comparison: MultiNest, NumPyro NUTS/SA/SMC-NUTS,
PyMC SMC, and jaxns ± grads.

Supports PSPL and PSBL models, narrow or open priors, and multiple injected
binary-lens scenarios. Produces JSON result records and an HTML report under
the output directory.
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

from bagle import fake_data
from bagle import model_fitter_jax as model_fitter
from bagle import model_jax as model
from bagle.b_sff_prior import phot_dataset_has_astrometry
from bagle.model_fitter_jax import MicrolensSolver, MicrolensSolverJaxLike

try:
    from report import write_html_report, write_suite_summary_html
except ImportError:  # pragma: no cover - package-style import
    from tests.psbl_sampler_compare.report import (
        write_html_report, write_suite_summary_html,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = HERE / "runs"

# Backend metadata for suite reports (JAX likelihood / gradient use).
BACKEND_META = {
    "multinest": {
        "jax": True, "grads": False, "family": "MultiNest",
        "notes": "JAX χ² lnL via MicrolensSolverJaxLike",
    },
    "multinest_host": {
        "jax": False, "grads": False, "family": "MultiNest",
        "notes": "Host (NumPy) lnL via MicrolensSolver",
    },
    "numpyro_nuts_grad": {
        "jax": True, "grads": True, "family": "NumPyro NUTS",
        "notes": "HMC-NUTS with JAX autodiff",
    },
    "numpyro_sa_nograd": {
        "jax": True, "grads": False, "family": "NumPyro SA",
        "notes": "Sample Adaptive MCMC; JAX lnL, no gradients",
    },
    "numpyro_smc_nuts": {
        "jax": True, "grads": True, "family": "NumPyro SMC-NUTS",
        "notes": "Tempered SMC with NUTS rejuvenation",
    },
    "pymc_smc": {
        "jax": True, "grads": True, "family": "PyMC SMC",
        "notes": "pm.sample_smc with JAX LogLikelihoodOp",
    },
    "pymc_smc_nojax": {
        "jax": False, "grads": False, "family": "PyMC SMC",
        "notes": "pm.sample_smc with host black-box lnL",
    },
    "jaxns_grad": {
        "jax": True, "grads": True, "family": "jaxns",
        "notes": "Nested sampling, gradient_guided=True",
    },
    "jaxns_nograd": {
        "jax": True, "grads": False, "family": "jaxns",
        "notes": "Nested sampling, gradient_guided=False",
    },
}

# Model families supported by this runner.
MODEL_CLASSES = {
    "psbl": model.PSBL_PhotAstrom_Par_Param1,
    "pspl": model.PSPL_PhotAstrom_noPar_Param1,
}

# Injected PhotAstrom Param1 binaries for open-prior robustness.
#
# Angle-suite note: with muL=(0,0), muS=(3,0) the relative PM is along +E
# (90° east of north). BAGLE ``alpha`` is the binary-axis PA east of north,
# so the directed angle from mu_rel to the binary axis is
#   φ ≡ (alpha - 90°) mod 360.
# Scenarios ``bulge_q0p5_phiXXX`` keep bulge-like masses/distances and a
# small |beta| so |u0|≪1 and the light curves show multiple caustic peaks.
SCENARIOS = {
    "bulge_q0p5": dict(
        description="Bulge-like q=0.5, sep≈θ_E, α=90°",
        seed=0,
        kwargs=dict(
            mLp=10.0, mLs=5.0, t0=57000.0, xS0_E=0.0, xS0_N=0.0, beta=2.0,
            muL_E=0.0, muL_N=0.0, muS_E=3.0, muS_N=0.0,
            dL=3000.0, dS=8000.0, sep=10.0, alpha=90.0,
            mag_src=14.0, b_sff=1.0, dmag_Lp_Ls=20.0,
            raL=259.5, decL=-29.0,
        ),
    ),
    "close_unequal": dict(
        description="Close unequal binary (q=0.1), α=35°, modest |β|",
        seed=1,
        kwargs=dict(
            mLp=8.0, mLs=0.8, t0=57120.0, xS0_E=0.0, xS0_N=0.0, beta=-1.2,
            muL_E=-1.5, muL_N=2.0, muS_E=2.5, muS_N=-0.5,
            dL=4000.0, dS=8000.0, sep=3.5, alpha=35.0,
            mag_src=15.5, b_sff=0.85, dmag_Lp_Ls=5.0,
            raL=268.0, decL=-29.5,
        ),
    ),
    "wide_near_equal": dict(
        description="Wider near-equal binary (q≈0.9), α=160°",
        seed=2,
        kwargs=dict(
            mLp=6.0, mLs=5.5, t0=56880.0, xS0_E=0.0, xS0_N=0.0, beta=3.5,
            muL_E=1.0, muL_N=-3.0, muS_E=4.0, muS_N=-1.0,
            dL=2500.0, dS=7500.0, sep=22.0, alpha=160.0,
            mag_src=16.0, b_sff=0.7, dmag_Lp_Ls=-2.0,
            raL=271.2, decL=-27.8,
        ),
    ),
    # --- mu_rel × binary-axis angle suite (open priors) ---
    "bulge_q0p5_phi000": dict(
        description="Bulge q=0.5 angle suite: φ=0° (α=90°, μ_rel∥+E), small |β|, multi-peak",
        seed=10,
        kwargs=dict(
            mLp=10.0, mLs=5.0, t0=57000.0, xS0_E=0.0, xS0_N=0.0, beta=0.4,
            muL_E=0.0, muL_N=0.0, muS_E=3.0, muS_N=0.0,
            dL=3000.0, dS=8000.0, sep=8.0, alpha=90.0,
            mag_src=14.0, b_sff=1.0, dmag_Lp_Ls=20.0,
            raL=259.5, decL=-29.0,
        ),
    ),
    "bulge_q0p5_phi045": dict(
        description="Bulge q=0.5 angle suite: φ=45° (α=135°, μ_rel∥+E), small |β|, multi-peak",
        seed=11,
        kwargs=dict(
            mLp=10.0, mLs=5.0, t0=57000.0, xS0_E=0.0, xS0_N=0.0, beta=0.4,
            muL_E=0.0, muL_N=0.0, muS_E=3.0, muS_N=0.0,
            dL=3000.0, dS=8000.0, sep=8.0, alpha=135.0,
            mag_src=14.0, b_sff=1.0, dmag_Lp_Ls=20.0,
            raL=259.5, decL=-29.0,
        ),
    ),
    "bulge_q0p5_phi090": dict(
        description="Bulge q=0.5 angle suite: φ=90° (α=180°, μ_rel∥+E), small |β|, multi-peak",
        seed=12,
        kwargs=dict(
            mLp=10.0, mLs=5.0, t0=57000.0, xS0_E=0.0, xS0_N=0.0, beta=0.4,
            muL_E=0.0, muL_N=0.0, muS_E=3.0, muS_N=0.0,
            dL=3000.0, dS=8000.0, sep=8.0, alpha=180.0,
            mag_src=14.0, b_sff=1.0, dmag_Lp_Ls=20.0,
            raL=259.5, decL=-29.0,
        ),
    ),
    "bulge_q0p5_phi135": dict(
        description="Bulge q=0.5 angle suite: φ=135° (α=225°, μ_rel∥+E), small |β|, multi-peak",
        seed=13,
        kwargs=dict(
            mLp=10.0, mLs=5.0, t0=57000.0, xS0_E=0.0, xS0_N=0.0, beta=0.4,
            muL_E=0.0, muL_N=0.0, muS_E=3.0, muS_N=0.0,
            dL=3000.0, dS=8000.0, sep=8.0, alpha=225.0,
            mag_src=14.0, b_sff=1.0, dmag_Lp_Ls=20.0,
            raL=259.5, decL=-29.0,
        ),
    ),

    # --- MB19284 / MOA-2019-BLG-284 PSBL candidate (Jessica approval pending) ---
    # Binary LENS (not binary source). Overleaf-BSPL scale (tE~910d, thetaE~6.3mas,
    # piE~0.1, M~7.7Msun, dL~1.3kpc) + q=1/18. Tuned for smooth multi-bump LC with
    # caustic APPROACH (no crossing). See reports/runs_open_mb19284_py314/examples/.
    # DO NOT launch sampler runs until approved.
    "mb19284_psbl_q018": dict(
        description=(
            "MB19284-like PSBL q=1/18, tE~910d, piE~0.1, smooth multi-bump, "
            "NO caustic crossing (candidate; approval pending)"
        ),
        seed=284,
        kwargs=dict(
            mLp=7.338449928004481,
            mLs=0.4076916626669156,
            t0=59405.0,
            xS0_E=0.0,
            xS0_N=0.0,
            beta=-1.6484,
            muL_E=0.0,
            muL_N=0.0,
            muS_E=0.25320485484664584,
            muS_N=-2.5320485484664585,
            dL=1298.7012987012986,
            dS=7528.419784687194,
            sep=2.853,
            alpha=60.0,
            mag_src=16.08,
            b_sff=1.0,
            dmag_Lp_Ls=12.0,
            raL=271.4795,
            decL=-30.3369,
        ),
    ),
}

# Single-lens PSPL scenario (fake_data1 injection).
PSPL_SCENARIOS = {
    "fake_data1": dict(
        description="PSPL PhotAstrom noPar (fake_data1)",
        seed=0,
    ),
}


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


def _bsff_phot_index(name):
    """Zero-based photometry index encoded in a ``b_sff`` name.

    Parameters
    ----------
    name : str
        Parameter name, for example ``b_sff1`` or bare ``b_sff``.

    Returns
    -------
    phot_idx : int
        ``b_sff1`` is 0. A name with no digits is dataset 0.
    """
    digits = "".join(c for c in name if c.isdigit())
    if digits:
        return int(digits) - 1

    return 0


def _clip_bsff_edges(fitter, name, lo, hi):
    """Cap a ``b_sff`` window when that dataset has astrometry.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver whose photometry-to-astrometry map selects the cap.
    name : str
        Parameter name, for example ``b_sff1``.
    lo : float
        Proposed lower edge.
    hi : float
        Proposed upper edge (truth plus half-width, or the open window).

    Returns
    -------
    lo : float
        Lower edge, kept strictly below ``hi``.
    hi : float
        Upper edge. At most 1.0 when this photometry dataset is paired
        with astrometry. Photometry-only datasets are left unchanged.

    Notes
    -----
    A truth of 1.0 plus a positive half-width would otherwise put the
    upper edge above 1. Clipping that edge can leave ``lo >= hi`` when
    the truth sits on the cap and the half-width is zero; a small gap
    is opened in that case so ``make_gen`` still receives a window.
    """
    lo = float(lo)
    hi = float(hi)
    if name.startswith("b_sff"):
        phot_idx = _bsff_phot_index(name)
        if phot_dataset_has_astrometry(fitter, phot_idx):
            hi = min(hi, 1.0)

    # Truth at the cap must not collapse the uniform interval.
    if lo >= hi:
        lo = hi - 1e-3

    return lo, hi


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
    widths["dL_dS"] = 0.01
    widths["mag_src1"] = 0.05
    widths["b_sff1"] = 0.05
    widths["dmag_Lp_Ls1"] = 0.5
    widths["alpha"] = 1.0
    widths["sep"] = 0.2
    widths["mL"] = 0.1
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
        # Absolute distances must stay positive; dL_dS is a ratio in (0, 1).
        if name in ("dL", "dS"):
            lo = max(lo, 1.0)
        if name == "dL_dS":
            lo = max(lo, 1e-3)
            hi = min(hi, 0.999)
        if name.startswith("sep") and lo <= 0:
            lo = 1e-3
        # Astrometry partners cannot draw b_sff > 1. Phot-only windows
        # keep truth ± half-width, including an edge above 1.
        if name.startswith("b_sff"):
            lo, hi = _clip_bsff_edges(fitter, name, lo, hi)
        fitter.priors[name] = make(name, lo, hi, stats_pkg=stats_pkg)

    return None


def apply_open_priors(fitter, p_in, stats_pkg="scipy"):
    """Apply open (much wider than narrow) priors for PhotAstrom Param1.

    Parameters
    ----------
    fitter : MicrolensSolver
        Solver whose ``priors`` are replaced.
    p_in : dict
        Injected fake-data parameters (used only to center wide windows).
    stats_pkg : {'scipy', 'numpyro'}, optional
        Prior backend matching the solver.

    Returns
    -------
    None

    Notes
    -----
    Priors are intentionally far wider than the narrow-around-truth
    comparison (~40× those half-widths for most lens params, with ``alpha``
    fully open on [0, 360)). Windows are still centered on the injected
    truth so a 17-D PSBL phot+astrom search remains tractable for a
    controlled backend comparison; the injected scenarios probe
    robustness across different binaries (including the φ angle suite). Data-driven generators are
    retained for ``xS0``, ``muS``, and ``mag_src``.
    """
    # Data-driven sites for well-measured astrometric / photometric params.
    fitter.make_default_priors(stats_pkg=stats_pkg)
    make = model_fitter.make_gen
    truth = _truth_dict(p_in, fitter.fitter_param_names)

    # Open absolute half-widths (~40× the narrow comparison).
    widths = {
        "mLp": 4.0, "mLs": 4.0, "t0": 200.0, "beta": 4.0,
        "sep": 8.0, "muL_E": 5.0, "muL_N": 5.0, "dL": 800.0, "dS": 1000.0,
        "b_sff1": 0.35, "dmag_Lp_Ls1": 12.0,
        "xS0_E": 5e-3, "xS0_N": 5e-3, "muS_E": 1.5, "muS_N": 1.5,
        "mag_src1": 1.0,
    }

    for name in fitter.fitter_param_names:
        # Fully open binary orientation.
        if name == "alpha":
            fitter.priors[name] = make(name, 0.0, 360.0, stats_pkg=stats_pkg)
            continue

        # Keep data-driven mag_src / xS0 / muS when already set, unless we
        # have an explicit open width override below.
        if name.startswith("mag_src") and name not in widths:
            digits = "".join(c for c in name if c.isdigit())
            filt = int(digits) if digits else 1
            fitter.priors[name] = model_fitter.make_mag_src_gen(
                name, fitter.data[f"mag{filt}"], stats_pkg=stats_pkg
            )
            continue

        half = widths.get(name)
        if half is None:
            # Fall back to any default prior already installed.
            if name not in fitter.priors:
                raise RuntimeError(f"Open priors missing width for: {name}")
            continue

        lo = truth[name] - half
        hi = truth[name] + half
        if name.startswith("mL") and lo <= 0:
            lo = 1e-3
        if name in ("dL", "dS"):
            lo = max(lo, 100.0)
        if name == "dL_dS":
            lo = max(lo, 1e-3)
            hi = min(hi, 0.999)
        if name.startswith("sep") and lo <= 0:
            lo = 1e-3
        if name.startswith("b_sff"):
            lo = max(lo, 0.01)
            phot_idx = _bsff_phot_index(name)
            # Photometry-only keeps the historical 1.5 cap. A dataset
            # paired with astrometry is clipped at 1.0.
            if phot_dataset_has_astrometry(fitter, phot_idx):
                hi = min(hi, 1.0)
            else:
                hi = min(hi, 1.5)
            lo, hi = _clip_bsff_edges(fitter, name, lo, hi)
        fitter.priors[name] = make(name, lo, hi, stats_pkg=stats_pkg)

    # Enforce dS > dL at the prior edges when both are free.
    if "dL" in fitter.priors and "dS" in fitter.priors:
        dL_hi = truth["dL"] + widths["dL"]
        dS_lo = max(truth["dS"] - widths["dS"], dL_hi + 100.0)
        dS_hi = max(truth["dS"] + widths["dS"], dS_lo + 100.0)
        fitter.priors["dS"] = make("dS", dS_lo, dS_hi, stats_pkg=stats_pkg)

    missing = [n for n in fitter.fitter_param_names if n not in fitter.priors]
    if missing:
        raise RuntimeError(f"Open priors missing for: {missing}")

    return None


def apply_priors(fitter, p_in, prior_mode, stats_pkg):
    """Dispatch narrow vs open prior setup.

    Parameters
    ----------
    fitter : MicrolensSolver
        Target solver.
    p_in : dict
        Injected parameters (used for narrow priors).
    prior_mode : {'narrow', 'open'}
        Prior width mode.
    stats_pkg : str
        Prior backend.

    Returns
    -------
    None
    """
    if prior_mode == "open":
        apply_open_priors(fitter, p_in, stats_pkg=stats_pkg)
    else:
        apply_narrow_priors(fitter, p_in, stats_pkg=stats_pkg)
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


def _dense_times(t_obs, cadence_days=10.0, pad_days=30.0):
    """Build a dense model time grid spanning the data (and seasonal gaps).

    Parameters
    ----------
    t_obs : array_like
        Observation epochs (MJD).
    cadence_days : float, optional
        Model sampling cadence through gaps (days).
    pad_days : float, optional
        Extra padding beyond the first/last observation.

    Returns
    -------
    t_mod : ndarray
        Sorted dense times in MJD.
    """
    t_obs = np.asarray(t_obs, dtype=float).ravel()
    t0 = float(np.nanmin(t_obs)) - pad_days
    t1 = float(np.nanmax(t_obs)) + pad_days
    return np.arange(t0, t1 + cadence_days, cadence_days)


def _save_trace_png(fitter, out_png, n_params=8):
    """Write a compact posterior-trace figure."""
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


def _save_model_data_png(fitter, best, truth_model, out_png, cadence_days=10.0):
    """Photometry + astrometry data vs oversampled best-fit / truth models.

    Parameters
    ----------
    fitter : MicrolensSolver
        Fitted solver (provides data).
    best : dict
        Best-fit parameter dictionary.
    truth_model : bagle model
        Injected-truth model.
    out_png : str or Path
        Output PNG path.
    cadence_days : float, optional
        Model oversampling cadence through seasonal gaps.

    Returns
    -------
    None
    """
    mod = fitter.get_model(best)
    data = fitter.data

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Photometry with dense model through gaps.
    t = data["t_phot1"]
    mag = data["mag1"]
    err = data["mag_err1"]
    t_mod = _dense_times(t, cadence_days=cadence_days)
    axes[0, 0].errorbar(
        t, mag, yerr=err, fmt=".", ms=2, alpha=0.5, color="k", label="data"
    )
    axes[0, 0].plot(
        t_mod, mod.get_photometry(t_mod), color="C1", lw=1.5, label="best fit"
    )
    axes[0, 0].plot(
        t_mod, truth_model.get_photometry(t_mod), color="C0", lw=1.0,
        ls="--", label="truth"
    )
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylabel("mag")
    axes[0, 0].set_title(f"Photometry (model Δt={cadence_days:g} d)")
    axes[0, 0].legend(fontsize=8)

    # Phot residuals at data epochs.
    mag_best = np.asarray(mod.get_photometry(t))
    axes[1, 0].errorbar(
        t, mag - mag_best, yerr=err, fmt=".", ms=2, alpha=0.5, color="k"
    )
    axes[1, 0].axhline(0.0, color="C1", lw=1)
    axes[1, 0].set_xlabel("MJD")
    axes[1, 0].set_ylabel("residual (mag)")

    # Astrometry on-sky: dense model tracks + data.
    ta = data["t_ast1"]
    xe, ye = data["xpos1"], data["ypos1"]
    xe_err, ye_err = data["xpos_err1"], data["ypos_err1"]
    t_ast_mod = _dense_times(ta, cadence_days=cadence_days)
    pos_b_dense = np.asarray(mod.get_astrometry(t_ast_mod))
    pos_t_dense = np.asarray(truth_model.get_astrometry(t_ast_mod))
    pos_b = np.asarray(mod.get_astrometry(ta))

    axes[0, 1].errorbar(
        xe * 1e3, ye * 1e3, xerr=xe_err * 1e3, yerr=ye_err * 1e3,
        fmt=".", ms=3, alpha=0.6, color="k", label="data", zorder=3
    )
    axes[0, 1].plot(
        pos_b_dense[:, 0] * 1e3, pos_b_dense[:, 1] * 1e3,
        color="C1", lw=1.5, label="best fit", zorder=2
    )
    axes[0, 1].plot(
        pos_t_dense[:, 0] * 1e3, pos_t_dense[:, 1] * 1e3,
        color="C0", lw=1.0, ls="--", label="truth", zorder=1
    )
    axes[0, 1].set_xlabel("East (mas)")
    axes[0, 1].set_ylabel("North (mas)")
    axes[0, 1].set_title(f"Astrometry (model Δt={cadence_days:g} d)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].invert_xaxis()

    # Astrom residuals vs time at data epochs.
    axes[1, 1].errorbar(
        ta, (xe - pos_b[:, 0]) * 1e3, yerr=xe_err * 1e3,
        fmt=".", ms=3, alpha=0.6, color="C3", label="E"
    )
    axes[1, 1].errorbar(
        ta, (ye - pos_b[:, 1]) * 1e3, yerr=ye_err * 1e3,
        fmt=".", ms=3, alpha=0.6, color="C0", label="N"
    )
    axes[1, 1].axhline(0.0, color="0.5", lw=1)
    axes[1, 1].set_xlabel("MJD")
    axes[1, 1].set_ylabel("residual (mas)")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return None


def run_one(label, factory, outdir, data, p_in, truth_model, resume=False,
            cadence_days=10.0, skip_fitter_plots=True, timeout_sec=None):
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
    cadence_days : float, optional
        Model oversampling cadence for model-vs-data plots.
    skip_fitter_plots : bool, optional
        If True, skip heavy ``plot_model_and_data`` PNG dumps.
    timeout_sec : float or None, optional
        Soft wall-clock limit for ``fitter.solve()`` via SIGALRM.

    Returns
    -------
    record : dict
        Summary metrics, paths, and best-fit parameters.
    """
    import signal

    print(f"\n===== Starting {label} =====", flush=True)
    run_dir = outdir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "label": label,
        "status": "running",
        "runtime_sec": None,
        "error": None,
    }

    def _alarm_handler(signum, frame):
        raise TimeoutError(f"{label} exceeded timeout of {timeout_sec:.0f}s")

    try:
        fitter = factory()
        t0 = time.time()
        old_handler = None
        if timeout_sec is not None and hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(int(timeout_sec))
        try:
            fitter.solve()
        finally:
            if timeout_sec is not None and hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
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
        _save_model_data_png(
            fitter, best, truth_model, model_png, cadence_days=cadence_days
        )

        if not skip_fitter_plots:
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
    """Construct sampler factory callables for the comparison suite.

    Parameters
    ----------
    data : dict
        Fake data dictionary.
    p_in : dict
        Injected truth parameters.
    outdir : Path
        Scenario output directory.
    args : argparse.Namespace
        CLI options (model family, sampler knobs, prior mode).

    Returns
    -------
    factories : list of (label, callable)
        Ordered backend factories.
    """
    model_kind = str(getattr(args, "model", "psbl")).lower()
    model_class = MODEL_CLASSES[model_kind]
    base = str(outdir) + os.sep
    prior_mode = args.prior_mode

    def multinest():
        # Wrap alpha on [0, 360) for MultiNest mode exploration (PSBL).
        n_dim = len(model_class.fitter_param_names)
        wrapped = [0] * n_dim
        if "alpha" in model_class.fitter_param_names:
            wrapped[model_class.fitter_param_names.index("alpha")] = 1

        fitter = MicrolensSolverJaxLike(
            data,
            model_class,
            n_live_points=args.mnest_live,
            max_iter=args.mnest_max_iter,
            evidence_tolerance=args.mnest_tol,
            # Open priors: constant-efficiency helps keep acceptance stable
            # on sharp PSBL peaks in wide prior volumes.
            sampling_efficiency=0.3 if args.prior_mode == "open" else 0.8,
            const_efficiency_mode=(args.prior_mode == "open"),
            multimodal=True,
            wrapped_params=wrapped,
            outputfiles_basename=base + "multinest_",
            dump_callback=None,
            verbose=bool(getattr(args, "verbose", False)),
            resume=args.resume,
        )
        apply_priors(fitter, p_in, prior_mode, stats_pkg="scipy")
        return fitter

    def multinest_host():
        # Classic MultiNest with host (NumPy) likelihood — no JAX lnL.
        n_dim = len(model_class.fitter_param_names)
        wrapped = [0] * n_dim
        if "alpha" in model_class.fitter_param_names:
            wrapped[model_class.fitter_param_names.index("alpha")] = 1

        fitter = MicrolensSolver(
            data,
            model_class,
            n_live_points=args.mnest_live,
            max_iter=args.mnest_max_iter,
            evidence_tolerance=args.mnest_tol,
            sampling_efficiency=0.3 if args.prior_mode == "open" else 0.8,
            const_efficiency_mode=(args.prior_mode == "open"),
            multimodal=True,
            wrapped_params=wrapped,
            outputfiles_basename=base + "multinest_host_",
            dump_callback=None,
            verbose=bool(getattr(args, "verbose", False)),
            resume=args.resume,
        )
        apply_priors(fitter, p_in, prior_mode, stats_pkg="scipy")
        return fitter

    # Progress bars / MultiNest chatter when --verbose is set.
    verb = bool(getattr(args, "verbose", False))

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
            max_tree_depth=8 if args.prior_mode == "open" else 10,
            random_seed=0,
            outputfiles_basename=base + "nuts_",
            verbose=verb,
        )
        apply_priors(fitter, p_in, prior_mode, stats_pkg="numpyro")
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
            verbose=verb,
        )
        apply_priors(fitter, p_in, prior_mode, stats_pkg="numpyro")
        return fitter

    def smc_nuts():
        # Tempered SMC with per-particle NUTS rejuvenation.
        fitter = model_fitter.MicrolensSolverNumPyro(
            data,
            model_class,
            sampler="smc_nuts",
            use_jax_grad=True,
            n_live_points=args.smc_particles,
            n_temperatures=args.smc_temperatures,
            tune=args.smc_nuts_tune,
            smc_rejuvenate_steps=args.smc_nuts_steps,
            max_tree_depth=8 if model_kind == "psbl" else 10,
            chain_method="sequential",
            random_seed=4,
            outputfiles_basename=base + "smc_nuts_",
            verbose=verb,
        )
        apply_priors(fitter, p_in, prior_mode, stats_pkg="numpyro")
        return fitter

    def pymc_smc():
        fitter = model_fitter.MicrolensSolverPyMC(
            data,
            model_class,
            sampler="smc",
            use_jax_grad=True,
            draws=args.pymc_smc_draws,
            chains=args.pymc_smc_chains,
            cores=1,
            pymc_random_seed=5,
            outputfiles_basename=base + "pymc_smc_",
            verbose=verb,
        )
        apply_priors(fitter, p_in, prior_mode, stats_pkg="scipy")
        return fitter

    def pymc_smc_nojax():
        fitter = model_fitter.MicrolensSolverPyMC(
            data,
            model_class,
            sampler="smc",
            use_jax_grad=False,
            draws=args.pymc_smc_draws,
            chains=args.pymc_smc_chains,
            cores=1,
            pymc_random_seed=6,
            outputfiles_basename=base + "pymc_smc_nojax_",
            verbose=verb,
        )
        apply_priors(fitter, p_in, prior_mode, stats_pkg="scipy")
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
            verbose=verb,
        )
        apply_priors(fitter, p_in, prior_mode, stats_pkg="numpyro")
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
            verbose=verb,
        )
        apply_priors(fitter, p_in, prior_mode, stats_pkg="numpyro")
        return fitter

    factories = [
        ("multinest", multinest),
        ("multinest_host", multinest_host),
        ("numpyro_nuts_grad", nuts),
        ("numpyro_sa_nograd", sa),
        ("numpyro_smc_nuts", smc_nuts),
        ("pymc_smc", pymc_smc),
        ("pymc_smc_nojax", pymc_smc_nojax),
        ("jaxns_grad", jaxns_grad),
        ("jaxns_nograd", jaxns_nograd),
    ]
    if args.only:
        keep = set(args.only.split(","))
        factories = [f for f in factories if f[0] in keep]
    return factories


def make_fake_data(scenario_name, model_kind="psbl", seed=None, outdir=None):
    """Generate noisy PhotAstrom fake data for one scenario.

    Parameters
    ----------
    scenario_name : str
        Key into :data:`SCENARIOS` (PSBL) or :data:`PSPL_SCENARIOS`.
    model_kind : {'psbl', 'pspl'}, optional
        Model family.
    seed : int or None, optional
        RNG seed override (defaults to the scenario seed).
    outdir : Path or None, optional
        Directory for optional fake-data figure dumps.

    Returns
    -------
    data, p_in, truth_model
        Fake data dict, injected parameter dict, and truth model instance.
    """
    model_kind = str(model_kind).lower()
    if model_kind == "pspl":
        if scenario_name not in PSPL_SCENARIOS:
            raise KeyError(
                f"Unknown PSPL scenario {scenario_name!r}; "
                f"choose from {sorted(PSPL_SCENARIOS)}"
            )
        sc = PSPL_SCENARIOS[scenario_name]
        if seed is None:
            seed = sc["seed"]
        np.random.seed(seed)
        outroot = str(outdir) + os.sep if outdir is not None else "./"
        data, p_in = fake_data.fake_data1(
            plot=False, verbose=False, outdir=outroot, target="pspl"
        )
        # Build truth model from injected parameters.
        truth_model = model.PSPL_PhotAstrom_noPar_Param1(
            p_in["mL"], p_in["t0"], p_in["beta"], p_in["dL"],
            p_in["dL_dS"],
            p_in["xS0_E"], p_in["xS0_N"],
            p_in["muL_E"], p_in["muL_N"], p_in["muS_E"], p_in["muS_N"],
            np.atleast_1d(p_in["b_sff"]).tolist(),
            np.atleast_1d(p_in["mag_src"]).tolist(),
        )
        return data, p_in, truth_model

    if scenario_name not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario {scenario_name!r}; "
            f"choose from {sorted(SCENARIOS)}"
        )
    sc = SCENARIOS[scenario_name]
    if seed is None:
        seed = sc["seed"]
    np.random.seed(seed)
    data, p_in, _, _ = fake_data.fake_data_PSBL(
        parallax=True, animate=False, **sc["kwargs"]
    )
    truth_model = model.PSBL_PhotAstrom_Par_Param1(
        p_in["mLp"], p_in["mLs"], p_in["t0"], p_in["xS0_E"], p_in["xS0_N"],
        p_in["beta"], p_in["muL_E"], p_in["muL_N"], p_in["muS_E"], p_in["muS_N"],
        p_in["dL"], p_in["dS"], p_in["sep"], p_in["alpha"],
        p_in["b_sff"], p_in["mag_src"], p_in["dmag_Lp_Ls"],
        raL=data["raL"], decL=data["decL"], root_tol=1e-8,
    )
    return data, p_in, truth_model


def _scenario_catalog(model_kind):
    """Return the scenario dict for the selected model family."""
    if str(model_kind).lower() == "pspl":
        return PSPL_SCENARIOS
    return SCENARIOS


def run_scenario(scenario_name, args, outdir):
    """Run all backends for one injected scenario and write its HTML report.

    Parameters
    ----------
    scenario_name : str
        Scenario key.
    args : argparse.Namespace
        CLI options.
    outdir : Path
        Scenario output directory.

    Returns
    -------
    results : list of dict
        Per-backend result records.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_kind = str(getattr(args, "model", "psbl")).lower()
    catalog = _scenario_catalog(model_kind)
    sc = catalog[scenario_name]
    print(
        f"\n######## {model_kind.upper()} scenario {scenario_name}: "
        f"{sc['description']} ########",
        flush=True,
    )

    data, p_in, truth_model = make_fake_data(
        scenario_name, model_kind=model_kind, seed=args.seed, outdir=outdir
    )
    with open(outdir / "fake_data.pkl", "wb") as f:
        pickle.dump(
            {
                "data": data,
                "p_in": p_in,
                "scenario": scenario_name,
                "description": sc["description"],
                "model": model_kind,
                "kwargs": sc.get("kwargs"),
            },
            f,
        )

    factories = build_factories(data, p_in, outdir, args)
    title = (
        f"{model_kind.upper()} {args.prior_mode}-prior comparison — "
        f"{scenario_name}"
    )
    subtitle = (
        f"{sc['description']} · model=<code>{model_kind}</code> · "
        f"prior_mode=<code>{args.prior_mode}</code> · "
        f"model oversampled at {args.model_cadence:g} d · "
        "MultiNest / NUTS / SA / SMC-NUTS / PyMC-SMC / jaxns±grad"
    )

    results = []
    # Reuse completed backend results when present (for resume after kills).
    for label, factory in factories:
        prior = outdir / label / "result.json"
        if args.resume and prior.exists():
            with open(prior) as f:
                record = json.load(f)
            # Only skip successful runs so failed/timed-out backends can retry.
            if record.get("status") == "ok":
                print(
                    f"===== Skipping {label} "
                    f"(resume, status={record.get('status')}) =====",
                    flush=True,
                )
                record["scenario"] = scenario_name
                record["prior_mode"] = args.prior_mode
                record["model"] = model_kind
                record.update(BACKEND_META.get(label, {}))
                results.append(record)
                write_html_report(
                    results, outdir / "comparison_report.html",
                    title=title, subtitle=subtitle,
                )
                continue
            print(
                f"===== Re-running {label} "
                f"(resume found status={record.get('status')}) =====",
                flush=True,
            )
        # Soft timeouts for known long / hanging open-prior backends.
        timeout = None
        if args.prior_mode == "open" and "nuts" in label and "smc" not in label:
            timeout = 1200.0
        elif args.prior_mode == "open" and "jaxns" in label:
            timeout = 7200.0
        elif "smc_nuts" in label:
            # Per-particle NUTS rejuvenation is expensive (esp. open / PSBL).
            timeout = 10800.0 if model_kind == "psbl" else 7200.0
        elif args.prior_mode == "open" and "pymc_smc" in label:
            timeout = 7200.0
        record = run_one(
            label, factory, outdir, data, p_in, truth_model,
            resume=args.resume, cadence_days=args.model_cadence,
            skip_fitter_plots=not args.fitter_plots,
            timeout_sec=timeout,
        )
        record["scenario"] = scenario_name
        record["prior_mode"] = args.prior_mode
        record["model"] = model_kind
        record.update(BACKEND_META.get(label, {}))
        results.append(record)
        write_html_report(
            results, outdir / "comparison_report.html",
            title=title, subtitle=subtitle,
        )

    with open(outdir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nReport: {outdir / 'comparison_report.html'}", flush=True)
    return results


def parse_args(argv=None):
    """Parse CLI arguments for the comparison runner."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--seed", type=int, default=None,
                   help="Override scenario RNG seed")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated subset of run labels")
    p.add_argument(
        "--model", choices=("psbl", "pspl"), default="psbl",
        help="Model family (default: psbl)",
    )
    p.add_argument(
        "--prior-mode", choices=("narrow", "open"), default="open",
        help="Prior width (default: open)",
    )
    p.add_argument(
        "--scenario", type=str, default="all",
        help="Scenario name, comma list, or 'all'",
    )
    p.add_argument("--model-cadence", type=float, default=10.0,
                   help="Model oversampling cadence in days")
    p.add_argument("--fitter-plots", action="store_true",
                   help="Also write fitter.plot_model_and_data PNGs")
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable sampler progress bars / verbose MultiNest output",
    )
    # Nested / MCMC knobs (open-prior defaults are more generous).
    p.add_argument("--mnest-live", type=int, default=200)
    p.add_argument("--mnest-max-iter", type=int, default=50000)
    p.add_argument("--mnest-tol", type=float, default=0.5)
    p.add_argument("--nuts-draws", type=int, default=1000)
    p.add_argument("--nuts-tune", type=int, default=1000)
    p.add_argument("--nuts-chains", type=int, default=2)
    p.add_argument("--sa-draws", type=int, default=6000)
    p.add_argument("--sa-tune", type=int, default=2500)
    p.add_argument("--jaxns-live", type=int, default=150)
    p.add_argument("--jaxns-max-samples", type=int, default=80000)
    p.add_argument("--jaxns-dlogz", type=float, default=0.5)
    p.add_argument("--jaxns-posterior", type=int, default=1500)
    # SMC knobs (NumPyro SMC-NUTS and PyMC SMC).
    p.add_argument("--smc-particles", type=int, default=40,
                   help="NumPyro SMC-NUTS particle count")
    p.add_argument("--smc-temperatures", type=int, default=6,
                   help="NumPyro SMC-NUTS tempering levels")
    p.add_argument("--smc-nuts-tune", type=int, default=40,
                   help="NUTS warmup steps per SMC rejuvenation")
    p.add_argument("--smc-nuts-steps", type=int, default=1,
                   help="NUTS draws kept per particle per SMC stage")
    p.add_argument("--pymc-smc-draws", type=int, default=500,
                   help="PyMC SMC particle count")
    p.add_argument("--pymc-smc-chains", type=int, default=2,
                   help="PyMC SMC independent chains")
    return p.parse_args(argv)


def main(argv=None):
    """Run sampler comparison(s) and write HTML report(s)."""
    args = parse_args(argv)
    root = args.outdir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    model_kind = str(args.model).lower()
    catalog = _scenario_catalog(model_kind)

    if args.scenario == "all":
        names = list(catalog.keys())
    else:
        names = [s.strip() for s in args.scenario.split(",") if s.strip()]
        for name in names:
            if name not in catalog:
                raise SystemExit(
                    f"Unknown scenario {name!r}; "
                    f"choose from {sorted(catalog)} or 'all'"
                )

    index = []
    for name in names:
        outdir = root / name
        results = run_scenario(name, args, outdir)
        index.append(
            {
                "scenario": name,
                "description": catalog[name]["description"],
                "report": str(outdir / "comparison_report.html"),
                "n_ok": sum(1 for r in results if r.get("status") == "ok"),
                "n_total": len(results),
            }
        )

    # Lightweight index page linking all scenario reports.
    index_path = root / "index.html"
    rows = []
    for item in index:
        rows.append(
            "<tr>"
            f"<td>{item['scenario']}</td>"
            f"<td>{item['description']}</td>"
            f"<td>{item['n_ok']}/{item['n_total']}</td>"
            f"<td><a href='{item['scenario']}/comparison_report.html'>"
            "open report</a></td>"
            "</tr>"
        )
    index_path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{model_kind.upper()} sampler comparisons</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 2rem; background: #f4f5f7; color: #20242a; }}
table {{ border-collapse: collapse; background: white; }}
th, td {{ padding: 10px 14px; border-bottom: 1px solid #dde1e8; text-align: left; }}
th {{ background: #eef1f5; }}
</style></head><body>
<h1>{model_kind.upper()} sampler comparisons</h1>
<p>Model: <code>{model_kind}</code> · Prior mode: <code>{args.prior_mode}</code> ·
model cadence: {args.model_cadence:g} d</p>
<table>
<tr><th>scenario</th><th>description</th><th>backends ok</th><th>report</th></tr>
{''.join(rows)}
</table>
</body></html>
"""
    )
    print(f"\nIndex: {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
