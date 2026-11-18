"""
Timing and accuracy of the njx FSPL host paths, JAX kernel and VBM.

Accuracy is measured against a high-precision reference: the uniform-source
magnification and centroid shift (along u-hat) from the Lee et al. (2010)
Eq. 10 one-dimensional integrals evaluated with ``scipy.integrate.quad``.
VBM (``ESPLMag2`` / ``astrox1``) is scored the same way; near the limb it is
only good to ~1e-3 in magnification.

The njx variants sweep the adaptive-mesh tolerance ``amg_tol`` (used when
``|u| < 2 rho``) and ``n_outline`` (uniform contour integration farther out).

Writes ``tests/plots/model_vs_vbm/fspl_fix_timing.{txt,png}``.

Run with::

    PYTHONPATH=src python tests/_bench_fspl_fixes.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax

import test_model_vs_vbm as T
from bagle.jax.fspl import fspl_magnification_centroid

N_REPEAT = 5

# (label, amg_tol, n_outline)
VARIANTS = [
    ("amg_tol=1e-3", 1e-3, 200),
    ("amg_tol=1e-4 (default)", 1e-4, 200),
    ("amg_tol=1e-5", 1e-5, 200),
    ("amg_tol=1e-4, n_outline=400", 1e-4, 400),
]


lee10_reference = T._lee10_quad_reference


def time_call(fn, n_repeat=N_REPEAT):
    """
    Median wall time of ``fn()`` after one warm-up call.

    Parameters
    ----------
    fn : callable
    n_repeat : int

    Returns
    -------
    t_med : float
        Median time in seconds.
    out : object
        Return value of the last call.
    """
    out = fn()
    times = []
    for _ in range(n_repeat):
        t_start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t_start)

    return np.median(times), out


def error_summary(shift, shift_ref, amp, amp_ref, u_amp, rho):
    """
    Max centroid-shift (along u-hat) and fractional magnification errors.

    Parameters
    ----------
    shift, shift_ref : ndarray, shape (N,)
        Shifts along u-hat (theta_E).
    amp, amp_ref : ndarray, shape (N,)
        Magnifications.
    u_amp : ndarray, shape (N,)
    rho : float

    Returns
    -------
    errs : dict
        Centroid errors ``outside`` (> 1.05 rho), ``limb`` (+-5%), ``inside``
        (< 0.95 rho); magnification errors ``phot`` (off-limb) and
        ``phot_limb``.
    """
    d = np.abs(shift - shift_ref)
    dA = np.abs(amp / amp_ref - 1)
    outside, limb, inside = T._limb_masks(u_amp, rho, band=0.05)
    errs = dict(outside=np.max(d[outside]), limb=np.max(d[limb]),
                inside=np.max(d[inside]) if inside.any() else np.nan,
                phot=np.max(dA[outside | inside]), phot_limb=np.max(dA[limb]))

    return errs


def run_case(u0, rho):
    """
    Time and score every njx variant, the JAX kernel and VBM on one event.

    Parameters
    ----------
    u0, rho : float

    Returns
    -------
    rows : list of (label, seconds, errs)
    n_t : int
        Number of epochs.
    """
    mod = T._get_model_Lee2010(u0, rho)
    th = mod.thetaE_amp
    rho = mod.source_radius_thetaE_units()

    # Lee et al. window, densely sampled across the source.
    tau = np.unique(np.concatenate([np.linspace(-1.5, 1.5, 301),
                                    np.linspace(-1.5 * rho, 1.5 * rho, 301)]))
    t = mod.t0 + mod.tE * tau
    u = mod.get_u(t)
    ua = np.linalg.norm(u, axis=1)

    # High-precision reference.
    ref = np.array([lee10_reference(x, rho) for x in ua])
    a_ref, s_ref = ref[:, 0], ref[:, 1]

    rows = []
    for label, tol, n_out in VARIANTS:
        mod.amg_tol = tol
        mod.n_outline = n_out

        def run_njx():
            a = mod.get_amplification(t)
            s = T._along_uhat(mod.get_centroid_shift(t) / th, u)
            return a, s

        sec, (a, s) = time_call(run_njx)
        rows.append((label, sec, error_summary(s, s_ref, a, a_ref, ua, rho)))
    del mod.amg_tol
    mod.n_outline = T.N_OUTLINE

    # JAX kernel (jitted; static limb).
    def run_jax():
        mag, cen = fspl_magnification_centroid(u, rho, T.N_OUTLINE)
        jax.block_until_ready(cen)
        shift = np.asarray(cen) - u
        return np.asarray(mag), T._along_uhat(shift, u)

    sec, (a, s) = time_call(run_jax)
    rows.append(("jax kernel (static limb)", sec,
                 error_summary(s, s_ref, a, a_ref, ua, rho)))

    # VBM ESPL (tabulated / contour hybrid).
    phot = T._kernel_model(abs(mod.u0_amp), rho, t0=mod.t0, tE=mod.tE)
    vbm, par = T._get_vbm(phot, tol=1e-6, rel_tol=1e-7, astrometry=True)

    def run_vbm():
        a = np.array(vbm.ESPLLightCurve(par, list(t))[0])
        return a, T._vbm_shift_uhat(vbm, ua, rho)

    sec, (a, s) = time_call(run_vbm)
    rows.append(("VBM ESPL", sec, error_summary(s, s_ref, a, a_ref, ua, rho)))

    return rows, len(t)


def main():
    """Run both cases, print and save the table and a summary plot."""
    cases = [(0.1, 0.5), (0.02, 0.05)]
    lines = []
    results = {}
    for u0, rho in cases:
        rows, n_t = run_case(u0, rho)
        results[(u0, rho)] = rows
        lines.append(f"u0={u0}, rho={rho}, {n_t} epochs "
                     f"(median of {N_REPEAT}, after warm-up)")
        lines.append(f"{'method':30s} {'time [ms]':>10s} {'shift out':>10s} "
                     f"{'limb':>9s} {'in':>9s} {'|dA/A|':>9s} {'limb':>9s}")
        for label, sec, e in rows:
            lines.append(
                f"{label:30s} {sec * 1e3:10.1f} {e['outside']:10.1e} "
                f"{e['limb']:9.1e} {e['inside']:9.1e} {e['phot']:9.1e} "
                f"{e['phot_limb']:9.1e}")
        lines.append("")

    text = ("FSPL time and max error vs Lee+2010 Eq. 10 quadrature "
            "(shift along u-hat in theta_E; limb = |u|/rho in 0.95-1.05)\n\n"
            + "\n".join(lines))
    print(text)
    (T.PLOT_DIR / "fspl_fix_timing.txt").write_text(text)

    # Time (left) and centroid errors by region (right) per method.
    fig, axes = plt.subplots(len(cases), 2, figsize=(13.0, 4.2 * len(cases)))
    for row_ax, (u0, rho) in zip(axes, cases):
        rows = results[(u0, rho)]
        labels = [r[0] for r in rows]
        y = np.arange(len(rows))
        row_ax[0].barh(y, [r[1] * 1e3 for r in rows], color="C0")
        row_ax[0].set_yticks(y, labels, fontsize=8)
        row_ax[0].invert_yaxis()
        row_ax[0].set_xscale("log")
        row_ax[0].set_xlabel("time [ms]")
        row_ax[0].set_title(f"u0={u0}, rho={rho}: compute time", fontsize=10)

        for k, (key, col) in enumerate([("outside", "C2"), ("limb", "C1"),
                                        ("inside", "C3")]):
            vals = [max(r[2][key], 1e-10) for r in rows]
            row_ax[1].barh(y + (k - 1) * 0.27, vals, height=0.27,
                           color=col, label=key)
        row_ax[1].axvline(1e-4, color="k", ls=":", lw=0.8)
        row_ax[1].set_xscale("log")
        row_ax[1].set_yticks(y, labels, fontsize=8)
        row_ax[1].invert_yaxis()
        row_ax[1].set_xlabel(r"max |shift - quad| [$\theta_E$]")
        row_ax[1].set_title("centroid error vs Lee Eq. 10 (dotted: 1e-4)",
                            fontsize=10)
        row_ax[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(T.PLOT_DIR / "fspl_fix_timing.png", dpi=120)
    plt.close(fig)

    return None


if __name__ == "__main__":
    main()
