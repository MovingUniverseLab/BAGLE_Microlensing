"""
Audit the t0 / x0 reference frames of every binary-lens parameterization.

Convention under test: internally, ``self.t0``, ``self.u0`` and ``self.xL0``
always refer to the geometric center (midpoint) of the lens pair; COM
quantities are only used to place orbiting lenses. Input parameters may be
given about the geometric center, the lens CoM, or the primary lens, and
must be honored in that frame.

For each Param class this script builds one no-parallax model (parallax is
switched off on Par-only classes), measures the actual source-lens geometry,
and writes ``binary_frame_audit.md`` plus one figure per representative case
into this directory.

Run with::

    PYTHONPATH=src python docs/audits/binary_frames/make_binary_frame_audit.py
"""
from __future__ import annotations

import inspect
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
from bagle import model_jax as model

OUTDIR = Path(__file__).resolve().parent

LEAF = re.compile(r"^(PSBL|FSBL|BSBL|BFSBL)_(Phot|PhotAstrom)(_Limb)?_(noPar|Par)_")

# Separations for PhotAstrom (mas) and phot-only (Einstein radii) models.
ASTRO = dict(sep=3.0, sepL=3.0, sepS=0.5, a=3.0, aL=3.0, aS=0.6)
PHOT = dict(sep=0.8, sep_L=0.8, sep_S=0.1, sep_SL=0.3)

# One value per constructor argument; each class takes what it needs.
VALS = dict(
    mLp=10.0, mLs=5.0, t0=57000.0, t0_com=57000.0, t0_p=57000.0,
    t0_prim=57000.0, u0_amp=0.1, u0_amp_com=0.1, u0_amp_prim=0.1,
    beta=0.4, beta_com=0.4, beta_p=0.4, tE=100.0, thetaE=5.0,
    log10_thetaE=np.log10(5.0), piS=0.25, piE_E=0.05, piE_N=0.03,
    piEN_piEE=0.6, xS0_E=0.0, xS0_N=0.01, muL_E=0.0, muL_N=0.0, muS_E=4.0,
    muS_N=-3.0, dL=2000.0, dS=4000.0, q=0.5, alpha=90.0, alphaL=90.0,
    alphaS=30.0, phi=40.0, phi_L=40.0, phi_S=20.0, omega_pri=30.0,
    big_omega_sec=20.0, i=40.0, e=0.2, tp=56950.0, omegaL_pri=30.0,
    big_omegaL_sec=20.0, iL=40.0, eL=0.2, tpL=56950.0, omegaS_pri=10.0,
    big_omegaS_sec=70.0, iS=20.0, eS=0.1, pS=300.0, tpS=56900.0,
    alephS=0.2, aleph_secS=0.4, mSp=1.0, mSs=0.5,
    aleph=0.4, aleph_sec=0.8, r_s=0.5, a_s=1.0, v_para=0.01, v_perp=0.005,
    v_rad=-0.03, accLsec_E=2.0, accLsec_N=1.0, accSsec_E=1.0, accSsec_N=0.5,
    delta_muL_sec_E=1.5, delta_muL_sec_N=-1.0, delta_muS_sec_E=0.5,
    delta_muS_sec_N=0.3, radiusS=1e-4, radiusS_pri=1e-4, radiusS_sec=1e-4,
    n_outline=20, fratio_bin=0.5, mag_src=18.0, mag_base=18.0,
    mag_src_pri=18.0, mag_src_sec=18.75, b_sff=1.0, dmag_Lp_Ls=20.0)
FILT = {"xS0_E", "xS0_N", "b_sff", "mag_src", "mag_base", "mag_src_pri",
        "mag_src_sec", "dmag_Lp_Ls", "fratio_bin"}
XS0_IN = np.array([0.0, 0.01])  # arcsec

FRAME_OF = {"t0": "geometric", "t0_com": "CoM", "t0_p": "primary",
            "t0_prim": "primary"}

# Representative class for each row of the summary table.
CASES = [
    ("geometric, static", "PSBL_PhotAstrom_noPar_Param1"),
    ("geometric, static (phot-only)", "PSBL_Phot_noPar_Param1"),
    ("geometric, static (binary source)", "BSBL_PhotAstrom_noPar_Param1"),
    ("CoM, static", "PSBL_PhotAstrom_noPar_Param4"),
    ("primary (t0_prim), static", "PSBL_PhotAstrom_Par_Param5"),
    ("primary (t0_p), static", "PSBL_PhotAstrom_Par_Param7"),
    ("geometric, Keplerian orbit", "PSBL_PhotAstrom_noPar_EllOrbs_Param2"),
    ("CoM (beta_com), Keplerian orbit", "PSBL_PhotAstrom_noPar_EllOrbs_Param1"),
    ("CoM (u0_amp_com), Keplerian orbit", "PSBL_PhotAstrom_noPar_EllOrbs_Param4"),
    ("primary, Keplerian orbit", "PSBL_PhotAstrom_noPar_EllOrbs_Param7"),
    ("phot-only Keplerian orbit", "PSBL_Phot_noPar_EllOrbs_Param1"),
    ("binary source, Keplerian orbit", "BSBL_PhotAstrom_noPar_EllOrbs_Param1"),
]


def param_class(leaf):
    """
    Param mixin of a composed leaf (the class that defines fitter names).

    Parameters
    ----------
    leaf : type
        Composed model class.

    Returns
    -------
    pc : type or None
        First base whose name ends in ``ParamN`` and defines
        ``fitter_param_names``.
    """
    for base in leaf.__mro__[1:]:
        if "fitter_param_names" in base.__dict__ and re.search(
                r"Param\d+$", base.__name__):
            return base
    return None


def build(leafname):
    """
    Instantiate a leaf from ``VALS`` with parallax switched off.

    Parameters
    ----------
    leafname : str
        Name of the composed class in :mod:`bagle.model_jax`.

    Returns
    -------
    m : model instance or str
        The model, or an error message.
    """
    leaf = getattr(model, leafname)
    sig = inspect.signature(param_class(leaf).__init__)
    vals = dict(VALS, **(PHOT if "_Phot_" in leafname else ASTRO))
    kw = {}
    for arg, par in sig.parameters.items():
        if arg in ("self", "obsLocation", "root_tol"):
            continue
        if arg in ("raL", "decL"):
            kw[arg] = 268.0 if arg == "raL" else -29.0
            continue
        if arg not in vals:
            if par.default is inspect.Parameter.empty:
                return f"no value for {arg}"
            continue
        kw[arg] = [vals[arg]] if arg in FILT else vals[arg]
    try:
        m = leaf(**kw)
    except Exception as err:  # noqa: BLE001 - report and move on
        return f"{type(err).__name__}: {str(err)[:80]}"

    # Rectilinear tracks only: closest approach is then well defined.
    m.parallaxFlag = False
    return m


def in_einstein(m, pos):
    """
    Sky positions in Einstein radii.

    Parameters
    ----------
    m : model instance
        BAGLE model.
    pos : ndarray
        Positions as returned by the model (arcsec for PhotAstrom,
        Einstein radii for phot-only).

    Returns
    -------
    pos_ein : ndarray
        Same positions in Einstein radii.
    """
    if m.astrometryFlag:
        return pos / (m.thetaE_amp * 1e-3)
    return pos


def source_pos(m, t):
    """
    Unlensed primary-source position (Einstein radii).

    Parameters
    ----------
    m : model instance
        BAGLE model.
    t : ndarray, shape (N_times,)
        Times (MJD).

    Returns
    -------
    xS : ndarray, shape (N_times, 2)
        Primary source (the only source for single-source models).
    """
    if not m.astrometryFlag:
        u = m.get_u(t)
        return u[:, 0] if u.ndim == 3 else u
    if hasattr(m, "get_resolved_source_astrometry_unlensed"):
        return in_einstein(m, m.get_resolved_source_astrometry_unlensed(t)[:, 0])
    return in_einstein(m, m.get_source_astrometry_unlensed(t))


def lens_point(m, t, frame):
    """
    Lens reference point (Einstein radii).

    Parameters
    ----------
    m : model instance
        BAGLE model.
    t : ndarray, shape (N_times,)
        Times (MJD).
    frame : {'geometric', 'CoM', 'primary'}
        Lens midpoint, mass-weighted CoM, or primary lens.

    Returns
    -------
    xL : ndarray, shape (N_times, 2)
        Reference positions.
    """
    # Shape of lens: [N_times, 2 lenses, 2].
    lens = in_einstein(m, m.get_resolved_lens_astrometry(t))
    if frame == "geometric":
        return 0.5 * (lens[:, 0] + lens[:, 1])
    if frame == "CoM":
        mp, ms = (m.mLp, m.mLs) if hasattr(m, "mLp") else (1.0, m.q)
        return (mp * lens[:, 0] + ms * lens[:, 1]) / (mp + ms)
    return lens[:, 0]


def closest_approach(m, frame, t_center):
    """
    Closest approach of the primary source to a lens reference point.

    Parameters
    ----------
    m : model instance
        BAGLE model with parallax off.
    frame : {'geometric', 'CoM', 'primary'}
        Reference point.
    t_center : float
        Center of the search window (MJD).

    Returns
    -------
    t_min : float
        Time of closest approach (MJD).
    d_min : float
        Separation then (Einstein radii).
    """
    # The midpoint and primary of an orbiting binary move non-linearly, so
    # their (t0, u0) are defined at an instant: at t_center the source is
    # u0 from the point's position then, perpendicular to thetaE_hat. Report
    # the along-track offset as a time shift and the cross-track distance.
    if m.orbitFlag and frame != "CoM":
        tt = np.array([t_center])
        rel = (source_pos(m, tt) - lens_point(m, tt, frame))[0]
        along = rel @ m.thetaE_hat
        return t_center - m.tE * along, np.hypot(*(rel - along * m.thetaE_hat))

    # Rectilinear tracks: fine grid over +-2 tE.
    t = np.linspace(t_center - 2.0 * m.tE, t_center + 2.0 * m.tE, 40001)
    d = np.hypot(*(source_pos(m, t) - lens_point(m, t, frame)).T)
    k = np.argmin(d)
    return t[k], d[k]


def input_frame(pc):
    """
    Input frame and parameter names from the Param constructor.

    Parameters
    ----------
    pc : type
        Param mixin.

    Returns
    -------
    t_name, u_name, frame : str
        Input time and impact-parameter names, and their frame.
    """
    args = list(inspect.signature(pc.__init__).parameters)
    t_name = next(a for a in args if a.startswith("t0"))
    u_name = next(a for a in args if a.startswith(("u0", "beta")))
    return t_name, u_name, FRAME_OF[t_name]


def xs0_doc_time(pc, default):
    """
    Time at which the docstring places ``xS0``.

    Parameters
    ----------
    pc : type
        Param mixin.
    default : str
        Fallback attribute name.

    Returns
    -------
    name : str
        ``t0`` / ``t0_com`` / ``t0_p`` ... as written in the docstring.
    """
    doc = inspect.getdoc(pc) or ""
    m = re.search(r"xS0_E\s*:.*?at t\s*=\s*(t0\w*)", doc, re.S)
    return m.group(1) if m else default


def measure(leafname):
    """
    Frame measurements for one class.

    Parameters
    ----------
    leafname : str
        Composed class name.

    Returns
    -------
    res : dict or str
        Measurements, or an error message.
    """
    m = build(leafname)
    if isinstance(m, str):
        return m
    pc = param_class(type(m))
    t_name, u_name, frame = input_frame(pc)

    # Input values; beta-style impact parameters are in mas.
    t_in = getattr(m, t_name)
    u_in = abs(getattr(m, u_name))
    if u_name.startswith("beta"):
        u_in = u_in / m.thetaE_amp

    # Input frame: closest approach to the named reference point.
    t_ci, d_ci = closest_approach(m, frame, t_in)

    # Internal: closest approach to the actual lens midpoint.
    t_cg, d_cg = closest_approach(m, "geometric", m.t0)

    res = dict(m=m, pc=pc, t_name=t_name, u_name=u_name, frame=frame,
               t_in=t_in, u_in=u_in, t0=m.t0, u0=abs(m.u0_amp), t_ci=t_ci,
               d_ci=d_ci, t_cg=t_cg, d_cg=d_cg, dxL0=None, dxS0=None)
    if not m.astrometryFlag:
        return res

    # Internal xL0 (first filter) vs the actual midpoint at t0.
    geo0 = lens_point(m, np.array([m.t0]), "geometric")[0]
    res["dxL0"] = np.hypot(*(in_einstein(m, m.xL0[0]) - geo0))

    # Source vs the input xS0 at its documented time (mas).
    res["xs0_time"] = xs0_doc_time(pc, t_name)
    t_doc = getattr(m, res["xs0_time"], t_in)
    xS_doc = source_pos(m, np.array([t_doc]))[0]
    res["dxS0"] = m.thetaE_amp * np.hypot(*(xS_doc - in_einstein(m, XS0_IN)))
    return res


def verdicts(r):
    """
    Pass / fail flags for the input and internal frames.

    Parameters
    ----------
    r : dict
        Output of :func:`measure`.

    Returns
    -------
    inp_ok, int_ok : bool
        Input frame honored; internal t0/u0/xL0 at the geometric center.
    """
    inp_ok = abs(r["t_ci"] - r["t_in"]) < 0.05 and abs(r["d_ci"] - r["u_in"]) < 2e-3
    if r["dxS0"] is not None:
        inp_ok = inp_ok and r["dxS0"] < 1e-3
    int_ok = abs(r["t_cg"] - r["t0"]) < 0.05 and abs(r["d_cg"] - r["u0"]) < 2e-3
    if r["dxL0"] is not None:
        int_ok = int_ok and r["dxL0"] < 2e-3
    return inp_ok, int_ok


def plot_case(label, leafname, r, outfile):
    """
    Lens positions, source track, and input vs internal t0 / x0.

    Parameters
    ----------
    label : str
        Case description.
    leafname : str
        Composed class name.
    r : dict
        Output of :func:`measure`.
    outfile : pathlib.Path
        Destination PNG.

    Returns
    -------
    None
    """
    m = r["m"]
    t0, t_in = r["t0"], r["t_in"]
    t0_arr = np.array([t0])

    # Plot in the frame co-moving with the lens CoM, pinned at internal t0,
    # so static lenses stay put and the source moves at mu_rel. Origin is
    # the lens midpoint at t0.
    com0 = lens_point(m, t0_arr, "CoM")[0]
    origin = lens_point(m, t0_arr, "geometric")[0]

    def to_plot(pos, t):
        """Shift positions at times t into the co-moving plot frame."""
        return pos - (lens_point(m, t, "CoM") - com0) - origin

    def at(fn, t, *args):
        """Plot-frame position of fn(m, t, *args) at a single time t."""
        t_arr = np.array([t])
        return to_plot(fn(m, t_arr, *args), t_arr)[0]

    fig, ax = plt.subplots(figsize=(11.0, 6.6))

    # Source track; the secondary source (binary-source models) is dotted.
    t_tr = np.linspace(t0 - 0.8 * m.tE, t0 + 0.8 * m.tE, 801)
    xS_tr = to_plot(source_pos(m, t_tr), t_tr)
    ax.plot(*xS_tr.T, color="0.55", lw=1.2, label="source track")
    if m.astrometryFlag and hasattr(m, "get_resolved_source_astrometry_unlensed"):
        xS2 = in_einstein(m, m.get_resolved_source_astrometry_unlensed(t_tr)[:, 1])
        ax.plot(*to_plot(xS2, t_tr).T, color="0.55", lw=1.0, ls=":",
                label="secondary source")
    k = len(t_tr) // 2
    ax.annotate("", xy=xS_tr[k + 20], xytext=xS_tr[k],
                arrowprops=dict(arrowstyle="->", color="0.55", lw=1.2))

    # Lens tracks over the window (orbits show up; static lenses are fixed).
    # Shape of lens_tr: [N_times, 2 lenses, 2].
    lens_tr = in_einstein(m, m.get_resolved_lens_astrometry(t_tr))
    xL1_tr = to_plot(lens_tr[:, 0], t_tr)
    xL2_tr = to_plot(lens_tr[:, 1], t_tr)
    moving = not np.allclose(xL1_tr[0], xL1_tr[-1])
    if moving:
        ax.plot(*xL1_tr.T, color="k", lw=0.6, alpha=0.4)
        ax.plot(*xL2_tr.T, color="0.5", lw=0.6, alpha=0.4)

    def lenses_at(t):
        """Primary and secondary lens in the plot frame at time t."""
        t_arr = np.array([t])
        return to_plot(in_einstein(m, m.get_resolved_lens_astrometry(t_arr))[0],
                       t_arr)

    def draw_lenses(t, filled, tag):
        """Two lenses, midpoint, and CoM at time t."""
        lens = lenses_at(t)
        face = {} if filled else dict(mfc="none")
        alpha = 1.0 if filled else 0.45
        ax.plot(*lens[0], "*", color="k", mec="k", ms=17,
                label=f"primary lens {tag}", **face)
        ax.plot(*lens[1], "*", color="0.45", mec="0.45", ms=13,
                label=f"secondary lens {tag}", **face)
        ax.plot(*at(lens_point, t, "geometric"), "x", color="k", ms=9, mew=1.5,
                alpha=alpha, label=f"lens midpoint {tag}")
        ax.plot(*at(lens_point, t, "CoM"), "+", color="k", ms=11, mew=1.5,
                alpha=alpha, label=f"lens CoM {tag}")
        return None

    draw_lenses(t0, True, "@ internal t0")
    if moving and t_in != t0:
        draw_lenses(t_in, False, f"@ input {r['t_name']}")

    # Input: reference point and source at the input time, and the actual
    # closest approach to that reference point.
    ref_in = at(lens_point, t_in, r["frame"])
    xS_in = at(source_pos, t_in)
    ax.plot(*ref_in, "o", mfc="none", mec="C0", ms=13, mew=2,
            label=f"input reference ({r['frame']}) @ {r['t_name']}")
    ax.plot(*xS_in, "o", color="C0", ms=8, label=f"source @ input {r['t_name']}")
    ax.plot(*np.stack([ref_in, xS_in]).T, "--", color="C0", lw=1.5)
    xS_ci = at(source_pos, r["t_ci"])
    ax.plot(*xS_ci, "v", color="C0", ms=9,
            label=f"actual closest approach to {r['frame']}")

    # Internal: xL0, source at t0, and the actual closest approach to the
    # lens midpoint. Phot-only models have no xL0 (lens origin = 0).
    xL0 = in_einstein(m, m.xL0[0]) - origin if m.astrometryFlag else -origin
    if m.astrometryFlag:
        ax.plot(*xL0, "D", mfc="none", mec="C3", ms=10, mew=2,
                label="internal xL0")
    xS_t0 = at(source_pos, t0)
    ax.plot(*xS_t0, "s", color="C3", ms=8, label="source @ internal t0")
    ax.plot(*np.stack([xL0, xS_t0]).T, "--", color="C3", lw=1.5)
    xS_cg = at(source_pos, r["t_cg"])
    ax.plot(*xS_cg, "^", color="C3", ms=9,
            label="actual closest approach to midpoint")

    # Input xS0 (PhotAstrom), moved into the plot frame at its documented time.
    pts = [ref_in, xS_in, xS_ci, xL0, xS_t0, xS_cg, *lenses_at(t0)]
    if m.astrometryFlag:
        t_doc = np.array([getattr(m, r["xs0_time"], t_in)])
        xs0 = to_plot(in_einstein(m, XS0_IN)[None, :], t_doc)[0]
        ax.plot(*xs0, "P", color="C2", ms=10,
                label=f"input xS0 (doc: at {r['xs0_time']})")
        pts.append(xs0)

    inp_ok, int_ok = verdicts(r)
    txt = (
        f"INPUT ({r['frame']}, {r['t_name']}, {r['u_name']})  "
        f"{'OK' if inp_ok else 'MISMATCH'}\n"
        f"  closest approach at {r['t_name']} {r['t_ci'] - t_in:+.2f} d\n"
        f"  |u| there {r['d_ci']:.4f}  (input {r['u_in']:.4f})\n"
        f"INTERNAL (geometric t0, u0_amp)  {'OK' if int_ok else 'MISMATCH'}\n"
        f"  closest approach at t0 {r['t_cg'] - t0:+.2f} d\n"
        f"  |u| there {r['d_cg']:.4f}  (u0_amp {r['u0']:.4f})")
    if r["dxL0"] is not None:
        txt += f"\n  |xL0 - midpoint| {r['dxL0']:.4f}"
    if r["dxS0"] is not None:
        txt += f"\n  source - xS0 @ {r['xs0_time']}: {r['dxS0']:.3f} mas"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=7.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    # Zoom on the lenses and the marked points.
    pts = np.array(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    half = max(0.5 * np.max(hi - lo) + 0.25, 0.5)
    mid = 0.5 * (lo + hi)
    ax.set_xlim(mid[0] + half, mid[0] - half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_aspect("equal")
    ax.set_xlabel(r"East ($\theta_E$, lens frame; origin = midpoint @ "
                  r"internal $t_0$)")
    ax.set_ylabel(r"North ($\theta_E$)")
    ax.set_title(f"{label}\n{leafname}", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)
    return None


def fmt(x, w=7, p=3):
    """Number or dash for a markdown cell."""
    return "–" if x is None else f"{x:{w}.{p}f}"


def main():
    """
    Write the figures and ``binary_frame_audit.md``.

    Returns
    -------
    None
    """
    # Representative cases with figures.
    case_rows = []
    for k, (label, leafname) in enumerate(CASES, start=1):
        r = measure(leafname)
        fig_name = f"case{k:02d}_{leafname}.png"
        plot_case(label, leafname, r, OUTDIR / fig_name)
        inp_ok, int_ok = verdicts(r)
        case_rows.append((k, label, leafname, r, fig_name, inp_ok, int_ok))
        print(f"case {k:2d} {leafname}: input {'OK' if inp_ok else 'X'} "
              f"internal {'OK' if int_ok else 'X'}")

    # Every Param class (one leaf each, no-parallax preferred).
    leaves = {}
    for name in sorted(dir(model)):
        if not LEAF.match(name) or "GP" in name:
            continue
        pc = param_class(getattr(model, name))
        if pc is None:
            continue
        if pc.__name__ not in leaves or "_noPar_" in name:
            leaves[pc.__name__] = name
    all_rows = []
    for pcname in sorted(leaves):
        r = measure(leaves[pcname])
        all_rows.append((pcname, r))

    write_report(case_rows, all_rows)
    return None


def write_report(case_rows, all_rows):
    """
    Markdown report with the summary, figures, and the full table.

    Parameters
    ----------
    case_rows : list of tuple
        Representative cases from :func:`main`.
    all_rows : list of tuple
        ``(param_class_name, measure(...))`` for every class.

    Returns
    -------
    None
    """
    ok = lambda b: "✅" if b else "❌"
    lines = []
    lines.append(TEMPLATE_HEAD)

    lines.append("## Representative cases\n")
    lines.append("| # | Case | Class | Input frame | Input honored | "
                 "Internal geometric |")
    lines.append("|---|---|---|---|---|---|")
    for k, label, leafname, r, fig, inp_ok, int_ok in case_rows:
        lines.append(f"| {k} | {label} | `{leafname}` | {r['frame']} "
                     f"(`{r['t_name']}`, `{r['u_name']}`) | {ok(inp_ok)} | "
                     f"{ok(int_ok)} |")
    lines.append("")
    for k, label, leafname, r, fig, inp_ok, int_ok in case_rows:
        lines.append(f"### {k}. {label}: `{leafname}`\n")
        lines.append(CASE_NOTES.get(leafname, "") + "\n")
        lines.append(f"![{leafname}]({fig})\n")

    lines.append("## Every binary-lens parameterization\n")
    lines.append(
        "Times in days, separations in Einstein radii, `xS0` offset in mas. "
        "`Δt_in` / `|u|_in`: closest approach to the input reference point "
        "relative to the input time, and the separation there (compare "
        "with `input |u|`). `Δt_geo` / `|u|_geo`: closest approach to the "
        "actual lens midpoint relative to the internal `t0`, and the "
        "separation there (compare with `u0_amp`). `xL0`: distance from "
        "the internal `xL0` to the actual midpoint at `t0`. `xS0`: source "
        "distance from the input `xS0` at its documented time.\n")
    lines.append("| Param class | Input | Δt_in | \\|u\\|_in | input \\|u\\| | "
                 "Δt_geo | \\|u\\|_geo | u0_amp | xL0 | xS0 (time) | In | Int |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for pcname, r in all_rows:
        if isinstance(r, str):
            lines.append(f"| `{pcname}` | error: {r} |  |  |  |  |  |  |  |  |  |  |")
            continue
        inp_ok, int_ok = verdicts(r)
        xs0 = "–" if r["dxS0"] is None else f"{r['dxS0']:.3f} ({r['xs0_time']})"
        lines.append(
            f"| `{pcname}` | {r['frame']} `{r['t_name']}` | "
            f"{r['t_ci'] - r['t_in']:+.2f} | {r['d_ci']:.4f} | {r['u_in']:.4f} | "
            f"{r['t_cg'] - r['t0']:+.2f} | {r['d_cg']:.4f} | {r['u0']:.4f} | "
            f"{fmt(r['dxL0'], 6, 4)} | {xs0} | {ok(inp_ok)} | {ok(int_ok)} |")
    lines.append("")
    lines.append(TEMPLATE_TAIL)
    (OUTDIR / "binary_frame_audit.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUTDIR / "binary_frame_audit.md")
    return None


TEMPLATE_HEAD = """# Binary-lens reference-frame audit

**Convention under test.** Internally, `self.t0`, `self.u0` / `u0_amp` and
`self.xL0` always describe the source's closest approach to the
**geometric midpoint** of the lens pair, and every class also stores the
lens **CoM** closest approach (`t0_com`, `u0_amp_com`, `xL0_com`) where
masses exist. Inputs may be given about the midpoint, the CoM, or the
**primary**, and must be honored in that frame. For orbiting lenses the
midpoint and primary move, so their `(t0, u0)` are defined at an instant:
at `t0` the unlensed source is `u0` from the point's position *at that
time*, perpendicular to the relative proper motion. The CoM moves on a
straight line, so its closest approach is exact. Lens orbits are always
evaluated about the CoM.

**Result.** All PSBL / FSBL parameterizations, static and Keplerian
(including phot-only), now honor their input frame and keep the internal
geometric frame. The remaining failures are the binary-source BSBL / BFSBL
classes and the primary-input linear / accelerated orbit classes (see
Remaining issues).

## Method

`make_binary_frame_audit.py` (this directory) builds one model per binary-
lens Param class in `bagle.model_jax` (PSBL, FSBL, BSBL, BFSBL; FSBL/BFSBL
exist only there), with parallax switched off so tracks are straight. For
each reference point it measures where the (primary) source passes:

* the **input reference point** (midpoint, CoM = mass-weighted, or primary)
  — the time should equal the input `t0*` and the separation the input
  `u0*` / `beta* / thetaE`;
* the **actual lens midpoint** — the time should equal the internal
  `self.t0`, the separation `|self.u0_amp|`, and `self.xL0` should be the
  midpoint at `self.t0`;
* the input **`xS0`** at the time named in the docstring.

Static points and the CoM use the true closest approach on a fine time grid.
Moving points (midpoint or primary of an orbiting binary) use the
instantaneous definition at the reference time: the along-track offset is
reported as a time shift and the cross-track distance as the separation.

Figures are in Einstein radii, in the frame co-moving with the lens CoM
(so the source track is the relative-motion line, static lenses stay fixed,
and orbiting lenses trace their orbits), centered on the lens midpoint at
the internal `t0`, with East to the left. Blue marks the input frame (open circle: input
reference point at the input time; filled circle: source then; ▼: actual
closest approach to that point). Red marks the internal frame (◇: internal
`xL0`; ■: source at internal `t0`; ▲: actual closest approach to the lens
midpoint). Stars are the lenses (filled at internal `t0`, open at the input
time when the lenses moved); × is the midpoint and + the CoM. Green ✚ is
the input `xS0`. A ✅ needs both markers of a color to coincide. For
orbiting lenses the ▼ / ▲ markers are the linear closest approach, so they
need not coincide with the instantaneous definition.
"""

CASE_NOTES = {
    "PSBL_PhotAstrom_noPar_Param1":
        "Geometric inputs (`t0`, `beta`), static lenses. Input and internal "
        "frames are the same and exact.",
    "PSBL_Phot_noPar_Param1":
        "Phot-only (Einstein radii), geometric inputs. Exact.",
    "BSBL_PhotAstrom_noPar_Param1":
        "Binary source, geometric lens inputs, measured with the primary "
        "source. Exact.",
    "PSBL_PhotAstrom_noPar_Param4":
        "CoM inputs (`t0_com`, `u0_amp_com`), converted with "
        "`convert_u0_amp_t0_psbl`. Both frames exact (the old sin φ sign "
        "error is fixed).",
    "PSBL_PhotAstrom_Par_Param5":
        "Primary inputs (`t0_prim`, `u0_amp_prim`). Both frames exact.",
    "PSBL_PhotAstrom_Par_Param7":
        "Primary inputs (`t0_p`, `beta_p`) via the rewritten "
        "`fc.convert_u0_t0_psbl`. Both frames exact.",
    "PSBL_PhotAstrom_noPar_EllOrbs_Param2":
        "Geometric inputs with a Keplerian orbit. The CoM closest approach "
        "comes from the orbit at `t0` (`convert_u0_amp_t0_psbl_orbit`), so "
        "`xL0` is the midpoint at `t0` and the inputs hold at that instant.",
    "PSBL_PhotAstrom_noPar_EllOrbs_Param1":
        "CoM inputs (`t0_com`, `beta_com`) with a Keplerian orbit. CoM inputs "
        "exact; the geometric `t0` solves for the instant when the source - "
        "midpoint vector is perpendicular to the motion.",
    "PSBL_PhotAstrom_noPar_EllOrbs_Param4":
        "CoM inputs (`t0_com`, `u0_amp_com`) with a Keplerian orbit. Both "
        "frames exact (geometric `t0` at the orbit configuration at `t0`).",
    "PSBL_PhotAstrom_noPar_EllOrbs_Param7":
        "Primary inputs with a Keplerian orbit: primary at `t0_p` → CoM → "
        "midpoint at `t0`. Both frames exact.",
    "PSBL_Phot_noPar_EllOrbs_Param1":
        "Phot-only Keplerian orbit. `t0` / `u0_amp` are geometric at `t0`; "
        "the source track (`get_complex_pos`, `get_u`) runs from the CoM "
        "closest approach `t0_com` / `u0_com`, since the lenses orbit the "
        "CoM.",
    "BSBL_PhotAstrom_noPar_EllOrbs_Param1":
        "Binary source and binary lens, both orbiting. Neither the input "
        "nor the internal frame is consistent; the BSBL/BFSBL orbit classes "
        "need their own design pass (CircOrbs Param3 is badly broken).",
}

TEMPLATE_TAIL = """## Fixes applied

1. **Static CoM / primary ↔ geometric shift.** The old scalar formula
   `u0_amp = u0_amp_com ∓ ½·qeff·sep·sin φ/θE` used one sign, but the sign
   of the shift depends on which side of the motion the positive-`u0_amp`
   direction lies, so it was right for southward motion only. The shift is
   now done on the closest-approach vector
   (`bagle.model.convert_u0_amp_t0_psbl`; JAX:
   `derive_psbl_param4_heliocentric`). Used by Param4/5/6/8.
2. **`fc.convert_u0_t0_psbl` rewritten** from the binary-axis geometry
   (it shifted u0 by component signs, and most `t0` branches had the wrong
   sign). Param7 also keeps the sign of `u0_amp`.
3. **Keplerian orbits** convert frames with the orbit at the reference
   time (`convert_u0_amp_t0_psbl_orbit`) instead of a snapshot at `tp` with
   separation `a`; `alpha` / `phi` / `sep_t0` are reported at `t0`. The
   traced PSTL EllOrbs Param2 path uses the same definition.
4. **Phot-only orbits** store `t0_com` / `u0_com` and run the source track
   about the CoM in `get_complex_pos` and `get_u`; `phi_rho1_rad` is now in
   radians.
5. **EllOrbs / CircOrbs Param1** list `beta_com` (not `beta`) in
   `fitter_param_names`.
6. **`xS0` is always at the input reference time** (`t0_com`, `t0` or
   `t0_p`); EllOrbs / CircOrbs Param3 and Param7 now follow Param1/2/4/8.
7. **Phot-only orbit constructors** take their arguments in
   `fitter_param_names` order (`t0, u0_amp, tE, piE_E, piE_N, q, sep,
   v_para, v_perp, v_rad, r_s, a_s, b_sff, mag_src`; FSBL adds `radiusS`),
   and derive `aleph` / `aleph_sec` from `sep`, `r_s`, `a_s` and `q`.
8. **Phot-only orbit elements** come from `bagle.orbits.relative_state_to_elements`,
   written in the `Orbit.oal2xy` (Thiele-Innes) frame, so the orbit
   reproduces `sep`, the axis angle, `r_s` and the velocities at `t0`
   (circular orbits included). A new `phi` (after `sep`, as in
   `PSBL_PhotParam1`) sets the binary axis; light curves match
   VBMicrolensing's `BinaryLightCurveKepler` (`w1, w2, w3, sz/s, a/r` =
   `v_para, v_perp, v_rad, r_s, a_s` per day).
9. **`fitter_param_names` match the constructors** (checked for every Param
   class by `tests/test_fitter_param_names.py`): Param4 / CircOrbs Param8
   use `t0_com`, `u0_amp_com`; the Param6 family uses `t0_prim`,
   `u0_amp_prim`; swapped or shifted arguments were fixed in
   `PSPL_PhotAstromParam5`, BSBL / BFSBL `LinOrbs` / `AccOrbs` Param1,
   BSBL / BFSBL `CircOrbs_Param3`, the BSPL GP orbit classes and the FSBL
   limb GP classes; `EllOrbs_Param4` passes a true `mag_base` to its parent.

Regression tests: `tests/test_binary_frames_vs_vbm.py` (static classes vs
VBMicrolensing in three motion directions), `tests/test_binary_orbit_frames.py`
(orbit definitions read back from the lens / source positions), and
`tests/test_frame_convert.py::test_convert_u0_t0_psbl_geometry`.

## Remaining issues

* **BSBL / BFSBL** static Param2 and all orbit classes are inconsistent in
  both frames.
* **Linear / accelerated primary-input classes** (`LinOrbs` / `AccOrbs`
  Param6 / Param7): internal frame correct, input frame not honored. Their
  `get_resolved_lens_astrometry` branch places the primary at
  `xL0 - (sep/2)(sin α, cos α)`, the opposite side from the static and
  Keplerian classes, so the static conversion lands on the wrong lens.

## Reproduce

```bash
PYTHONPATH=src python docs/audits/binary_frames/make_binary_frame_audit.py
```
"""


if __name__ == "__main__":
    main()
