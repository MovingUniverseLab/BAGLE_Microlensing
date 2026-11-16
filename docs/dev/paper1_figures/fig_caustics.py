"""
Reproduce paper 1 Figures ``magmaps_varyq.png`` and ``magmaps_varysep.png``.

PAPER (captions / Sec. results_q): PSBL, thetaE = 6 mas, plotted at tp.
``varyq``: lens separation 5 mas, q = 1, 0.5, 0.1.
``varysep``: q = 1, a = 3, 6, 9 mas.

ASSUMED (read off the published figures): the lens centre of mass sits at
(60, -9.5) mas with the primary due North of the secondary at tp; that is a
face-on circular orbit (i = 0, e = 0) with omega = Omega = 0, so the
separation at tp equals a. Total lens mass 10 Msun and dS = 8 kpc (only
used to set dL so that thetaE = 6 mas).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bagle import model

from common import dist_for_thetaE, save, sky_axes

T0 = 60000.0
COM = (60.0, -9.5)
M_TOT = 10.0
DS = 8000.0
THETA_E = 6.0


def make_psbl(q, a_mas):
    """
    Build a face-on circular PSBL whose CoM is at ``COM`` at tp = t0_com.

    Parameters
    ----------
    q : float
        Mass ratio mLs / mLp.
    a_mas : float
        Total semi-major axis (mas); equals the separation for e = 0.

    Returns
    -------
    psbl : PSBL_PhotAstrom_noPar_EllOrbs_Param1
        Model instance.
    """

    # Split the mass and pick dL so that thetaE = 6 mas.
    mLp = M_TOT / (1 + q)
    mLs = M_TOT - mLp
    dL = dist_for_thetaE(M_TOT, THETA_E, DS)

    # Stationary lens and source, source on top of the lens CoM at t0_com.
    psbl = model.PSBL_PhotAstrom_noPar_EllOrbs_Param1(
        mLp, mLs, T0, COM[0] * 1e-3, COM[1] * 1e-3, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, T0, a_mas, 1.0, 0.0, dL, DS,
        [1.0], [20.0], 0.0)
    return psbl


def draw(ax, psbl, title):
    """
    Plot caustics, critical curves and lenses at tp in mas.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    psbl : PSBL model
        Model to draw.
    title : str
        Panel title.

    Returns
    -------
    None
    """

    t = np.array([T0])
    thE = float(psbl.thetaE_amp)

    # Curves come back in Einstein units on the sky; convert to mas.
    cc = np.asarray(psbl.get_critical_curves(t, N_pts=800)).ravel() * thE
    ca = np.asarray(psbl.get_caustics(t, N_pts=800)).ravel() * thE
    xL = np.asarray(psbl.get_resolved_lens_astrometry(t)) * 1e3

    ax.scatter(cc.real, cc.imag, s=1, color="#56C8F5")
    ax.scatter(ca.real, ca.imag, s=1, color="#F4A6B5")
    ax.plot(xL[0, 0, 0], xL[0, 0, 1], "o", color="k", ms=9)
    ax.plot(xL[0, 1, 0], xL[0, 1, 1], "^", color="purple", ms=10)
    ax.set_xlim(75, 45)
    ax.set_ylim(-20, 0)
    ax.set_aspect("equal")
    ax.set_title(title)
    sky_axes(ax)
    return None


def legend(fig):
    """
    Add the shared legend used in the paper figures.

    Parameters
    ----------
    fig : matplotlib Figure
        Figure to decorate.

    Returns
    -------
    None
    """
    handles = [
        Patch(fc="#F4A6B5", ec="k", label="Caustics"),
        Patch(fc="#56C8F5", ec="k", label="Critical Curves"),
        Line2D([], [], marker="o", color="k", ls="none", ms=10,
               label="Primary Lens"),
        Line2D([], [], marker="^", color="purple", ls="none", ms=10,
               label="Secondary Lens")]
    fig.legend(handles=handles, loc="upper center", ncol=4)
    return None


def main():
    """
    Make the vary-q and vary-a caustic panels.

    Returns
    -------
    None
    """

    # Vary q at a fixed 5 mas separation.
    fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    for ax, q in zip(axs, [1.0, 0.5, 0.1]):
        psbl = make_psbl(q, 5.0)
        draw(ax, psbl, f"q = {q:g}")
        print(f"q={q}: thetaE={float(psbl.thetaE_amp):.3f} mas")
    for ax in axs[1:]:
        ax.set_ylabel("")
    legend(fig)
    save(fig, "magmaps_varyq")

    # Vary a at fixed q = 1.
    fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    for ax, a in zip(axs, [3.0, 6.0, 9.0]):
        draw(ax, make_psbl(1.0, a), f"a = {a:g} mas")
    for ax in axs[1:]:
        ax.set_ylabel("")
    legend(fig)
    save(fig, "magmaps_varysep")
    return None


if __name__ == "__main__":
    main()
