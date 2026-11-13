"""
Reproduce paper 1 Figure ``linorbs.png`` (BSPL linear / accelerated orbits).

PAPER (caption + code listing): BSPL_PhotAstrom_noPar_LinOrbs_Param2 and
AccOrbs_Param2; muS = [6, 3] mas/yr, delta_muS_sec = [9, 7] mas/yr,
accSsec = [0.5, -2] mas/yr^2, tE = 269 d, thetaE = 4.9 mas, u0 = 1.01,
lens 10 Msun and stationary, mag_pri = 16, mag_sec = 17;
t = arange(t0 - 10 tE, t0 + 10 tE, 1).

DERIVED: a stationary lens means mu_rel = muS, so piE points along muS with
|piE| = pi_rel / thetaE, pi_rel = thetaE^2 / (kappa M) = 0.295 mas.

ASSUMED: dS = 8 kpc (piS = 0.125 mas); xS0 = (0, 0) at t0, which puts the
lens at (-2.2, 4.4) mas as in the published figure; the secondary offset at
t0 (sep, alpha) is not given, so we place it ``SEC0`` = (2.7, -3.5) mas from
the primary (alpha is the PA of the secondary measured from the primary).

FINDING: the published secondary track has slope 0.77 (pixel-measured)
and its curvature matches a secondary moving at (9, 7) mas/yr, i.e.
delta_muS_sec = (3, 4), not muS + (9, 7) = (15, 10) as Eq. (linear_motion2)
gives with the caption values. ``linorbs.png`` uses the caption values;
``linorbs_match.png`` uses delta_muS_sec = (3, 4).
"""
import numpy as np
import matplotlib.pyplot as plt

from bagle import model

from common import KAPPA, save
from trajectories import draw_bspl

T0 = 60000.0
TE = 269.0
THETA_E = 4.9
M_L = 10.0
MU_S = np.array([6.0, 3.0])
DMU_SEC = np.array([9.0, 7.0])
DMU_SEC_MATCH = np.array([3.0, 4.0])
ACC_SEC = np.array([0.5, -2.0])
SEC0 = np.array([2.7, -3.5])


def build(acc, dmu):
    """
    Build the linear or accelerated BSPL model.

    Parameters
    ----------
    acc : bool
        Use AccOrbs_Param2 (True) or LinOrbs_Param2 (False).
    dmu : numpy array, shape = [2]
        Secondary proper motion relative to the primary (mas/yr).

    Returns
    -------
    bspl : BSPL model
        Model instance.
    """

    # Parallax vector along mu_rel = muS for a stationary lens.
    piRel = THETA_E ** 2 / (KAPPA * M_L)
    piE = piRel / THETA_E * MU_S / np.hypot(*MU_S)
    piS = 1e3 / 8000.0

    # Secondary offset at t0 -> (sep, alpha East of North).
    sep = np.hypot(*SEC0)
    alpha = np.degrees(np.arctan2(SEC0[0], SEC0[1])) % 360
    fratio = 10 ** (-0.4 * (17.0 - 16.0))
    mag_base = 16.0 - 2.5 * np.log10(1 + fratio)

    args = [T0, 1.01, TE, THETA_E, piS, piE[0], piE[1], 0.0, 0.0,
            MU_S[0], MU_S[1], dmu[0], dmu[1]]
    if acc:
        args += [ACC_SEC[0], ACC_SEC[1]]
        cls = model.BSPL_PhotAstrom_noPar_AccOrbs_Param2
    else:
        cls = model.BSPL_PhotAstrom_noPar_LinOrbs_Param2
    args += [sep, alpha, fratio, [mag_base], [1.0]]
    return cls(*args)


def draw(ax, bspl, title):
    """
    Plot unlensed sources, lens, +/- images and the observed centroid.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    bspl : BSPL model
        Model to draw.
    title : str
        Panel title.

    Returns
    -------
    None
    """

    # Time grid from the paper's code listing.
    t = np.arange(T0 - 10 * TE, T0 + 10 * TE, 1)

    draw_bspl(ax, bspl, t)
    ax.set_xlim(21, -21)
    ax.set_ylim(-21, 21)
    ax.set_title(title)
    return None


def main():
    """
    Make the two-panel linear / accelerated BSPL figure.

    Returns
    -------
    None
    """

    for name, dmu in [("linorbs", DMU_SEC), ("linorbs_match", DMU_SEC_MATCH)]:
        fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        for ax, acc, title in zip(axs, [False, True],
                                  ["Linear Orbits", "Accelerated Orbits"]):
            bspl = build(acc, dmu)
            draw(ax, bspl, title)
        axs[0].set_xlabel("")
        axs[0].legend(loc="center left", bbox_to_anchor=(1.05, -0.1))
        fig.suptitle(rf"$\Delta\mu_{{S,sec}}$ = ({dmu[0]:g}, {dmu[1]:g}) "
                     "mas/yr", y=1.02)
        xL = bspl.get_lens_astrometry(np.array([T0]))[0] * 1e3
        print(f"{name}: lens at t0 {np.round(xL, 2)} mas, "
              f"muS_sec = {bspl.muS_sec}")
        save(fig, name)
    return None


if __name__ == "__main__":
    main()
