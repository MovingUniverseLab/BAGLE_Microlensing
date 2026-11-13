"""
Reproduce paper 1 Figures ``psbl_keplerian.png``, ``bspl_keplerian.png`` and
``bsbl_keplerian.png`` (Keplerian binary lens / source / both).

psbl_keplerian -- PAPER: PSBL_PhotAstrom_noPar_EllOrbs_Param1 (code
listing); mLp = 10, mLs = 5 Msun, thetaE = 4.5 mas, omega = 30, Omega = 10,
i = 90, e = 0.6, P = 1054.41 d, "5 mas between the two lenses at t0_com",
tE = 412.02 d, stationary source, mag_src = 16.
DERIVED: a = 5 mas reproduces P = 1054.41 d only if dL = 1 kpc
(a_AU = 5 AU), so the quoted 5 mas is the semi-major axis a; thetaE = 4.5
then fixes dS = 1198.7 pc and tE fixes |muL| = 3.99 mas/yr.
ASSUMED (published figure): muL along +East (the broad secondary-lens peaks
sit 0.50 of the way between the sharp troughs in the published PNG; +East
gives 0.48 and -East 0.32), source at (0, 0) and the lens CoM passing 2 mas
South (beta_com = +2), tp = t0_com (the sharp periastron features sit at
Delta alpha* = 0 and repeat every |muL| P = 11.5 mas).

bspl_keplerian -- PAPER: omega = 30, Omega = 10, i = 90, e = 0.6,
P = 1000 d, aleph = 3, aleph_sec = 8 mas, thetaE = 2.9 mas, tE = 208.47 d,
stationary 8 Msun lens, mags 16 / 17.
ASSUMED: BSPL_PhotAstrom_noPar_EllOrbs_Param1, dS = 8 kpc, muS along +East
(|muS| = thetaE / tE; the published secondary-source peaks sit 0.37 of the
way between troughs, +East gives 0.39), tp = t0, xS0 = (0, 0) and
beta = +1 mas, which with the current code puts the lens at (0, -1) and the
source CoM at (0.5, 3.0) mas as in the published figure.
FINDING: the primary is then at (0.7, 4.0) mas at t0, not at the input
xS0 = (0, 0): Param1 builds xS0_com = xS0 + aleph * (pri - sec unit vector
at tp) instead of subtracting the primary's orbital offset at t0.

bsbl_keplerian -- PAPER: thetaE = 4.9 mas, tE = 231.15 d, mLp = 10,
mLs = 8 Msun (q = 0.8); lens orbit omega = 30, Omega = 10, i = 90, e = 0.2,
P = 2722.46 d (a_AU = 10 AU); source orbit omega = 30, Omega = 10, i = 90,
e = 0.4, P = 6000 d; mags 16 / 17.
ASSUMED: BSBL_PhotAstrom_noPar_EllOrbs_Param2; aL = 9 mas (dL = 1.11 kpc,
matching the drawn lens wiggle), dS from thetaE; lens CoM moving at
(-6.2, 0) and source CoM at (0, -4.3) mas/yr (|mu_rel| = thetaE / tE);
alephS = 4, aleph_secS = 5 mas; tpL = tpS = t0_com; xS0 = (0, 0).
"""
import numpy as np
import matplotlib.pyplot as plt

from bagle import model

from common import KAPPA, save
from trajectories import draw_binary_lens, draw_bspl

T0 = 60000.0


def psbl_keplerian():
    """
    Keplerian binary lens, stationary source.

    Returns
    -------
    None
    """

    mLp, mLs, thetaE, tE = 10.0, 5.0, 4.5, 412.02
    dL = 1000.0
    piS = 1e3 / dL - thetaE ** 2 / (KAPPA * (mLp + mLs))
    dS = 1e3 / piS
    mu = thetaE / (tE / 365.25)

    psbl = model.PSBL_PhotAstrom_noPar_EllOrbs_Param1(
        mLp, mLs, T0, 0.0, 0.0, 2.0, mu, 0.0, 30.0, 10.0, 90.0, 0.6,
        T0, 5.0, 0.0, 0.0, dL, dS, [1.0], [16.0], 0.0)
    print(f"psbl_keplerian: dS = {dS:.1f} pc, P = {float(psbl.p):.2f} d, "
          f"thetaE = {float(psbl.thetaE_amp):.3f}, tE = {float(psbl.tE):.2f}")

    t = np.arange(T0 - 10 * tE, T0 + 10 * tE, 2)
    fig, ax = plt.subplots(figsize=(11, 4))
    draw_binary_lens(ax, psbl, t)
    ax.set_xlim(15, -15)
    ax.set_ylim(-15, 15)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5)
    save(fig, "psbl_keplerian")
    return None


def bspl_keplerian():
    """
    Keplerian binary source, stationary lens.

    Returns
    -------
    None
    """

    mL, thetaE, tE, dS = 8.0, 2.9, 208.47, 8000.0
    piL = thetaE ** 2 / (KAPPA * mL) + 1e3 / dS
    dL = 1e3 / piL
    mu = thetaE / (tE / 365.25)

    bspl = model.BSPL_PhotAstrom_noPar_EllOrbs_Param1(
        mL, T0, 1.0, dL, dL / dS, 0.0, 0.0, 0.0, 0.0, mu, 0.0,
        30.0, 10.0, 90.0, 0.6, 1000.0, T0, 3.0, 8.0, [16.0], [17.0], [1.0])
    xS = bspl.get_resolved_source_astrometry_unlensed(np.array([T0]))[0] * 1e3
    print(f"bspl_keplerian: tE = {float(bspl.tE):.2f} d, "
          f"lens = {np.round(bspl.xL0.ravel() * 1e3, 2)}, "
          f"xS0 input = (0, 0) but primary at t0 = {np.round(xS[0], 2)}, "
          f"xS0_com = {np.round(np.ravel(bspl.xS0_com) * 1e3, 2)}")

    t = np.arange(T0 - 10 * tE, T0 + 10 * tE, 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    draw_bspl(ax, bspl, t)
    ax.set_xlim(16, -16)
    ax.set_ylim(-16, 16)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    save(fig, "bspl_keplerian")
    return None


def bsbl_keplerian():
    """
    Keplerian binary source and Keplerian binary lens.

    Returns
    -------
    None
    """

    mLp, mLs, thetaE, tE = 10.0, 8.0, 4.9, 231.15
    aL = 9.0
    dL = 10.0 / aL * 1e3
    piS = 1e3 / dL - thetaE ** 2 / (KAPPA * (mLp + mLs))
    muL = np.array([-6.2, 0.0])
    muS_amp = np.sqrt((thetaE / (tE / 365.25)) ** 2 - muL[0] ** 2)
    muS = np.array([0.0, -muS_amp])

    # piE along mu_rel = muS - muL.
    muRel = muS - muL
    piRel = 1e3 / dL - piS
    piE = piRel / thetaE * muRel / np.hypot(*muRel)
    fratio = 10 ** (-0.4 * (17.0 - 16.0))
    mag_base = 16.0 - 2.5 * np.log10(1 + fratio)

    bsbl = model.BSBL_PhotAstrom_noPar_EllOrbs_Param2(
        T0, 0.3, tE, thetaE, piS, piE[0], piE[1], mLs / mLp, 0.0, 0.0,
        muS[0], muS[1], 30.0, 10.0, 90.0, 0.2, T0, aL,
        30.0, 10.0, 90.0, 0.4, 6000.0, T0, 4.0, 5.0,
        fratio, [mag_base], [1.0], 0.0)
    print(f"bsbl_keplerian: dL = {dL:.0f} pc, dS = {1e3 / piS:.0f} pc, "
          f"pL = {float(getattr(bsbl, 'pL', np.nan)):.2f} d, "
          f"muL = {np.round(bsbl.muL, 2)}")

    t = np.arange(T0 - 10 * tE, T0 + 10 * tE, 2)
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_binary_lens(ax, bsbl, t, img_color="olivedrab")
    ax.set_xlim(50, -50)
    ax.set_ylim(-50, 50)
    ax.set_title("Binary Sources and Binary Lenses in Microlensing")
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    save(fig, "bsbl_keplerian")
    return None


if __name__ == "__main__":
    psbl_keplerian()
    bspl_keplerian()
    bsbl_keplerian()
