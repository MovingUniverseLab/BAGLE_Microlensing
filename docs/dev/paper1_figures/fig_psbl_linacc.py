"""
Reproduce paper 1 Figure ``lin_lenses.png`` (PSBL linear / accelerated).

PAPER (caption): PSBL_PhotAstrom_noPar_LinOrbs_Param1 / AccOrbs_Param1;
stationary source; muL (primary) = [-3.76, -3.76] mas/yr, secondary lens
proper motion [-2.76, -2.76] mas/yr (delta_muL_sec = [1, 1]);
accLsec = [1, -1] mas/yr^2 (the caption says mas/yr); tE = 412 d,
thetaE = 6 mas, u0 = 0.5, mag_src = 16.

DERIVED: |mu_rel| = thetaE / tE = 5.32 mas/yr = |muL|, as expected for a
stationary source.

ASSUMED: mLp = 10, mLs = 5 Msun, dS = 8 kpc (dL from thetaE); source at
(0, 0); beta = +3 mas (u0 thetaE) puts the lens midpoint North-West of the
source as in the figure; sep = 4 mas with the secondary offset towards the
North-West of the primary, as in the figure; t = t0 +/- 10 tE.

FINDING: the static PSBL classes put the primary at +sep/2 along alpha
(alpha points from the secondary to the primary, as in Appendix A), but the
linear / accelerated branch of ``PSBL_PhotAstrom.get_resolved_lens_astrometry``
puts the primary at -sep/2. To place the secondary North-West of the primary
we therefore pass alpha = 315 deg here (135 deg in the static convention).
"""
import numpy as np
import matplotlib.pyplot as plt

from bagle import model

from common import dist_for_thetaE, save
from trajectories import draw_binary_lens

T0 = 60000.0
TE = 412.0
THETA_E = 6.0
MLP, MLS = 10.0, 5.0
DS = 8000.0
MU_L = np.array([-3.76, -3.76])
DMU_L_SEC = np.array([1.0, 1.0])
ACC_L_SEC = np.array([1.0, -1.0])
SEP, ALPHA = 4.0, 315.0


def build(acc):
    """
    Build the linear or accelerated PSBL model.

    Parameters
    ----------
    acc : bool
        Use AccOrbs_Param1 (True) or LinOrbs_Param1 (False).

    Returns
    -------
    psbl : PSBL model
        Model instance.
    """

    dL = dist_for_thetaE(MLP + MLS, THETA_E, DS)
    beta = 0.5 * THETA_E

    # Shared leading / trailing arguments of the two parameterizations.
    args = [MLP, MLS, T0, 0.0, 0.0, beta, MU_L[0], MU_L[1],
            DMU_L_SEC[0], DMU_L_SEC[1]]
    if acc:
        args += [ACC_L_SEC[0], ACC_L_SEC[1]]
        cls = model.PSBL_PhotAstrom_noPar_AccOrbs_Param1
    else:
        cls = model.PSBL_PhotAstrom_noPar_LinOrbs_Param1
    args += [0.0, 0.0, dL, DS, SEP, ALPHA, [1.0], [16.0], 0.0]
    return cls(*args)


def main():
    """
    Make the two-panel linear / accelerated PSBL figure.

    Returns
    -------
    None
    """

    t = np.arange(T0 - 10 * TE, T0 + 10 * TE, 2)

    fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for ax, acc, title, ylim in zip(
            axs, [False, True], ["Linear Orbits", "Accelerated Orbits"],
            [(-35, 35), (-20, 20)]):
        psbl = build(acc)
        draw_binary_lens(ax, psbl, t)
        ax.set_xlim(20, -20)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        print(f"{title}: thetaE = {float(psbl.thetaE_amp):.2f} mas, "
              f"tE = {float(psbl.tE):.1f} d, u0 = {np.round(psbl.u0, 3)}")
    axs[0].set_xlabel("")
    axs[0].legend(loc="center left", bbox_to_anchor=(1.03, -0.1))
    save(fig, "lin_lenses")
    return None


if __name__ == "__main__":
    main()
