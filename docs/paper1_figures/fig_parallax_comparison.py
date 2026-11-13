"""
Reproduce paper 1 Figures ``parallax_comparison.png`` (PSBL) and
``parallax_comparison_bspl.png`` (BSPL): BAGLE vs MulensModel vs
VBMicrolensing with parallax and no orbital motion.

PAPER (Sec. comparison): PSBL_Phot_Par_Param1 with t0 = 60478.49,
u0 = 1, tE = 100.4 d, q = 0.3, piE = [0.3, 0.2], phi = 125 deg, sep = 0.8;
geo-projected t0 = 60464.64, u0 = 0.98, tE = 65.06, piE = [-0.34, -0.13];
after the midpoint -> CoM shift t0 = 60460.41, u0 = 0.78; t0_par = t0.
BSPL_Phot_Par_Param1 with t0_p = 60478.49, t0_s = 60476.09, u0_p = 1,
u0_s = 1.25, tE = 100.4, flux ratio 0.16, piE = [0.3, 0.2].

DERIVED: the paper only says "Galactic Bulge". RA = 267.333, Dec = -21.283
deg reproduces all five quoted geo-projected values (u0 = 0.983); bulge
fields at Dec ~ -29 give u0_geo ~ 0.93. With these coordinates the CoM
shift gives t0 = 60459.41 (paper: 60460.41, a 1.00 d offset) and u0 = 0.783.
BSPL sep / phi are solved from the quoted secondary t0 / u0.

FINDING: phot-only ``BSPL.get_u`` builds u_sec = u0_sec + tau_pri muRel_hat,
keeping only the separation component along u0_hat, so the secondary
always reaches closest approach at the primary t0 (here 60478.49, not the
quoted 60476.09). The paper's geo values (t0_s = 60461.01, u0_s = 1.22)
are the geo projection of (60478.49, 1.25), i.e. the comparison was run on
what BAGLE realised. We feed MulensModel / VBM the realised secondary.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from bagle import frame_convert as fc
from bagle import model

from common import save

RA, DEC = 267.333, -21.283
T0, U0, TE = 60478.49, 1.0, 100.4
PIE = np.array([0.3, 0.2])
T = np.linspace(59880.0, 61080.0, 2000)


def vbm_setup(coords):
    """
    VBMicrolensing instance with parallax in (N, E) and t0_par = t0.

    Parameters
    ----------
    coords : SkyCoord
        Event coordinates.

    Returns
    -------
    vbm : VBMicrolensing.VBMicrolensing
        Configured instance.
    t_vbm : numpy array
        Model times in VBM's HJD' - 2450000 convention.
    """
    import VBMicrolensing

    vbm = VBMicrolensing.VBMicrolensing()
    vbm.SetObjectCoordinates(coords.to_string("hmsdms", sep=":"))
    vbm.parallaxsystem = 1
    vbm.t0_par_fixed = 1
    vbm.t0_par = fc.mjd_prime_to_vbm_time(T0, coords)
    t_vbm = fc.mjd_prime_to_vbm_time(T, coords)
    return vbm, t_vbm


def plot_comparison(A_b, A_v, A_m, title, name):
    """
    Two-panel light curve and residual figure in the paper's style.

    Parameters
    ----------
    A_b, A_v, A_m : numpy array
        BAGLE, VBMicrolensing and MulensModel magnifications.
    title : str
        Figure title.
    name : str
        Output name.

    Returns
    -------
    None
    """

    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axs[0].plot(T, A_v, ":", color="limegreen", lw=2, label="VBMicrolensing")
    axs[0].plot(T, A_m, "-.", color="tab:red", lw=2, label="MulensModel")
    axs[0].plot(T, A_b, "-", color="darkviolet", lw=2, label="BAGLE")
    axs[0].set_ylabel("Magnification")
    axs[0].set_title(title)
    axs[0].legend()
    axs[1].plot(T, A_v - A_b, ":", color="limegreen", lw=2,
                label="VBMicrolensing - BAGLE")
    axs[1].plot(T, A_m - A_b, "-.", color="tab:red", lw=2,
                label="MulensModel - BAGLE")
    axs[1].set_xlabel("Time (MJD)")
    axs[1].legend()
    print(f"{name}: max|VBM-BAGLE| = {np.max(np.abs(A_v - A_b)):.2e}, "
          f"max|MM-BAGLE| = {np.max(np.abs(A_m - A_b)):.2e}")
    save(fig, name)
    return None


def psbl():
    """
    PSBL with parallax: BAGLE vs MulensModel vs VBMicrolensing.

    Returns
    -------
    None
    """
    import MulensModel as mm

    q, sep, phi = 0.3, 0.8, 125.0
    coords = SkyCoord(RA, DEC, unit="deg")

    # BAGLE (heliocentric, geometric-midpoint frame).
    mod = model.PSBL_Phot_Par_Param1(T0, U0, TE, PIE[0], PIE[1], q, sep, phi,
                                     [1.0], [22.0], raL=RA, decL=DEC)
    A_b = mod.get_amplification(T)

    # Geo-projected values before / after the midpoint -> CoM shift.
    geo = fc.convert_helio_geo_phot(RA, DEC, T0, U0, TE, PIE[0], PIE[1], T0,
                                    murel_in="SL", murel_out="LS",
                                    coord_in="EN", coord_out="EN", plot=False)
    out = fc.convert_bagle_mulens_psbl_phot(RA, DEC, T0, U0, TE, PIE[0],
                                            PIE[1], T0, q, phi, sep,
                                            mod_in="bagle", plot=False)
    t0_m, u0_m, tE_m, piEE_m, piEN_m, q_m, alpha_m, sep_m = out
    print("PSBL geo (midpoint):", np.round(np.ravel(geo).astype(float), 3))
    print(f"PSBL geo (CoM): t0 = {t0_m:.2f}, u0 = {u0_m:.3f}, "
          f"alpha = {alpha_m:.2f}")

    # MulensModel.
    mm_mod = mm.Model({"t_0": t0_m + 2400000.5, "u_0": u0_m, "t_E": tE_m,
                       "pi_E_N": piEN_m, "pi_E_E": piEE_m, "q": q_m,
                       "s": sep_m, "alpha": alpha_m,
                       "t_0_par": T0 + 2400000.5}, coords=coords)
    A_m = mm_mod.get_magnification(T + 2400000.5)

    # VBMicrolensing; pick the alpha mapping that matches MulensModel.
    vbm, t_vbm = vbm_setup(coords)
    t0_v = fc.mjd_prime_to_vbm_time(t0_m, coords)
    best = None
    for lab, a in [("alpha", alpha_m), ("-alpha", -alpha_m),
                   ("alpha+180", alpha_m + 180), ("180-alpha", 180 - alpha_m)]:
        par = [np.log(sep_m), np.log(q_m), u0_m, np.radians(a), np.log(1e-5),
               np.log(tE_m), t0_v, piEN_m, piEE_m]
        A_v = np.asarray(vbm.BinaryLightCurveParallax(par, list(t_vbm))[0])
        d = np.max(np.abs(A_v - A_m))
        if best is None or d < best[0]:
            best = (d, lab, A_v)
    print(f"PSBL VBM alpha mapping: {best[1]} (max|VBM-MM| = {best[0]:.2e})")

    plot_comparison(A_b, best[2], A_m, "PSBL with Parallax",
                    "parallax_comparison")
    return None


def bspl():
    """
    BSPL with parallax: BAGLE vs MulensModel vs VBMicrolensing.

    Returns
    -------
    None
    """
    import MulensModel as mm

    coords = SkyCoord(RA, DEC, unit="deg")
    t0_s, u0_s, qflux = 60476.09, 1.25, 0.16

    # Solve sep / phi from the secondary closest approach (Einstein units):
    # sep_vec = -dtau * mu_hat + du0 * u0_hat, with mu_hat along piE.
    probe = model.BSPL_Phot_Par_Param1(T0, U0, TE, PIE[0], PIE[1], 0.1, 0.0,
                                       [22.0], [24.0], [1.0], raL=RA, decL=DEC)
    mu_hat = PIE / np.hypot(*PIE)
    sep_vec = -(t0_s - T0) / TE * mu_hat + (u0_s - U0) * probe.u0_hat
    sep = np.hypot(*sep_vec)
    alpha = np.degrees(np.arctan2(sep_vec[0], sep_vec[1]))
    phi = (alpha - np.degrees(np.arctan2(PIE[0], PIE[1]))) % 360
    mag_sec = 22.0 - 2.5 * np.log10(qflux)

    # BAGLE magnification = total flux / unlensed total flux.
    mod = model.BSPL_Phot_Par_Param1(T0, U0, TE, PIE[0], PIE[1], sep, phi,
                                     [22.0], [mag_sec], [1.0], raL=RA,
                                     decL=DEC)
    mag = mod.get_photometry(T)
    base = 22.0 - 2.5 * np.log10(1 + qflux)
    A_b = 10 ** (-0.4 * (mag - base))

    # Secondary closest approach actually realised by BAGLE (heliocentric):
    # the u_sec - u_pri offset d is constant for a static binary source.
    nopar = model.BSPL_Phot_noPar_Param1(T0, U0, TE, PIE[0], PIE[1], sep,
                                         phi, [22.0], [mag_sec], [1.0])
    u = np.asarray(nopar.get_u(np.array([T0])))[0]
    d = u[1] - u[0]
    t0_s_model = T0 - np.dot(d, mu_hat) * TE
    u0_s_model = U0 + np.dot(d, probe.u0_hat)
    print(f"BSPL: sep = {sep:.4f}, phi = {phi:.2f} deg; secondary intended "
          f"(t0, u0) = ({t0_s:.2f}, {u0_s:.3f}), realised by BAGLE = "
          f"({t0_s_model:.2f}, {u0_s_model:.3f})")

    # Geo-project each source (as BAGLE realises it) with the same t0_par.
    geo = []
    for t0_i, u0_i in [(T0, U0), (t0_s_model, u0_s_model)]:
        g = fc.convert_helio_geo_phot(RA, DEC, t0_i, u0_i, TE, PIE[0], PIE[1],
                                      T0, murel_in="SL", murel_out="LS",
                                      coord_in="EN", coord_out="EN",
                                      plot=False)
        geo.append(np.ravel(g).astype(float))
    (t0p_g, u0p_g, tE_g, piEE_g, piEN_g), (t0s_g, u0s_g) = geo[0], geo[1][:2]
    print(f"BSPL geo: t0p = {t0p_g:.2f}, t0s = {t0s_g:.2f}, u0p = {u0p_g:.3f}, "
          f"u0s = {u0s_g:.3f}, tE = {tE_g:.2f}, piE = ({piEE_g:.3f}, "
          f"{piEN_g:.3f})")

    # MulensModel two-source model.
    mm_mod = mm.Model({"t_0_1": t0p_g + 2400000.5, "u_0_1": u0p_g,
                       "t_0_2": t0s_g + 2400000.5, "u_0_2": u0s_g,
                       "t_E": tE_g, "pi_E_N": piEN_g, "pi_E_E": piEE_g,
                       "t_0_par": T0 + 2400000.5}, coords=coords)
    A_m = mm_mod.get_magnification(T + 2400000.5, source_flux_ratio=qflux)

    # VBMicrolensing.
    vbm, t_vbm = vbm_setup(coords)
    par = [np.log(tE_g), np.log(qflux), u0p_g, u0s_g,
           fc.mjd_prime_to_vbm_time(t0p_g, coords),
           fc.mjd_prime_to_vbm_time(t0s_g, coords), piEN_g, piEE_g]
    A_v = np.asarray(vbm.BinSourceLightCurveParallax(par, list(t_vbm))[0])

    plot_comparison(A_b, A_v, A_m, "BSPL with Parallax",
                    "parallax_comparison_bspl")
    return None


if __name__ == "__main__":
    psbl()
    bspl()
