"""Roman GBTDS mock of a dark 8 solar-mass black hole plus an Earth-mass planet.

This is a standalone demonstration for Prof. Jessica Lu. It does not modify
BAGLE. The lens is an 8 solar-mass black hole at 3 kpc with a 1 Earth-mass
planet at a 3D separation of 0.1 AU. Both lens components are dark. The
source is at 8 kpc with F146W = 19. The source is aimed through the central
caustic, which also puts the closest approach well inside 0.1 mas.

BAGLE has no working finite-source binary-lens class (the FSBL block in
``model.py`` is commented out). The light curve is therefore the point-source
binary lens. A uniform-disk point-lens estimate is overplotted only as a
scale comparison for the black-hole peak.

Run from the repository root with::

    PYTHONPATH=src python scratch/bh_planet_roman/make_bh_planet_roman_demo.py
"""

import contextlib
import io
import os
import sys
import types
import warnings
from pathlib import Path

# Parallax cache must be set before bagle.parallax is imported.
os.environ.setdefault('PARALLAX_CACHE_DIR', '/tmp/bagle_parallax_cache')

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.time import Time
from matplotlib.patches import Circle


def _install_pymultinest_stub():
    """Register a pymultinest stub so ``bagle.fake_data`` can be imported.

    ``bagle.fake_data`` imports ``bagle.model_fitter``, which imports
    ``pymultinest``. That import calls ``sys.exit`` when ``libmultinest``
    is absent. This demo never samples a posterior, so a stub is enough.

    Returns
    -------
    None
    """
    # Drop a half-initialized real package if the failed import left one.
    for name in list(sys.modules):
        if name == 'pymultinest' or name.startswith('pymultinest.'):
            del sys.modules[name]

    run = types.ModuleType('pymultinest.run')

    def _unavailable(*args, **kwargs):
        raise RuntimeError('MultiNest is not used by this demo.')

    run.run = _unavailable
    analyse = types.ModuleType('pymultinest.analyse')
    analyse.Analyzer = type('Analyzer', (), {})
    solve = types.ModuleType('pymultinest.solve')
    solve.Solver = type('Solver', (), {})
    solve.solve = _unavailable
    package = types.ModuleType('pymultinest')
    package.run = run
    package.solve = solve
    package.analyse = analyse
    sys.modules['pymultinest'] = package
    sys.modules['pymultinest.run'] = run
    sys.modules['pymultinest.solve'] = solve
    sys.modules['pymultinest.analyse'] = analyse
    return None


def _import_bagle():
    """Import BAGLE model, parallax, and fake-data helpers.

    Returns
    -------
    fake_data : module
        BAGLE fake-data module (Roman cadence and noise helpers).
    model : module
        BAGLE microlensing model module.
    parallax : module
        BAGLE parallax module.

    Notes
    -----
    MultiNest's shared library is optional here. If it is missing, a stub
    is installed before ``bagle.fake_data`` is imported.
    """
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            from pymultinest.solve import Solver  # noqa: F401
        except BaseException:
            _install_pymultinest_stub()
        from bagle import fake_data
        from bagle import model
        from bagle import parallax
    return fake_data, model, parallax


fake_data, model, parallax = _import_bagle()

# BAGLE's internal conversion from mas/yr to days.
DAYS_PER_YEAR = float(model.days_per_year)

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'output' / 'v1_static_0p1AU'
ARTIFACT_DIR = Path('/opt/cursor/artifacts')

# Galactic Center line of sight used by get_times_roman_gbtds.
RA_DEG = (17.0 + 40.0 / 60.0 + 40.04 / 3600.0) * 15.0
DEC_DEG = -(29.0 + 28.0 / 3600.0)


def configure_matplotlib():
    """Set publication-style plotting defaults.

    Returns
    -------
    None
    """
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 11,
        'legend.fontsize': 8.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.3,
        'figure.dpi': 120,
        'savefig.dpi': 160,
        'savefig.bbox': 'tight',
        'mathtext.default': 'regular',
    })
    return None


def earth_mass_msun():
    """Return the Earth/Sun mass ratio used for the planet.

    Returns
    -------
    mass_ratio : float
        ``M_earth / M_sun`` from CODATA constants in astropy.
    """
    ratio = (const.M_earth / const.M_sun).to(u.dimensionless_unscaled)
    return float(ratio.value)


def source_angular_radius_mas(distance_pc, radius_rsun):
    """Angular radius of a star.

    Parameters
    ----------
    distance_pc : float
        Distance in parsecs.
    radius_rsun : float
        Stellar radius in solar radii.

    Returns
    -------
    theta_mas : float
        Angular radius in milliarcseconds.

    Notes
    -----
    One solar radius at one parsec is ``R_sun / 1 AU`` arcseconds, because
    1 AU at 1 pc subtends 1 arcsecond. Multiply by 1000 to convert to mas.
    """
    radius_au = float((radius_rsun * const.R_sun / const.au).value)
    theta_arcsec = radius_au / distance_pc
    theta_mas = theta_arcsec * 1.0e3
    return theta_mas


def orbital_period_days(a_au, mass_msun):
    """Keplerian period of a circular orbit.

    Parameters
    ----------
    a_au : float
        Semi-major axis in AU. For this demo the projected separation is
        taken equal to this value (face-on orbit).
    mass_msun : float
        Total mass in solar masses.

    Returns
    -------
    period_days : float
        Orbital period in days.

    Notes
    -----
    Uses ``P = 2 pi sqrt(a^3 / GM)`` with astropy constants, then converts
    to days. In astronomical units this is ``365.25 * sqrt(a^3 / M)``.
    """
    a = a_au * u.AU
    mass = mass_msun * u.Msun
    period = 2.0 * np.pi * np.sqrt(a ** 3 / (const.G * mass))
    period_days = period.to(u.day).value
    return float(period_days)


def point_lens_magnification(u_einstein):
    """Point-lens magnification for a point source.

    Parameters
    ----------
    u_einstein : float or ndarray
        Lens-source separation in units of ``theta_E``. Shape ``(N,)``
        or scalar.

    Returns
    -------
    magnification : ndarray
        Magnification ``A(u) = (u^2 + 2) / (u sqrt(u^2 + 4))``.
        Same shape as the input.

    Notes
    -----
    A floor of ``1e-18`` avoids a divide-by-zero at the exact lens
    position. That floor is far inside the central caustic of the
    requested planet and is used only for the comparison curve.
    """
    u_arr = np.asarray(u_einstein, dtype=float)
    u_safe = np.maximum(u_arr, 1.0e-18)
    magnification = (u_safe ** 2 + 2.0) / (
        u_safe * np.sqrt(u_safe ** 2 + 4.0)
    )
    return magnification


def point_lens_centroid(source_arcsec, lens_arcsec, theta_e_mas):
    """Unresolved centroid of a dark point-lens event.

    Parameters
    ----------
    source_arcsec : ndarray, shape (N, 2)
        Unlensed source position in arcseconds, East then North.
    lens_arcsec : ndarray, shape (N, 2)
        Point-lens position in arcseconds, East then North.
    theta_e_mas : float
        Einstein radius in milliarcseconds.

    Returns
    -------
    centroid_arcsec : ndarray, shape (N, 2)
        Flux-weighted image centroid in arcseconds.

    Notes
    -----
    This is the ``b_sff = 1`` limit of BAGLE's point-lens astrometry.
    The centroid shift relative to the unlensed source is
    ``theta_S / (u^2 + 2)``, with ``theta_S`` the source-minus-lens
    vector.
    """
    theta_s = source_arcsec - lens_arcsec
    theta_e_arcsec = theta_e_mas * 1.0e-3
    u_vec = theta_s / theta_e_arcsec
    u_amp = np.linalg.norm(u_vec, axis=1)
    u_safe = np.maximum(u_amp, 1.0e-18)
    shift = theta_s / (u_safe ** 2 + 2.0)[:, None]
    centroid_arcsec = source_arcsec + shift
    return centroid_arcsec


def uniform_disk_peak_magnification(rho):
    """Peak magnification of a uniform disk centered on a point lens.

    Parameters
    ----------
    rho : float
        Source angular radius in units of ``theta_E``.

    Returns
    -------
    magnification : float
        Disk-averaged magnification. For ``u = 0`` this is
        ``sqrt(rho^2 + 4) / rho``, which is ``2 / rho`` when
        ``rho << 1``.

    Notes
    -----
    The integral of ``A(u) u`` from 0 to ``rho`` is
    ``0.5 rho sqrt(rho^2 + 4)``. Dividing by the disk area gives the
    expression above. This is a point-lens result, not a binary-lens
    finite-source magnification.
    """
    magnification = np.sqrt(rho ** 2 + 4.0) / rho
    return float(magnification)


def roman_f146_uncertainties(mag, apply_debug_factor=False):
    """Roman F146 photometric and astrometric uncertainties.

    Parameters
    ----------
    mag : ndarray, shape (N,)
        F146W magnitude at each epoch.
    apply_debug_factor : bool, optional
        If True, multiply both uncertainties by ``1e-5``, which is what
        ``tests/test_model.py:test_roman_lightcurve`` currently does.
        The default is False. See Notes.

    Returns
    -------
    mag_err : ndarray, shape (N,)
        Photometric uncertainty in magnitudes.
    ast_err_mas : ndarray, shape (N,)
        Astrometric uncertainty in milliarcseconds, same value in East
        and North. Floored at 0.1 mas.

    Notes
    -----
    BAGLE's Roman test adopts a zero point of 28 mag at SNR = 1 and
    converts flux to SNR as ``sqrt(flux)``. The magnitude error used
    here is the exact derivative ``(2.5 / ln(10)) / SNR``. BAGLE's
    ``fake_data.add_photometric_noise`` uses the rounded factor 1.087
    instead of 1.0857; the noise realization below calls that function.

    Astrometry follows the same test: ``FWHM / (2 SNR)`` with
    ``FWHM = 0.25 * lambda_um / D_m``, ``lambda = 1.49`` um (the test's
    W149 wavelength; F146 is centered near 1.46 um), and ``D = 2.4`` m.
    The test then multiplies both errors by ``1e-5``. That factor pulls
    every astrometric error under the 0.1 mas floor and makes a 19th
    magnitude star precise at the ``1e-7`` mag level, which is not a
    physical F146W = 19 error budget. The factor is omitted unless
    ``apply_debug_factor`` is set. The 0.1 mas floor is kept.
    """
    mag_arr = np.asarray(mag, dtype=float)
    # Zero point where a single photoelectron would have SNR = 1.
    zp_f146 = 28.0
    flux = 10.0 ** ((mag_arr - zp_f146) / -2.5)
    snr = np.sqrt(np.maximum(flux, 1.0e-30))

    # dmag = (2.5 / ln(10)) dF / F, and dF/F = 1/SNR for Poisson noise.
    mag_err = (2.5 / np.log(10.0)) / snr

    # Diffraction estimate copied from the Roman test (W149 / F146).
    telescope_diam_m = 2.4
    lambda_um = 1.49
    fwhm_arcsec = 0.25 * lambda_um / telescope_diam_m
    ast_err_mas = (fwhm_arcsec * 1.0e3) / (2.0 * snr)
    ast_err_mas = np.maximum(ast_err_mas, 0.1)

    if apply_debug_factor:
        mag_err = mag_err * 1.0e-5
        ast_err_mas = np.maximum(ast_err_mas * 1.0e-5, 0.1)

    return mag_err, ast_err_mas


def add_roman_photometric_noise(mag_model, rng_seed=146):
    """Add Poisson photometric noise with BAGLE's fake-data helper.

    Parameters
    ----------
    mag_model : ndarray, shape (N,)
        Noise-free F146W magnitudes.
    rng_seed : int, optional
        Seed for the global NumPy random stream. ``add_photometric_noise``
        draws with ``np.random.randn``.

    Returns
    -------
    mag_obs : ndarray, shape (N,)
        Noisy magnitudes.
    mag_err : ndarray, shape (N,)
        BAGLE magnitude uncertainties (factor 1.087 / SNR).

    Notes
    -----
    The electron scale is BAGLE's Roman zero point: SNR = 1 at magnitude
    28, so an object at magnitude 19 has
    ``flux0 = 10 ** ((28 - 19) / 2.5)`` electrons. This is the same flux
    scale as ``test_roman_lightcurve``.
    """
    mag_ref = 19.0
    zp_f146 = 28.0
    # Electrons for a 19th-mag star if mag 28 is one electron.
    flux0 = 10.0 ** ((zp_f146 - mag_ref) / 2.5)
    np.random.seed(rng_seed)
    mag_obs, mag_err = fake_data.add_photometric_noise(
        flux0, mag_ref, np.asarray(mag_model, dtype=float)
    )
    return np.asarray(mag_obs, dtype=float), np.asarray(mag_err, dtype=float)


def fast_seasons(times_mjd):
    """Group a Roman time series into visibility seasons.

    Parameters
    ----------
    times_mjd : ndarray, shape (N,)
        Observation times in MJD.

    Returns
    -------
    seasons : list of tuple
        Each entry is ``(t_start, t_stop, n_points, is_fast)``. A season
        is "fast" when it contains more than 100 samples. In the default
        GBTDS helper those seasons are sampled every 12.8 minutes; the
        other seasons are one F146 visit every few days.

    Notes
    -----
    Season edges are gaps longer than 2 days, which is how the visibility
    windows come out of ``get_times_roman_gbtds``.
    """
    times = np.sort(np.asarray(times_mjd, dtype=float))
    if len(times) == 0:
        return []

    gaps = np.where(np.diff(times) > 2.0)[0]
    starts = np.r_[0, gaps + 1]
    stops = np.r_[gaps, len(times) - 1]
    seasons = []
    for i_start, i_stop in zip(starts, stops):
        n_points = int(i_stop - i_start + 1)
        t_start = float(times[i_start])
        t_stop = float(times[i_stop])
        is_fast = n_points > 100
        seasons.append((t_start, t_stop, n_points, is_fast))
    return seasons


def build_psbl(t0, beta_mas, sep_mas, m_bh, m_planet):
    """Build a dark point-source binary lens with parallax.

    Parameters
    ----------
    t0 : float
        Barycentric time of closest approach, MJD.
    beta_mas : float
        Signed barycentric impact parameter relative to the geometric
        center of the binary, in milliarcseconds. With the proper motion
        used here, positive beta is North.
    sep_mas : float
        Instantaneous sky-projected binary separation in milliarcseconds.
    m_bh : float
        Black-hole mass in solar masses (lens primary).
    m_planet : float
        Planet mass in solar masses (lens secondary).

    Returns
    -------
    psbl : bagle.model.PSBL_PhotAstrom_Par_Param1
        Static binary lens. Photometry, astrometry, and annual parallax
        are all enabled. Lens flux is zero.

    Notes
    -----
    ``PSBL_PhotAstrom_Par_Param1`` takes physical inputs (masses,
    distances, separation, proper motions). That matches the requested
    system directly. ``b_sff = 1`` and ``dmag_Lp_Ls = 0`` are the class
    docstring's prescription for two dark lenses.

    The source proper motion is 7 mas/yr due East and the lens proper
    motion is zero in this frame, so the relative proper motion is
    7 mas/yr. That is a representative disk-bulge relative motion. The
    observer is ``'earth'``. Roman sits at Sun-Earth L2, about 0.01 AU
    from Earth, which shifts the parallax by roughly ``0.01 * pi_rel``
    (about 0.002 mas). That is negligible for the black hole and still
    vastly larger than the 0.1 AU planetary caustic.
    """
    # alpha = 90 deg puts the primary East of the geometric center and
    # the planet West of it, along the source's direction of motion.
    alpha_deg = 90.0
    mu_s_east = 7.0
    mu_s_north = 0.0
    mu_l_east = 0.0
    mu_l_north = 0.0
    d_l_pc = 3000.0
    d_s_pc = 8000.0
    x_s0_east = 0.0
    x_s0_north = 0.0
    b_sff = [1.0]
    mag_src = [19.0]
    dmag_lp_ls = [0.0]

    psbl = model.PSBL_PhotAstrom_Par_Param1(
        m_bh,
        m_planet,
        t0,
        x_s0_east,
        x_s0_north,
        beta_mas,
        mu_l_east,
        mu_l_north,
        mu_s_east,
        mu_s_north,
        d_l_pc,
        d_s_pc,
        sep_mas,
        alpha_deg,
        b_sff,
        mag_src,
        dmag_lp_ls,
        raL=RA_DEG,
        decL=DEC_DEG,
        obsLocation=['earth'],
        root_tol=1.0e-8,
    )
    return psbl


def com_offset_mas(sep_mas, m_primary, m_secondary, alpha_deg=90.0):
    """Center-of-mass offset from the geometric center.

    Parameters
    ----------
    sep_mas : float
        Projected separation in milliarcseconds.
    m_primary : float
        Primary mass. Units cancel in the mass ratio.
    m_secondary : float
        Secondary mass, same units as ``m_primary``.
    alpha_deg : float, optional
        BAGLE binary angle, degrees East of North. Default 90, so the
        primary lies due East of the geometric center.

    Returns
    -------
    offset_mas : ndarray, shape (2,)
        ``[East, North]`` offset of the center of mass from the
        geometric center, in milliarcseconds.

    Notes
    -----
    BAGLE places the primary at ``+sep/2`` along ``(sin alpha, cos alpha)``
    and the secondary at the opposite point. The center of mass is pulled
    off the primary toward the planet by ``sep * q / (1 + q)``.
    """
    alpha = np.deg2rad(alpha_deg)
    primary_offset = 0.5 * sep_mas * np.array(
        [np.sin(alpha), np.cos(alpha)]
    )
    # COM = (m1 x1 + m2 x2) / (m1 + m2), with x2 = -x1.
    mass_factor = (m_primary - m_secondary) / (m_primary + m_secondary)
    offset_mas = primary_offset * mass_factor
    return offset_mas


def aim_at_sky_offset(t_cross, target_east_mas, target_north_mas,
                      pi_rel_mas, mu_east=7.0):
    """Choose ``t0`` and ``beta`` so the geocentric source hits a point.

    Parameters
    ----------
    t_cross : float
        MJD at which the geocentric source should sit on the target.
    target_east_mas : float
        Desired East offset of the source from the binary geometric
        center at ``t_cross``, in milliarcseconds.
    target_north_mas : float
        Desired North offset, in milliarcseconds.
    pi_rel_mas : float
        Relative parallax in milliarcseconds.
    mu_east : float, optional
        Relative proper motion, mas/yr, entirely in the East direction.

    Returns
    -------
    t0 : float
        Barycentric closest-approach time, MJD.
    beta_mas : float
        Barycentric impact parameter, milliarcseconds (North).
    parallax_au : ndarray, shape (2,)
        BAGLE parallax vector at ``t_cross``, in AU, ``[East, North]``.

    Notes
    -----
    With this proper-motion direction, BAGLE's ``thetaS0`` is
    ``[0, beta]``. The geocentric source-minus-geometric-center vector
    in mas is::

        theta(t) = thetaS0 + (t - t0) / 365.25 * mu_rel
                   - pi_rel * parallax_vector(t)

    Solving that at ``t_cross`` gives ``t0`` and ``beta``. Parallax is
    included because a Roman bulge event with this lens has
    ``pi_E ~ 0.06``, and the parallax shift (tenths of a mas) is enormous
    compared with the planetary caustic.
    """
    parallax_au = parallax.parallax_in_direction(
        RA_DEG, DEC_DEG, np.array([t_cross]), obsLocation='earth'
    )[0]
    t0 = t_cross - (
        (target_east_mas + pi_rel_mas * parallax_au[0])
        * DAYS_PER_YEAR
        / mu_east
    )
    beta_mas = target_north_mas + pi_rel_mas * parallax_au[1]
    return float(t0), float(beta_mas), np.asarray(parallax_au, dtype=float)


def evaluate_event(psbl, times_mjd):
    """Evaluate images, photometry, and astrometry.

    Parameters
    ----------
    psbl : bagle model
        Instantiated PSBL model.
    times_mjd : ndarray, shape (N,)
        Times in MJD.

    Returns
    -------
    result : dict
        ``mag`` shape ``(N,)`` in magnitudes,
        ``magnification`` shape ``(N,)``,
        ``n_images`` shape ``(N,)``,
        ``centroid_arcsec`` shape ``(N, 2)``,
        ``source_arcsec`` shape ``(N, 2)``,
        ``primary_arcsec`` shape ``(N, 2)``,
        ``secondary_arcsec`` shape ``(N, 2)``.
        Positions are East, North.

    Notes
    -----
    Image solving is BAGLE's complex fifth-order polynomial. Invalid
    roots are masked before the magnification sum, which is the same
    path ``get_photometry`` uses.
    """
    times = np.asarray(times_mjd, dtype=float)
    images, amps = psbl.get_all_arrays(times)
    amp_masked = np.ma.masked_invalid(amps)
    magnification = np.array(np.sum(amp_masked, axis=1), dtype=float)
    n_images = np.sum(np.isfinite(np.real(images)), axis=1).astype(int)
    mag = np.array(
        psbl.get_photometry(times, amp_arr=amps), dtype=float
    )
    centroid = np.array(
        psbl.get_astrometry(times, image_arr=images, amp_arr=amps),
        dtype=float,
    )
    source = np.array(
        psbl.get_source_astrometry_unlensed(times), dtype=float
    )
    primary, secondary = psbl.get_resolved_lens_astrometry(times)
    result = {
        'mag': mag,
        'magnification': magnification,
        'n_images': n_images,
        'centroid_arcsec': centroid,
        'source_arcsec': source,
        'primary_arcsec': np.array(primary, dtype=float),
        'secondary_arcsec': np.array(secondary, dtype=float),
    }
    return result


def source_minus_com_mas(evaluated, m_primary, m_secondary):
    """Source position relative to the lens center of mass.

    Parameters
    ----------
    evaluated : dict
        Output of :func:`evaluate_event`.
    m_primary : float
        Primary mass (units cancel).
    m_secondary : float
        Secondary mass (same units).

    Returns
    -------
    offset_mas : ndarray, shape (N, 2)
        ``[East, North]`` source-minus-COM vector in milliarcseconds.
    """
    com = (
        m_primary * evaluated['primary_arcsec']
        + m_secondary * evaluated['secondary_arcsec']
    ) / (m_primary + m_secondary)
    offset_mas = (evaluated['source_arcsec'] - com) * 1.0e3
    return offset_mas


def caustic_points_theta_e(psbl, time_mjd, n_pts=2500):
    """BAGLE caustic points relative to the center of mass.

    Parameters
    ----------
    psbl : bagle model
        Binary-lens model with ``get_caustics``.
    time_mjd : float
        Time at which the caustic is evaluated. A static binary's caustic
        is fixed to the lenses, so this only sets the sky location.
    n_pts : int, optional
        Number of phase samples along the critical curve. BAGLE returns
        four polynomial branches.

    Returns
    -------
    points : ndarray, shape (M, 2)
        ``[East, North]`` caustic coordinates in units of ``theta_E``,
        relative to the center of mass. Non-finite samples are dropped.

    Notes
    -----
    ``get_caustics`` returns complex sky positions divided by ``theta_E``
    (real = East, imaginary = North), the same convention as
    ``get_complex_pos``.
    """
    time = np.array([time_mjd], dtype=float)
    caustic = psbl.get_caustics(time, N_pts=n_pts)[0]
    primary, secondary = psbl.get_resolved_lens_astrometry(time)
    com = (
        psbl.mLp * primary + psbl.mLs * secondary
    ) / (psbl.mLp + psbl.mLs)
    # theta_E in arcsec, so that sky arcsec / theta_E is dimensionless.
    theta_e_arcsec = psbl.thetaE_amp / 1.0e3
    com_complex = (com[0, 0] + 1j * com[0, 1]) / theta_e_arcsec
    branches = []
    for branch in range(caustic.shape[1]):
        relative = caustic[:, branch] - com_complex
        good = np.isfinite(relative)
        if np.any(good):
            kept = relative[good]
            coords = np.column_stack([kept.real, kept.imag])
            branches.append(coords)
    if len(branches) == 0:
        return np.zeros((0, 2), dtype=float)
    points = np.vstack(branches)
    return points


def critical_curve_theta_e(psbl, time_mjd, n_pts=1200):
    """Critical curve relative to the black hole, in Einstein units.

    Parameters
    ----------
    psbl : bagle model
        Binary-lens model with ``get_critical_curves``.
    time_mjd : float
        Evaluation time, MJD.
    n_pts : int, optional
        Phase samples per branch.

    Returns
    -------
    points : ndarray, shape (M, 2)
        ``[East, North]`` critical-curve coordinates in units of
        ``theta_E``, relative to the primary (the black hole).
    """
    time = np.array([time_mjd], dtype=float)
    critical = psbl.get_critical_curves(time, N_pts=n_pts)[0]
    primary, _secondary = psbl.get_resolved_lens_astrometry(time)
    theta_e_arcsec = psbl.thetaE_amp / 1.0e3
    primary_complex = (
        primary[0, 0] + 1j * primary[0, 1]
    ) / theta_e_arcsec
    branches = []
    for branch in range(critical.shape[1]):
        relative = critical[:, branch] - primary_complex
        good = np.isfinite(relative)
        if np.any(good):
            kept = relative[good]
            coords = np.column_stack([kept.real, kept.imag])
            branches.append(coords)
    if len(branches) == 0:
        return np.zeros((0, 2), dtype=float)
    points = np.vstack(branches)
    return points


def central_caustic_width_theta_e(q_mass, s_einstein):
    """Analytic full width of a planetary central caustic.

    Parameters
    ----------
    q_mass : float
        Planet-to-host mass ratio.
    s_einstein : float
        Projected separation in units of ``theta_E``.

    Returns
    -------
    width : float
        Cusp-to-cusp width along the binary axis, in units of
        ``theta_E``. The approximation is
        ``4 q / (s - 1/s)^2`` (Chung et al. / Han planetary limit).
        It is not valid in the resonant regime ``s ~ 1``, where it
        diverges.

    Notes
    -----
    For ``s << 1`` this reduces to ``4 q s^2``.
    """
    factor = s_einstein - 1.0 / s_einstein
    width = 4.0 * q_mass / factor ** 2
    return float(width)


def analytic_astroid(half_width, n_pts=400):
    """Four-cusped astroid of a given half-width.

    Parameters
    ----------
    half_width : float
        Distance from center to an on-axis cusp, same units as the plot.
    n_pts : int, optional
        Samples around the curve.

    Returns
    -------
    east : ndarray, shape (n_pts,)
        East coordinate.
    north : ndarray, shape (n_pts,)
        North coordinate.

    Notes
    -----
    The curve is ``x = a cos^3 phi``, ``y = a sin^3 phi``. For the
    close-planet limit the on-axis and perpendicular widths are nearly
    equal, which is what BAGLE returns for ``s ~ 0.009``.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_pts)
    east = half_width * np.cos(phi) ** 3
    north = half_width * np.sin(phi) ** 3
    return east, north


def model_time_grid(t_cross, t_min, t_max):
    """Time grid that resolves the black-hole peak and the baseline.

    Parameters
    ----------
    t_cross : float
        Geocentric caustic-crossing / peak time, MJD.
    t_min : float
        First time to cover, MJD.
    t_max : float
        Last time to cover, MJD.

    Returns
    -------
    times : ndarray, shape (N,)
        Sorted unique MJD values. Daily over the whole survey, 0.05 day
        within 40 days of the peak, and 0.002 day within 3 days.

    Notes
    -----
    The point-source planetary spike is ~0.2 seconds wide and is not
    sampled here. The grid does resolve the day-long black-hole peak.
    """
    daily = np.arange(t_min, t_max + 0.5, 1.0)
    near = np.arange(t_cross - 40.0, t_cross + 40.0, 0.05)
    inner = np.arange(t_cross - 3.0, t_cross + 3.0, 0.002)
    times = np.unique(np.concatenate([daily, near, inner]))
    return times


def style_mag_axis(axis, faint, bright):
    """Label a magnitude axis with bright stars toward the top.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis whose y values are magnitudes.
    faint : float
        Larger magnitude, drawn at the bottom.
    bright : float
        Smaller magnitude, drawn at the top.

    Returns
    -------
    None
    """
    axis.set_ylim(faint, bright)
    axis.set_ylabel('F146W (mag)')
    return None


def plot_photometry(t_data, mag_data, mag_err, t_model, mag_model,
                    mag_point, t_cross, peak_note, residual_max,
                    outpath):
    """Plot the F146 light curve, a peak zoom, and the planet residual.

    Parameters
    ----------
    t_data : ndarray, shape (N,)
        Roman F146 observation times, MJD.
    mag_data : ndarray, shape (N,)
        Noisy mock magnitudes.
    mag_err : ndarray, shape (N,)
        Photometric uncertainties in magnitudes.
    t_model : ndarray, shape (M,)
        Dense model times, MJD.
    mag_model : ndarray, shape (M,)
        BAGLE point-source binary-lens magnitudes.
    mag_point : ndarray, shape (M,)
        Point-lens magnitudes on the same grid, primary mass at the
        black-hole position.
    t_cross : float
        Geocentric peak time, MJD.
    peak_note : str
        Short note on the finite-source cap, drawn on the zoom panel.
    residual_max : float
        Maximum absolute planet-minus-point-lens residual on the Roman
        times, in magnitudes. Written on the residual panel.
    outpath : Path
        PNG destination.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.4, 9.4),
        sharex=False,
        gridspec_kw={'height_ratios': [1.05, 1.15, 0.72]},
    )
    ax_full, ax_zoom, ax_res = axes
    # The single model sample on the caustic (sub-second) is outside the
    # polynomial solver's trustworthy domain. Drop it so it does not draw
    # a spurious spike. The Roman data cadence cannot land on it either.
    safe = np.abs(t_model - t_cross) > 1.0e-4
    t_model = t_model[safe]
    mag_model = mag_model[safe]
    mag_point = mag_point[safe]

    # Full survey. Downsample the error bars for readability; the model
    # line still uses every model epoch.
    step = max(1, len(t_data) // 4000)
    ax_full.errorbar(
        t_data[::step],
        mag_data[::step],
        yerr=mag_err[::step],
        fmt='.',
        ms=1.6,
        color='0.25',
        ecolor='0.55',
        elinewidth=0.4,
        alpha=0.45,
        rasterized=True,
        zorder=2,
        label='Roman F146 mock',
    )
    ax_full.plot(
        t_model,
        mag_model,
        color='#b2182b',
        lw=1.2,
        zorder=3,
        label='BAGLE PSBL (point source)',
    )
    ax_full.plot(
        t_model,
        mag_point,
        color='#2166ac',
        lw=1.0,
        ls='--',
        zorder=4,
        label='Point lens (same BH trajectory)',
    )
    style_mag_axis(ax_full, 19.8, 14.0)
    ax_full.set_xlim(t_data.min() - 20.0, t_data.max() + 20.0)
    ax_full.set_xlabel('Time (MJD)')
    ax_full.text(
        0.02,
        0.04,
        'Point-source peak is brighter than this panel (see zoom).',
        transform=ax_full.transAxes,
        fontsize=8,
        va='bottom',
    )
    ax_full.set_title(
        '8 M$_\\odot$ black hole + 1 M$_\\oplus$ at 0.1 AU, '
        'dark lens, F146W = 19'
    )
    ax_full.legend(loc='lower right', frameon=False)

    zoom = (t_model > t_cross - 8.0) & (t_model < t_cross + 8.0)
    data_zoom = (t_data > t_cross - 8.0) & (t_data < t_cross + 8.0)
    ax_zoom.errorbar(
        t_data[data_zoom],
        mag_data[data_zoom],
        yerr=mag_err[data_zoom],
        fmt='.',
        ms=2.2,
        color='0.15',
        ecolor='0.45',
        elinewidth=0.5,
        alpha=0.7,
        rasterized=True,
        zorder=2,
        label='Roman F146 mock',
    )
    ax_zoom.plot(
        t_model[zoom],
        mag_model[zoom],
        color='#b2182b',
        lw=1.4,
        label='BAGLE PSBL',
    )
    ax_zoom.plot(
        t_model[zoom],
        mag_point[zoom],
        color='#2166ac',
        lw=1.1,
        ls='--',
        label='Point lens',
    )
    style_mag_axis(ax_zoom, 19.6, 5.5)
    ax_zoom.set_xlim(t_cross - 8.0, t_cross + 8.0)
    ax_zoom.set_xlabel('Time (MJD)')
    ax_zoom.set_title(
        'Zoom on the peak. The planetary caustic spike is 0.002 s wide '
        'and is not separately visible.'
    )
    ax_zoom.text(
        0.02,
        0.04,
        peak_note,
        transform=ax_zoom.transAxes,
        fontsize=8,
        va='bottom',
        ha='left',
    )
    ax_zoom.legend(loc='lower right', frameon=False)

    delta_mmag = 1.0e3 * (mag_model - mag_point)
    ax_res.plot(
        t_model,
        delta_mmag,
        color='#b2182b',
        lw=1.0,
        label='PSBL $-$ point lens',
    )
    ax_res.axhline(0.0, color='0.4', lw=0.6)
    # Baseline Roman sigma is ~17 mmag. Show that band for scale.
    ax_res.axhspan(
        -17.0, 17.0, color='#2166ac', alpha=0.12, label='±1σ at F146 = 19'
    )
    ax_res.set_xlim(t_data.min() - 20.0, t_data.max() + 20.0)
    ax_res.set_ylim(-40.0, 40.0)
    ax_res.set_ylabel('Residual (mmag)')
    ax_res.set_xlabel('Time (MJD)')
    ax_res.set_title(
        'Planet versus a point lens. '
        f'Max |Δm| on the Roman grid = {residual_max * 1.0e3:.3e} mmag.'
    )
    ax_res.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def plot_astrometry(t_model, source_mas, centroid_mas, point_mas,
                    primary_mas, t_data, data_mas, data_err_mas,
                    t_cross, outpath):
    """Plot the sky track and the astrometric shift versus time.

    Parameters
    ----------
    t_model : ndarray, shape (M,)
        Model times, MJD.
    source_mas : ndarray, shape (M, 2)
        Unlensed source position, mas, East then North, relative to the
        black hole at ``t_cross``.
    centroid_mas : ndarray, shape (M, 2)
        Lensed centroid, same frame.
    point_mas : ndarray, shape (M, 2)
        Point-lens centroid, same frame.
    primary_mas : ndarray, shape (M, 2)
        Black-hole position, same frame.
    t_data : ndarray, shape (N,)
        Observation times, MJD.
    data_mas : ndarray, shape (N, 2)
        Noisy centroid, same frame.
    data_err_mas : ndarray, shape (N,)
        Astrometric uncertainty per coordinate, mas.
    t_cross : float
        Geocentric peak time, MJD.
    outpath : Path
        PNG destination.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.4))
    ax_sky, ax_zoom, ax_east, ax_north = axes.ravel()
    # Same sub-second cut as the light curve: the on-caustic sample is
    # not a converged binary-lens solution.
    safe = np.abs(t_model - t_cross) > 1.0e-4
    t_model = t_model[safe]
    source_mas = source_mas[safe]
    centroid_mas = centroid_mas[safe]
    point_mas = point_mas[safe]
    primary_mas = primary_mas[safe]

    step = max(1, len(t_data) // 2500)
    ax_sky.errorbar(
        data_mas[::step, 0],
        data_mas[::step, 1],
        xerr=data_err_mas[::step],
        yerr=data_err_mas[::step],
        fmt='.',
        ms=1.5,
        color='0.35',
        ecolor='0.7',
        elinewidth=0.3,
        alpha=0.35,
        rasterized=True,
        zorder=2,
        label='Roman mock',
    )
    ax_sky.plot(
        source_mas[:, 0],
        source_mas[:, 1],
        color='0.45',
        lw=1.0,
        label='Unlensed source',
    )
    ax_sky.plot(
        centroid_mas[:, 0],
        centroid_mas[:, 1],
        color='#1b9e77',
        lw=1.3,
        label='Lensed centroid (PSBL)',
    )
    ax_sky.plot(
        primary_mas[:, 0],
        primary_mas[:, 1],
        color='0.15',
        lw=0.8,
        label='Black hole (parallax)',
    )
    # Mark the black hole at the peak. The planet is 0.033 mas away.
    i_peak = int(np.argmin(np.abs(t_model - t_cross)))
    ax_sky.scatter(
        [primary_mas[i_peak, 0]],
        [primary_mas[i_peak, 1]],
        marker='*',
        s=80,
        color='black',
        zorder=5,
        label='BH at peak',
    )
    ax_sky.invert_xaxis()
    ax_sky.set_aspect('equal', adjustable='datalim')
    ax_sky.set_xlabel(r'East offset, $\Delta\alpha\cos\delta$ (mas)')
    ax_sky.set_ylabel(r'North offset, $\Delta\delta$ (mas)')
    ax_sky.set_title('Sky track, relative to the BH at peak')
    ax_sky.legend(loc='best', frameon=False, fontsize=7.5)

    # Zoom: a few Einstein radii around the peak so the ~1 mas shift shows.
    window = (t_model > t_cross - 250.0) & (t_model < t_cross + 250.0)
    data_window = (t_data > t_cross - 250.0) & (t_data < t_cross + 250.0)
    ax_zoom.plot(
        source_mas[window, 0],
        source_mas[window, 1],
        color='0.45',
        lw=1.1,
        label='Unlensed',
    )
    ax_zoom.plot(
        point_mas[window, 0],
        point_mas[window, 1],
        color='#2166ac',
        lw=1.2,
        ls='--',
        label='Point-lens centroid',
    )
    ax_zoom.plot(
        centroid_mas[window, 0],
        centroid_mas[window, 1],
        color='#1b9e77',
        lw=1.4,
        label='PSBL centroid',
    )
    zoom_step = max(1, int(np.sum(data_window)) // 800)
    ax_zoom.errorbar(
        data_mas[data_window][::zoom_step, 0],
        data_mas[data_window][::zoom_step, 1],
        xerr=data_err_mas[data_window][::zoom_step],
        yerr=data_err_mas[data_window][::zoom_step],
        fmt='.',
        ms=2.0,
        color='0.2',
        ecolor='0.6',
        elinewidth=0.4,
        alpha=0.45,
        rasterized=True,
        label='Roman mock',
    )
    ax_zoom.scatter(
        [0.0],
        [0.0],
        marker='*',
        s=90,
        color='black',
        zorder=5,
        label='BH at peak',
    )
    ax_zoom.invert_xaxis()
    ax_zoom.set_aspect('equal', adjustable='box')
    ax_zoom.set_xlim(8.0, -8.0)
    ax_zoom.set_ylim(-8.0, 8.0)
    ax_zoom.set_xlabel(r'East offset (mas)')
    ax_zoom.set_ylabel(r'North offset (mas)')
    ax_zoom.set_title('±250 days around the peak')
    ax_zoom.legend(loc='best', frameon=False, fontsize=7.5)

    shift_psbl = centroid_mas - source_mas
    shift_point = point_mas - source_mas
    for axis, component, name in (
        (ax_east, 0, 'East'),
        (ax_north, 1, 'North'),
    ):
        axis.plot(
            t_model,
            shift_psbl[:, component],
            color='#1b9e77',
            lw=1.3,
            label='PSBL',
        )
        axis.plot(
            t_model,
            shift_point[:, component],
            color='#2166ac',
            lw=1.0,
            ls='--',
            label='Point lens',
        )
        axis.axvline(t_cross, color='0.5', lw=0.6, ls=':')
        axis.set_xlabel('Time (MJD)')
        axis.set_ylabel(f'{name} shift (mas)')
        axis.set_title(f'Centroid minus unlensed source ({name})')
        axis.legend(loc='best', frameon=False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def plot_caustics(critical, caustic_central, trajectory_rho,
                  trajectory_caustic, rho, s_einstein, width_theta_e,
                  half_width, outpath):
    """Plot critical curves and the central caustic with the trajectory.

    Parameters
    ----------
    critical : ndarray, shape (M, 2)
        Critical curve relative to the black hole, in ``theta_E``.
    caustic_central : ndarray, shape (K, 2)
        Central-caustic samples relative to the COM, in ``theta_E``.
    trajectory_rho : ndarray, shape (N, 2)
        Trajectory on the source-radius scale, in ``theta_E``.
    trajectory_caustic : ndarray, shape (N, 2)
        Trajectory on the caustic scale, in ``theta_E``.
    rho : float
        Source radius in units of ``theta_E``.
    s_einstein : float
        Projected separation in units of ``theta_E``.
    width_theta_e : float
        Analytic full central-caustic width in ``theta_E``.
    half_width : float
        Half of ``width_theta_e``, the astroid cusp distance.
    outpath : Path
        PNG destination.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.15))
    ax_crit, ax_rho, ax_cau = axes

    # Image plane. The Einstein ring is the point-lens critical curve.
    if len(critical) > 0:
        ring = np.linalg.norm(critical, axis=1) > 0.2
        planet_loop = ~ring
        ax_crit.scatter(
            critical[ring, 0],
            critical[ring, 1],
            s=2,
            c='#542788',
            rasterized=True,
            label='Critical curve',
        )
        if np.any(planet_loop):
            ax_crit.scatter(
                critical[planet_loop, 0],
                critical[planet_loop, 1],
                s=8,
                c='#e66101',
                label='Near the planet',
            )
    phi = np.linspace(0.0, 2.0 * np.pi, 400)
    ax_crit.plot(np.cos(phi), np.sin(phi), color='0.6', lw=0.7, ls=':')
    ax_crit.scatter([0.0], [0.0], marker='*', s=70, c='black', label='BH',
                    zorder=5)
    ax_crit.scatter(
        [-s_einstein],
        [0.0],
        marker='o',
        s=18,
        c='#e66101',
        label='Planet',
        zorder=5,
    )
    ax_crit.set_aspect('equal', adjustable='box')
    ax_crit.set_xlim(1.6, -1.6)
    ax_crit.set_ylim(-1.6, 1.6)
    ax_crit.set_xlabel(r'East / $\theta_E$ (image plane)')
    ax_crit.set_ylabel(r'North / $\theta_E$')
    ax_crit.set_title('Critical curves, relative to the BH')
    ax_crit.legend(loc='upper right', frameon=False, fontsize=7)

    # Source-radius scale. The caustic is a dot; the star is the circle.
    # Draw the star first so the trajectory remains visible across it.
    source_disk = Circle(
        (0.0, 0.0),
        rho,
        facecolor='#fdae61',
        edgecolor='#e66101',
        lw=1.0,
        alpha=0.8,
        label=r'Source (1 $R_\odot$)',
        zorder=2,
    )
    ax_rho.add_patch(source_disk)
    if len(trajectory_rho) > 0:
        ax_rho.plot(
            trajectory_rho[:, 0],
            trajectory_rho[:, 1],
            color='#1b9e77',
            lw=1.4,
            zorder=3,
            label='Source center',
        )
    ax_rho.scatter(
        [0.0],
        [0.0],
        marker='x',
        s=36,
        c='#542788',
        zorder=4,
        label='Caustic (unresolved)',
    )
    limit = 4.0 * rho
    ax_rho.set_aspect('equal', adjustable='box')
    ax_rho.set_xlim(limit, -limit)
    ax_rho.set_ylim(-limit, limit)
    ax_rho.set_xlabel(r'East / $\theta_E$ (source plane)')
    ax_rho.set_ylabel(r'North / $\theta_E$')
    ax_rho.set_title(r'Source scale: $\rho \gg$ caustic')
    ax_rho.legend(loc='upper right', frameon=False, fontsize=7)

    # Caustic scale. BAGLE's samples plus the analytic astroid.
    if len(caustic_central) > 0:
        ax_cau.scatter(
            caustic_central[:, 0],
            caustic_central[:, 1],
            s=14,
            c='#542788',
            zorder=3,
            label='BAGLE caustic',
        )
    ast_e, ast_n = analytic_astroid(half_width)
    ax_cau.plot(
        ast_e,
        ast_n,
        color='#e66101',
        lw=1.5,
        label='Analytic astroid',
    )
    if len(trajectory_caustic) > 0:
        ax_cau.plot(
            trajectory_caustic[:, 0],
            trajectory_caustic[:, 1],
            color='#1b9e77',
            lw=1.2,
            label='Source center',
            zorder=4,
        )
    span = 3.2 * half_width
    ax_cau.set_aspect('equal', adjustable='box')
    ax_cau.set_xlim(span, -span)
    ax_cau.set_ylim(-span, span)
    ax_cau.set_xlabel(r'East / $\theta_E$ (source plane)')
    ax_cau.set_ylabel(r'North / $\theta_E$')
    ax_cau.set_title(
        f'Central caustic, full width {width_theta_e:.2e} '
        r'$\theta_E$'
    )
    ax_cau.legend(loc='upper right', frameon=False, fontsize=7)
    fig.suptitle(
        'Caustic crossing is a point-source statement. '
        'The stellar disk in the middle panel is ~10$^{6}\\times$ larger.',
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def plot_optional(s_grid, width_grid, s_requested, width_requested,
                  s_resonant, width_resonant, rho, theta_e_mas, r_e_au,
                  caustic_pts, trajectory, t_opt, mag_bin, mag_pt,
                  mag_err_ref, outpath):
    """Optional figure: separation that makes an Earth-mass caustic visible.

    Parameters
    ----------
    s_grid : ndarray, shape (N,)
        Projected separations in units of ``theta_E``, excluding the
        resonant neighborhood of ``s = 1``.
    width_grid : ndarray, shape (N,)
        Analytic central-caustic full width, in ``theta_E``.
    s_requested : float
        Requested separation, ``0.1 AU / r_E``.
    width_requested : float
        Analytic width at ``s_requested``, in ``theta_E``.
    s_resonant : float
        Separation of the optional BAGLE example (``s = 1``).
    width_resonant : float
        BAGLE-measured full East-West extent at ``s = 1``, in ``theta_E``.
    rho : float
        Source radius in ``theta_E``.
    theta_e_mas : float
        Einstein radius in mas, used only to label AU on the top axis.
    r_e_au : float
        Einstein radius at the lens, in AU.
    caustic_pts : ndarray, shape (K, 2)
        Optional-model caustic relative to the COM, in ``theta_E``.
    trajectory : ndarray, shape (M, 2)
        Source-center track relative to the COM, in ``theta_E``.
    t_opt : ndarray, shape (M,)
        Times for the optional light curve, MJD.
    mag_bin : ndarray, shape (M,)
        BAGLE PSBL magnitudes.
    mag_pt : ndarray, shape (M,)
        Point-lens magnitudes.
    mag_err_ref : float
        Representative F146 uncertainty at the baseline, magnitudes.
    outpath : Path
        PNG destination.

    Returns
    -------
    None
    """
    # The s = 1 caustic is ~10:1 (East-West over North-South). Give it
    # a full-width row so equal aspect still leaves the cusps visible.
    fig = plt.figure(figsize=(11.6, 6.6))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.25, 1.0], width_ratios=[1.15, 1.0]
    )
    ax_w = fig.add_subplot(grid[0, 0])
    ax_l = fig.add_subplot(grid[0, 1])
    ax_g = fig.add_subplot(grid[1, :])

    ax_w.plot(
        s_grid,
        width_grid,
        color='#542788',
        lw=1.3,
        label=r'Analytic $4q/(s-1/s)^{2}$',
    )
    ax_w.scatter(
        [s_requested],
        [width_requested],
        s=40,
        c='#b2182b',
        zorder=4,
        label='Requested 0.1 AU',
    )
    ax_w.scatter(
        [s_resonant],
        [width_resonant],
        s=46,
        marker='*',
        c='#e66101',
        zorder=4,
        label='BAGLE at $s = 1$',
    )
    ax_w.axhline(rho, color='#e66101', ls='--', lw=0.9, label=r'$\rho$ (1 $R_\odot$)')
    ax_w.axhline(2.0 * rho, color='#e66101', ls=':', lw=0.8, label=r'source diameter')
    ax_w.axvspan(0.9, 1.1, color='0.85', alpha=0.8, label='resonant (formula fails)')
    ax_w.set_xscale('log')
    ax_w.set_yscale('log')
    ax_w.set_xlabel(r'Projected separation $s$ ($\theta_E$)')
    ax_w.set_ylabel(r'Central-caustic full width ($\theta_E$)')
    ax_w.set_title('What separation would be large enough?')
    ax_w.legend(loc='lower right', frameon=False, fontsize=6.5)
    top = ax_w.twiny()
    top.set_xscale('log')
    top.set_xlim(ax_w.get_xlim())
    # Tick in AU at a few decades. s * r_E.
    au_ticks = np.array([0.1, 1.0, 11.0])
    s_ticks = au_ticks / r_e_au
    top.set_xticks(s_ticks)
    top.set_xticklabels([f'{value:g} AU' for value in au_ticks])
    top.set_xlabel(r'Projected separation ($r_E = %.2f$ AU)' % r_e_au)

    disk = Circle(
        (0.0, 0.0),
        rho,
        facecolor='#fdae61',
        edgecolor='#e66101',
        alpha=0.9,
        zorder=3,
        label=r'1 $R_\odot$ source',
    )
    ax_g.add_patch(disk)
    if len(caustic_pts) > 0:
        ax_g.scatter(
            caustic_pts[:, 0],
            caustic_pts[:, 1],
            s=8,
            c='#542788',
            rasterized=True,
            zorder=2,
            label='BAGLE caustic',
        )
    if len(trajectory) > 0:
        ax_g.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color='#1b9e77',
            lw=1.5,
            zorder=4,
            label='Source center',
        )
    # Zoom to the caustic. At s = 1 and q ~ 4e-7 the resonant caustic
    # is much wider East-West than North-South; a fixed ±0.014 box
    # squeezes that shape into a few pixels.
    if len(caustic_pts) > 0:
        half_east = 0.5 * float(
            caustic_pts[:, 0].max() - caustic_pts[:, 0].min()
        )
        half_north = 0.5 * float(
            caustic_pts[:, 1].max() - caustic_pts[:, 1].min()
        )
    else:
        half_east = 0.01
        half_north = 0.002
    pad_east = max(3.0 * rho, 0.2 * half_east)
    pad_north = max(4.0 * rho, 0.45 * half_north)
    ax_g.set_aspect('equal', adjustable='box')
    ax_g.set_xlim(half_east + pad_east, -(half_east + pad_east))
    ax_g.set_ylim(-(half_north + pad_north), half_north + pad_north)
    ax_g.set_xlabel(r'East / $\theta_E$')
    ax_g.set_ylabel(r'North / $\theta_E$')
    ax_g.set_title(r'OPTIONAL: $s = 1$ caustic ($a \approx r_E$)')
    ax_g.legend(loc='upper right', frameon=False, fontsize=7)

    ax_l.plot(t_opt, mag_bin, color='#b2182b', lw=1.3, label='BAGLE PSBL')
    ax_l.plot(
        t_opt, mag_pt, color='#2166ac', lw=1.1, ls='--', label='Point lens'
    )
    # One Roman-sized error bar at baseline, for scale.
    t_bar = t_opt[0] + 0.15 * (t_opt[-1] - t_opt[0])
    ax_l.errorbar(
        [t_bar],
        [19.15],
        yerr=[mag_err_ref],
        fmt='o',
        ms=3,
        color='0.2',
        label=f'Roman σ at mag 19 ({mag_err_ref:.3f})',
    )
    bright = float(np.nanpercentile(np.concatenate([mag_bin, mag_pt]), 0.5))
    style_mag_axis(ax_l, 19.7, min(bright - 0.4, 14.0))
    ax_l.set_xlabel('Time (MJD)')
    ax_l.set_title('Same Earth mass, separation = $r_E$')
    ax_l.legend(loc='lower right', frameon=False, fontsize=7)
    fig.suptitle(
        'OPTIONAL — not the requested 0.1 AU system. '
        'Earth-mass planet moved to $s = 1$ so a caustic crossing is real.',
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def plot_parameter_table(rows, outpath):
    """Render the parameter table as a figure.

    Parameters
    ----------
    rows : list of tuple
        ``(label, value)`` strings.
    outpath : Path
        PNG destination.

    Returns
    -------
    None
    """
    fig, axis = plt.subplots(figsize=(8.8, 12.4))
    axis.axis('off')
    lines = [f'{label:<46s} {value}' for label, value in rows]
    axis.text(
        0.01,
        0.99,
        '\n'.join(lines),
        va='top',
        ha='left',
        family='monospace',
        fontsize=8.0,
        transform=axis.transAxes,
    )
    axis.set_title('Input and derived parameters', loc='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def write_tables(rows, outdir):
    """Write the parameter table as text and CSV.

    Parameters
    ----------
    rows : list of tuple
        ``(label, value)`` strings.
    outdir : Path
        Directory for ``parameters.txt`` and ``parameters.csv``.

    Returns
    -------
    None
    """
    text_path = outdir / 'parameters.txt'
    csv_path = outdir / 'parameters.csv'
    with text_path.open('w', encoding='utf-8') as handle:
        handle.write('BAGLE Roman BH + Earth-mass planet demo\n')
        handle.write('=' * 72 + '\n')
        for label, value in rows:
            handle.write(f'{label:<46s} {value}\n')
    with csv_path.open('w', encoding='utf-8') as handle:
        handle.write('quantity,value\n')
        for label, value in rows:
            safe_label = '"' + label.replace('"', "'") + '"'
            safe_value = '"' + str(value).replace('"', "'") + '"'
            handle.write(f'{safe_label},{safe_value}\n')
    return None


def copy_artifacts(outdir):
    """Copy output files into the agent artifact directory, if present.

    Parameters
    ----------
    outdir : Path
        Directory that holds the PNG and table files.

    Returns
    -------
    None
    """
    if not ARTIFACT_DIR.is_dir():
        return None
    for path in sorted(outdir.iterdir()):
        if path.is_file():
            target = ARTIFACT_DIR / path.name
            target.write_bytes(path.read_bytes())
    return None


def separation_grid(q_mass):
    """Analytic central-caustic width on either side of resonance.

    Parameters
    ----------
    q_mass : float
        Planet-to-host mass ratio.

    Returns
    -------
    s_values : ndarray
        Separations in ``theta_E``, excluding ``0.9 < s < 1.1``.
    widths : ndarray
        Analytic full widths in ``theta_E``.
    """
    left = np.logspace(np.log10(0.004), np.log10(0.9), 200)
    right = np.logspace(np.log10(1.1), np.log10(3.0), 120)
    s_values = np.concatenate([left, right])
    widths = np.array([
        central_caustic_width_theta_e(q_mass, float(s_value))
        for s_value in s_values
    ])
    return s_values, widths


def main():
    """Generate the mock Roman event, figures, and parameter table.

    Returns
    -------
    None
    """
    warnings.filterwarnings('ignore')
    configure_matplotlib()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    m_earth = earth_mass_msun()
    m_bh = 8.0
    m_planet = m_earth
    q_mass = m_planet / m_bh
    d_l_pc = 3000.0
    d_s_pc = 8000.0
    a_au = 0.1
    radius_rsun = 1.0

    # Trial model: theta_E and pi_rel do not depend on beta or separation,
    # only on the total mass and the two distances.
    trial = build_psbl(57000.0, 0.01, 0.03, m_bh, m_planet)
    theta_e_mas = float(trial.thetaE_amp)
    pi_rel_mas = float(trial.piRel)
    pi_l_mas = float(trial.piL)
    pi_s_mas = float(trial.piS)
    # 1 mas at 1 kpc is 1 AU, so r_E [AU] = theta_E [mas] * D_L [kpc].
    r_e_au = theta_e_mas * (d_l_pc / 1000.0)
    # Face-on circular orbit: the sky projection equals the 3D separation.
    s_einstein = a_au / r_e_au
    sep_mas = s_einstein * theta_e_mas

    print('Computing Roman GBTDS F146 times...', flush=True)
    t_f146, t_f087 = fake_data.get_times_roman_gbtds()
    t_f146 = np.asarray(t_f146, dtype=float)
    t_f087 = np.asarray(t_f087, dtype=float)
    seasons = fast_seasons(t_f146)
    fast = [season for season in seasons if season[3]]
    if len(fast) == 0:
        raise RuntimeError('No fast Roman season was found.')
    t_season_start, t_season_stop, n_fast, _is_fast = fast[0]
    # Geocentric crossing is placed in the middle of the first fast season
    # (2027 February–April for the default GBTDS window).
    t_cross = 0.5 * (t_season_start + t_season_stop)

    com_mas = com_offset_mas(sep_mas, m_bh, m_planet, alpha_deg=90.0)
    t0, beta_mas, pvec = aim_at_sky_offset(
        t_cross, com_mas[0], com_mas[1], pi_rel_mas, mu_east=7.0
    )
    psbl = build_psbl(t0, beta_mas, sep_mas, m_bh, m_planet)

    # Confirm the geocentric source sits on the center of mass.
    check = evaluate_event(psbl, np.array([t_cross]))
    offset_check = source_minus_com_mas(check, m_bh, m_planet)[0]
    miss_mas = float(np.linalg.norm(offset_check))
    print(f'Geocentric miss from COM at t_cross: {miss_mas:.3e} mas')
    if miss_mas > 1.0e-4:
        raise RuntimeError(
            'Source was not aimed through the central caustic.'
        )

    print(f'Evaluating PSBL on {len(t_f146)} Roman epochs...', flush=True)
    roman = evaluate_event(psbl, t_f146)
    t_model = model_time_grid(t_cross, float(t_f146.min()), float(t_f146.max()))
    print(f'Evaluating PSBL on {len(t_model)} model times...', flush=True)
    curve = evaluate_event(psbl, t_model)

    # Point-lens comparison: separation from the black hole, theta_E of
    # the total mass. Away from the caustic this matches BAGLE to ~1e-8.
    def point_lens_mag_from_eval(evaluated):
        delta_mas = (
            evaluated['source_arcsec'] - evaluated['primary_arcsec']
        ) * 1.0e3
        u_einstein = np.linalg.norm(delta_mas, axis=1) / theta_e_mas
        return point_lens_magnification(u_einstein), u_einstein

    mag_src = 19.0
    a_point_model, u_model = point_lens_mag_from_eval(curve)
    mag_point_model = mag_src - 2.5 * np.log10(a_point_model)
    a_point_roman, u_roman = point_lens_mag_from_eval(roman)
    mag_point_roman = mag_src - 2.5 * np.log10(a_point_roman)

    mag_noisy, mag_err = add_roman_photometric_noise(roman['mag'], rng_seed=146)
    _mag_err_formula, ast_err_mas = roman_f146_uncertainties(roman['mag'])

    # Astrometric noise in the floored Roman error, local generator so it
    # does not consume the photometric random stream.
    rng = np.random.default_rng(146)
    ast_noise = rng.normal(
        scale=ast_err_mas[:, None],
        size=roman['centroid_arcsec'].shape,
    )
    data_centroid_arcsec = roman['centroid_arcsec'] + ast_noise * 1.0e-3

    # Reference frame: black hole at the peak sits at the origin.
    i_peak = int(np.argmin(np.abs(t_model - t_cross)))
    origin_arcsec = curve['primary_arcsec'][i_peak]

    def to_mas(arcsec):
        return (arcsec - origin_arcsec) * 1.0e3

    source_mas = to_mas(curve['source_arcsec'])
    centroid_mas = to_mas(curve['centroid_arcsec'])
    primary_mas = to_mas(curve['primary_arcsec'])
    point_centroid = point_lens_centroid(
        curve['source_arcsec'], curve['primary_arcsec'], theta_e_mas
    )
    point_mas = to_mas(point_centroid)
    data_mas = to_mas(data_centroid_arcsec)

    delta_mag = roman['mag'] - mag_point_roman
    residual_max = float(np.nanmax(np.abs(delta_mag)))
    chi2_phot = float(np.nansum((delta_mag / mag_err) ** 2))
    delta_pos_mas = (
        roman['centroid_arcsec'] - point_lens_centroid(
            roman['source_arcsec'], roman['primary_arcsec'], theta_e_mas
        )
    ) * 1.0e3
    delta_pos_amp = np.linalg.norm(delta_pos_mas, axis=1)
    residual_pos_max = float(np.nanmax(delta_pos_amp))
    shift_mas = np.linalg.norm(centroid_mas - source_mas, axis=1)
    # Ignore the single on-caustic sample; the solver is not converged there.
    shift_ok = np.abs(t_model - t_cross) > 1.0e-4
    max_shift_mas = float(np.nanmax(shift_mas[shift_ok]))
    chi2_ast = float(np.nansum((delta_pos_mas / ast_err_mas[:, None]) ** 2))

    # Caustic geometry.
    print('Computing caustics...', flush=True)
    caustic = caustic_points_theta_e(psbl, t_cross, n_pts=2500)
    critical = critical_curve_theta_e(psbl, t_cross, n_pts=900)
    radius_caustic = np.linalg.norm(caustic, axis=1) if len(caustic) else np.array([])
    central = caustic[radius_caustic < 1.0e-8] if len(caustic) else caustic
    if len(central) == 0:
        raise RuntimeError('BAGLE did not return a central caustic.')
    measured_full_east = float(central[:, 0].max() - central[:, 0].min())
    measured_full_north = float(central[:, 1].max() - central[:, 1].min())
    width_analytic = central_caustic_width_theta_e(q_mass, s_einstein)
    half_width = 0.5 * width_analytic

    # Trajectories relative to the COM, in Einstein units.
    com_track = source_minus_com_mas(curve, m_bh, m_planet) / theta_e_mas
    # Long enough that the track extends past the stellar disk.
    rho_window = np.abs(t_model - t_cross) < 0.25
    # Caustic-scale track: a few caustic-widths on either side.
    # Width ~ 1e-10 theta_E and t_E ~ 192 d, so 1e-8 day is plenty.
    t_caustic = t_cross + np.linspace(-2.0e-8, 2.0e-8, 41)
    caustic_eval = evaluate_event(psbl, t_caustic)
    track_caustic = (
        source_minus_com_mas(caustic_eval, m_bh, m_planet) / theta_e_mas
    )

    theta_star_mas = source_angular_radius_mas(d_s_pc, radius_rsun)
    rho = theta_star_mas / theta_e_mas
    a_fs_peak = uniform_disk_peak_magnification(rho)
    mag_fs_peak = mag_src - 2.5 * np.log10(a_fs_peak)
    period_days = orbital_period_days(a_au, m_bh + m_planet)
    # Time to cross the analytic full width at the relative proper motion.
    mu_rel = float(psbl.muRel_amp)
    t_e_days = float(psbl.tE)
    duration_caustic_days = width_analytic * t_e_days
    duration_source_days = 2.0 * rho * t_e_days
    u0_bary = float(psbl.u0_amp)
    # Minimum geocentric separation from the black hole on the Roman grid.
    min_sep_bh_mas = float(np.min(u_roman) * theta_e_mas)
    min_sep_com_mas = miss_mas

    # Fractional agreement outside the numerically broken zone |u| < 1e-8.
    trustworthy = u_roman > 1.0e-6
    frac = np.abs(
        roman['magnification'][trustworthy] - a_point_roman[trustworthy]
    ) / a_point_roman[trustworthy]
    max_frac = float(np.nanmax(frac))

    peak_note = (
        f'1 Rsun uniform-disk point-lens cap: '
        f'A ≈ {a_fs_peak:.0f}, F146 ≈ {mag_fs_peak:.2f}. '
        'Not a BAGLE model (no FSBL).'
    )

    print('Writing figures...', flush=True)
    plot_photometry(
        t_f146,
        mag_noisy,
        mag_err,
        t_model,
        curve['mag'],
        mag_point_model,
        t_cross,
        peak_note,
        residual_max,
        OUTDIR / 'fig_photometry_f146.png',
    )
    plot_astrometry(
        t_model,
        source_mas,
        centroid_mas,
        point_mas,
        primary_mas,
        t_f146,
        data_mas,
        ast_err_mas,
        t_cross,
        OUTDIR / 'fig_astrometry.png',
    )
    plot_caustics(
        critical,
        central,
        com_track[rho_window],
        track_caustic,
        rho,
        s_einstein,
        width_analytic,
        half_width,
        OUTDIR / 'fig_caustic.png',
    )

    # Optional: same planet mass, separation equal to the Einstein radius.
    print('Building optional s = 1 example...', flush=True)
    sep_opt_mas = theta_e_mas
    s_opt = 1.0
    com_opt = com_offset_mas(sep_opt_mas, m_bh, m_planet, alpha_deg=90.0)
    t0_opt, beta_opt, _pvec_opt = aim_at_sky_offset(
        t_cross, com_opt[0], com_opt[1], pi_rel_mas, mu_east=7.0
    )
    psbl_opt = build_psbl(t0_opt, beta_opt, sep_opt_mas, m_bh, m_planet)
    t_opt = np.arange(t_cross - 3.0, t_cross + 3.0, 12.8 / 1440.0)
    opt_curve = evaluate_event(psbl_opt, t_opt)
    a_opt_point, _u_opt = point_lens_mag_from_eval(opt_curve)
    # point_lens_mag_from_eval uses the outer theta_e_mas, which matches
    # because the planet mass did not change.
    mag_opt_point = mag_src - 2.5 * np.log10(a_opt_point)
    opt_delta = np.abs(opt_curve['mag'] - mag_opt_point)
    opt_max_delta = float(np.nanmax(opt_delta))
    n_five = int(np.sum(opt_curve['n_images'] >= 5))
    caustic_opt = caustic_points_theta_e(psbl_opt, t_cross, n_pts=1800)
    if len(caustic_opt):
        resonant_width = float(
            caustic_opt[:, 0].max() - caustic_opt[:, 0].min()
        )
    else:
        resonant_width = np.nan
    track_opt = source_minus_com_mas(opt_curve, m_bh, m_planet) / theta_e_mas
    s_grid, width_grid = separation_grid(q_mass)
    _mag_err_19, ast_err_19 = roman_f146_uncertainties(np.array([19.0]))
    # Photometric error actually returned by BAGLE's noise helper at mag 19.
    _dummy_noisy, mag_err_19 = add_roman_photometric_noise(
        np.array([19.0, 19.0]), rng_seed=1
    )
    mag_err_19_value = float(np.median(mag_err_19))

    plot_optional(
        s_grid,
        width_grid,
        s_einstein,
        width_analytic,
        s_opt,
        resonant_width,
        rho,
        theta_e_mas,
        r_e_au,
        caustic_opt,
        track_opt,
        t_opt,
        opt_curve['mag'],
        mag_opt_point,
        mag_err_19_value,
        OUTDIR / 'fig_optional_s1_earth.png',
    )

    # Mass that would make the close-limit central caustic as large as rho.
    # w = 4 q s^2 = rho  =>  q = rho / (4 s^2). Not valid if q is not << 1.
    q_for_rho = rho / (4.0 * s_einstein ** 2)
    m_for_rho = q_for_rho * m_bh

    t0_iso = Time(t0, format='mjd').isot
    t_cross_iso = Time(t_cross, format='mjd').isot
    fast_dt_min = float(
        np.median(np.diff(np.sort(t_f146)[:n_fast])) * 1440.0
    )
    pi_e_amp = float(psbl.piE_amp)
    rows = [
        ('Model class', 'PSBL_PhotAstrom_Par_Param1'),
        ('Finite-source binary lens', 'unavailable (FSBL commented out)'),
        ('Orbital-motion classes', 'yes, not used (see notes below)'),
        ('Cadence helper', 'fake_data.get_times_roman_gbtds'),
        ('Photometric noise', 'fake_data.add_photometric_noise, zp=28'),
        ('Astrometric error', 'FWHM/(2 SNR), floor 0.1 mas; no 1e-5'),
        ('M_BH (Msun)', f'{m_bh:.1f}'),
        ('M_planet (Msun)', f'{m_planet:.6e}'),
        ('M_planet', '1 M_earth'),
        ('q = M_planet/M_BH', f'{q_mass:.6e}'),
        ('D_L (kpc)', f'{d_l_pc / 1000.0:.1f}'),
        ('D_S (kpc)', f'{d_s_pc / 1000.0:.1f}'),
        ('a_3D (AU)', f'{a_au:.1f}'),
        ('Projection', 'face-on; a_proj = a_3D'),
        ('s = a_proj / r_E', f'{s_einstein:.6e}'),
        ('sep (mas)', f'{sep_mas:.6e}'),
        ('alpha (deg E of N)', '90 (planet due West of BH)'),
        ('b_sff', '1 (no blend, no lens flux)'),
        ('dmag_Lp_Ls', '0 (both components dark)'),
        ('mag_src F146W', '19'),
        ('mu_rel (mas/yr)', f'{mu_rel:.1f} due East'),
        ('mu_L (mas/yr)', '0, 0 in this frame'),
        ('RA (deg)', f'{RA_DEG:.6f}'),
        ('Dec (deg)', f'{DEC_DEG:.6f}'),
        ('Observer', 'earth (Roman L2 ~ 0.01 AU away)'),
        ('t0 barycentric (MJD)', f'{t0:.5f} ({t0_iso})'),
        ('t_cross geocentric (MJD)', f'{t_cross:.5f} ({t_cross_iso})'),
        ('beta barycentric (mas)', f'{beta_mas:.6e}'),
        ('u0 barycentric', f'{u0_bary:.6e}'),
        ('min sep from BH, Roman grid (mas)', f'{min_sep_bh_mas:.6e}'),
        ('sep from COM at t_cross (mas)', f'{min_sep_com_mas:.6e}'),
        ('theta_E (mas)', f'{theta_e_mas:.6f}'),
        ('r_E at lens (AU)', f'{r_e_au:.4f}'),
        ('t_E (days)', f'{t_e_days:.4f}'),
        ('pi_L (mas)', f'{pi_l_mas:.6f}'),
        ('pi_S (mas)', f'{pi_s_mas:.6f}'),
        ('pi_rel (mas)', f'{pi_rel_mas:.6f}'),
        ('pi_E', f'{pi_e_amp:.6f}'),
        ('R_source assumed', '1 R_sun'),
        ('theta_source (mas)', f'{theta_star_mas:.6e}'),
        ('rho = theta_*/theta_E', f'{rho:.6e}'),
        ('central caustic width analytic (theta_E)', f'{width_analytic:.6e}'),
        ('central caustic width analytic (mas)', f'{width_analytic * theta_e_mas:.6e}'),
        ('BAGLE caustic East extent (theta_E)', f'{measured_full_east:.6e}'),
        ('BAGLE caustic North extent (theta_E)', f'{measured_full_north:.6e}'),
        ('point-source caustic duration (days)', f'{duration_caustic_days:.6e}'),
        (
            'point-source caustic duration (seconds)',
            f'{duration_caustic_days * 86400.0:.4f}',
        ),
        ('source-diameter crossing (days)', f'{duration_source_days:.5f}'),
        ('finite-source peak A (point lens, u=0)', f'{a_fs_peak:.1f}'),
        ('finite-source peak F146 (mag)', f'{mag_fs_peak:.3f}'),
        ('planet period (days)', f'{period_days:.4f}'),
        ('N F146 epochs', f'{len(t_f146)}'),
        ('N F087 epochs (not simulated)', f'{len(t_f087)}'),
        ('fast-cadence median (minutes)', f'{fast_dt_min:.3f}'),
        ('F146 mag err at mag 19', f'{mag_err_19_value:.5f}'),
        ('F146 ast err at mag 19 (mas)', f'{float(ast_err_19[0]):.4f}'),
        ('max |A_PSBL-A_point|/A outside caustic', f'{max_frac:.3e}'),
        ('max |dm| planet vs point lens (mag)', f'{residual_max:.6e}'),
        ('chi2 phot of that residual', f'{chi2_phot:.6e}'),
        ('max |dpos| planet vs point lens (mas)', f'{residual_pos_max:.6e}'),
        ('chi2 ast of that residual', f'{chi2_ast:.6e}'),
        ('peak BH centroid shift (mas)', f'{max_shift_mas:.4f}'),
        ('q to make w=rho at 0.1 AU (invalid if ~1)', f'{q_for_rho:.3f}'),
        ('that companion mass (Msun)', f'{m_for_rho:.2f}'),
        ('optional s', '1 (a_proj = r_E)'),
        ('optional BAGLE caustic EW extent (theta_E)', f'{resonant_width:.6e}'),
        ('optional max |dm| vs point lens (mag)', f'{opt_max_delta:.3f}'),
        ('optional epochs with >=5 images', f'{n_five} / {len(t_opt)}'),
        ('Notes', 'Static binary is not consistent with a 4-day orbit.'),
        ('Notes 2', 'BAGLE has CircOrbs, EllOrbs, LinOrbs, AccOrbs PSBL classes.'),
        ('Notes 3', '1e-5 factor in test_roman_lightcurve was not applied.'),
    ]
    write_tables(rows, OUTDIR)
    plot_parameter_table(rows, OUTDIR / 'fig_parameter_table.png')
    copy_artifacts(OUTDIR)

    print('max |dA/A| outside caustic zone:', max_frac)
    print('max |dm| mag:', residual_max, 'chi2_phot:', chi2_phot)
    print('max |dpos| mas:', residual_pos_max, 'chi2_ast:', chi2_ast)
    print('optional max |dm|:', opt_max_delta, 'n_five:', n_five)
    print('Wrote', OUTDIR)
    return None


if __name__ == '__main__':
    main()
