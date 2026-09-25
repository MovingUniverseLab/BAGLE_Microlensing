"""Roman GBTDS mock: static 8 solar-mass black hole plus an Earth at 6 AU.

This is the fourth demonstration for Prof. Jessica Lu. It does not modify
BAGLE. The lens is an 8 solar-mass black hole at 5 kpc. A 1 Earth-mass
planet has a sky-projected separation of 6 AU. Both components are dark.
There is no orbital motion: the binary is the static point-source model
``PSBL_PhotAstrom_Par_Param1`` with parallax. A face-on orbit would keep
this projected separation, but that orbit is not integrated.

The source is at 8 kpc with F146W = 19. Its trajectory is due East at
7 mas/yr and is aimed through one of the two triangular planetary
caustics. The impact parameter is kept small enough that the black hole
still produces a clear photometric peak, and the peak and the caustic
crossing are placed in consecutive Roman fast seasons.

BAGLE has no working finite-source binary class (``FSBL`` is commented
out; ``FSPL`` is a single lens). The curves are point-source. A 1 solar
radius is used only to draw the stellar disk and to compare ``rho`` with
the caustic.

Run from the repository root with::

    PYTHONPATH=src python scratch/bh_planet_roman/make_bh_planet_roman_v4.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_bh_planet_roman_demo as v1


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'output' / 'v4_static_6AU_5kpc'
ARTIFACT_DIR = Path('/opt/cursor/artifacts')

# Due East, same bulge-like relative motion as the earlier demos.
MU_EAST = 7.0
MU_NORTH = 0.0
# Barycentric closest-approach target, in Einstein units. This is the
# North coordinate of the crossed caustic. It is a choice: see Notes in
# ``choose_crossing_time``.
U0_TARGET = 0.10
D_L_PC = 5000.0
D_S_PC = 8000.0
A_AU = 6.0


def build_static(t0, beta_mas, sep_mas, alpha_deg, m_bh, m_planet):
    """Build a dark static binary lens with annual parallax.

    Parameters
    ----------
    t0 : float
        Barycentric time of closest approach to the geometric center,
        MJD.
    beta_mas : float
        Signed barycentric impact parameter, milliarcseconds. With an
        Eastward proper motion this is the North offset.
    sep_mas : float
        Sky-projected separation, milliarcseconds.
    alpha_deg : float
        BAGLE binary angle, degrees East of North. The primary lies at
        position angle ``alpha_deg`` from the geometric center.
    m_bh : float
        Black-hole mass in solar masses.
    m_planet : float
        Planet mass in solar masses.

    Returns
    -------
    psbl : bagle.model.PSBL_PhotAstrom_Par_Param1
        Static point-source binary. Lens flux is zero.

    Notes
    -----
    ``b_sff = 1`` and ``dmag_Lp_Ls = 0`` are the dark-lens prescription.
    The projected separation is an input, not an orbit. Calling it
    face-on means a circular orbit in the plane of the sky would have
    this separation at every epoch; the model itself does not move the
    planet.
    """
    psbl = v1.model.PSBL_PhotAstrom_Par_Param1(
        m_bh,
        m_planet,
        t0,
        0.0,
        0.0,
        beta_mas,
        0.0,
        0.0,
        MU_EAST,
        MU_NORTH,
        D_L_PC,
        D_S_PC,
        sep_mas,
        alpha_deg,
        [1.0],
        [19.0],
        [0.0],
        raL=v1.RA_DEG,
        decL=v1.DEC_DEG,
        obsLocation=['earth'],
        root_tol=1.0e-8,
    )
    return psbl


def mass_fractions(psbl):
    """Mass fractions and the total Einstein radius.

    Parameters
    ----------
    psbl : bagle model
        Binary lens. ``m1`` and ``m2`` are squared Einstein radii of
        each component in arcsec^2, not mass fractions.

    Returns
    -------
    m1 : float
        Primary mass fraction. ``m1 + m2 = 1``.
    m2 : float
        Secondary mass fraction.
    theta_arcsec : float
        Total Einstein radius in arcseconds.
    """
    m_sum = float(psbl.m1 + psbl.m2)
    m1 = float(psbl.m1 / m_sum)
    m2 = float(psbl.m2 / m_sum)
    theta_arcsec = float(np.sqrt(m_sum))
    return m1, m2, theta_arcsec


def _newton_root(z, w, z1, z2, m1, m2):
    """Polish one complex root of the binary lens equation.

    Parameters
    ----------
    z : complex
        Starting image position, Einstein units.
    w, z1, z2 : complex
        Source, primary, and planet positions, Einstein units.
    m1, m2 : float
        Mass fractions.

    Returns
    -------
    z_out : complex
        Polished position. Not a root if the iteration was singular.
    converged : bool
        True when the lens-equation residual is below ``1e-9``.
    """
    z = complex(z)
    converged = False
    for _step in range(25):
        d1 = z - z1
        d2 = z - z2
        if min(abs(d1), abs(d2)) < 1.0e-18:
            break
        f = z - m1 / np.conj(d1) - m2 / np.conj(d2) - w
        if abs(f) < 1.0e-13:
            converged = True
            break
        fp = m1 / np.conj(d1) ** 2 + m2 / np.conj(d2) ** 2
        denom = 1.0 - abs(fp) ** 2
        # On a fold the Jacobian vanishes and this step is undefined.
        if abs(denom) < 1.0e-22:
            break
        step = -(f - fp * np.conj(f)) / denom
        if abs(step) > 1.0:
            step = step / abs(step)
        z = z + step
        if abs(step) < 1.0e-15:
            f = z - m1 / np.conj(z - z1) - m2 / np.conj(z - z2) - w
            converged = abs(f) < 1.0e-9
            break
    return z, converged


def solve_binary(psbl, times_mjd, m1, m2, theta_arcsec):
    """Point-source binary magnification and centroid.

    Parameters
    ----------
    psbl : bagle model
        Static PSBL model. Supplies positions and the polynomial solver.
    times_mjd : ndarray, shape (N,)
        Evaluation times, MJD.
    m1, m2 : float
        Mass fractions from :func:`mass_fractions`.
    theta_arcsec : float
        Total Einstein radius in arcseconds.

    Returns
    -------
    magnification : ndarray, shape (N,)
        Sum of image magnifications.
    n_images : ndarray, shape (N,)
        Number of roots that satisfy the lens equation.
    centroid_arcsec : ndarray, shape (N, 2)
        Flux-weighted centroid, East then North, arcseconds.
    source_arcsec : ndarray, shape (N, 2)
        Unlensed source, arcseconds.
    primary_arcsec : ndarray, shape (N, 2)
        Black hole, arcseconds.
    secondary_arcsec : ndarray, shape (N, 2)
        Planet, arcseconds.

    Notes
    -----
    Positions and parallax come from BAGLE. Image positions are roots
    of BAGLE's fifth-order polynomial (``get_image_pos_arr``), solved
    in Einstein units with the binary rotated onto the real axis, then
    Newton-polished on the lens equation.

    The default ``get_all_arrays`` path rescales that tolerance by the
    coordinate scale. For this mass ratio the planetary-image residuals
    in the rescaled frame are about ``1e-4``, so ``root_tol = 1e-8``
    throws those images out and the caustic disappears. In Einstein
    units the same polynomial returns residuals below ``1e-6`` inside
    the caustic. Roots are still required to satisfy the lens equation
    to ``1e-9`` after polishing. A missing planetary image far from the
    caustic has negligible flux: the sum then matches the point lens.
    """
    times = np.asarray(times_mjd, dtype=float)
    w_as, z1_as, z2_as = psbl.get_complex_pos(times)
    n_times = len(times)
    magnification = np.empty(n_times, dtype=float)
    n_images = np.empty(n_times, dtype=int)
    centroid = np.empty((n_times, 2), dtype=float)

    for i in range(n_times):
        w = w_as[i] / theta_arcsec
        z1 = z1_as[i] / theta_arcsec
        z2 = z2_as[i] / theta_arcsec
        mid = 0.5 * (z1 + z2)
        axis = z2 - z1
        rot = np.exp(-1j * np.angle(axis))
        w_r = (w - mid) * rot
        z1_r = complex(((z1 - mid) * rot).real)
        z2_r = complex(((z2 - mid) * rot).real)
        raw = psbl.get_image_pos_arr(
            np.array([w_r]),
            np.array([z1_r]),
            np.array([z2_r]),
            m1,
            m2,
            check_sols=False,
        )[0]
        found = []
        for z0 in raw:
            z_pol, ok = _newton_root(z0, w_r, z1_r, z2_r, m1, m2)
            if not ok:
                continue
            if any(abs(z_pol - prev) < 1.0e-8 for prev in found):
                continue
            found.append(z_pol)
        if len(found) == 0:
            magnification[i] = np.nan
            n_images[i] = 0
            centroid[i] = np.nan
            continue
        amps = []
        for z_img in found:
            dw = m1 / (z_img - z1_r) ** 2 + m2 / (z_img - z2_r) ** 2
            amps.append(1.0 / abs(1.0 - abs(dw) ** 2))
        amps = np.asarray(amps, dtype=float)
        weights = np.asarray(found, dtype=np.complex128)
        center_r = np.sum(amps * weights) / np.sum(amps)
        center = center_r / rot + mid
        magnification[i] = float(np.sum(amps))
        n_images[i] = len(found)
        centroid[i, 0] = center.real * theta_arcsec
        centroid[i, 1] = center.imag * theta_arcsec

    source = np.column_stack([w_as.real, w_as.imag])
    primary = np.column_stack([z1_as.real, z1_as.imag])
    secondary = np.column_stack([z2_as.real, z2_as.imag])
    return (
        magnification,
        n_images,
        centroid,
        source,
        primary,
        secondary,
    )


def split_planetary_caustics(points):
    """Split close-topology caustic samples into the two triangles.

    Parameters
    ----------
    points : ndarray, shape (M, 2)
        Caustic coordinates in Einstein units, East then North,
        relative to the center of mass.

    Returns
    -------
    first, second : ndarray
        The two planetary branches. Empty arrays if none are found.
    central : ndarray
        Samples of the central caustic.

    Notes
    -----
    For ``s ~ 0.5`` the planetary caustics sit near ``1/s - s`` and the
    central caustic sits on the center of mass. A cut at 0.2 Einstein
    radii separates them. The two triangles are then the two ends of
    the long axis of the planetary cloud.
    """
    if len(points) == 0:
        empty = np.zeros((0, 2), dtype=float)
        return empty, empty, empty
    radius = np.linalg.norm(points, axis=1)
    central = points[radius <= 0.2]
    far = points[radius > 0.2]
    if len(far) < 10:
        empty = np.zeros((0, 2), dtype=float)
        return empty, empty, central
    center = far.mean(axis=0)
    cov = np.cov((far - center).T)
    _evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(_evals))]
    proj = (far - center) @ axis
    first = far[proj >= 0.0]
    second = far[proj < 0.0]
    return first, second, central


def axis_angle_deg(primary_arcsec, secondary_arcsec):
    """Position angle of the primary as seen from the secondary.

    Parameters
    ----------
    primary_arcsec : ndarray, shape (2,)
        Primary position, East then North, arcseconds.
    secondary_arcsec : ndarray, shape (2,)
        Planet position, same frame.

    Returns
    -------
    angle_deg : float
        Degrees East of North, in ``[0, 360)``.
    """
    delta = np.asarray(primary_arcsec) - np.asarray(secondary_arcsec)
    angle = np.rad2deg(np.arctan2(delta[0], delta[1]))
    return float(np.mod(angle, 360.0))


def angle_between_deg(angle_a, angle_b):
    """Smallest angle between two undirected axes.

    Parameters
    ----------
    angle_a, angle_b : float
        Position angles in degrees.

    Returns
    -------
    separation_deg : float
        Angle in ``[0, 90]``.
    """
    raw = abs((angle_a - angle_b + 180.0) % 360.0 - 180.0)
    if raw > 90.0:
        raw = 180.0 - raw
    return float(raw)


def orient_binary(sep_mas, m_bh, m_planet):
    """Rotate the binary so one planetary caustic has North = ``U0_TARGET``.

    Parameters
    ----------
    sep_mas : float
        Projected separation, milliarcseconds.
    m_bh : float
        Black-hole mass, solar masses.
    m_planet : float
        Planet mass, solar masses.

    Returns
    -------
    alpha_deg : float
        Binary angle, degrees East of North.
    chosen : ndarray, shape (K, 2)
        Caustic samples of the branch the source will cross, in
        Einstein units relative to the center of mass.
    other : ndarray, shape (L, 2)
        The other planetary caustic.
    central : ndarray, shape (M, 2)
        Central caustic.
    theta_e_mas : float
        Einstein radius, milliarcseconds.

    Notes
    -----
    The source moves due East, so the impact parameter is the North
    coordinate of the point it is aimed through. Putting that caustic
    at North = 0.10 Einstein radii gives a primary peak of about 10
    while the along-track offset stays near ``1/s - s``. The caustic
    is placed at positive East, so the crossing comes after the peak.
    """
    trial = build_static(60000.0, 0.1, sep_mas, 90.0, m_bh, m_planet)
    points = v1.caustic_points_theta_e(trial, 60000.0, n_pts=2500)
    first, second, _central = split_planetary_caustics(points)
    if len(first) == 0 or len(second) == 0:
        raise RuntimeError('BAGLE did not return two planetary caustics.')
    # At alpha = 90 the two triangles are North and South of the axis.
    if first[:, 1].mean() >= second[:, 1].mean():
        north_branch = first
    else:
        north_branch = second
    centroid = north_branch.mean(axis=0)
    radius = float(np.linalg.norm(centroid))
    if U0_TARGET >= radius:
        raise RuntimeError('Requested u0 is outside the planetary caustic.')
    target_pa = float(np.arccos(U0_TARGET / radius))
    current_pa = float(np.arctan2(centroid[0], centroid[1]))
    alpha_deg = 90.0 + np.rad2deg(target_pa - current_pa)

    oriented = build_static(
        60000.0, 0.1, sep_mas, alpha_deg, m_bh, m_planet
    )
    points = v1.caustic_points_theta_e(oriented, 60000.0, n_pts=4000)
    first, second, central = split_planetary_caustics(points)
    # The branch nearest the requested North coordinate.
    branches = [first, second]
    north_coord = [float(branch[:, 1].mean()) for branch in branches]
    chosen_i = int(np.argmin(np.abs(np.array(north_coord) - U0_TARGET)))
    chosen = branches[chosen_i]
    other = branches[1 - chosen_i]
    theta_e_mas = float(oriented.thetaE_amp)
    return alpha_deg, chosen, other, central, theta_e_mas


def aim_at_caustic(t_cross, target_east_mas, target_north_mas,
                   sep_mas, alpha_deg, m_bh, m_planet, pi_rel_mas):
    """Place the geocentric source on a sky offset at ``t_cross``.

    Parameters
    ----------
    t_cross : float
        MJD at which the source should sit on the caustic.
    target_east_mas, target_north_mas : float
        Source-minus-COM offset at that time, milliarcseconds.
    sep_mas : float
        Projected separation, milliarcseconds.
    alpha_deg : float
        Binary angle, degrees.
    m_bh, m_planet : float
        Component masses, solar masses.
    pi_rel_mas : float
        Relative parallax, milliarcseconds.

    Returns
    -------
    psbl : bagle model
        Aimed static binary.
    t0 : float
        Barycentric closest-approach time, MJD.
    beta_mas : float
        Barycentric impact parameter, milliarcseconds.
    miss_mas : float
        Residual distance from the target, milliarcseconds.
    """
    t0, beta_mas, _parallax = v1.aim_at_sky_offset(
        t_cross,
        target_east_mas,
        target_north_mas,
        pi_rel_mas,
        mu_east=MU_EAST,
    )
    psbl = None
    miss = np.array([np.nan, np.nan])
    for _step in range(5):
        psbl = build_static(
            t0, beta_mas, sep_mas, alpha_deg, m_bh, m_planet
        )
        evaluated = v1.evaluate_event(psbl, np.array([t_cross]))
        miss = v1.source_minus_com_mas(evaluated, m_bh, m_planet)[0]
        t0 = t0 + (miss[0] - target_east_mas) * v1.DAYS_PER_YEAR / MU_EAST
        beta_mas = beta_mas - (miss[1] - target_north_mas)
    miss_mas = float(np.linalg.norm(
        miss - np.array([target_east_mas, target_north_mas])
    ))
    return psbl, float(t0), float(beta_mas), miss_mas


def fast_season_list(times_mjd):
    """Fast Roman visibility windows, earliest first.

    Parameters
    ----------
    times_mjd : ndarray, shape (N,)
        F146 sample times, MJD.

    Returns
    -------
    seasons : list of tuple
        ``(t_start, t_stop, n_points)`` for each window with more than
        100 samples.
    """
    seasons = []
    for t_start, t_stop, n_points, is_fast in v1.fast_seasons(times_mjd):
        if is_fast:
            seasons.append((float(t_start), float(t_stop), int(n_points)))
    return seasons


def choose_crossing_time(seasons, delay_days):
    """Put the peak and the crossing in two consecutive fast seasons.

    Parameters
    ----------
    seasons : list of tuple
        Fast windows ``(t_start, t_stop, n_points)``.
    delay_days : float
        Time from the geocentric peak to the caustic crossing. Positive
        when the caustic is East of the black hole and the source moves
        East.

    Returns
    -------
    t_cross : float
        Requested crossing time, MJD.
    t_peak_goal : float
        Requested geocentric peak time, MJD.

    Notes
    -----
    The along-track distance to a caustic at ``u_c ~ 1.3`` is about
    ``t_E * 1.3`` (here ~150 days) when ``u0`` is small. One fast season
    is ~72 days and the gap before the next fast season is ~113 days, so
    both events do not fit in one season. They do fit in season 0
    (2027 February–April) and season 1 (2027 August–October) if the peak
    is in the second half of season 0.

    A smaller delay, short enough for one season, would need
    ``u0 >~ 1.1`` and a primary amplitude of only ~0.25 mag. This run
    keeps ``u0 = 0.10`` (amplitude ~10, about 2.5 mag) and uses two
    seasons instead.
    """
    if len(seasons) < 2:
        raise RuntimeError('Need two fast Roman seasons.')
    t_start, t_stop, _n_points = seasons[0]
    # 10 days of margin before the season ends, so the bright core of
    # the peak is inside the fast cadence.
    t_peak_goal = t_stop - 10.0
    t_cross = t_peak_goal + delay_days
    t1_start, t1_stop, _n1 = seasons[1]
    if not (t1_start + 5.0 < t_cross < t1_stop - 5.0):
        raise RuntimeError(
            'Caustic crossing does not land inside the second fast season.'
        )
    if not (t_start + 5.0 < t_peak_goal < t_stop - 5.0):
        raise RuntimeError('Primary peak does not land inside season 0.')
    return float(t_cross), float(t_peak_goal)


def geocentric_peak(psbl, t_guess, m1, m2, theta_arcsec, theta_e_mas):
    """Time and separation of closest geocentric approach to the BH.

    Parameters
    ----------
    psbl : bagle model
        Static binary.
    t_guess : float
        Starting guess, MJD.
    m1, m2 : float
        Mass fractions.
    theta_arcsec : float
        Einstein radius, arcseconds.
    theta_e_mas : float
        Einstein radius, milliarcseconds.

    Returns
    -------
    t_peak : float
        MJD of the minimum source–black-hole separation.
    u_min : float
        That separation in Einstein units.
    magnification : float
        Binary magnification at ``t_peak``.
    """
    grid = t_guess + np.linspace(-8.0, 8.0, 321)
    _mag, _n, _cen, source, primary, _sec = solve_binary(
        psbl, grid, m1, m2, theta_arcsec
    )
    separation = np.linalg.norm(source - primary, axis=1) * 1.0e3
    u_einstein = separation / theta_e_mas
    index = int(np.argmin(u_einstein))
    # Refine to 0.01 day.
    fine = grid[index] + np.linspace(-0.15, 0.15, 61)
    _mag, _n, _cen, source, primary, _sec = solve_binary(
        psbl, fine, m1, m2, theta_arcsec
    )
    separation = np.linalg.norm(source - primary, axis=1) * 1.0e3
    u_einstein = separation / theta_e_mas
    index = int(np.argmin(u_einstein))
    return float(fine[index]), float(u_einstein[index]), float(_mag[index])


def contiguous_inside(times, n_images, t_cross, max_gap_days=30.0 / 86400.0):
    """Duration of the 5-image interval that contains ``t_cross``.

    Parameters
    ----------
    times : ndarray, shape (N,)
        MJD values, not necessarily sorted.
    n_images : ndarray, shape (N,)
        Image counts. Five means the source center is inside a caustic.
    t_cross : float
        Time that should fall inside the interval, MJD.
    max_gap_days : float, optional
        Gaps shorter than this are numerical dropouts (the Jacobian is
        singular on the fold, so a sample can come back with 4 images)
        and do not end the interval.

    Returns
    -------
    t_enter, t_exit : float
        First and last 5-image sample of that interval, MJD.
    """
    times = np.asarray(times, dtype=float)
    order = np.argsort(times)
    times = times[order]
    inside = np.asarray(n_images)[order] >= 5
    indices = np.where(inside)[0]
    if len(indices) == 0:
        raise RuntimeError('Fine grid never entered the planetary caustic.')
    gaps = np.diff(times[indices])
    breaks = np.where(gaps > max_gap_days)[0]
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks, len(indices) - 1]
    for i_start, i_stop in zip(starts, stops):
        t_enter = float(times[indices[i_start]])
        t_exit = float(times[indices[i_stop]])
        if t_enter <= t_cross <= t_exit:
            return t_enter, t_exit
    raise RuntimeError('t_cross is not inside a 5-image interval.')


def plot_photometry(t_data, mag_data, mag_err, t_model, mag_model,
                    mag_point, t_peak, t_cross, t_enter, t_exit,
                    roman_max_mag, outpath):
    """Plot the light curve, both zooms, and the point-lens residual.

    Parameters
    ----------
    t_data : ndarray, shape (N,)
        Roman times, MJD.
    mag_data, mag_err : ndarray, shape (N,)
        Noisy mock magnitudes and their uncertainties.
    t_model : ndarray, shape (M,)
        Model times, MJD.
    mag_model, mag_point : ndarray, shape (M,)
        Point-source binary and point-lens magnitudes.
    t_peak, t_cross : float
        Geocentric peak and caustic-crossing times, MJD.
    t_enter, t_exit : float
        Edges of the 5-image interval, MJD.
    roman_max_mag : float
        Maximum |binary − point lens| on the Roman grid, magnitudes.
    outpath : Path
        PNG destination.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(
        4, 1, figsize=(8.6, 12.2),
        gridspec_kw={'height_ratios': [1.05, 1.0, 1.15, 0.95]},
    )
    ax_full, ax_peak, ax_cau, ax_res = axes
    step = max(1, len(t_data) // 3500)
    ax_full.errorbar(
        t_data[::step], mag_data[::step], yerr=mag_err[::step],
        fmt='.', ms=1.5, color='0.25', ecolor='0.55', elinewidth=0.4,
        alpha=0.4, rasterized=True, zorder=2, label='Roman F146 mock',
    )
    ax_full.plot(
        t_model, mag_model, color='#b2182b', lw=1.15, zorder=3,
        label='BAGLE PSBL (point source)',
    )
    ax_full.plot(
        t_model, mag_point, color='#2166ac', lw=1.0, ls='--', zorder=4,
        label='Point lens (same BH trajectory)',
    )
    ax_full.axvline(t_cross, color='#542788', lw=0.7, ls=':')
    v1.style_mag_axis(ax_full, 19.85, 16.05)
    ax_full.set_xlim(t_data.min() - 20.0, t_data.max() + 20.0)
    ax_full.set_xlabel('Time (MJD)')
    ax_full.set_title(
        r'8 M$_\odot$ BH at 5 kpc + 1 M$_\oplus$ at 6 AU, static binary'
    )
    ax_full.text(
        0.02, 0.05,
        'Caustic spike is narrower than this panel (see third panel).',
        transform=ax_full.transAxes, fontsize=8, va='bottom',
    )
    ax_full.legend(loc='lower right', frameon=False, fontsize=8)

    half = 30.0
    zoom = np.abs(t_model - t_peak) < half
    data_zoom = np.abs(t_data - t_peak) < half
    ax_peak.errorbar(
        t_data[data_zoom], mag_data[data_zoom], yerr=mag_err[data_zoom],
        fmt='.', ms=2.0, color='0.15', ecolor='0.45', elinewidth=0.45,
        alpha=0.55, rasterized=True, zorder=2, label='Roman F146 mock',
    )
    ax_peak.plot(
        t_model[zoom], mag_model[zoom], color='#b2182b', lw=1.4,
        label='PSBL',
    )
    ax_peak.plot(
        t_model[zoom], mag_point[zoom], color='#2166ac', lw=1.1, ls='--',
        label='Point lens',
    )
    v1.style_mag_axis(ax_peak, 19.7, 16.15)
    ax_peak.set_xlim(t_peak - half, t_peak + half)
    ax_peak.set_xlabel('Time (MJD)')
    ax_peak.set_title(
        'Primary peak in the 2027 February–April fast season. '
        'The planet is not in this window.'
    )
    ax_peak.legend(loc='lower right', frameon=False, fontsize=8)

    cau_half = 0.02
    zoom = np.abs(t_model - t_cross) < cau_half
    data_zoom = np.abs(t_data - t_cross) < cau_half
    ax_cau.axvspan(
        t_enter, t_exit, color='#542788', alpha=0.15, zorder=0,
        label='Source center inside caustic',
    )
    ax_cau.errorbar(
        t_data[data_zoom], mag_data[data_zoom], yerr=mag_err[data_zoom],
        fmt='o', ms=3.5, color='0.1', ecolor='0.35', elinewidth=0.6,
        zorder=4, label='Roman F146 mock',
    )
    ax_cau.plot(
        t_model[zoom], mag_model[zoom], color='#b2182b', lw=1.4,
        label='PSBL point source',
    )
    ax_cau.plot(
        t_model[zoom], mag_point[zoom], color='#2166ac', lw=1.1, ls='--',
        label='Point lens',
    )
    v1.style_mag_axis(ax_cau, 19.15, 16.3)
    ax_cau.set_xlim(t_cross - cau_half, t_cross + cau_half)
    ax_cau.set_xlabel('Time (MJD)')
    ax_cau.set_title(
        'Outer-caustic crossing. The shaded interval is the 5-image '
        'region; the fold spikes are unresolved.'
    )
    ax_cau.legend(loc='lower right', frameon=False, fontsize=7.5)

    res_half = 1.0
    res = np.abs(t_model - t_cross) < res_half
    delta = mag_model - mag_point
    ax_res.axvspan(t_enter, t_exit, color='#542788', alpha=0.15, zorder=0)
    ax_res.plot(
        t_model[res], delta[res], color='#b2182b', lw=1.2,
        label='PSBL $-$ point lens',
    )
    ax_res.axhline(0.0, color='0.4', lw=0.6)
    ax_res.axhspan(
        -0.017, 0.017, color='#2166ac', alpha=0.15,
        label=r'$\pm 1\sigma$ at F146 = 19',
    )
    ax_res.set_xlim(t_cross - res_half, t_cross + res_half)
    ax_res.set_ylim(0.15, -1.35)
    ax_res.set_ylabel(r'$\Delta$F146 (mag)')
    ax_res.set_xlabel('Time (MJD)')
    ax_res.set_title(
        'Planet versus a point lens near the crossing. '
        f'Roman-grid max |Δm| = {roman_max_mag:.3f} mag.'
    )
    ax_res.legend(loc='lower right', frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def plot_astrometry(t_model, source_mas, centroid_mas, point_mas,
                    primary_mas, t_data, data_mas, data_err_mas,
                    t_peak, t_cross, outpath):
    """Plot the sky track and the centroid shift, including the crossing.

    Parameters
    ----------
    t_model : ndarray, shape (M,)
        Model times, MJD.
    source_mas, centroid_mas, point_mas, primary_mas : ndarray
        East, North positions in mas relative to the black hole at the
        geocentric peak. Shapes ``(M, 2)``.
    t_data : ndarray, shape (N,)
        Roman times, MJD.
    data_mas : ndarray, shape (N, 2)
        Noisy centroid, same frame.
    data_err_mas : ndarray, shape (N,)
        Astrometric uncertainty per coordinate, mas.
    t_peak, t_cross : float
        Geocentric peak and crossing, MJD.
    outpath : Path
        PNG destination.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.6))
    ax_sky, ax_zoom, ax_full, ax_cau = axes.ravel()
    # The point-source fold makes a few centroid samples numerically
    # useless (the Jacobian vanishes). Keep them out of the survey
    # panels; the zoom still shows the resolved kick.
    survey = np.abs(t_model - t_cross) > 0.02
    sane = np.linalg.norm(centroid_mas - point_mas, axis=1) < 8.0
    step = max(1, len(t_data) // 2500)
    ax_sky.errorbar(
        data_mas[::step, 0], data_mas[::step, 1],
        xerr=data_err_mas[::step], yerr=data_err_mas[::step],
        fmt='.', ms=1.4, color='0.35', ecolor='0.7', elinewidth=0.3,
        alpha=0.3, rasterized=True, zorder=2, label='Roman mock',
    )
    ax_sky.plot(
        source_mas[survey, 0], source_mas[survey, 1], color='0.45', lw=1.0,
        label='Unlensed source',
    )
    ax_sky.plot(
        centroid_mas[survey, 0], centroid_mas[survey, 1], color='#1b9e77',
        lw=1.2, label='Lensed centroid',
    )
    ax_sky.plot(
        primary_mas[survey, 0], primary_mas[survey, 1], color='0.15', lw=0.8,
        label='Black hole (parallax)',
    )
    i_peak = int(np.argmin(np.abs(t_model - t_peak)))
    ax_sky.scatter(
        [primary_mas[i_peak, 0]], [primary_mas[i_peak, 1]],
        marker='*', s=80, color='black', zorder=5, label='BH at peak',
    )
    ax_sky.invert_xaxis()
    ax_sky.set_aspect('equal', adjustable='datalim')
    ax_sky.set_xlabel(r'East offset, $\Delta\alpha\cos\delta$ (mas)')
    ax_sky.set_ylabel(r'North offset, $\Delta\delta$ (mas)')
    ax_sky.set_title('Sky track, relative to the BH at the peak')
    ax_sky.legend(loc='best', frameon=False, fontsize=7)

    window = (np.abs(t_model - t_cross) < 3.0) & sane
    data_window = np.abs(t_data - t_cross) < 3.0
    ax_zoom.plot(
        source_mas[window, 0], source_mas[window, 1],
        color='0.45', lw=1.2, label='Unlensed',
    )
    ax_zoom.plot(
        point_mas[window, 0], point_mas[window, 1],
        color='#2166ac', lw=1.2, ls='--', label='Point-lens centroid',
    )
    ax_zoom.plot(
        centroid_mas[window, 0], centroid_mas[window, 1],
        color='#1b9e77', lw=1.5, label='PSBL centroid',
    )
    ax_zoom.errorbar(
        data_mas[data_window, 0], data_mas[data_window, 1],
        xerr=data_err_mas[data_window], yerr=data_err_mas[data_window],
        fmt='.', ms=3.0, color='0.15', ecolor='0.55', elinewidth=0.5,
        alpha=0.7, zorder=3, label='Roman mock',
    )
    ax_zoom.invert_xaxis()
    ax_zoom.set_aspect('equal', adjustable='datalim')
    ax_zoom.set_xlabel('East offset (mas)')
    ax_zoom.set_ylabel('North offset (mas)')
    ax_zoom.set_title('±3 days around the caustic crossing')
    ax_zoom.legend(loc='best', frameon=False, fontsize=7)

    shift = centroid_mas - source_mas
    shift_point = point_mas - source_mas
    ax_full.plot(
        t_model[survey], shift[survey, 0], color='#1b9e77', lw=1.2,
        label='PSBL East',
    )
    ax_full.plot(
        t_model[survey], shift[survey, 1], color='#1b9e77', lw=1.0, ls=':',
        label='PSBL North',
    )
    ax_full.plot(
        t_model[survey], shift_point[survey, 0], color='#2166ac', lw=1.0,
        ls='--', label='Point lens East',
    )
    ax_full.axvline(t_peak, color='0.5', lw=0.6, ls=':')
    ax_full.axvline(t_cross, color='#542788', lw=0.7, ls=':')
    ax_full.set_xlim(t_data.min() - 20.0, t_data.max() + 20.0)
    ax_full.set_ylim(-1.15, 1.15)
    ax_full.set_xlabel('Time (MJD)')
    ax_full.set_ylabel('Centroid − source (mas)')
    ax_full.set_title('Black-hole astrometric shift (caustic kick omitted)')
    ax_full.legend(loc='best', frameon=False, fontsize=7)

    near = (np.abs(t_model - t_cross) < 1.0) & sane
    ax_cau.plot(
        t_model[near], shift[near, 0], color='#1b9e77', lw=1.3,
        label='PSBL East',
    )
    ax_cau.plot(
        t_model[near], shift[near, 1], color='#e66101', lw=1.2,
        label='PSBL North',
    )
    ax_cau.plot(
        t_model[near], shift_point[near, 0], color='#2166ac', lw=1.0,
        ls='--', label='Point lens East',
    )
    ax_cau.plot(
        t_model[near], shift_point[near, 1], color='#2166ac', lw=1.0,
        ls=':', label='Point lens North',
    )
    ax_cau.axvline(t_cross, color='#542788', lw=0.7, ls=':')
    ax_cau.set_xlim(t_cross - 1.0, t_cross + 1.0)
    ax_cau.set_xlabel('Time (MJD)')
    ax_cau.set_ylabel('Centroid − source (mas)')
    ax_cau.set_title('Planet astrometric kick at the outer caustic')
    ax_cau.legend(loc='best', frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def plot_caustics(critical, chosen, other, central, track, source_cross,
                  rho, planet_theta, half_width_central, outpath):
    """Plot critical curves and the crossed planetary caustic.

    Parameters
    ----------
    critical : ndarray, shape (M, 2)
        Critical curve relative to the black hole, Einstein units.
    chosen, other, central : ndarray
        Caustic samples relative to the center of mass, Einstein units.
        ``chosen`` is the crossed triangle.
    track : ndarray, shape (N, 2)
        Source-center track relative to the center of mass, Einstein
        units.
    source_cross : ndarray, shape (2,)
        Source center at the crossing, same frame.
    rho : float
        Stellar radius in Einstein units.
    planet_theta : ndarray, shape (2,)
        Planet minus black hole, Einstein units.
    half_width_central : float
        Central-caustic cusp radius, Einstein units.
    outpath : Path
        PNG destination.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.0))
    ax_crit, ax_wide, ax_both, ax_zoom = axes.ravel()

    if len(critical) > 0:
        ring = np.linalg.norm(critical, axis=1) > 0.2
        ax_crit.scatter(
            critical[ring, 0], critical[ring, 1], s=2, c='#542788',
            rasterized=True, label='Critical curve',
        )
    phi = np.linspace(0.0, 2.0 * np.pi, 400)
    ax_crit.plot(np.cos(phi), np.sin(phi), color='0.65', lw=0.7, ls=':')
    ax_crit.scatter([0.0], [0.0], marker='*', s=70, c='black', label='BH', zorder=5)
    ax_crit.scatter(
        [planet_theta[0]], [planet_theta[1]], marker='o', s=18,
        c='#e66101', label='Planet', zorder=5,
    )
    ax_crit.set_aspect('equal', adjustable='box')
    ax_crit.set_xlim(1.8, -1.8)
    ax_crit.set_ylim(-1.8, 1.8)
    ax_crit.set_xlabel(r'East / $\theta_E$ (image plane)')
    ax_crit.set_ylabel(r'North / $\theta_E$')
    ax_crit.set_title('Critical curves')
    ax_crit.legend(loc='upper right', frameon=False, fontsize=7)

    ax_wide.plot(
        track[:, 0], track[:, 1], color='#1b9e77', lw=1.3, label='Source center',
        zorder=3,
    )
    ax_wide.scatter(
        [0.0], [0.0], marker='*', s=70, c='black', label='COM / BH', zorder=5,
    )
    if len(chosen):
        ax_wide.scatter(
            chosen[:, 0], chosen[:, 1], s=8, c='#542788', zorder=4,
            label='Crossed caustic',
        )
    if len(other):
        ax_wide.scatter(
            other[:, 0], other[:, 1], s=8, c='#e66101', zorder=4,
            label='Other planetary caustic',
        )
    ax_wide.scatter(
        [source_cross[0]], [source_cross[1]], marker='x', s=40,
        c='#b2182b', zorder=6, label='Source at crossing',
    )
    span = 2.2
    ax_wide.set_aspect('equal', adjustable='box')
    ax_wide.set_xlim(span, -span)
    ax_wide.set_ylim(-span, span)
    ax_wide.set_xlabel(r'East / $\theta_E$ (source plane)')
    ax_wide.set_ylabel(r'North / $\theta_E$')
    ax_wide.set_title(
        f'Close topology. Central caustic radius {half_width_central:.1e}'
        r' $\theta_E$ (not crossed).'
    )
    ax_wide.legend(loc='upper right', frameon=False, fontsize=6.5)

    # Both triangles. The disk is small on this scale but visible.
    mid = source_cross.copy()
    limit = 0.008
    ax_both.plot(
        track[:, 0], track[:, 1], color='#1b9e77', lw=1.2, zorder=2,
        label='Source center',
    )
    if len(chosen):
        ax_both.scatter(
            chosen[:, 0], chosen[:, 1], s=10, c='#542788', zorder=3,
            label='Crossed caustic',
        )
    if len(other):
        ax_both.scatter(
            other[:, 0], other[:, 1], s=10, c='#e66101', zorder=3,
            label='Other triangle',
        )
    disk = Circle(
        (source_cross[0], source_cross[1]), rho,
        facecolor='#fdae61', edgecolor='#e66101', lw=0.8, alpha=0.9,
        zorder=4, label=r'1 $R_\odot$ disk',
    )
    ax_both.add_patch(disk)
    ax_both.set_aspect('equal', adjustable='box')
    ax_both.set_xlim(mid[0] + limit, mid[0] - limit)
    ax_both.set_ylim(mid[1] - limit, mid[1] + limit)
    ax_both.set_xlabel(r'East / $\theta_E$')
    ax_both.set_ylabel(r'North / $\theta_E$')
    ax_both.set_title('Both planetary caustics. The track hits only one.')
    ax_both.legend(loc='upper right', frameon=False, fontsize=6.5)

    # Disk-scale zoom. The triangle sits inside the star.
    local_limit = 2.4 * rho
    ax_zoom.scatter(
        chosen[:, 0], chosen[:, 1], s=12, c='#542788', zorder=3,
        label='Caustic',
    )
    near = np.linalg.norm(track - source_cross, axis=1) < 5.0 * rho
    if np.any(near):
        ax_zoom.plot(
            track[near, 0], track[near, 1], color='#1b9e77', lw=1.4,
            zorder=2, label='Source center',
        )
    disk_zoom = Circle(
        (source_cross[0], source_cross[1]), rho,
        facecolor='#fdae61', edgecolor='#e66101', lw=1.0, alpha=0.55,
        zorder=1, label=r'1 $R_\odot$ at 8 kpc',
    )
    ax_zoom.add_patch(disk_zoom)
    ax_zoom.scatter(
        [source_cross[0]], [source_cross[1]], marker='x', c='#b2182b',
        s=36, zorder=5,
    )
    ax_zoom.set_aspect('equal', adjustable='box')
    ax_zoom.set_xlim(
        source_cross[0] + local_limit, source_cross[0] - local_limit
    )
    ax_zoom.set_ylim(
        source_cross[1] - local_limit, source_cross[1] + local_limit
    )
    ax_zoom.set_xlabel(r'East / $\theta_E$')
    ax_zoom.set_ylabel(r'North / $\theta_E$')
    ax_zoom.set_title(r'Crossed caustic with the stellar disk to scale')
    ax_zoom.legend(loc='upper right', frameon=False, fontsize=7)
    fig.suptitle(
        'Static close binary. The source crosses one outer caustic '
        'after the black-hole peak.',
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def copy_artifacts(outdir):
    """Copy v4 products into the artifact directory with a ``v4_`` prefix.

    Parameters
    ----------
    outdir : Path
        Directory of PNG and table files.

    Returns
    -------
    None

    Notes
    -----
    The prefix keeps the v1 and v2 artifact filenames from being reused.
    """
    if not ARTIFACT_DIR.is_dir():
        return None
    for path in sorted(outdir.iterdir()):
        if path.is_file():
            target = ARTIFACT_DIR / ('v4_' + path.name)
            target.write_bytes(path.read_bytes())
    return None


def main():
    """Generate the static 6 AU Roman mock, figures, and table.

    Returns
    -------
    None
    """
    warnings.filterwarnings('ignore')
    v1.configure_matplotlib()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    m_earth = v1.earth_mass_msun()
    m_bh = 8.0
    m_planet = m_earth
    q_mass = m_planet / m_bh
    # 1 mas at 1 kpc is 1 AU, so sep [mas] = a [AU] / D_L [kpc].
    sep_mas = A_AU / (D_L_PC / 1000.0)

    print('Computing Roman GBTDS F146 times...', flush=True)
    t_f146, t_f087 = v1.fake_data.get_times_roman_gbtds()
    t_f146 = np.asarray(t_f146, dtype=float)
    t_f087 = np.asarray(t_f087, dtype=float)
    seasons = fast_season_list(t_f146)

    print('Orienting the binary toward one outer caustic...', flush=True)
    alpha_deg, chosen, other, central, theta_e_mas = orient_binary(
        sep_mas, m_bh, m_planet
    )
    centroid = chosen.mean(axis=0)
    target_east_mas = float(centroid[0] * theta_e_mas)
    target_north_mas = float(centroid[1] * theta_e_mas)
    delay_days = target_east_mas * v1.DAYS_PER_YEAR / MU_EAST
    t_cross, t_peak_goal = choose_crossing_time(seasons, delay_days)

    # pi_rel does not depend on the impact parameter.
    probe = build_static(60000.0, 0.1, sep_mas, alpha_deg, m_bh, m_planet)
    pi_rel_mas = float(probe.piRel)
    m1, m2, theta_arcsec = mass_fractions(probe)

    print('Aiming the source through the caustic...', flush=True)
    psbl, t0, beta_mas, miss_mas = aim_at_caustic(
        t_cross, target_east_mas, target_north_mas,
        sep_mas, alpha_deg, m_bh, m_planet, pi_rel_mas,
    )
    print(f'Aim residual {miss_mas:.3e} mas')
    if miss_mas > 1.0e-6:
        raise RuntimeError('Source was not placed on the caustic.')

    # Parallax moves the geocentric peak relative to the straight-line
    # estimate. Shift the whole event so the measured peak stays in season.
    t_peak, u_min, a_peak = geocentric_peak(
        psbl, t_peak_goal, m1, m2, theta_arcsec, theta_e_mas
    )
    season0 = seasons[0]
    shift = (season0[1] - 10.0) - t_peak
    if abs(shift) > 0.2:
        t_cross = t_cross + shift
        psbl, t0, beta_mas, miss_mas = aim_at_caustic(
            t_cross, target_east_mas, target_north_mas,
            sep_mas, alpha_deg, m_bh, m_planet, pi_rel_mas,
        )
        t_peak, u_min, a_peak = geocentric_peak(
            psbl, t_peak_goal + shift, m1, m2, theta_arcsec, theta_e_mas
        )
    print(
        f't_peak {t_peak:.3f} u_min {u_min:.4f} A {a_peak:.3f}  '
        f't_cross {t_cross:.3f}'
    )
    if not (season0[0] < t_peak < season0[1]):
        raise RuntimeError('Geocentric peak is outside the first fast season.')
    if not (seasons[1][0] < t_cross < seasons[1][1]):
        raise RuntimeError('Crossing is outside the second fast season.')

    # Refresh caustics on the aimed model (static, so the shape matches).
    cau_pts = v1.caustic_points_theta_e(psbl, t_cross, n_pts=4000)
    chosen, other, central = split_planetary_caustics(cau_pts)
    if chosen[:, 1].mean() < other[:, 1].mean():
        # Keep ``chosen`` as the northern triangle the source was aimed at.
        if abs(chosen[:, 1].mean() - U0_TARGET) > abs(other[:, 1].mean() - U0_TARGET):
            chosen, other = other, chosen
    cau_radius = float(np.linalg.norm(chosen - chosen.mean(0), axis=1).max())
    cau_east = float(chosen[:, 0].max() - chosen[:, 0].min())
    cau_north = float(chosen[:, 1].max() - chosen[:, 1].min())
    if len(central):
        half_central = float(np.linalg.norm(central, axis=1).max())
    else:
        half_central = np.nan

    print('Solving the light curve...', flush=True)
    daily = np.arange(float(t_f146.min()), float(t_f146.max()) + 0.5, 1.0)
    near_peak = np.arange(t_peak - 40.0, t_peak + 40.0, 0.05)
    inner_peak = np.arange(t_peak - 3.0, t_peak + 3.0, 0.01)
    wings = np.arange(t_cross - 2.0, t_cross + 2.0, 0.005)
    fine = np.linspace(t_cross - 0.01, t_cross + 0.01, 2001)
    t_model = np.unique(np.concatenate(
        [daily, near_peak, inner_peak, wings, fine, np.array([t_peak, t_cross])]
    ))
    mag_a, n_model, cen_model, src_model, pri_model, sec_model = solve_binary(
        psbl, t_model, m1, m2, theta_arcsec
    )
    rom_a, n_roman, cen_roman, src_roman, pri_roman, sec_roman = solve_binary(
        psbl, t_f146, m1, m2, theta_arcsec
    )

    u_model = (
        np.linalg.norm(src_model - pri_model, axis=1) * 1.0e3 / theta_e_mas
    )
    u_roman = (
        np.linalg.norm(src_roman - pri_roman, axis=1) * 1.0e3 / theta_e_mas
    )
    a_point_model = v1.point_lens_magnification(u_model)
    a_point_roman = v1.point_lens_magnification(u_roman)
    mag_src = 19.0
    mag_model = mag_src - 2.5 * np.log10(mag_a)
    mag_point_model = mag_src - 2.5 * np.log10(a_point_model)
    mag_roman = mag_src - 2.5 * np.log10(rom_a)
    mag_point_roman = mag_src - 2.5 * np.log10(a_point_roman)

    # Solver checks against the analytic point lens and against BAGLE's
    # default photometry, away from the planetary caustic.
    i_peak = int(np.argmin(np.abs(t_model - t_peak)))
    frac_peak = abs(mag_a[i_peak] - a_point_model[i_peak]) / a_point_model[i_peak]
    far = np.abs(t_model - t_cross) > 5.0
    frac_far = np.nanmax(
        np.abs(mag_a[far] - a_point_model[far]) / a_point_model[far]
    )
    bagle_check_t = np.array([t_peak, t_cross - 10.0, t_cross])
    bagle_eval = v1.evaluate_event(psbl, bagle_check_t)
    solved_check, n_check, *_rest = solve_binary(
        psbl, bagle_check_t, m1, m2, theta_arcsec
    )
    print(
        'fractional |A-Apoint| at peak', f'{frac_peak:.3e}',
        'far', f'{frac_far:.3e}',
    )
    print(
        'BAGLE default vs solver',
        bagle_eval['magnification'] - solved_check,
        'n_solver', n_check,
        'n_bagle', bagle_eval['n_images'],
    )
    if frac_peak > 1.0e-4:
        raise RuntimeError('Peak magnification disagrees with the point lens.')
    i_cross = int(np.argmin(np.abs(t_model - t_cross)))
    if n_model[i_cross] < 5:
        raise RuntimeError('Source center is not inside the planetary caustic.')

    t_enter, t_exit = contiguous_inside(t_model, n_model, t_cross)
    # The fine grid is the part with sub-minute spacing.
    fine_mask = np.abs(t_model - t_cross) <= 0.01
    duration_days = t_exit - t_enter
    in_caustic = (t_f146 >= t_enter) & (t_f146 <= t_exit)

    delta_mag = mag_roman - mag_point_roman
    mag_noisy, mag_err = v1.add_roman_photometric_noise(mag_roman, rng_seed=146)
    _formula_err, ast_err_mas = v1.roman_f146_uncertainties(mag_roman)
    chi2_phot = float(np.nansum((delta_mag / mag_err) ** 2))
    roman_max = float(np.nanmax(np.abs(delta_mag)))
    fine_delta = mag_model[fine_mask] - mag_point_model[fine_mask]
    fine_max = float(np.nanmax(np.abs(fine_delta)))
    # Interior plateau: 5-image samples, excluding the outer 15% of the
    # interval where the folds dominate the point-source spike.
    span = t_exit - t_enter
    interior = (
        (n_model >= 5)
        & (t_model > t_enter + 0.15 * span)
        & (t_model < t_exit - 0.15 * span)
    )
    if np.any(interior):
        interior_dm = float(np.median(
            mag_model[interior] - mag_point_model[interior]
        ))
    else:
        interior_dm = float(mag_model[i_cross] - mag_point_model[i_cross])

    wing = np.abs(delta_mag) > 0.01
    strong = np.abs(delta_mag) > 0.1
    # Demagnification: binary fainter than the point lens.
    demag = delta_mag > 0.0
    max_demag = float(np.nanmax(delta_mag)) if np.any(demag) else 0.0

    point_cen_model = v1.point_lens_centroid(src_model, pri_model, theta_e_mas)
    point_cen_roman = v1.point_lens_centroid(src_roman, pri_roman, theta_e_mas)
    dpos_roman = (cen_roman - point_cen_roman) * 1.0e3
    dpos_norm = np.linalg.norm(dpos_roman, axis=1)
    ast_max = float(np.nanmax(dpos_norm))
    chi2_ast = float(np.nansum((dpos_roman / ast_err_mas[:, None]) ** 2))
    dpos_fine = np.linalg.norm(
        (cen_model[fine_mask] - point_cen_model[fine_mask]) * 1.0e3, axis=1
    )
    ast_fine_max = float(np.nanmax(dpos_fine))

    rng = np.random.default_rng(146)
    data_centroid = cen_roman + rng.normal(
        scale=ast_err_mas[:, None], size=cen_roman.shape
    ) * 1.0e-3

    i_origin = int(np.argmin(np.abs(t_model - t_peak)))
    origin = pri_model[i_origin]

    def to_mas(arcsec):
        return (arcsec - origin) * 1.0e3

    # Drop non-finite samples before plotting.
    good_model = np.isfinite(mag_model) & np.isfinite(cen_model[:, 0])
    t_plot = t_model[good_model]
    mag_plot = mag_model[good_model]
    mag_point_plot = mag_point_model[good_model]
    src_mas = to_mas(src_model[good_model])
    cen_mas = to_mas(cen_model[good_model])
    point_mas = to_mas(point_cen_model[good_model])
    pri_mas = to_mas(pri_model[good_model])
    data_mas = to_mas(data_centroid)

    # Track relative to the COM for the caustic figure, Einstein units.
    com_track = v1.source_minus_com_mas(
        {
            'source_arcsec': src_model[good_model],
            'primary_arcsec': pri_model[good_model],
            'secondary_arcsec': sec_model[good_model],
        },
        m_bh,
        m_planet,
    ) / theta_e_mas
    # Only the approach from the peak through the crossing, plus a margin.
    track_window = (t_plot > t_peak - 20.0) & (t_plot < t_cross + 5.0)
    cross_eval = v1.evaluate_event(psbl, np.array([t_cross]))
    source_cross = v1.source_minus_com_mas(cross_eval, m_bh, m_planet)[0]
    source_cross = source_cross / theta_e_mas
    alpha_cross = axis_angle_deg(
        cross_eval['primary_arcsec'][0],
        cross_eval['secondary_arcsec'][0],
    )
    angle_to_axis = angle_between_deg(90.0, alpha_cross)
    planet_theta = (
        cross_eval['secondary_arcsec'][0] - cross_eval['primary_arcsec'][0]
    ) * 1.0e3 / theta_e_mas

    theta_star_mas = v1.source_angular_radius_mas(D_S_PC, 1.0)
    rho = theta_star_mas / theta_e_mas
    r_e_au = theta_e_mas * (D_L_PC / 1000.0)
    s_einstein = A_AU / r_e_au
    t_e_days = float(psbl.tE)
    pi_e = float(psbl.piE_amp)
    pi_l = float(psbl.piL)
    pi_s = float(psbl.piS)
    u0_bary = float(psbl.u0_amp)
    width_analytic = v1.central_caustic_width_theta_e(q_mass, s_einstein)
    mu_theta_per_day = (MU_EAST / v1.DAYS_PER_YEAR) / theta_e_mas
    # Time for a 1 Rsun disk to sweep its diameter across a fixed caustic.
    fs_duration_days = (2.0 * rho) / mu_theta_per_day
    shift_bh = np.linalg.norm(cen_mas - src_mas, axis=1)
    # Peak BH shift, away from the planetary spike.
    away = np.abs(t_plot - t_cross) > 1.0
    max_shift_mas = float(np.nanmax(shift_bh[away]))

    in_fast = (t_f146 >= seasons[1][0]) & (t_f146 <= seasons[1][1])
    fast_dt_min = float(np.median(np.diff(np.sort(t_f146[in_fast]))) * 1440.0)
    _err19, ast19 = v1.roman_f146_uncertainties(np.array([19.0]))
    _noisy19, mag_err_19 = v1.add_roman_photometric_noise(
        np.array([19.0, 19.0]), rng_seed=1
    )
    mag_err_19_value = float(np.median(mag_err_19))

    # BAGLE's default photometry at the aim epoch, for the numerical note.
    bagle_on = float(bagle_eval['magnification'][2])
    solver_on = float(solved_check[2])

    print('Writing figures...', flush=True)
    plot_photometry(
        t_f146, mag_noisy, mag_err, t_plot, mag_plot, mag_point_plot,
        t_peak, t_cross, t_enter, t_exit, roman_max,
        OUTDIR / 'fig_photometry_f146.png',
    )
    plot_astrometry(
        t_plot, src_mas, cen_mas, point_mas, pri_mas,
        t_f146, data_mas, ast_err_mas, t_peak, t_cross,
        OUTDIR / 'fig_astrometry.png',
    )
    critical = v1.critical_curve_theta_e(psbl, t_cross, n_pts=800)
    plot_caustics(
        critical, chosen, other, central,
        com_track[track_window], source_cross, rho, planet_theta,
        half_central, OUTDIR / 'fig_caustic.png',
    )

    t_cross_iso = Time(t_cross, format='mjd').isot
    t_peak_iso = Time(t_peak, format='mjd').isot
    t0_iso = Time(t0, format='mjd').isot
    n_bad = int(np.sum(~np.isfinite(mag_a)))
    rows = [
        ('Model class', 'PSBL_PhotAstrom_Par_Param1'),
        ('Orbit', 'none (static projected separation)'),
        ('Image solver', 'BAGLE 5th-order poly, Einstein units'),
        ('Why not get_all_arrays', 'root_tol drops planetary images'),
        ('Finite-source binary', 'unavailable (FSBL commented out)'),
        ('Source treatment', 'point source; disk drawn for scale'),
        ('Cadence helper', 'fake_data.get_times_roman_gbtds'),
        ('Photometric noise', 'fake_data.add_photometric_noise, zp=28'),
        ('Astrometric error', 'FWHM/(2 SNR), floor 0.1 mas; no 1e-5'),
        ('M_BH (Msun)', f'{m_bh:.1f}'),
        ('M_planet (Msun)', f'{m_planet:.6e}'),
        ('M_planet', '1 M_earth'),
        ('q = M_planet/M_BH', f'{q_mass:.6e}'),
        ('D_L (kpc)', f'{D_L_PC / 1000.0:.1f}'),
        ('D_S (kpc)', f'{D_S_PC / 1000.0:.1f}'),
        ('a_proj (AU)', f'{A_AU:.1f} (face-on snapshot)'),
        ('sep (mas)', f'{sep_mas:.6f}'),
        ('s = a / r_E', f'{s_einstein:.6f}'),
        ('topology', 'close (s < 1), two planetary caustics'),
        ('1/s - s', f'{1.0 / s_einstein - s_einstein:.6f}'),
        ('alpha (deg E of N)', f'{alpha_cross:.4f}'),
        ('mu direction (deg E of N)', '90 (due East)'),
        ('angle(trajectory, axis)', f'{angle_to_axis:.4f} deg'),
        ('b_sff', '1 (no blend, no lens flux)'),
        ('dmag_Lp_Ls', '0 (both components dark)'),
        ('mag_src F146W', '19'),
        ('mu_rel (mas/yr)', f'{MU_EAST:.1f} due East'),
        ('t0 barycentric (MJD)', f'{t0:.5f} ({t0_iso})'),
        ('geocentric peak (MJD)', f'{t_peak:.5f} ({t_peak_iso})'),
        ('peak season', '1st fast window, 2027 Feb 9 to Apr 22'),
        ('crossing (MJD)', f'{t_cross:.5f} ({t_cross_iso})'),
        ('crossing season', '2nd fast window, 2027 Aug 13 to Oct 26'),
        ('peak-to-crossing (days)', f'{t_cross - t_peak:.3f}'),
        ('beta (mas)', f'{beta_mas:.6e}'),
        ('u0 barycentric', f'{u0_bary:.6e}'),
        ('u at geocentric peak', f'{u_min:.6f}'),
        ('A at geocentric peak', f'{a_peak:.4f}'),
        ('F146 at geocentric peak', f'{mag_src - 2.5 * np.log10(a_peak):.3f}'),
        ('aim residual (mas)', f'{miss_mas:.3e}'),
        ('within 0.1 mas?', 'no; outer caustic is ~3 mas from the BH'),
        ('theta_E (mas)', f'{theta_e_mas:.6f}'),
        ('r_E at lens (AU)', f'{r_e_au:.4f}'),
        ('t_E (days)', f'{t_e_days:.4f}'),
        ('pi_L (mas)', f'{pi_l:.6f}'),
        ('pi_S (mas)', f'{pi_s:.6f}'),
        ('pi_rel (mas)', f'{pi_rel_mas:.6f}'),
        ('pi_E', f'{pi_e:.6f}'),
        ('R_source', '1 R_sun (not in the PSBL integral)'),
        ('theta_source (mas)', f'{theta_star_mas:.6e}'),
        ('rho = theta_*/theta_E', f'{rho:.6e}'),
        ('crossed caustic radius (theta_E)', f'{cau_radius:.6e}'),
        ('caustic East span (theta_E)', f'{cau_east:.6e}'),
        ('caustic North span (theta_E)', f'{cau_north:.6e}'),
        ('rho / caustic radius', f'{rho / cau_radius:.3f}'),
        ('central caustic radius (theta_E)', f'{half_central:.6e}'),
        ('central width analytic (theta_E)', f'{width_analytic:.6e}'),
        ('5-image duration (seconds)', f'{duration_days * 86400.0:.2f}'),
        ('1 Rsun sweep time (minutes)', f'{fs_duration_days * 1440.0:.2f}'),
        ('images at the aim epoch', f'{int(n_model[i_cross])}'),
        ('interior median dm (mag)', f'{interior_dm:.4f}'),
        ('fine-grid max |dm| (mag)', f'{fine_max:.4f}'),
        ('fine-grid max |dpos| (mas)', f'{ast_fine_max:.4f}'),
        ('N F146 inside the 5-image window', f'{int(np.sum(in_caustic))}'),
        ('N F146 with |dm| > 0.01 mag', f'{int(np.sum(wing))}'),
        ('N F146 with |dm| > 0.1 mag', f'{int(np.sum(strong))}'),
        ('max |dm| on Roman grid (mag)', f'{roman_max:.6f}'),
        ('chi2 of that residual', f'{chi2_phot:.4f}'),
        ('max demagnification (mag)', f'{max_demag:.6f}'),
        ('max |dpos| on Roman grid (mas)', f'{ast_max:.6f}'),
        ('chi2 ast of that residual', f'{chi2_ast:.4f}'),
        ('peak BH centroid shift (mas)', f'{max_shift_mas:.4f}'),
        ('BAGLE default A at crossing', f'{bagle_on:.6f}'),
        ('solver A at crossing', f'{solver_on:.6f}'),
        ('non-finite model samples', f'{n_bad}'),
        ('N F146 epochs', f'{len(t_f146)}'),
        ('N F087 epochs (not simulated)', f'{len(t_f087)}'),
        ('fast-cadence median (minutes)', f'{fast_dt_min:.3f}'),
        ('F146 mag err at mag 19', f'{mag_err_19_value:.5f}'),
        ('F146 ast err at mag 19 (mas)', f'{float(ast19[0]):.4f}'),
        ('|A-Apoint|/A at the peak', f'{frac_peak:.3e}'),
        ('|A-Apoint|/A, >5 d from caustic', f'{frac_far:.3e}'),
        ('Trade-off', 'u0=0.10; peak and crossing in consecutive seasons'),
        ('Trough', 'trajectory crosses one triangle, not the gap'),
        ('FSBL re-enable', 'uncomment, match PSBL API, integrate the disk'),
        ('Notes', '1e-5 factor in test_roman_lightcurve was not applied.'),
        ('Notes 2', 'chi2 is the residual against Roman errors, not a refit.'),
        ('Notes 3', 'Fine-grid max |dm| samples the point-source fold.'),
    ]
    v1.write_tables(rows, OUTDIR)
    fig_h = 0.175 * len(rows) + 0.9
    fig, axis = plt.subplots(figsize=(9.2, fig_h))
    axis.axis('off')
    lines = [f'{label:<44s} {value}' for label, value in rows]
    axis.text(
        0.01, 0.995, '\n'.join(lines), va='top', ha='left',
        family='monospace', fontsize=7.2, linespacing=1.15,
        transform=axis.transAxes, clip_on=False,
    )
    axis.set_title('Input and derived parameters', loc='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig_parameter_table.png')
    plt.close(fig)
    copy_artifacts(OUTDIR)

    print('duration_s', duration_days * 86400.0)
    print('roman inside', int(np.sum(in_caustic)), 'wing>0.01', int(np.sum(wing)))
    print('max |dm|', roman_max, 'chi2', chi2_phot)
    print('max |dpos|', ast_max, 'chi2_ast', chi2_ast)
    print('interior dm', interior_dm, 'fine max', fine_max)
    print('alpha', alpha_cross, 'angle', angle_to_axis, 'u_min', u_min)
    print('Wrote', OUTDIR)
    return None


if __name__ == '__main__':
    main()
