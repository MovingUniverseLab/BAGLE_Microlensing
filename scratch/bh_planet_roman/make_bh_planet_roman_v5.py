"""Roman GBTDS mock: static 8 solar-mass black hole plus an Earth at 10 AU.

This is the fifth demonstration for Prof. Jessica Lu. It does not modify
BAGLE. The lens is an 8 solar-mass black hole at 5 kpc. A 1 Earth-mass
planet has a sky-projected separation of 10 AU. Both components are dark.
There is no orbital motion: the binary is the static point-source model
``PSBL_PhotAstrom_Par_Param1`` with parallax.

The source is at 8 kpc with F146W = 19. Its trajectory is due East at
7 mas/yr and is aimed through one triangular planetary caustic, with a
barycentric impact parameter near 0.10 Einstein radii so the black-hole
peak is obvious. At this separation ``s ~ 0.90``, so the caustic sits
near ``|s - 1/s| ~ 0.20`` and the peak and the crossing fit in one Roman
fast season.

BAGLE has no working finite-source binary class (``FSBL`` is commented
out; ``FSPL`` is a single lens). The curves are point-source. A uniform
1 solar-radius disk average is computed in this script only, on a short
grid around the crossing, to estimate how much the disk would smooth the
point-source spike.

Run from the repository root with::

    PYTHONPATH=src python scratch/bh_planet_roman/make_bh_planet_roman_v5.py
"""

import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_bh_planet_roman_demo as v1


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'output' / 'v5_static_10AU_5kpc'
ARTIFACT_DIR = Path('/opt/cursor/artifacts')

# Due East, same bulge-like relative motion as the earlier demos.
MU_EAST = 7.0
MU_NORTH = 0.0
# North coordinate of the crossed caustic, in Einstein units. This is
# the barycentric impact parameter of a due-East trajectory.
U0_TARGET = 0.10
D_L_PC = 5000.0
D_S_PC = 8000.0
A_AU = 10.0
# Days after the fast season opens to place the geocentric peak. The
# crossing follows by ~20 days, and the season is ~72 days long.
LEAD_IN_DAYS = 24.0
# v4 (6 AU, s = 0.54) triangle radius, Einstein units. Used only to
# report how much larger the 10 AU caustic is. Not recomputed here.
V4_CAUSTIC_RADIUS = 8.584340e-05
# Central caustic vs planetary triangles. The triangles sit at
# |s - 1/s| ~ 0.20, so a cut of 0.2 (the v4 value) would swallow them.
CENTRAL_CUT = 0.02
CROSS_COLOR = '#c51b7d'
PSBL_COLOR = '#b2182b'
POINT_COLOR = '#2166ac'
TRACK_COLOR = '#1b9e77'


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
    The projected separation is an input, not an orbit.
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


def _solve_einstein(psbl, w, z1, z2, m1, m2):
    """Images of one source position in Einstein units.

    Parameters
    ----------
    psbl : bagle model
        Supplies ``get_image_pos_arr``. Positions themselves come from
        the caller.
    w, z1, z2 : complex
        Source, primary, and planet, Einstein units, unrotated.
    m1, m2 : float
        Mass fractions.

    Returns
    -------
    magnification : float
        Sum of image magnifications. NaN if no root survives.
    n_images : int
        Number of accepted roots.
    centroid : complex
        Flux-weighted centroid, Einstein units, unrotated. NaN if no
        root survives.

    Notes
    -----
    The fifth-order polynomial is solved with the binary rotated onto
    the real axis, then each root is Newton-polished on the lens
    equation and kept only if the residual is below ``1e-9``. This is
    the same path as v4. ``get_all_arrays`` is not used: it rescales
    ``root_tol`` and drops the planetary images.
    """
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
        return np.nan, 0, np.nan + 0j

    amps = []
    for z_img in found:
        dw = m1 / (z_img - z1_r) ** 2 + m2 / (z_img - z2_r) ** 2
        amps.append(1.0 / abs(1.0 - abs(dw) ** 2))
    amps = np.asarray(amps, dtype=float)
    weights = np.asarray(found, dtype=np.complex128)
    center_r = np.sum(amps * weights) / np.sum(amps)
    center = center_r / rot + mid
    return float(np.sum(amps)), len(found), center


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
    """
    times = np.asarray(times_mjd, dtype=float)
    w_as, z1_as, z2_as = psbl.get_complex_pos(times)
    n_times = len(times)
    magnification = np.empty(n_times, dtype=float)
    n_images = np.empty(n_times, dtype=int)
    centroid = np.empty((n_times, 2), dtype=float)

    for i in range(n_times):
        amp, n_img, center = _solve_einstein(
            psbl, w_as[i] / theta_arcsec, z1_as[i] / theta_arcsec,
            z2_as[i] / theta_arcsec, m1, m2,
        )
        magnification[i] = amp
        n_images[i] = n_img
        if n_img == 0:
            centroid[i] = np.nan
            continue
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


def disk_average_magnification(psbl, w, z1, z2, m1, m2, rho,
                               n_rad=5, n_phi=12):
    """Uniform-disk average of the point-source magnification.

    Parameters
    ----------
    psbl : bagle model
        Supplies the polynomial solver.
    w, z1, z2 : complex
        Source center, primary, and planet, Einstein units.
    m1, m2 : float
        Mass fractions.
    rho : float
        Stellar angular radius in Einstein units.
    n_rad : int, optional
        Number of equal-area radial rings.
    n_phi : int, optional
        Azimuth samples on each ring.

    Returns
    -------
    magnification : float
        Area-weighted mean magnification. NaN if every sample failed.
    n_ok : int
        Samples that returned a finite magnification.
    n_try : int
        Samples requested.

    Notes
    -----
    Rings are spaced so each has the same area, and each ring has the
    same number of azimuth samples, so every sample has equal weight.
    This is a scratch-script estimate. It is not a BAGLE finite-source
    model, and it has no limb darkening.
    """
    total = 0.0
    n_ok = 0
    n_try = int(n_rad * n_phi)
    for k in range(n_rad):
        radius = rho * np.sqrt((k + 0.5) / float(n_rad))
        for j in range(n_phi):
            phi = 2.0 * np.pi * (j + 0.5) / float(n_phi)
            sample = w + radius * np.exp(1j * phi)
            amp, _n_img, _center = _solve_einstein(
                psbl, sample, z1, z2, m1, m2,
            )
            if np.isfinite(amp):
                total += amp
                n_ok += 1
    if n_ok == 0:
        return np.nan, 0, n_try
    return total / float(n_ok), n_ok, n_try


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
    For ``s ~ 0.90`` the planetary caustics sit near ``|s - 1/s| ~ 0.20``
    and the central caustic sits on the center of mass (radius
    ``~ 4e-5``). The cut is ``CENTRAL_CUT = 0.02``, not the v4 value of
    0.2, which would classify the triangles as central. The two
    triangles are the two ends of the long axis of the planetary cloud.
    """
    if len(points) == 0:
        empty = np.zeros((0, 2), dtype=float)
        return empty, empty, empty
    radius = np.linalg.norm(points, axis=1)
    central = points[radius <= CENTRAL_CUT]
    far = points[radius > CENTRAL_CUT]
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


def _pick_branch(first, second):
    """Return the branch whose North coordinate is closer to ``U0_TARGET``.

    Parameters
    ----------
    first, second : ndarray
        Planetary-caustic samples, East then North.

    Returns
    -------
    chosen, other : ndarray
        The crossed branch and the remaining branch.
    """
    north = np.array([
        float(first[:, 1].mean()),
        float(second[:, 1].mean()),
    ])
    chosen_i = int(np.argmin(np.abs(north - U0_TARGET)))
    branches = (first, second)
    chosen = branches[chosen_i]
    other = branches[1 - chosen_i]
    return chosen, other


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
    coordinate of the aim point. The caustic is placed at positive East,
    so the crossing comes after the peak. The along-track offset is
    ``sqrt(R^2 - u0^2)``, with ``R ~ |s - 1/s|``.
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
    chosen, other = _pick_branch(first, second)
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


def choose_crossing_time(seasons, delay_days, lead_days):
    """Put the peak and the crossing in the same fast season.

    Parameters
    ----------
    seasons : list of tuple
        Fast windows ``(t_start, t_stop, n_points)``.
    delay_days : float
        Time from the geocentric peak to the caustic crossing. Positive
        when the caustic is East of the black hole and the source moves
        East.
    lead_days : float
        Days after the season opens to place the peak.

    Returns
    -------
    t_cross : float
        Requested crossing time, MJD.
    t_peak_goal : float
        Requested geocentric peak time, MJD.

    Notes
    -----
    At ``s ~ 0.90`` and ``u0 = 0.10`` the along-track delay is about
    20 days. One fast season is about 72 days, so both events fit if
    the peak is placed early in the window rather than at its end.
    """
    if len(seasons) < 1:
        raise RuntimeError('Need a fast Roman season.')
    t_start, t_stop, _n_points = seasons[0]
    t_peak_goal = t_start + lead_days
    t_cross = t_peak_goal + delay_days
    if not (t_start + 5.0 < t_peak_goal < t_stop - 5.0):
        raise RuntimeError('Primary peak does not land inside season 0.')
    if not (t_start + 5.0 < t_cross < t_stop - 5.0):
        raise RuntimeError(
            'Caustic crossing does not land in the same fast season.'
        )
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
    # Refine to about 0.005 day.
    fine = grid[index] + np.linspace(-0.15, 0.15, 61)
    mag, _n, _cen, source, primary, _sec = solve_binary(
        psbl, fine, m1, m2, theta_arcsec
    )
    separation = np.linalg.norm(source - primary, axis=1) * 1.0e3
    u_einstein = separation / theta_e_mas
    index = int(np.argmin(u_einstein))
    return float(fine[index]), float(u_einstein[index]), float(mag[index])


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


def roman_phase_shift(t_enter, t_exit, times_mjd):
    """Shift that places a Roman epoch at the center of the 5-image window.

    Parameters
    ----------
    t_enter, t_exit : float
        Edges of the 5-image interval, MJD.
    times_mjd : ndarray
        Roman sample times, MJD.

    Returns
    -------
    shift_days : float
        Added to the event time. Zero when an epoch is already inside.
        The Roman times stay fixed; the event moves.

    Notes
    -----
    The fast cadence is 12.8 minutes. A window shorter than that can
    fall between samples. The shift is at most half a cadence, and it
    is refused if the nearest epoch is more than 0.01 day away.
    """
    times = np.asarray(times_mjd, dtype=float)
    inside = (times >= t_enter) & (times <= t_exit)
    if np.any(inside):
        return 0.0
    mid = 0.5 * (t_enter + t_exit)
    nearest = float(times[int(np.argmin(np.abs(times - mid)))])
    shift = float(nearest - mid)
    if abs(shift) > 0.01:
        raise RuntimeError(
            'No Roman epoch within 0.01 day of the caustic window.'
        )
    return shift


def _mag_span(values, low_q, high_q, pad_faint, pad_bright):
    """Faint and bright magnitude limits from percentiles.

    Parameters
    ----------
    values : ndarray
        Magnitudes.
    low_q, high_q : float
        Percentiles. The low percentile is the bright end.
    pad_faint, pad_bright : float
        Padding in magnitudes.

    Returns
    -------
    faint, bright : float
        Limits for :func:`v1.style_mag_axis`.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    bright = float(np.percentile(finite, low_q) - pad_bright)
    faint = float(np.percentile(finite, high_q) + pad_faint)
    if faint < bright + 0.35:
        faint = bright + 0.35
    return faint, bright


def plot_photometry(t_data, mag_data, mag_err, t_model, mag_model,
                    mag_point, t_peak, t_cross, t_enter, t_exit,
                    t_disk, mag_disk, n_inside, n_wing, season_label,
                    outpath):
    """Plot the light curve with the caustic crossing highlighted.

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
    t_disk, mag_disk : ndarray
        Times and uniform-disk magnitudes around the crossing.
    n_inside : int
        Roman epochs with the source center inside the caustic.
    n_wing : int
        Roman epochs outside that window with ``|dm| > 0.01``.
    season_label : str
        Human-readable fast-season range.
    outpath : Path
        PNG destination.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(
        4, 1, figsize=(8.6, 12.6),
        gridspec_kw={'height_ratios': [1.15, 1.05, 1.2, 0.95]},
    )
    ax_full, ax_peak, ax_cau, ax_res = axes
    step = max(1, len(t_data) // 3500)
    duration = max(float(t_exit - t_enter), 1.0 / 86400.0)
    # Wide enough that the shaded 5-image interval is a visible band,
    # and tight enough that individual Roman epochs can be seen.
    cau_half = max(0.10, 2.4 * duration)

    ax_full.axvspan(
        t_enter, t_exit, color=CROSS_COLOR, alpha=0.55, zorder=1,
        label='Caustic crossing',
    )
    ax_full.errorbar(
        t_data[::step], mag_data[::step], yerr=mag_err[::step],
        fmt='.', ms=1.5, color='0.25', ecolor='0.55', elinewidth=0.4,
        alpha=0.35, rasterized=True, zorder=2, label='Roman F146 mock',
    )
    ax_full.plot(
        t_model, mag_model, color=PSBL_COLOR, lw=1.15, zorder=3,
        label='BAGLE PSBL (point source)',
    )
    ax_full.plot(
        t_model, mag_point, color=POINT_COLOR, lw=1.0, ls='--', zorder=4,
        label='Point lens (same BH trajectory)',
    )
    v1.style_mag_axis(ax_full, 19.85, 16.05)
    ax_full.set_xlim(t_data.min() - 20.0, t_data.max() + 20.0)
    ax_full.set_xlabel('Time (MJD)')
    ax_full.set_title(
        r'8 M$_\odot$ BH at 5 kpc + 1 M$_\oplus$ at 10 AU, static binary'
    )
    ax_full.annotate(
        'caustic crossing (see inset)',
        xy=(t_cross, 16.85),
        xytext=(t_cross + 420.0, 16.40),
        color=CROSS_COLOR, fontsize=8,
        arrowprops=dict(arrowstyle='->', color=CROSS_COLOR, lw=0.8),
        zorder=6,
    )
    ax_full.legend(loc='lower right', frameon=False, fontsize=7.5)

    # The 5-image window is hours long, so on a five-year axis the
    # shade is thinner than a pixel. The inset is the readable copy.
    inset = ax_full.inset_axes([0.55, 0.38, 0.42, 0.52])
    zoom = np.abs(t_model - t_cross) < cau_half
    data_zoom = np.abs(t_data - t_cross) < cau_half
    inset.axvspan(t_enter, t_exit, color=CROSS_COLOR, alpha=0.22, zorder=0)
    inset.plot(
        t_model[zoom], mag_point[zoom], color=POINT_COLOR, lw=1.0, ls='--',
    )
    inset.plot(t_model[zoom], mag_model[zoom], color=PSBL_COLOR, lw=1.3)
    if len(t_disk):
        inset.plot(t_disk, mag_disk, color='0.15', lw=1.15, ls='-.')
    inset.errorbar(
        t_data[data_zoom], mag_data[data_zoom], yerr=mag_err[data_zoom],
        fmt='o', ms=2.2, color='0.15', ecolor='0.45', elinewidth=0.4,
        zorder=4,
    )
    faint, bright = _mag_span(mag_model[zoom], 3.0, 97.0, 0.12, 0.2)
    v1.style_mag_axis(inset, faint, bright)
    inset.set_xlim(t_cross - cau_half, t_cross + cau_half)
    inset.tick_params(labelsize=7)
    inset.set_title('Caustic crossing', fontsize=8)
    inset.set_xlabel('')

    half = 40.0
    zoom = np.abs(t_model - t_peak) < half
    data_zoom = np.abs(t_data - t_peak) < half
    ax_peak.axvspan(
        t_enter, t_exit, color=CROSS_COLOR, alpha=0.35, zorder=0,
        label='Caustic crossing',
    )
    ax_peak.errorbar(
        t_data[data_zoom], mag_data[data_zoom], yerr=mag_err[data_zoom],
        fmt='.', ms=2.0, color='0.15', ecolor='0.45', elinewidth=0.45,
        alpha=0.45, rasterized=True, zorder=2, label='Roman F146 mock',
    )
    ax_peak.plot(
        t_model[zoom], mag_model[zoom], color=PSBL_COLOR, lw=1.4,
        label='PSBL', zorder=3,
    )
    ax_peak.plot(
        t_model[zoom], mag_point[zoom], color=POINT_COLOR, lw=1.1, ls='--',
        label='Point lens', zorder=4,
    )
    v1.style_mag_axis(ax_peak, 19.55, 15.4)
    ax_peak.set_xlim(t_peak - half, t_peak + half)
    ax_peak.set_xlabel('Time (MJD)')
    ax_peak.set_title(
        f'{season_label}: black-hole peak and the planetary crossing'
    )
    ax_peak.legend(loc='lower right', frameon=False, fontsize=8)

    zoom = np.abs(t_model - t_cross) < cau_half
    data_zoom = np.abs(t_data - t_cross) < cau_half
    inside_epoch = data_zoom & (t_data >= t_enter) & (t_data <= t_exit)
    ax_cau.axvspan(
        t_enter, t_exit, color=CROSS_COLOR, alpha=0.22, zorder=0,
        label='Source center inside caustic',
    )
    ax_cau.errorbar(
        t_data[data_zoom], mag_data[data_zoom], yerr=mag_err[data_zoom],
        fmt='o', ms=3.2, color='0.25', ecolor='0.55', elinewidth=0.55,
        zorder=3, label='Roman F146 mock',
    )
    if np.any(inside_epoch):
        ax_cau.scatter(
            t_data[inside_epoch], mag_data[inside_epoch],
            s=28, facecolors='none', edgecolors=CROSS_COLOR, linewidths=1.2,
            zorder=5, label='Epoch inside caustic',
        )
    ax_cau.plot(
        t_model[zoom], mag_model[zoom], color=PSBL_COLOR, lw=1.4,
        label='PSBL point source', zorder=4,
    )
    ax_cau.plot(
        t_model[zoom], mag_point[zoom], color=POINT_COLOR, lw=1.1, ls='--',
        label='Point lens',
    )
    if len(t_disk):
        ax_cau.plot(
            t_disk, mag_disk, color='0.1', lw=1.5, ls='-.',
            label=r'1 $R_\odot$ disk average', zorder=4,
        )
    faint, bright = _mag_span(mag_model[zoom], 2.5, 97.0, 0.1, 0.15)
    v1.style_mag_axis(ax_cau, faint, bright)
    ax_cau.set_xlim(t_cross - cau_half, t_cross + cau_half)
    ax_cau.set_xlabel('Time (MJD)')
    ax_cau.set_title(
        f'Shaded 5-image window. {n_inside} Roman epochs inside, '
        f'{n_wing} more in the wings.'
    )
    ax_cau.legend(loc='lower right', frameon=False, fontsize=7)

    res_half = 1.0
    res = np.abs(t_model - t_cross) < res_half
    delta = mag_model - mag_point
    ax_res.axvspan(t_enter, t_exit, color=CROSS_COLOR, alpha=0.22, zorder=0)
    ax_res.plot(
        t_model[res], delta[res], color=PSBL_COLOR, lw=1.2,
        label='PSBL $-$ point lens',
    )
    if len(t_disk):
        # Disk times are a subset of the model grid, so the point-lens
        # magnitude is the model value at the nearest sample.
        point_on_disk = np.interp(t_disk, t_model, mag_point)
        ax_res.plot(
            t_disk, mag_disk - point_on_disk, color='0.1', lw=1.4, ls='-.',
            label=r'1 $R_\odot$ disk $-$ point lens',
        )
    ax_res.axhline(0.0, color='0.4', lw=0.6)
    ax_res.axhspan(
        -0.017, 0.017, color=POINT_COLOR, alpha=0.12,
        label=r'$\pm 1\sigma$ at F146 = 19',
    )
    finite = delta[res]
    finite = finite[np.isfinite(finite)]
    bright_dm = float(np.percentile(finite, 4))
    faint_dm = float(np.percentile(finite, 96))
    pad = 0.12 * max(faint_dm - bright_dm, 0.3)
    # Brighter (negative residual) is up, matching the magnitude panels.
    ax_res.set_ylim(faint_dm + pad, bright_dm - pad)
    ax_res.set_xlim(t_cross - res_half, t_cross + res_half)
    ax_res.set_ylabel(r'$\Delta$F146 (mag)')
    ax_res.set_xlabel('Time (MJD)')
    ax_res.set_title(
        'Planet versus a point lens. The shade is the 5-image window; '
        'the black curve is the disk average.'
    )
    ax_res.legend(loc='lower right', frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def plot_astrometry(t_model, source_mas, centroid_mas, point_mas,
                    primary_mas, t_data, data_mas, data_err_mas,
                    t_peak, t_cross, t_enter, t_exit, outpath):
    """Plot the sky track with the caustic-crossing segment highlighted.

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
    t_enter, t_exit : float
        Edges of the 5-image interval, MJD.
    outpath : Path
        PNG destination.

    Returns
    -------
    None

    Notes
    -----
    A few point-source samples land on the fold, where the centroid is
    not meaningful. Those with ``|centroid - point lens| > 8`` mas are
    left off the axes. The real 5-image segment is drawn, in a separate
    color, on every panel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.8))
    ax_sky, ax_zoom, ax_full, ax_cau = axes.ravel()
    sane = np.linalg.norm(centroid_mas - point_mas, axis=1) < 8.0
    sane = sane & np.isfinite(centroid_mas[:, 0])
    crossing = (t_model >= t_enter) & (t_model <= t_exit) & sane
    body = sane & ~crossing
    step = max(1, len(t_data) // 2500)

    ax_sky.errorbar(
        data_mas[::step, 0], data_mas[::step, 1],
        xerr=data_err_mas[::step], yerr=data_err_mas[::step],
        fmt='.', ms=1.4, color='0.35', ecolor='0.7', elinewidth=0.3,
        alpha=0.28, rasterized=True, zorder=2, label='Roman mock',
    )
    ax_sky.plot(
        source_mas[sane, 0], source_mas[sane, 1], color='0.55', lw=0.9,
        label='Unlensed source',
    )
    ax_sky.plot(
        centroid_mas[body, 0], centroid_mas[body, 1], color=TRACK_COLOR,
        lw=1.15, label='Lensed centroid',
    )
    if np.any(crossing):
        ax_sky.plot(
            centroid_mas[crossing, 0], centroid_mas[crossing, 1],
            color=CROSS_COLOR, lw=2.2, zorder=5, label='Caustic crossing',
        )
    ax_sky.plot(
        primary_mas[sane, 0], primary_mas[sane, 1], color='0.15', lw=0.8,
        label='Black hole (parallax)',
    )
    i_peak = int(np.argmin(np.abs(t_model - t_peak)))
    ax_sky.scatter(
        [primary_mas[i_peak, 0]], [primary_mas[i_peak, 1]],
        marker='*', s=80, color='black', zorder=6, label='BH at peak',
    )
    ax_sky.invert_xaxis()
    ax_sky.set_aspect('equal', adjustable='datalim')
    ax_sky.set_xlabel(r'East offset, $\Delta\alpha\cos\delta$ (mas)')
    ax_sky.set_ylabel(r'North offset, $\Delta\delta$ (mas)')
    ax_sky.set_title('Sky track, relative to the BH at the peak')
    ax_sky.legend(loc='best', frameon=False, fontsize=7)

    window = (np.abs(t_model - t_cross) < 1.5) & sane
    data_window = np.abs(t_data - t_cross) < 1.5
    inside_data = (
        data_window & (t_data >= t_enter) & (t_data <= t_exit)
    )
    ax_zoom.plot(
        source_mas[window, 0], source_mas[window, 1],
        color='0.45', lw=1.2, label='Unlensed',
    )
    ax_zoom.plot(
        point_mas[window, 0], point_mas[window, 1],
        color=POINT_COLOR, lw=1.15, ls='--', label='Point-lens centroid',
    )
    body_w = window & ~crossing
    ax_zoom.plot(
        centroid_mas[body_w, 0], centroid_mas[body_w, 1],
        color=TRACK_COLOR, lw=1.4, label='PSBL centroid',
    )
    if np.any(crossing):
        ax_zoom.plot(
            centroid_mas[crossing, 0], centroid_mas[crossing, 1],
            color=CROSS_COLOR, lw=2.4, zorder=4, label='Inside caustic',
        )
    ax_zoom.errorbar(
        data_mas[data_window, 0], data_mas[data_window, 1],
        xerr=data_err_mas[data_window], yerr=data_err_mas[data_window],
        fmt='.', ms=3.0, color='0.35', ecolor='0.65', elinewidth=0.45,
        alpha=0.7, zorder=3, label='Roman mock',
    )
    if np.any(inside_data):
        ax_zoom.scatter(
            data_mas[inside_data, 0], data_mas[inside_data, 1],
            s=36, facecolors='none', edgecolors=CROSS_COLOR, linewidths=1.3,
            zorder=6, label='Epoch inside caustic',
        )
    ax_zoom.invert_xaxis()
    ax_zoom.set_aspect('equal', adjustable='datalim')
    ax_zoom.set_xlabel('East offset (mas)')
    ax_zoom.set_ylabel('North offset (mas)')
    ax_zoom.set_title('Sky track within ±1.5 days of the crossing')
    ax_zoom.legend(loc='best', frameon=False, fontsize=6.5)

    shift = centroid_mas - source_mas
    shift_point = point_mas - source_mas
    ax_full.axvspan(
        t_enter, t_exit, color=CROSS_COLOR, alpha=0.35, zorder=0,
        label='Caustic crossing',
    )
    ax_full.plot(
        t_model[body], shift[body, 0], color=TRACK_COLOR, lw=1.15,
        label='PSBL East',
    )
    ax_full.plot(
        t_model[body], shift[body, 1], color=TRACK_COLOR, lw=1.0, ls=':',
        label='PSBL North',
    )
    if np.any(crossing):
        ax_full.plot(
            t_model[crossing], shift[crossing, 0], color=CROSS_COLOR,
            lw=2.0, zorder=4, label='Crossing East',
        )
        ax_full.plot(
            t_model[crossing], shift[crossing, 1], color=CROSS_COLOR,
            lw=1.5, ls=':', zorder=4, label='Crossing North',
        )
    ax_full.plot(
        t_model[sane], shift_point[sane, 0], color=POINT_COLOR, lw=0.9,
        ls='--', label='Point lens East',
    )
    ax_full.axvline(t_peak, color='0.45', lw=0.6, ls=':')
    shown = shift[sane]
    ymax = float(np.nanpercentile(np.abs(shown), 99.5))
    ymax = max(ymax * 1.2, 1.05)
    ax_full.set_ylim(-ymax, ymax)
    ax_full.set_xlim(t_data.min() - 20.0, t_data.max() + 20.0)
    ax_full.set_xlabel('Time (MJD)')
    ax_full.set_ylabel('Centroid − source (mas)')
    ax_full.set_title(
        'Astrometric shift. The colored segment is the caustic crossing.'
    )
    ax_full.legend(loc='best', frameon=False, fontsize=6.5)

    near = (np.abs(t_model - t_cross) < 1.0) & sane
    near_body = near & ~crossing
    ax_cau.axvspan(t_enter, t_exit, color=CROSS_COLOR, alpha=0.22, zorder=0)
    ax_cau.plot(
        t_model[near_body], shift[near_body, 0], color=TRACK_COLOR, lw=1.3,
        label='PSBL East',
    )
    ax_cau.plot(
        t_model[near_body], shift[near_body, 1], color='#e66101', lw=1.15,
        label='PSBL North',
    )
    if np.any(crossing):
        ax_cau.plot(
            t_model[crossing], shift[crossing, 0], color=CROSS_COLOR, lw=2.2,
            label='Crossing East',
        )
        ax_cau.plot(
            t_model[crossing], shift[crossing, 1], color=CROSS_COLOR, lw=1.6,
            ls=':', label='Crossing North',
        )
    ax_cau.plot(
        t_model[near], shift_point[near, 0], color=POINT_COLOR, lw=1.0,
        ls='--', label='Point lens East',
    )
    ax_cau.plot(
        t_model[near], shift_point[near, 1], color=POINT_COLOR, lw=1.0,
        ls=':', label='Point lens North',
    )
    ax_cau.set_xlim(t_cross - 1.0, t_cross + 1.0)
    ax_cau.set_xlabel('Time (MJD)')
    ax_cau.set_ylabel('Centroid − source (mas)')
    ax_cau.set_title('Planet astrometric kick. Shade: source center inside.')
    ax_cau.legend(loc='best', frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def plot_caustics(critical, chosen, other, central, track, source_cross,
                  rho, planet_theta, half_width_central, s_einstein,
                  outpath):
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
    s_einstein : float
        Projected separation in Einstein units.
    outpath : Path
        PNG destination.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.0))
    ax_crit, ax_wide, ax_both, ax_zoom = axes.ravel()

    if len(critical) > 0:
        ax_crit.scatter(
            critical[:, 0], critical[:, 1], s=2, c='#542788',
            rasterized=True, label='Critical curve',
        )
    phi = np.linspace(0.0, 2.0 * np.pi, 400)
    ax_crit.plot(np.cos(phi), np.sin(phi), color='0.65', lw=0.7, ls=':')
    ax_crit.scatter(
        [0.0], [0.0], marker='*', s=70, c='black', label='BH', zorder=5,
    )
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
        track[:, 0], track[:, 1], color=TRACK_COLOR, lw=1.3,
        label='Source center', zorder=3,
    )
    ax_wide.scatter(
        [0.0], [0.0], marker='*', s=70, c='black', label='COM / BH',
        zorder=5,
    )
    if len(track):
        i_close = int(np.argmin(np.linalg.norm(track, axis=1)))
        ax_wide.scatter(
            [track[i_close, 0]], [track[i_close, 1]], marker='o', s=28,
            facecolors='none', edgecolors='0.15', zorder=6,
            label='BH peak',
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
        c=PSBL_COLOR, zorder=6, label='Source at crossing',
    )
    reach = float(np.percentile(np.linalg.norm(track, axis=1), 99))
    span = max(0.45, reach * 1.35)
    ax_wide.set_aspect('equal', adjustable='box')
    ax_wide.set_xlim(span, -span)
    ax_wide.set_ylim(-span, span)
    ax_wide.set_xlabel(r'East / $\theta_E$ (source plane)')
    ax_wide.set_ylabel(r'North / $\theta_E$')
    ax_wide.set_title(
        f'Close topology, s = {s_einstein:.3f}. Central radius '
        f'{half_width_central:.1e} (not crossed).'
    )
    ax_wide.legend(loc='upper right', frameon=False, fontsize=6.5)

    stack = [arr for arr in (chosen, other) if len(arr)]
    cloud = np.vstack(stack)
    extent = float(np.max(np.linalg.norm(cloud - source_cross, axis=1)))
    limit = max(1.55 * extent, 6.0 * rho)
    ax_both.plot(
        track[:, 0], track[:, 1], color=TRACK_COLOR, lw=1.2, zorder=2,
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
    ax_both.set_xlim(source_cross[0] + limit, source_cross[0] - limit)
    ax_both.set_ylim(source_cross[1] - limit, source_cross[1] + limit)
    ax_both.set_xlabel(r'East / $\theta_E$')
    ax_both.set_ylabel(r'North / $\theta_E$')
    ax_both.set_title('Both planetary caustics. The track hits only one.')
    ax_both.legend(loc='upper right', frameon=False, fontsize=6.5)

    half_box = float(np.max(np.linalg.norm(chosen - chosen.mean(0), axis=1)))
    local_limit = max(1.9 * half_box, 2.8 * rho)
    ax_zoom.scatter(
        chosen[:, 0], chosen[:, 1], s=12, c='#542788', zorder=3,
        label='Caustic',
    )
    near = np.linalg.norm(track - source_cross, axis=1) < 4.0 * local_limit
    if np.any(near):
        ax_zoom.plot(
            track[near, 0], track[near, 1], color=TRACK_COLOR, lw=1.4,
            zorder=2, label='Source center',
        )
    disk_zoom = Circle(
        (source_cross[0], source_cross[1]), rho,
        facecolor='#fdae61', edgecolor='#e66101', lw=1.0, alpha=0.55,
        zorder=1, label=r'1 $R_\odot$ at 8 kpc',
    )
    ax_zoom.add_patch(disk_zoom)
    ax_zoom.scatter(
        [source_cross[0]], [source_cross[1]], marker='x', c=PSBL_COLOR,
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
    # ``central`` is plotted only as a size in the title; keep the name
    # used so a future overlay has the samples in hand.
    _ = central
    fig.suptitle(
        'Static close binary. One planetary caustic is crossed after '
        'the black-hole peak, in the same fast season.',
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def copy_artifacts(outdir):
    """Copy v5 products into the artifact directory.

    Parameters
    ----------
    outdir : Path
        Directory of PNG and table files. Names already use the ``v5_``
        prefix; a missing prefix is added rather than doubled.

    Returns
    -------
    None
    """
    if not ARTIFACT_DIR.is_dir():
        return None
    for path in sorted(outdir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if not name.startswith('v5_'):
            name = 'v5_' + name
        target = ARTIFACT_DIR / name
        target.write_bytes(path.read_bytes())
    return None


def _caustic_metrics(chosen):
    """Radius and spans of one planetary caustic.

    Parameters
    ----------
    chosen : ndarray, shape (K, 2)
        Caustic samples, Einstein units.

    Returns
    -------
    radius, east_span, north_span : float
        Maximum distance from the centroid, and the coordinate spans.
    """
    centroid = chosen.mean(axis=0)
    radius = float(np.linalg.norm(chosen - centroid, axis=1).max())
    east_span = float(chosen[:, 0].max() - chosen[:, 0].min())
    north_span = float(chosen[:, 1].max() - chosen[:, 1].min())
    return radius, east_span, north_span


def main():
    """Generate the static 10 AU Roman mock, figures, and table.

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
    season0 = seasons[0]

    print('Orienting the binary toward one planetary caustic...', flush=True)
    alpha_deg, chosen, other, central, theta_e_mas = orient_binary(
        sep_mas, m_bh, m_planet
    )
    centroid = chosen.mean(axis=0)
    target_east_mas = float(centroid[0] * theta_e_mas)
    target_north_mas = float(centroid[1] * theta_e_mas)
    delay_days = target_east_mas * v1.DAYS_PER_YEAR / MU_EAST
    t_cross, t_peak_goal = choose_crossing_time(
        seasons, delay_days, LEAD_IN_DAYS
    )

    probe = build_static(60000.0, 0.1, sep_mas, alpha_deg, m_bh, m_planet)
    pi_rel_mas = float(probe.piRel)
    m1, m2, theta_arcsec = mass_fractions(probe)

    print('Aiming the source through the caustic...', flush=True)
    psbl = None
    t0 = np.nan
    beta_mas = np.nan
    miss_mas = np.nan
    t_peak = t_peak_goal
    u_min = np.nan
    a_peak = np.nan
    for _attempt in range(4):
        psbl, t0, beta_mas, miss_mas = aim_at_caustic(
            t_cross, target_east_mas, target_north_mas,
            sep_mas, alpha_deg, m_bh, m_planet, pi_rel_mas,
        )
        t_peak, u_min, a_peak = geocentric_peak(
            psbl, t_peak_goal, m1, m2, theta_arcsec, theta_e_mas
        )
        shift = t_peak_goal - t_peak
        print(
            f'  peak {t_peak:.3f} goal {t_peak_goal:.3f} '
            f'shift {shift:.3f} d',
            flush=True,
        )
        if abs(shift) < 0.05:
            break
        t_cross = t_cross + shift
    if miss_mas > 1.0e-6:
        raise RuntimeError('Source was not placed on the caustic.')
    if abs(t_peak - t_peak_goal) > 0.25:
        raise RuntimeError('Geocentric peak did not land on the season goal.')
    if not (season0[0] < t_peak < season0[1]):
        raise RuntimeError('Geocentric peak is outside the first fast season.')
    if not (season0[0] < t_cross < season0[1]):
        raise RuntimeError('Crossing is outside the first fast season.')

    # Measure the 5-image window before the full Roman grid, so a phase
    # nudge can still move the event onto a sample.
    print('Measuring the 5-image window...', flush=True)
    t_probe = np.linspace(t_cross - 0.20, t_cross + 0.20, 4001)
    _mag_p, n_probe, _cen_p, _src_p, _pri_p, _sec_p = solve_binary(
        psbl, t_probe, m1, m2, theta_arcsec
    )
    if int(n_probe[int(np.argmin(np.abs(t_probe - t_cross)))]) < 5:
        raise RuntimeError('Source center is not inside the planetary caustic.')
    t_enter, t_exit = contiguous_inside(t_probe, n_probe, t_cross)
    phase_days = roman_phase_shift(t_enter, t_exit, t_f146)
    if abs(phase_days) > 0.0:
        print(
            f'Phasing nudge {phase_days * 86400.0:.1f} s '
            'so a Roman epoch sits in the caustic.',
            flush=True,
        )
        t_cross = t_cross + phase_days
        psbl, t0, beta_mas, miss_mas = aim_at_caustic(
            t_cross, target_east_mas, target_north_mas,
            sep_mas, alpha_deg, m_bh, m_planet, pi_rel_mas,
        )
        t_peak, u_min, a_peak = geocentric_peak(
            psbl, t_peak, m1, m2, theta_arcsec, theta_e_mas
        )
        t_probe = np.linspace(t_cross - 0.20, t_cross + 0.20, 4001)
        _mag_p, n_probe, _cen_p, _src_p, _pri_p, _sec_p = solve_binary(
            psbl, t_probe, m1, m2, theta_arcsec
        )
        t_enter, t_exit = contiguous_inside(t_probe, n_probe, t_cross)
    print(
        f't_peak {t_peak:.3f} u_min {u_min:.4f} A {a_peak:.3f}  '
        f't_cross {t_cross:.3f}  '
        f'window {(t_exit - t_enter) * 86400.0:.1f} s',
        flush=True,
    )
    if not (season0[0] < t_peak < season0[1] and season0[0] < t_cross < season0[1]):
        raise RuntimeError('Peak and crossing are not in the same fast season.')

    cau_pts = v1.caustic_points_theta_e(psbl, t_cross, n_pts=4000)
    first, second, central = split_planetary_caustics(cau_pts)
    chosen, other = _pick_branch(first, second)
    cau_radius, cau_east, cau_north = _caustic_metrics(chosen)
    if cau_east > 0.02 or cau_north > 0.02:
        raise RuntimeError('Planetary caustic branch is not a compact triangle.')
    if len(central):
        half_central = float(np.linalg.norm(central, axis=1).max())
    else:
        half_central = np.nan
    if not np.isfinite(half_central) or half_central > 0.005:
        raise RuntimeError('Central caustic cut swallowed the triangles.')

    print('Solving the light curve...', flush=True)
    daily = np.arange(float(t_f146.min()), float(t_f146.max()) + 0.5, 1.0)
    near_peak = np.arange(t_peak - 45.0, t_peak + 45.0, 0.05)
    inner_peak = np.arange(t_peak - 3.0, t_peak + 3.0, 0.01)
    wings = np.arange(t_cross - 2.0, t_cross + 2.0, 0.005)
    fine = np.linspace(t_cross - 0.20, t_cross + 0.20, 4001)
    t_model = np.unique(np.concatenate([
        daily, near_peak, inner_peak, wings, fine,
        np.array([t_peak, t_cross]),
    ]))
    mag_a, n_model, cen_model, src_model, pri_model, sec_model = solve_binary(
        psbl, t_model, m1, m2, theta_arcsec
    )
    print('Solving the Roman grid...', flush=True)
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

    i_peak = int(np.argmin(np.abs(t_model - t_peak)))
    frac_peak = abs(mag_a[i_peak] - a_point_model[i_peak]) / a_point_model[i_peak]
    far = np.abs(t_model - t_cross) > 5.0
    frac_far = np.nanmax(
        np.abs(mag_a[far] - a_point_model[far]) / a_point_model[far]
    )
    bagle_check_t = np.array([t_peak, t_cross - 5.0, t_cross])
    bagle_eval = v1.evaluate_event(psbl, bagle_check_t)
    solved_check, n_check, *_rest = solve_binary(
        psbl, bagle_check_t, m1, m2, theta_arcsec
    )
    print(
        'fractional |A-Apoint| at peak', f'{frac_peak:.3e}',
        'far', f'{frac_far:.3e}',
        flush=True,
    )
    print(
        'BAGLE default vs solver',
        bagle_eval['magnification'] - solved_check,
        'n_solver', n_check,
        'n_bagle', bagle_eval['n_images'],
        flush=True,
    )
    if frac_peak > 1.0e-4:
        raise RuntimeError('Peak magnification disagrees with the point lens.')
    i_cross = int(np.argmin(np.abs(t_model - t_cross)))
    if n_model[i_cross] < 5:
        raise RuntimeError('Source center is not inside the planetary caustic.')

    # Re-measure on the full model grid so the shaded interval matches
    # the curve that is plotted. Bridge only sub-30 s dropouts.
    t_enter, t_exit = contiguous_inside(t_model, n_model, t_cross)
    duration_days = t_exit - t_enter
    in_caustic = (t_f146 >= t_enter) & (t_f146 <= t_exit)

    # Uniform-disk average on a short grid around the crossing.
    print('Disk-averaging the point-source map...', flush=True)
    t_clock = time.perf_counter()
    disk_half = 0.5 * duration_days + 0.06
    disk_idx = np.where(np.abs(t_model - t_cross) <= disk_half)[0]
    if len(disk_idx) > 70:
        stride = int(np.ceil(len(disk_idx) / 70.0))
        disk_idx = disk_idx[::stride]
    theta_star_mas = v1.source_angular_radius_mas(D_S_PC, 1.0)
    rho = theta_star_mas / theta_e_mas
    mag_disk = np.empty(len(disk_idx), dtype=float)
    n_disk_ok = 0
    n_disk_try = 0
    for k, i_time in enumerate(disk_idx):
        w = (src_model[i_time, 0] + 1j * src_model[i_time, 1]) / theta_arcsec
        z1 = (pri_model[i_time, 0] + 1j * pri_model[i_time, 1]) / theta_arcsec
        z2 = (sec_model[i_time, 0] + 1j * sec_model[i_time, 1]) / theta_arcsec
        amp, n_ok, n_try = disk_average_magnification(
            psbl, w, z1, z2, m1, m2, rho,
        )
        mag_disk[k] = mag_src - 2.5 * np.log10(amp)
        n_disk_ok += n_ok
        n_disk_try += n_try
    t_disk = t_model[disk_idx]
    disk_seconds = time.perf_counter() - t_clock
    print(
        f'  disk average {disk_seconds:.1f} s, '
        f'{n_disk_ok}/{n_disk_try} samples finite',
        flush=True,
    )

    delta_mag = mag_roman - mag_point_roman
    mag_noisy, mag_err = v1.add_roman_photometric_noise(mag_roman, rng_seed=146)
    _formula_err, ast_err_mas = v1.roman_f146_uncertainties(mag_roman)
    chi2_phot = float(np.nansum((delta_mag / mag_err) ** 2))
    roman_max = float(np.nanmax(np.abs(delta_mag)))
    fine_mask = np.abs(t_model - t_cross) <= 0.01
    fine_delta = mag_model[fine_mask] - mag_point_model[fine_mask]
    fine_max = float(np.nanmax(np.abs(fine_delta)))
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

    disk_delta = mag_disk - np.interp(t_disk, t_model, mag_point_model)
    disk_max = float(np.nanmax(np.abs(disk_delta)))
    disk_interior = (t_disk > t_enter + 0.15 * span) & (
        t_disk < t_exit - 0.15 * span
    )
    if np.any(disk_interior):
        disk_interior_dm = float(np.median(disk_delta[disk_interior]))
    else:
        disk_interior_dm = float(np.interp(t_cross, t_disk, disk_delta))
    # Smoothed duration: disk residual above 0.05 mag.
    above = np.abs(disk_delta) > 0.05
    if np.any(above):
        disk_span_days = float(t_disk[above].max() - t_disk[above].min())
    else:
        disk_span_days = 0.0

    wing = (np.abs(delta_mag) > 0.01) & ~in_caustic
    strong = np.abs(delta_mag) > 0.1
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

    good_mag = np.isfinite(mag_model)
    good_ast = good_mag & np.isfinite(cen_model[:, 0]) & np.isfinite(cen_model[:, 1])
    t_plot = t_model[good_mag]
    mag_plot = mag_model[good_mag]
    mag_point_plot = mag_point_model[good_mag]
    # Astrometry panels use the finite-centroid subset. Photometry keeps
    # the fold spikes, which are finite magnitudes with wild centroids.
    src_mas = to_mas(src_model[good_ast])
    cen_mas = to_mas(cen_model[good_ast])
    point_mas = to_mas(point_cen_model[good_ast])
    pri_mas = to_mas(pri_model[good_ast])
    data_mas = to_mas(data_centroid)
    t_ast = t_model[good_ast]

    com_track = v1.source_minus_com_mas(
        {
            'source_arcsec': src_model[good_mag],
            'primary_arcsec': pri_model[good_mag],
            'secondary_arcsec': sec_model[good_mag],
        },
        m_bh,
        m_planet,
    ) / theta_e_mas
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

    r_e_au = theta_e_mas * (D_L_PC / 1000.0)
    s_einstein = A_AU / r_e_au
    t_e_days = float(psbl.tE)
    pi_e = float(psbl.piE_amp)
    pi_l = float(psbl.piL)
    pi_s = float(psbl.piS)
    u0_bary = float(psbl.u0_amp)
    width_analytic = v1.central_caustic_width_theta_e(q_mass, s_einstein)
    mu_theta_per_day = (MU_EAST / v1.DAYS_PER_YEAR) / theta_e_mas
    fs_duration_days = (2.0 * rho) / mu_theta_per_day
    shift_bh = np.linalg.norm(cen_mas - src_mas, axis=1)
    away = np.abs(t_ast - t_cross) > 1.0
    max_shift_mas = float(np.nanmax(shift_bh[away]))
    source_bh_mas = float(np.linalg.norm(source_cross) * theta_e_mas)

    in_fast = (t_f146 >= season0[0]) & (t_f146 <= season0[1])
    fast_dt_min = float(np.median(np.diff(np.sort(t_f146[in_fast]))) * 1440.0)
    _err19, ast19 = v1.roman_f146_uncertainties(np.array([19.0]))
    _noisy19, mag_err_19 = v1.add_roman_photometric_noise(
        np.array([19.0, 19.0]), rng_seed=1
    )
    mag_err_19_value = float(np.median(mag_err_19))

    bagle_on = float(bagle_eval['magnification'][2])
    solver_on = float(solved_check[2])
    n_bagle_on = int(bagle_eval['n_images'][2])
    n_solver_on = int(n_check[2])
    misses_images = n_bagle_on < 5
    size_factor = cau_radius / V4_CAUSTIC_RADIUS
    narrow = min(cau_east, cau_north)
    disk_vs_narrow = (2.0 * rho) / narrow
    if misses_images:
        miss_text = (
            f'yes: default n={n_bagle_on}, A={bagle_on:.3f}; '
            f'solver n={n_solver_on}, A={solver_on:.3f}'
        )
    else:
        miss_text = (
            f'no: default n={n_bagle_on}, A={bagle_on:.3f}; '
            f'solver n={n_solver_on}, A={solver_on:.3f}'
        )
    if abs(phase_days) > 0.0:
        phase_text = (
            f'shifted by {phase_days * 86400.0:.1f} s so an epoch is inside'
        )
    else:
        phase_text = 'no nudge; a Roman epoch already falls inside'
    smooth_text = (
        f'disk radius is {rho / cau_radius:.2f}x the caustic radius; '
        f'diameter is {disk_vs_narrow:.2f}x the narrow axis. '
        f'Point-source max |dm| {fine_max:.2f} mag vs disk {disk_max:.2f} mag; '
        f'interior median {interior_dm:.2f} vs {disk_interior_dm:.2f} mag.'
    )

    season_start = Time(season0[0], format='mjd').iso[:10]
    season_stop = Time(season0[1], format='mjd').iso[:10]
    season_label = f'{season_start} to {season_stop}'
    t_cross_iso = Time(t_cross, format='mjd').isot
    t_peak_iso = Time(t_peak, format='mjd').isot
    t0_iso = Time(t0, format='mjd').isot
    n_bad = int(np.sum(~np.isfinite(mag_a)))
    n_inside = int(np.sum(in_caustic))
    n_wing = int(np.sum(wing))

    print('Writing figures...', flush=True)
    plot_photometry(
        t_f146, mag_noisy, mag_err, t_plot, mag_plot, mag_point_plot,
        t_peak, t_cross, t_enter, t_exit, t_disk, mag_disk,
        n_inside, n_wing, season_label,
        OUTDIR / 'v5_fig_photometry_f146.png',
    )
    plot_astrometry(
        t_ast, src_mas, cen_mas, point_mas, pri_mas,
        t_f146, data_mas, ast_err_mas, t_peak, t_cross, t_enter, t_exit,
        OUTDIR / 'v5_fig_astrometry.png',
    )
    critical = v1.critical_curve_theta_e(psbl, t_cross, n_pts=800)
    plot_caustics(
        critical, chosen, other, central,
        com_track[track_window], source_cross, rho, planet_theta,
        half_central, s_einstein, OUTDIR / 'v5_fig_caustic.png',
    )

    rows = [
        ('Model class', 'PSBL_PhotAstrom_Par_Param1'),
        ('Orbit', 'none (static projected separation)'),
        ('Image solver', 'BAGLE 5th-order poly, Einstein units'),
        ('default get_all_arrays misses images?', miss_text),
        ('Finite-source binary', 'unavailable (FSBL commented out)'),
        ('Source treatment', 'point source; disk average is scratch-only'),
        ('Cadence helper', 'fake_data.get_times_roman_gbtds'),
        ('Photometric noise', 'fake_data.add_photometric_noise, zp=28'),
        ('Astrometric error', 'FWHM/(2 SNR), floor 0.1 mas; no 1e-5'),
        ('M_BH (Msun)', f'{m_bh:.1f}'),
        ('M_planet (Msun)', f'{m_planet:.6e}'),
        ('M_planet', '1 M_earth'),
        ('q = M_planet/M_BH', f'{q_mass:.6e}'),
        ('D_L (kpc)', f'{D_L_PC / 1000.0:.1f}'),
        ('D_S (kpc)', f'{D_S_PC / 1000.0:.1f}'),
        ('a_proj (AU)', f'{A_AU:.1f}'),
        ('sep (mas)', f'{sep_mas:.6f}'),
        ('s = a / r_E', f'{s_einstein:.6f}'),
        ('topology', 'close (s < 1), two planetary caustics'),
        ('|s - 1/s|', f'{abs(s_einstein - 1.0 / s_einstein):.6f}'),
        ('alpha (deg E of N)', f'{alpha_cross:.4f}'),
        ('mu direction (deg E of N)', '90 (due East)'),
        ('angle(trajectory, axis)', f'{angle_to_axis:.4f} deg'),
        ('b_sff', '1 (no blend, no lens flux)'),
        ('dmag_Lp_Ls', '0 (both components dark)'),
        ('mag_src F146W', '19'),
        ('mu_rel (mas/yr)', f'{MU_EAST:.1f} due East'),
        ('t0 barycentric (MJD)', f'{t0:.5f} ({t0_iso})'),
        ('geocentric peak (MJD)', f'{t_peak:.5f} ({t_peak_iso})'),
        ('crossing (MJD)', f'{t_cross:.5f} ({t_cross_iso})'),
        ('peak and crossing season', season_label),
        ('peak-to-crossing (days)', f'{t_cross - t_peak:.3f}'),
        ('beta (mas)', f'{beta_mas:.6e}'),
        ('u0 barycentric', f'{u0_bary:.6e}'),
        ('u at geocentric peak', f'{u_min:.6f}'),
        ('A at geocentric peak', f'{a_peak:.4f}'),
        ('F146 at geocentric peak', f'{mag_src - 2.5 * np.log10(a_peak):.3f}'),
        ('aim residual (mas)', f'{miss_mas:.3e}'),
        ('source-BH at crossing (mas)', f'{source_bh_mas:.4f}'),
        ('theta_E (mas)', f'{theta_e_mas:.6f}'),
        ('r_E at lens (AU)', f'{r_e_au:.4f}'),
        ('t_E (days)', f'{t_e_days:.4f}'),
        ('pi_L (mas)', f'{pi_l:.6f}'),
        ('pi_S (mas)', f'{pi_s:.6f}'),
        ('pi_rel (mas)', f'{pi_rel_mas:.6f}'),
        ('pi_E', f'{pi_e:.6f}'),
        ('R_source', '1 R_sun (disk average is not in BAGLE)'),
        ('theta_source (mas)', f'{theta_star_mas:.6e}'),
        ('rho = theta_*/theta_E', f'{rho:.6e}'),
        ('crossed caustic radius (theta_E)', f'{cau_radius:.6e}'),
        ('caustic East span (theta_E)', f'{cau_east:.6e}'),
        ('caustic North span (theta_E)', f'{cau_north:.6e}'),
        ('rho / caustic radius', f'{rho / cau_radius:.3f}'),
        ('vs 6 AU caustic radius', f'{size_factor:.2f} times larger'),
        ('central caustic radius (theta_E)', f'{half_central:.6e}'),
        ('central width analytic (theta_E)', f'{width_analytic:.6e}'),
        ('5-image duration (seconds)', f'{duration_days * 86400.0:.2f}'),
        ('5-image duration (hours)', f'{duration_days * 24.0:.3f}'),
        ('1 Rsun sweep time (minutes)', f'{fs_duration_days * 1440.0:.2f}'),
        ('Roman phase nudge', phase_text),
        ('images at the aim epoch', f'{int(n_model[i_cross])}'),
        ('interior median dm (mag)', f'{interior_dm:.4f}'),
        ('fine-grid max |dm| (mag)', f'{fine_max:.4f}'),
        ('fine-grid max |dpos| (mas)', f'{ast_fine_max:.4f}'),
        ('N F146 inside the 5-image window', f'{n_inside}'),
        ('N F146 in the wings, |dm|>0.01', f'{n_wing}'),
        ('N F146 with |dm| > 0.1 mag', f'{int(np.sum(strong))}'),
        ('max |dm| on Roman grid (mag)', f'{roman_max:.6f}'),
        ('chi2 of that residual', f'{chi2_phot:.4f}'),
        ('max demagnification (mag)', f'{max_demag:.6f}'),
        ('max |dpos| on Roman grid (mas)', f'{ast_max:.6f}'),
        ('chi2 ast of that residual', f'{chi2_ast:.4f}'),
        ('peak BH centroid shift (mas)', f'{max_shift_mas:.4f}'),
        ('disk grid', '5 equal-area rings x 12 angles'),
        ('disk samples finite', f'{n_disk_ok}/{n_disk_try}'),
        ('disk-average max |dm| (mag)', f'{disk_max:.4f}'),
        ('disk interior median dm (mag)', f'{disk_interior_dm:.4f}'),
        ('disk |dm|>0.05 span (hours)', f'{disk_span_days * 24.0:.3f}'),
        ('1 Rsun smoothing', smooth_text),
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
        ('Placement', 'u0=0.10; peak and crossing share one fast season'),
        ('Notes', '1e-5 factor in test_roman_lightcurve was not applied.'),
        ('Notes 2', 'chi2 is the residual against Roman errors, not a refit.'),
        ('Notes 3', 'Fine-grid max |dm| samples the point-source fold.'),
    ]
    v1.write_tables(rows, OUTDIR)
    (OUTDIR / 'parameters.txt').rename(OUTDIR / 'v5_parameters.txt')
    (OUTDIR / 'parameters.csv').rename(OUTDIR / 'v5_parameters.csv')

    fig_h = 0.175 * len(rows) + 0.9
    fig, axis = plt.subplots(figsize=(10.4, fig_h))
    axis.axis('off')
    lines = [f'{label:<44s} {value}' for label, value in rows]
    axis.text(
        0.01, 0.995, '\n'.join(lines), va='top', ha='left',
        family='monospace', fontsize=7.0, linespacing=1.12,
        transform=axis.transAxes, clip_on=False,
    )
    axis.set_title('Input and derived parameters', loc='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'v5_fig_parameter_table.png')
    plt.close(fig)
    copy_artifacts(OUTDIR)

    print('duration_s', duration_days * 86400.0)
    print('roman inside', n_inside, 'wing>0.01', n_wing)
    print('max |dm|', roman_max, 'chi2', chi2_phot)
    print('max |dpos|', ast_max, 'chi2_ast', chi2_ast)
    print('interior dm', interior_dm, 'disk', disk_interior_dm, 'disk max', disk_max)
    print('caustic radius', cau_radius, 'factor vs v4', size_factor)
    print('rho/radius', rho / cau_radius, 'misses images', misses_images)
    print('Wrote', OUTDIR)
    return None


if __name__ == '__main__':
    main()
