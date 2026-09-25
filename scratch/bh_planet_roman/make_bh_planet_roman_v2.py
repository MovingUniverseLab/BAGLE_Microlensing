"""Roman GBTDS mock of an 8 solar-mass black hole with an orbiting Earth.

This is the second demonstration for Prof. Jessica Lu. It does not modify
BAGLE. The lens is an 8 solar-mass black hole at 1.5 kpc. A 1 Earth-mass
planet orbits it on a face-on circular orbit with a semi-major axis of
1 AU. Both lens components are dark. The source is at 8 kpc with
F146W = 19. The geocentric trajectory crosses the central caustic in the
middle of the second GBTDS fast season, at 45 degrees to the binary axis
at that epoch.

The v1 static 0.1 AU run is left in ``output/v1_static_0p1AU/``. This
script writes ``output/v2_orbit_1AU_1p5kpc/``.

BAGLE has no working finite-source binary-lens class. The light curve is
the point-source binary lens with Keplerian orbital motion. A uniform-disk
point-lens peak is quoted only as a scale comparison.

Run from the repository root with::

    PYTHONPATH=src python scratch/bh_planet_roman/make_bh_planet_roman_v2.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.patches import Circle

# The v1 demo lives next to this file and is not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_bh_planet_roman_demo as v1


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'output' / 'v2_orbit_1AU_1p5kpc'
ARTIFACT_DIR = Path('/opt/cursor/artifacts')

# Face-on circular orbit. Projected separation then equals 1 AU at all
# times, so "1 AU" is the sky separation and not an inclined projection.
INCLINATION_DEG = 0.0
# Argument of periastron. For a circular orbit this only sets the phase,
# together with the time of periastron.
OMEGA_PRI_DEG = 0.0
# Longitude of the ascending node of the secondary. With i = 0 and
# omega_pri = 0, BAGLE's binary angle at the time of periastron is this
# value (degrees East of North).
BIG_OMEGA_DEG = 45.0
# Relative proper motion, mas/yr, due East. Same bulge-like value as v1.
MU_EAST = 7.0
MU_NORTH = 0.0


def second_fast_season(times_mjd):
    """Return the second high-cadence Roman visibility window.

    Parameters
    ----------
    times_mjd : ndarray, shape (N,)
        F146 observation times from ``get_times_roman_gbtds``, MJD.

    Returns
    -------
    t_start : float
        First F146 sample of that window, MJD.
    t_stop : float
        Last F146 sample of that window, MJD.
    n_points : int
        Number of F146 samples in the window.
    t_mid : float
        Midpoint ``0.5 * (t_start + t_stop)``, MJD.

    Notes
    -----
    BAGLE's season index 0 is the first fast window (2027 February–April).
    Season index 1 is the next visibility window (2027 August–October).
    Slow-cadence visits are one exposure every few days, so a 2-day gap
    cut splits them into single-point groups. A "season" here is a window
    with more than 100 samples, which is the fast cadence.
    """
    seasons = v1.fast_seasons(times_mjd)
    fast = [season for season in seasons if season[3]]
    if len(fast) < 2:
        raise RuntimeError('Need at least two fast Roman seasons.')
    t_start, t_stop, n_points, _is_fast = fast[1]
    t_mid = 0.5 * (t_start + t_stop)
    return float(t_start), float(t_stop), int(n_points), float(t_mid)


def build_circorbs(t0_com, beta_com, tp, a_mas, m_bh, m_planet):
    """Build a dark circular-orbit binary lens with parallax.

    Parameters
    ----------
    t0_com : float
        Barycentric time of closest approach of the source to the lens
        center of mass, MJD.
    beta_com : float
        Signed barycentric impact parameter relative to the center of
        mass, in milliarcseconds. With an Eastward proper motion, this
        is the North offset.
    tp : float
        Time of periastron, MJD. For zero eccentricity this is only a
        phase label. The binary angle is ``BIG_OMEGA_DEG`` at this time.
    a_mas : float
        Semi-major axis in milliarcseconds. At the lens distance, 1 mas
        per kpc is 1 AU, so 1 AU at 1.5 kpc is ``1/1.5`` mas.
    m_bh : float
        Black-hole mass in solar masses (primary).
    m_planet : float
        Planet mass in solar masses (secondary).

    Returns
    -------
    psbl : bagle.model.PSBL_PhotAstrom_Par_CircOrbs_Param1
        Circular Keplerian binary. Photometry, astrometry, and annual
        parallax are enabled. Lens flux is zero.

    Notes
    -----
    ``CircOrbs`` is the elliptical Keplerian model with eccentricity
    fixed at 0. Linear and accelerated orbit classes are expansions for
    times much shorter than the period. The period here is about 130
    days and ``t_E`` is about 310 days, so the Keplerian class is the
    one that stays valid across the event.

    Inclination is 0 (face-on), so the sky-projected separation equals
    the 1 AU semi-major axis at every epoch. An inclined orbit would
    make the projected separation smaller than 1 AU for part of the
    period.

    ``b_sff = 1`` and ``dmag_Lp_Ls = 0`` are the static-model
    prescription for two dark lenses. With ``b_sff = 1`` the image
    centroid has no lens light.
    """
    psbl = v1.model.PSBL_PhotAstrom_Par_CircOrbs_Param1(
        m_bh,
        m_planet,
        t0_com,
        0.0,
        0.0,
        beta_com,
        0.0,
        0.0,
        OMEGA_PRI_DEG,
        BIG_OMEGA_DEG,
        INCLINATION_DEG,
        tp,
        a_mas,
        MU_EAST,
        MU_NORTH,
        1500.0,
        8000.0,
        [1.0],
        [19.0],
        [0.0],
        raL=v1.RA_DEG,
        decL=v1.DEC_DEG,
        obsLocation=['earth'],
        root_tol=1.0e-8,
    )
    return psbl


def aim_through_com(t_cross, pi_rel_mas, a_mas, m_bh, m_planet):
    """Place the geocentric source on the center of mass at ``t_cross``.

    Parameters
    ----------
    t_cross : float
        MJD of the requested geocentric caustic crossing.
    pi_rel_mas : float
        Relative parallax in milliarcseconds.
    a_mas : float
        Angular semi-major axis in milliarcseconds.
    m_bh : float
        Black-hole mass in solar masses.
    m_planet : float
        Planet mass in solar masses.

    Returns
    -------
    psbl : bagle model
        CircOrbs model whose geocentric source sits on the center of
        mass at ``t_cross``.
    t0_com : float
        Barycentric closest-approach time, MJD.
    beta_com : float
        Barycentric impact parameter, milliarcseconds.
    miss_mas : float
        Residual source-COM separation at ``t_cross``, milliarcseconds.

    Notes
    -----
    Periastron is set equal to ``t_cross``, so the binary axis is at
    45 degrees East of North at the geocentric crossing. BAGLE anchors
    the source ephemeris on the geometric-center ``t0``, which differs
    from ``t0_com`` by the time needed to travel from the center of
    mass to the geometric center. One correction along the proper
    motion removes that offset. The central caustic of a close binary
    sits on the center of mass, so this aim is a caustic crossing.
    """
    t0_com, beta_com, _pvec = v1.aim_at_sky_offset(
        t_cross, 0.0, 0.0, pi_rel_mas, mu_east=MU_EAST
    )
    psbl = None
    miss = np.array([np.nan, np.nan])
    for _step in range(5):
        psbl = build_circorbs(
            t0_com, beta_com, t_cross, a_mas, m_bh, m_planet
        )
        evaluated = v1.evaluate_event(psbl, np.array([t_cross]))
        miss = v1.source_minus_com_mas(evaluated, m_bh, m_planet)[0]
        # East residual: increasing t0_com moves the closest approach
        # later, which moves the source West at a fixed clock time.
        t0_com = t0_com + miss[0] * v1.DAYS_PER_YEAR / MU_EAST
        beta_com = beta_com - miss[1]
    miss_mas = float(np.linalg.norm(miss))
    return psbl, float(t0_com), float(beta_com), miss_mas


def axis_angle_deg(primary_arcsec, secondary_arcsec):
    """Position angle of the primary as seen from the secondary.

    Parameters
    ----------
    primary_arcsec : ndarray, shape (2,)
        Primary position, East then North, in arcseconds.
    secondary_arcsec : ndarray, shape (2,)
        Secondary position, same frame and units.

    Returns
    -------
    angle_deg : float
        Degrees East of North, in ``[0, 360)``. This is BAGLE's
        ``alpha`` for a static binary.
    """
    delta = np.asarray(primary_arcsec) - np.asarray(secondary_arcsec)
    angle = np.rad2deg(np.arctan2(delta[0], delta[1]))
    angle_deg = float(np.mod(angle, 360.0))
    return angle_deg


def angle_between_deg(angle_a, angle_b):
    """Smallest angle between two undirected axes.

    Parameters
    ----------
    angle_a : float
        First position angle, degrees.
    angle_b : float
        Second position angle, degrees.

    Returns
    -------
    separation_deg : float
        Angle in ``[0, 90]``. A binary axis has no arrowhead, so
        180 degrees is the same axis.
    """
    raw = abs((angle_a - angle_b + 180.0) % 360.0 - 180.0)
    if raw > 90.0:
        raw = 180.0 - raw
    return float(raw)


def rotated_astroid(half_width, axis_deg, n_pts=400):
    """Astroid with a cusp along a chosen binary axis.

    Parameters
    ----------
    half_width : float
        Cusp distance from the center, in the same units as the plot.
    axis_deg : float
        Position angle of an on-axis cusp, degrees East of North.
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
    The axis-aligned astroid has cusps on the cardinal directions. It is
    rotated so that one cusp lies at ``axis_deg`` East of North, which
    is the close-planet orientation (cusps along the binary axis and
    perpendicular to it).
    """
    east0, north0 = v1.analytic_astroid(half_width, n_pts=n_pts)
    alpha = np.deg2rad(axis_deg)
    east = east0 * np.sin(alpha) - north0 * np.cos(alpha)
    north = east0 * np.cos(alpha) + north0 * np.sin(alpha)
    return east, north


def build_static_match(orbit, t_cross, m_bh, m_planet, pi_rel_mas):
    """Static PSBL frozen at the orbit's geometry at one epoch.

    Parameters
    ----------
    orbit : bagle model
        CircOrbs model.
    t_cross : float
        Epoch at which the static separation and angle are taken, MJD.
    m_bh : float
        Black-hole mass in solar masses.
    m_planet : float
        Planet mass in solar masses.
    pi_rel_mas : float
        Relative parallax in milliarcseconds.

    Returns
    -------
    static : bagle.model.PSBL_PhotAstrom_Par_Param1
        Static binary whose lenses and source match ``orbit`` at
        ``t_cross``, up to a shared sky offset from the different
        time anchors.
    sep_mas : float
        Projected separation at ``t_cross``, milliarcseconds.
    alpha_deg : float
        Binary angle at ``t_cross``, degrees East of North.

    Notes
    -----
    This is the "orbital motion switched off" comparison. The static
    class has no ``tp`` or inclination. Separation and ``alpha`` are
    the instantaneous CircOrbs values.
    """
    evaluated = v1.evaluate_event(orbit, np.array([t_cross]))
    primary = evaluated['primary_arcsec'][0]
    secondary = evaluated['secondary_arcsec'][0]
    delta_mas = (primary - secondary) * 1.0e3
    sep_mas = float(np.linalg.norm(delta_mas))
    alpha_deg = axis_angle_deg(primary, secondary)
    geom = 0.5 * (primary + secondary)
    source_minus_geom = (evaluated['source_arcsec'][0] - geom) * 1.0e3
    t0_static, beta_static, _pvec = v1.aim_at_sky_offset(
        t_cross,
        float(source_minus_geom[0]),
        float(source_minus_geom[1]),
        pi_rel_mas,
        mu_east=MU_EAST,
    )
    static = v1.model.PSBL_PhotAstrom_Par_Param1(
        m_bh,
        m_planet,
        t0_static,
        0.0,
        0.0,
        beta_static,
        0.0,
        0.0,
        MU_EAST,
        MU_NORTH,
        1500.0,
        8000.0,
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
    return static, sep_mas, alpha_deg


def plot_photometry(t_data, mag_data, mag_err, t_model, mag_model,
                    mag_point, t_cross, t_disk, peak_note,
                    residual_max, outpath):
    """Plot the F146 light curve, a peak zoom, and the planet residual.

    Parameters
    ----------
    t_data : ndarray, shape (N,)
        Roman F146 times, MJD.
    mag_data : ndarray, shape (N,)
        Noisy mock magnitudes.
    mag_err : ndarray, shape (N,)
        Photometric uncertainties in magnitudes.
    t_model : ndarray, shape (M,)
        Dense model times, MJD.
    mag_model : ndarray, shape (M,)
        CircOrbs point-source magnitudes.
    mag_point : ndarray, shape (M,)
        Point-lens magnitudes on the same grid.
    t_cross : float
        Geocentric caustic-crossing time, MJD.
    t_disk : float
        Half-duration of the stellar-disk crossing, days. Times within
        this of ``t_cross`` are inside the assumed 1 solar-radius star.
    peak_note : str
        Finite-source scale note drawn on the zoom panel.
    residual_max : float
        Maximum absolute planet-minus-point-lens residual on the Roman
        times, in magnitudes.
    outpath : Path
        PNG destination.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.4, 9.6),
        sharex=False,
        gridspec_kw={'height_ratios': [1.05, 1.15, 0.78]},
    )
    ax_full, ax_zoom, ax_res = axes
    # Drop the on-caustic sample. The polynomial solver saturates there
    # instead of diverging, and the Roman cadence cannot land on it.
    safe = np.abs(t_model - t_cross) > 1.0e-4
    t_model = t_model[safe]
    mag_model = mag_model[safe]
    mag_point = mag_point[safe]

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
        t_model, mag_model, color='#b2182b', lw=1.2, zorder=3,
        label='BAGLE CircOrbs (point source)',
    )
    ax_full.plot(
        t_model, mag_point, color='#2166ac', lw=1.0, ls='--', zorder=4,
        label='Point lens (same BH trajectory)',
    )
    v1.style_mag_axis(ax_full, 19.8, 14.0)
    ax_full.set_xlim(t_data.min() - 20.0, t_data.max() + 20.0)
    ax_full.set_xlabel('Time (MJD)')
    ax_full.text(
        0.02, 0.04,
        'Point-source peak is brighter than this panel (see zoom).',
        transform=ax_full.transAxes, fontsize=8, va='bottom',
    )
    ax_full.set_title(
        r'8 M$_\odot$ BH + 1 M$_\oplus$ at 1 AU, $D_L$ = 1.5 kpc, '
        'face-on orbit'
    )
    ax_full.legend(loc='lower right', frameon=False)

    half = 8.0
    zoom = (t_model > t_cross - half) & (t_model < t_cross + half)
    data_zoom = (t_data > t_cross - half) & (t_data < t_cross + half)
    ax_zoom.axvspan(
        t_cross - t_disk, t_cross + t_disk, color='#fdae61', alpha=0.25,
        label='Inside 1 $R_\\odot$ disk', zorder=0,
    )
    ax_zoom.errorbar(
        t_data[data_zoom], mag_data[data_zoom], yerr=mag_err[data_zoom],
        fmt='.', ms=2.2, color='0.15', ecolor='0.45', elinewidth=0.5,
        alpha=0.7, rasterized=True, zorder=2, label='Roman F146 mock',
    )
    ax_zoom.plot(
        t_model[zoom], mag_model[zoom], color='#b2182b', lw=1.4,
        label='BAGLE CircOrbs',
    )
    ax_zoom.plot(
        t_model[zoom], mag_point[zoom], color='#2166ac', lw=1.1, ls='--',
        label='Point lens',
    )
    v1.style_mag_axis(ax_zoom, 19.6, 5.5)
    ax_zoom.set_xlim(t_cross - half, t_cross + half)
    ax_zoom.set_xlabel('Time (MJD)')
    ax_zoom.set_title(
        'Zoom on the second-season peak. The caustic spike is '
        'a fraction of a second wide.'
    )
    ax_zoom.text(
        0.02, 0.04, peak_note, transform=ax_zoom.transAxes,
        fontsize=8, va='bottom', ha='left',
    )
    ax_zoom.legend(loc='lower right', frameon=False, fontsize=7.5)

    delta_mmag = 1.0e3 * (mag_model - mag_point)
    ax_res.axvspan(
        t_cross - t_disk, t_cross + t_disk, color='#fdae61', alpha=0.35,
        label='Inside stellar disk', zorder=0,
    )
    ax_res.plot(
        t_model, delta_mmag, color='#b2182b', lw=1.0,
        label='CircOrbs $-$ point lens',
    )
    ax_res.axhline(0.0, color='0.4', lw=0.6)
    ax_res.axhspan(
        -17.0, 17.0, color='#2166ac', alpha=0.12,
        label=r'$\pm 1\sigma$ at F146 = 19',
    )
    ax_res.set_xlim(t_data.min() - 20.0, t_data.max() + 20.0)
    ax_res.set_ylim(-40.0, 40.0)
    ax_res.set_ylabel('Residual (mmag)')
    ax_res.set_xlabel('Time (MJD)')
    ax_res.set_title(
        'Planet versus a point lens. '
        f'Max |Δm| on the Roman grid = {residual_max * 1.0e3:.3f} mmag.'
    )
    ax_res.legend(loc='upper right', frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def plot_caustics(critical, caustic_epochs, trajectory_rho,
                  trajectory_caustic, rho, s_einstein, axis_deg,
                  half_width, planet_theta_e, outpath):
    """Plot critical curves and the rotating central caustic.

    Parameters
    ----------
    critical : ndarray, shape (M, 2)
        Critical curve at the crossing epoch, relative to the black
        hole, in units of ``theta_E``. Columns are East, North.
    caustic_epochs : list of tuple
        Each entry is ``(label, points)`` with ``points`` shaped
        ``(K, 2)`` in ``theta_E`` relative to the center of mass.
        The first entry is the crossing epoch.
    trajectory_rho : ndarray, shape (N, 2)
        Source-center track on the stellar-disk scale, ``theta_E``.
    trajectory_caustic : ndarray, shape (N, 2)
        Source-center track on the caustic scale, ``theta_E``.
    rho : float
        Source radius in units of ``theta_E``.
    s_einstein : float
        Projected separation in units of ``theta_E``.
    axis_deg : float
        Binary-axis position angle at the crossing, degrees East of
        North.
    half_width : float
        Half-width of the central caustic in ``theta_E``, used for the
        analytic astroid and the zoom.
    planet_theta_e : float
        Sky position of the planet relative to the black hole at the
        crossing, in ``theta_E``, shape ``(2,)`` as East, North.
    outpath : Path
        PNG destination.

    Returns
    -------
    None

    Notes
    -----
    The 45 degree convention is the angle at the geocentric crossing
    between the source proper motion (due East) and the binary axis.
    Later epochs show the same caustic after the orbit has rotated it.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.3))
    ax_crit, ax_rho, ax_cau = axes

    if len(critical) > 0:
        ring = np.linalg.norm(critical, axis=1) > 0.2
        ax_crit.scatter(
            critical[ring, 0], critical[ring, 1], s=2, c='#542788',
            rasterized=True, label='Critical curve',
        )
        if np.any(~ring):
            ax_crit.scatter(
                critical[~ring, 0], critical[~ring, 1], s=8, c='#e66101',
                label='Near the planet',
            )
    phi = np.linspace(0.0, 2.0 * np.pi, 400)
    ax_crit.plot(np.cos(phi), np.sin(phi), color='0.6', lw=0.7, ls=':')
    ax_crit.scatter(
        [0.0], [0.0], marker='*', s=70, c='black', label='BH', zorder=5,
    )
    ax_crit.scatter(
        [planet_theta_e[0]], [planet_theta_e[1]], marker='o', s=18,
        c='#e66101', label='Planet at $t_0$', zorder=5,
    )
    ax_crit.set_aspect('equal', adjustable='box')
    ax_crit.set_xlim(1.6, -1.6)
    ax_crit.set_ylim(-1.6, 1.6)
    ax_crit.set_xlabel(r'East / $\theta_E$ (image plane)')
    ax_crit.set_ylabel(r'North / $\theta_E$')
    ax_crit.set_title('Critical curves at the crossing')
    ax_crit.legend(loc='upper right', frameon=False, fontsize=7)

    # Source scale. The caustic is unresolved; the star is the circle.
    limit = 4.0 * rho
    axis_len = 0.92 * limit
    alpha = np.deg2rad(axis_deg)
    ax_rho.plot(
        [-axis_len * np.sin(alpha), axis_len * np.sin(alpha)],
        [-axis_len * np.cos(alpha), axis_len * np.cos(alpha)],
        color='0.45', lw=0.8, ls='--', label='Binary axis at $t_0$',
        zorder=1,
    )
    disk = Circle(
        (0.0, 0.0), rho, facecolor='#fdae61', edgecolor='#e66101',
        lw=1.0, alpha=0.85, label=r'Source (1 $R_\odot$)', zorder=2,
    )
    ax_rho.add_patch(disk)
    if len(trajectory_rho) > 0:
        ax_rho.plot(
            trajectory_rho[:, 0], trajectory_rho[:, 1], color='#1b9e77',
            lw=1.4, zorder=3, label='Source center (due East)',
        )
    ax_rho.scatter(
        [0.0], [0.0], marker='x', s=36, c='#542788', zorder=4,
        label='Caustic (unresolved)',
    )
    ax_rho.set_aspect('equal', adjustable='box')
    ax_rho.set_xlim(limit, -limit)
    ax_rho.set_ylim(-limit, limit)
    ax_rho.set_xlabel(r'East / $\theta_E$ (source plane)')
    ax_rho.set_ylabel(r'North / $\theta_E$')
    ax_rho.set_title(r'Source scale: $\rho \gg$ caustic')
    ax_rho.legend(loc='upper right', frameon=False, fontsize=6.5)

    colors = ['#542788', '#b2182b', '#2166ac', '#1a1a1a']
    for index, (label, points) in enumerate(caustic_epochs):
        if len(points) == 0:
            continue
        ax_cau.scatter(
            points[:, 0], points[:, 1], s=10, c=colors[index % 4],
            zorder=3, label=label,
        )
    ast_e, ast_n = rotated_astroid(half_width, axis_deg)
    ax_cau.plot(
        ast_e, ast_n, color='#e66101', lw=1.2, label='Analytic astroid',
    )
    if len(trajectory_caustic) > 0:
        ax_cau.plot(
            trajectory_caustic[:, 0], trajectory_caustic[:, 1],
            color='#1b9e77', lw=1.3, zorder=4, label='Source center',
        )
    span = 3.4 * half_width
    ax_cau.set_aspect('equal', adjustable='box')
    ax_cau.set_xlim(span, -span)
    ax_cau.set_ylim(-span, span)
    ax_cau.set_xlabel(r'East / $\theta_E$ (source plane)')
    ax_cau.set_ylabel(r'North / $\theta_E$')
    ax_cau.set_title(
        f'Central caustic, width {2.0 * half_width:.2e} '
        r'$\theta_E$'
    )
    ax_cau.legend(loc='upper right', frameon=False, fontsize=6.5)
    fig.suptitle(
        'Caustic crossing at 45$^\\circ$ to the binary axis. '
        'The axis rotates with the 129-day orbit.',
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return None


def select_central(points, radius_max=1.0e-4):
    """Keep caustic samples that belong to the central caustic.

    Parameters
    ----------
    points : ndarray, shape (M, 2)
        Caustic coordinates relative to the center of mass, in
        ``theta_E``.
    radius_max : float, optional
        Maximum radius kept. The planetary caustic for this system sits
        near ``1/s - s ~ 9``, so ``1e-4`` isolates the central branch.

    Returns
    -------
    central : ndarray, shape (K, 2)
        Central-caustic samples. Empty if none survive.
    """
    if len(points) == 0:
        return np.zeros((0, 2), dtype=float)
    radius = np.linalg.norm(points, axis=1)
    central = points[radius < radius_max]
    return central


def copy_artifacts(outdir):
    """Copy v2 products into the artifact directory with a prefix.

    Parameters
    ----------
    outdir : Path
        Directory of PNG and table files.

    Returns
    -------
    None

    Notes
    -----
    The prefix keeps the v1 artifact filenames in place.
    """
    if not ARTIFACT_DIR.is_dir():
        return None
    for path in sorted(outdir.iterdir()):
        if path.is_file():
            target = ARTIFACT_DIR / ('v2_' + path.name)
            target.write_bytes(path.read_bytes())
    return None


def main():
    """Generate the orbiting-planet Roman mock, figures, and table.

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
    d_l_pc = 1500.0
    d_s_pc = 8000.0
    a_au = 1.0
    # 1 mas at 1 kpc is 1 AU, so a [mas] = a [AU] / D_L [kpc].
    a_mas = a_au / (d_l_pc / 1000.0)

    print('Computing Roman GBTDS F146 times...', flush=True)
    t_f146, t_f087 = v1.fake_data.get_times_roman_gbtds()
    t_f146 = np.asarray(t_f146, dtype=float)
    t_f087 = np.asarray(t_f087, dtype=float)
    t_season_start, t_season_stop, n_fast, t_cross = second_fast_season(
        t_f146
    )

    # Trial model only to read theta_E and pi_rel. Those do not depend
    # on the impact parameter.
    trial = build_circorbs(57000.0, 0.01, 57000.0, a_mas, m_bh, m_planet)
    theta_e_mas = float(trial.thetaE_amp)
    pi_rel_mas = float(trial.piRel)
    pi_l_mas = float(trial.piL)
    pi_s_mas = float(trial.piS)
    period_days = float(trial.p)
    r_e_au = theta_e_mas * (d_l_pc / 1000.0)
    s_einstein = a_au / r_e_au

    print('Aiming the source through the center of mass...', flush=True)
    orbit, t0_com, beta_com, miss_mas = aim_through_com(
        t_cross, pi_rel_mas, a_mas, m_bh, m_planet
    )
    print(f'Geocentric miss from COM at t_cross: {miss_mas:.3e} mas')
    if miss_mas > 1.0e-6:
        raise RuntimeError('Source was not aimed through the caustic.')

    evaluated_cross = v1.evaluate_event(orbit, np.array([t_cross]))
    alpha_cross = axis_angle_deg(
        evaluated_cross['primary_arcsec'][0],
        evaluated_cross['secondary_arcsec'][0],
    )
    # Proper motion is due East, position angle 90 deg.
    trajectory_deg = 90.0
    angle_to_axis = angle_between_deg(trajectory_deg, alpha_cross)
    if abs(angle_to_axis - 45.0) > 0.05:
        raise RuntimeError(
            f'Binary axis is {angle_to_axis:.3f} deg from the trajectory.'
        )

    # Confirm a point-source caustic crossing: four images inside a
    # window narrower than the Roman cadence.
    fine = t_cross + np.array([-1.0e-7, 0.0, 1.0e-7])
    fine_eval = v1.evaluate_event(orbit, fine)
    n_inside = int(np.max(fine_eval['n_images']))
    print(f'Image count within 1e-7 day of the crossing: {n_inside}')
    if n_inside < 4:
        raise RuntimeError('Orbital model did not produce a caustic crossing.')

    print(f'Evaluating CircOrbs on {len(t_f146)} Roman epochs...', flush=True)
    roman = v1.evaluate_event(orbit, t_f146)
    t_model = v1.model_time_grid(
        t_cross, float(t_f146.min()), float(t_f146.max())
    )
    print(f'Evaluating CircOrbs on {len(t_model)} model times...', flush=True)
    curve = v1.evaluate_event(orbit, t_model)

    def point_lens_from(evaluated):
        delta_mas = (
            evaluated['source_arcsec'] - evaluated['primary_arcsec']
        ) * 1.0e3
        u_einstein = np.linalg.norm(delta_mas, axis=1) / theta_e_mas
        magnification = v1.point_lens_magnification(u_einstein)
        return magnification, u_einstein

    mag_src = 19.0
    a_point_model, u_model = point_lens_from(curve)
    mag_point_model = mag_src - 2.5 * np.log10(a_point_model)
    a_point_roman, u_roman = point_lens_from(roman)
    mag_point_roman = mag_src - 2.5 * np.log10(a_point_roman)

    mag_noisy, mag_err = v1.add_roman_photometric_noise(
        roman['mag'], rng_seed=146
    )
    _formula_err, ast_err_mas = v1.roman_f146_uncertainties(roman['mag'])
    rng = np.random.default_rng(146)
    ast_noise = rng.normal(
        scale=ast_err_mas[:, None], size=roman['centroid_arcsec'].shape
    )
    data_centroid = roman['centroid_arcsec'] + ast_noise * 1.0e-3

    i_peak = int(np.argmin(np.abs(t_model - t_cross)))
    origin_arcsec = curve['primary_arcsec'][i_peak]

    def to_mas(arcsec):
        return (arcsec - origin_arcsec) * 1.0e3

    source_mas = to_mas(curve['source_arcsec'])
    centroid_mas = to_mas(curve['centroid_arcsec'])
    primary_mas = to_mas(curve['primary_arcsec'])
    point_centroid = v1.point_lens_centroid(
        curve['source_arcsec'], curve['primary_arcsec'], theta_e_mas
    )
    point_mas = to_mas(point_centroid)
    data_mas = to_mas(data_centroid)

    delta_mag = roman['mag'] - mag_point_roman
    residual_max = float(np.nanmax(np.abs(delta_mag)))
    chi2_phot = float(np.nansum((delta_mag / mag_err) ** 2))
    theta_star_mas = v1.source_angular_radius_mas(d_s_pc, 1.0)
    sep_bh_mas = u_roman * theta_e_mas
    outside_star = sep_bh_mas > theta_star_mas
    residual_outside = float(np.nanmax(np.abs(delta_mag[outside_star])))
    chi2_outside = float(
        np.nansum((delta_mag[outside_star] / mag_err[outside_star]) ** 2)
    )
    n_inside_star = int(np.sum(~outside_star))

    delta_pos_mas = (
        roman['centroid_arcsec'] - v1.point_lens_centroid(
            roman['source_arcsec'], roman['primary_arcsec'], theta_e_mas
        )
    ) * 1.0e3
    residual_pos_max = float(np.linalg.norm(delta_pos_mas, axis=1).max())
    chi2_ast = float(np.nansum((delta_pos_mas / ast_err_mas[:, None]) ** 2))
    shift_mas = np.linalg.norm(centroid_mas - source_mas, axis=1)
    shift_ok = np.abs(t_model - t_cross) > 1.0e-4
    max_shift_mas = float(np.nanmax(shift_mas[shift_ok]))

    trustworthy = u_roman > 1.0e-6
    frac = np.abs(
        roman['magnification'][trustworthy] - a_point_roman[trustworthy]
    ) / a_point_roman[trustworthy]
    max_frac = float(np.nanmax(frac))

    # Frozen static binary at the crossing epoch.
    print('Comparing to a static binary frozen at t0...', flush=True)
    static, sep_mas, _alpha_static = build_static_match(
        orbit, t_cross, m_bh, m_planet, pi_rel_mas
    )
    # A day on either side, skipping the unconverged caustic sample.
    t_cmp = t_cross + np.linspace(-1.0, 1.0, 401)
    t_cmp = t_cmp[np.abs(t_cmp - t_cross) > 1.0e-4]
    orbit_cmp = v1.evaluate_event(orbit, t_cmp)
    static_cmp = v1.evaluate_event(static, t_cmp)
    dmag_static = np.abs(orbit_cmp['mag'] - static_cmp['mag'])
    max_dmag_static = float(np.nanmax(dmag_static))
    drel = (
        (orbit_cmp['source_arcsec'] - orbit_cmp['primary_arcsec'])
        - (static_cmp['source_arcsec'] - static_cmp['primary_arcsec'])
    ) * 1.0e3
    max_drel_mas = float(np.linalg.norm(drel, axis=1).max())
    # Planet position one quarter period later: the static model holds
    # it fixed, the orbit model has rotated it by 90 degrees.
    t_quarter = np.array([t_cross + 0.25 * period_days])
    orbit_q = v1.evaluate_event(orbit, t_quarter)
    static_q = v1.evaluate_event(static, t_quarter)
    planet_orbit = (
        orbit_q['secondary_arcsec'][0] - orbit_q['primary_arcsec'][0]
    )
    planet_static = (
        static_q['secondary_arcsec'][0] - static_q['primary_arcsec'][0]
    )
    planet_shift_mas = float(
        np.linalg.norm((planet_orbit - planet_static) * 1.0e3)
    )
    alpha_quarter = axis_angle_deg(
        orbit_q['primary_arcsec'][0], orbit_q['secondary_arcsec'][0]
    )

    print('Computing caustics...', flush=True)
    caustic_now = select_central(
        v1.caustic_points_theta_e(orbit, t_cross, n_pts=2000)
    )
    if len(caustic_now) == 0:
        raise RuntimeError('BAGLE did not return a central caustic.')
    width_east = float(caustic_now[:, 0].max() - caustic_now[:, 0].min())
    width_north = float(caustic_now[:, 1].max() - caustic_now[:, 1].min())
    # Cusp-to-center distance from the most distant central-caustic sample.
    half_width = float(np.linalg.norm(caustic_now, axis=1).max())
    width_analytic = v1.central_caustic_width_theta_e(q_mass, s_einstein)
    # P/4 is a 90 degree turn. The four-cusp caustic looks the same
    # after that rotation, so the figure uses P/8 (45 degrees).
    caustic_minus = select_central(
        v1.caustic_points_theta_e(
            orbit, t_cross - 0.125 * period_days, n_pts=1200
        )
    )
    caustic_plus = select_central(
        v1.caustic_points_theta_e(
            orbit, t_cross + 0.125 * period_days, n_pts=1200
        )
    )
    critical = v1.critical_curve_theta_e(orbit, t_cross, n_pts=900)

    # Planetary-caustic distance. Those branches sit near 1/s - s.
    all_caustic = v1.caustic_points_theta_e(orbit, t_cross, n_pts=1500)
    far_radius = np.linalg.norm(all_caustic, axis=1)
    planetary = all_caustic[far_radius > 1.0]
    if len(planetary):
        planetary_radius = float(np.median(np.linalg.norm(planetary, axis=1)))
    else:
        planetary_radius = np.nan
    # Source track over the survey, in theta_E relative to the COM.
    com_track = v1.source_minus_com_mas(curve, m_bh, m_planet) / theta_e_mas
    if len(planetary):
        # One epoch is enough for a lower bound: the caustic orbits at
        # fixed radius, and the source's closest survey sample is the
        # minimum over the model track to that radius.
        source_radius = np.linalg.norm(com_track, axis=1)
        miss_planetary = float(np.min(np.abs(source_radius - planetary_radius)))
    else:
        miss_planetary = np.nan

    rho = theta_star_mas / theta_e_mas
    t_e_days = float(orbit.tE)
    t_disk = rho * t_e_days
    duration_caustic_days = (2.0 * half_width) * t_e_days
    a_fs_peak = v1.uniform_disk_peak_magnification(rho)
    mag_fs_peak = mag_src - 2.5 * np.log10(a_fs_peak)
    mu_rel = float(orbit.muRel_amp)
    pi_e_amp = float(orbit.piE_amp)
    u0_bary = float(orbit.u0_amp_com)
    min_sep_bh_mas = float(np.min(sep_bh_mas))

    rho_window = np.abs(t_model - t_cross) < max(0.2, 6.0 * t_disk)
    t_caustic = t_cross + np.linspace(-2.0e-5, 2.0e-5, 41)
    caustic_eval = v1.evaluate_event(orbit, t_caustic)
    track_caustic = (
        v1.source_minus_com_mas(caustic_eval, m_bh, m_planet) / theta_e_mas
    )
    planet_theta = (
        evaluated_cross['secondary_arcsec'][0]
        - evaluated_cross['primary_arcsec'][0]
    ) * 1.0e3 / theta_e_mas

    peak_note = (
        f'1 Rsun uniform-disk point-lens cap: '
        f'A ≈ {a_fs_peak:.0f}, F146 ≈ {mag_fs_peak:.2f}. '
        'Not a BAGLE model (no FSBL).'
    )
    _mag_err_19, ast_err_19 = v1.roman_f146_uncertainties(np.array([19.0]))
    _dummy, mag_err_19 = v1.add_roman_photometric_noise(
        np.array([19.0, 19.0]), rng_seed=1
    )
    mag_err_19_value = float(np.median(mag_err_19))
    in_season = (t_f146 >= t_season_start) & (t_f146 <= t_season_stop)
    fast_dt_min = float(
        np.median(np.diff(np.sort(t_f146[in_season]))) * 1440.0
    )

    print('Writing figures...', flush=True)
    plot_photometry(
        t_f146, mag_noisy, mag_err, t_model, curve['mag'], mag_point_model,
        t_cross, t_disk, peak_note, residual_max,
        OUTDIR / 'fig_photometry_f146.png',
    )
    v1.plot_astrometry(
        t_model, source_mas, centroid_mas, point_mas, primary_mas,
        t_f146, data_mas, ast_err_mas, t_cross,
        OUTDIR / 'fig_astrometry.png',
    )
    plot_caustics(
        critical,
        [
            ('BAGLE at $t_0$', caustic_now),
            ('$t_0 - P/8$', caustic_minus),
            ('$t_0 + P/8$', caustic_plus),
        ],
        com_track[rho_window],
        track_caustic,
        rho,
        s_einstein,
        alpha_cross,
        half_width,
        planet_theta,
        OUTDIR / 'fig_caustic.png',
    )

    t_cross_iso = Time(t_cross, format='mjd').isot
    t0_iso = Time(t0_com, format='mjd').isot
    t_geom_iso = Time(float(orbit.t0), format='mjd').isot
    rows = [
        ('Model class', 'PSBL_PhotAstrom_Par_CircOrbs_Param1'),
        ('Orbit', 'circular Keplerian, face-on, e = 0'),
        ('Why not LinOrbs/AccOrbs', 'P ~ 129 d is not << t_E'),
        ('Finite-source binary lens', 'unavailable (FSBL commented out)'),
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
        ('Inclination (deg)', f'{INCLINATION_DEG:.1f} (face-on)'),
        ('omega_pri (deg)', f'{OMEGA_PRI_DEG:.1f}'),
        ('big_omega_sec (deg)', f'{BIG_OMEGA_DEG:.1f}'),
        ('tp', 'equals the geocentric crossing'),
        ('Projection', 'face-on; a_proj = a_3D at all t'),
        ('s = a / r_E', f'{s_einstein:.6e}'),
        ('sep at t0 (mas)', f'{sep_mas:.6e}'),
        ('alpha at crossing (deg E of N)', f'{alpha_cross:.4f}'),
        ('mu direction (deg E of N)', f'{trajectory_deg:.1f}'),
        ('angle(trajectory, axis) at t0', f'{angle_to_axis:.4f} deg'),
        ('alpha at t0+P/4 (deg)', f'{alpha_quarter:.4f}'),
        ('b_sff', '1 (no blend, no lens flux)'),
        ('dmag_Lp_Ls', '0 (both components dark)'),
        ('mag_src F146W', '19'),
        ('mu_rel (mas/yr)', f'{mu_rel:.1f} due East'),
        ('t0 geocentric crossing (MJD)', f'{t_cross:.5f} ({t_cross_iso})'),
        ('Season', '2nd fast window, 2027 Aug 13 to Oct 26'),
        ('t0_com barycentric (MJD)', f'{t0_com:.5f} ({t0_iso})'),
        ('t0 geometric barycentric (MJD)', f'{float(orbit.t0):.5f} ({t_geom_iso})'),
        ('45 deg convention', 'at the geocentric crossing, not t0_com'),
        ('beta_com (mas)', f'{beta_com:.6e}'),
        ('u0_com barycentric', f'{u0_bary:.6e}'),
        ('miss from COM at crossing (mas)', f'{miss_mas:.6e}'),
        ('min sep from BH, Roman grid (mas)', f'{min_sep_bh_mas:.6e}'),
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
        ('BAGLE caustic East extent (theta_E)', f'{width_east:.6e}'),
        ('BAGLE caustic North extent (theta_E)', f'{width_north:.6e}'),
        ('BAGLE cusp radius (theta_E)', f'{half_width:.6e}'),
        ('point-source caustic duration (s)', f'{duration_caustic_days * 86400.0:.4f}'),
        ('source-radius crossing (days)', f'{t_disk:.5f}'),
        ('finite-source peak A (point lens)', f'{a_fs_peak:.1f}'),
        ('finite-source peak F146 (mag)', f'{mag_fs_peak:.3f}'),
        ('planet period (days)', f'{period_days:.4f}'),
        ('orbits per t_E', f'{t_e_days / period_days:.3f}'),
        ('orbits in the 5-year survey', f'{5.0 * 365.25 / period_days:.2f}'),
        ('N F146 in the 2nd fast season', f'{n_fast}'),
        ('N F146 epochs', f'{len(t_f146)}'),
        ('N F087 epochs (not simulated)', f'{len(t_f087)}'),
        ('fast-cadence median (minutes)', f'{fast_dt_min:.3f}'),
        ('F146 mag err at mag 19', f'{mag_err_19_value:.5f}'),
        ('F146 ast err at mag 19 (mas)', f'{float(ast_err_19[0]):.4f}'),
        ('max |A_PSBL-A_point|/A outside caustic', f'{max_frac:.3e}'),
        ('max |dm| planet vs point lens (mag)', f'{residual_max:.6e}'),
        ('chi2 phot of that residual', f'{chi2_phot:.6e}'),
        ('Roman epochs inside the stellar disk', f'{n_inside_star}'),
        ('max |dm| outside the stellar disk (mag)', f'{residual_outside:.6e}'),
        ('chi2 phot outside the stellar disk', f'{chi2_outside:.6e}'),
        ('max |dpos| planet vs point lens (mas)', f'{residual_pos_max:.6e}'),
        ('chi2 ast of that residual', f'{chi2_ast:.6e}'),
        ('peak BH centroid shift (mas)', f'{max_shift_mas:.4f}'),
        ('max |dm| vs static within 1 day (mag)', f'{max_dmag_static:.6e}'),
        ('max |d(source-BH)| vs static (mas)', f'{max_drel_mas:.6e}'),
        ('planet shift vs static at P/4 (mas)', f'{planet_shift_mas:.4f}'),
        ('planetary caustic radius (theta_E)', f'{planetary_radius:.4f}'),
        ('min |u_source - r_caustic| (theta_E)', f'{miss_planetary:.4f}'),
        ('images within 1e-7 day of crossing', f'{n_inside}'),
        ('Notes', 'Angle is defined at the geocentric crossing.'),
        ('Notes 2', 'Axis rotates by 360 deg each orbital period.'),
        ('Notes 3', '1e-5 factor in test_roman_lightcurve was not applied.'),
    ]
    v1.write_tables(rows, OUTDIR)
    # Taller than the v1 table: this run has the orbit parameters too.
    fig, axis = plt.subplots(figsize=(8.8, 12.2))
    axis.axis('off')
    lines = [f'{label:<46s} {value}' for label, value in rows]
    axis.text(
        0.01, 0.995, '\n'.join(lines), va='top', ha='left',
        family='monospace', fontsize=7.6, transform=axis.transAxes,
    )
    axis.set_title('Input and derived parameters', loc='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fig_parameter_table.png')
    plt.close(fig)
    copy_artifacts(OUTDIR)

    print('angle to axis deg', angle_to_axis, 'alpha', alpha_cross)
    print('max |dm|', residual_max, 'chi2', chi2_phot)
    print('outside star |dm|', residual_outside, 'chi2', chi2_outside)
    print('max |dpos|', residual_pos_max, 'chi2_ast', chi2_ast)
    print('vs static |dm|', max_dmag_static, 'planet shift', planet_shift_mas)
    print('Wrote', OUTDIR)
    return None


if __name__ == '__main__':
    main()
