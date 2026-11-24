import math

import numpy as np
import joblib
import os
from astropy import units, units as u
from astropy.coordinates import SkyCoord, get_body_barycentric, get_body_barycentric_posvel, solar_system_ephemeris, \
    CartesianRepresentation
from astropy.time import Time

# Use the JPL DE440s ephemeris (2020; covers 1849-2150, ~32 MB download on
# first use). astropy's 'jpl' alias still means the older DE430.
solar_system_ephemeris.set('de440s')

# Setup a parallax cache
try:
    cache_dir = os.environ['PARALLAX_CACHE_DIR']
except:
    cache_dir = os.path.dirname(__file__) + '/parallax_cache/'

# Setup cache with max size of 1 GB
if joblib.__version__ < '1.5':
    cache_memory = joblib.Memory(cache_dir, verbose=0, bytes_limit='1G')
    cache_memory.reduce_size()
else:
    cache_memory = joblib.Memory(cache_dir, verbose=0)
    cache_memory.reduce_size(bytes_limit='1G')

@cache_memory.cache()
def parallax_in_direction(RA, Dec, mjd, obsLocation='earth'):
    """
    Observer parallax vector in the ICRS East/North tangent plane.

    The input time is a BAGLE MJD on the TDB scale. The JPL ephemeris is
    evaluated at ``JD_TDB = MJD + 2400000.5``. The returned vector is the
    barycentric position of the observer, from the observer toward the
    Solar System barycenter, in AU.

    Parameters
    ----------
    RA : float
        Right ascension in degrees, ICRS (J2000).
    Dec : float
        Declination in degrees, ICRS (J2000).
    mjd : float or array_like
        Time in TDB MJD.
    obsLocation : str, optional
        Observer body. ``'earth'`` or a JPL Horizons body name.

    Returns
    -------
    pvec : numpy array
        Parallax vector ``[E, N]`` in AU. Shape ``(2,)`` for a scalar time
        and ``(N_times, 2)`` for an array. Equations follow MulensModel,
        in the barycentric convention.
    """
    # Ephemeris version 3: JPL DE440s for astropy bodies; Horizons observers
    # are ICRF-equatorial and Hermite-interpolated on a fixed grid. joblib
    # clears this function's disk cache when its source changes, so bump
    # this version whenever get_observer_barycentric changes its output.

    # Munge inputs into astropy format.
    times = Time(mjd + 2400000.5, format='jd', scale='tdb')
    coord = SkyCoord(RA, Dec, unit=(units.deg, units.deg))

    direction = coord.cartesian.xyz.value
    north = np.array([0., 0., 1.])
    _east_projected = np.cross(north, direction) / np.linalg.norm(np.cross(north, direction))
    _north_projected = np.cross(direction, _east_projected) / np.linalg.norm(np.cross(direction, _east_projected))

    obs_pos = get_observer_barycentric(obsLocation, times)

    # Old Code for Heliocentric convention
    #sun_pos = get_body_barycentric(body='sun', time=times)
    #sun_obs_pos = sun_pos - obs_pos
    #pos = sun_obs_pos.xyz.T.to(units.au)

    # New Code for Barycentric convention
    bary_obs_pos = -obs_pos
    pos = bary_obs_pos.xyz.T.to(units.au)

    e = np.dot(pos, _east_projected)
    n = np.dot(pos, _north_projected)

    pvec = np.array([e.value, n.value]).T

    return pvec


def dparallax_dt_in_direction(RA, Dec, mjd, obsLocation='earth'):
    """
    Time derivative of the East/North parallax vector.

    Parameters
    ----------
    RA : float
        Right ascension in degrees, ICRS (J2000).
    Dec : float
        Declination in degrees, ICRS (J2000).
    mjd : float or array_like
        Time in TDB MJD. The ephemeris is evaluated at that TDB instant.
    obsLocation : str, optional
        Observer body. ``'earth'`` or a JPL Horizons body name.

    Returns
    -------
    dpvec_dt : numpy array
        ``d(E, N) / dt`` in AU/yr. Shape matches ``parallax_in_direction``.
        Uses the same barycentric convention as ``parallax_in_direction``
        (vector from the observer toward the Solar System barycenter), so
        this is the exact time derivative of that function's output.
    """
    # Munge inputs into astropy format.
    times = Time(mjd + 2400000.5, format='jd', scale='tdb')
    coord = SkyCoord(RA, Dec, unit=(units.deg, units.deg))

    direction = coord.cartesian.xyz.value
    north = np.array([0., 0., 1.])
    _east_projected = np.cross(north, direction) / np.linalg.norm(np.cross(north, direction))
    _north_projected = np.cross(direction, _east_projected) / np.linalg.norm(np.cross(direction, _east_projected))

    obs_vel = get_observer_barycentric(obsLocation, times, velocity=True)[1]

    # Barycentric convention, matching parallax_in_direction (pos = -obs_pos).
    # The old heliocentric form (v_sun - v_obs) differed by the Sun's
    # barycentric velocity (~10 m/s).
    vel = (-obs_vel).xyz.T.to(units.au / units.year)

    e = np.dot(vel, _east_projected)
    n = np.dot(vel, _north_projected)

    dpvec_dt = np.array([e.value, n.value]).T

    return dpvec_dt


def get_observer_barycentric(body, times, ephem_step=1.0, velocity=False):
    """
    Barycentric ICRS position (and optionally velocity) of an observer.

    Bodies in astropy's JPL ephemeris (e.g. ``'earth'``) are evaluated
    directly with ``get_body_barycentric(_posvel)``. Any other body (e.g. a
    spacecraft such as JWST, ``'-170'``) is fetched from JPL Horizons on a
    fixed, fine time grid spanning ``times`` and interpolated onto ``times``
    with a cubic Hermite spline that uses the Horizons velocities as the
    derivatives.

    Parameters
    ----------
    body : str
        Observer body. An astropy solar-system body name, or any JPL
        Horizons target id/name.
    times : astropy.time.Time
        Scalar or array of times. Converted to TDB internally.
    ephem_step : float, optional
        Horizons grid step in days (default 1.0). The grid is fixed and does
        NOT depend on the spacing of ``times``. A 1-day grid with Hermite
        interpolation reproduces Earth's orbit to well under 1 km. Use a
        smaller step only for observers on short-period orbits (e.g. LEO).
    velocity : bool, optional
        If True, also return the velocity.

    Returns
    -------
    pos : astropy.coordinates.CartesianRepresentation
        Barycentric ICRS (equatorial) position, same shape as ``times``.
    vel : astropy.coordinates.CartesianRepresentation
        Only if ``velocity=True``. Barycentric ICRS velocity.
    """
    # Capitalized Earth is the same builtin body. Do not alias
    # 'l2' or a Roman Horizons id; those stay caller-supplied.
    _observer_aliases = {'Earth': 'earth', 'EARTH': 'earth'}
    if isinstance(body, str):
        body = _observer_aliases.get(body, body)

    if body in solar_system_ephemeris.bodies:
        if velocity:
            return get_body_barycentric_posvel(body=body, time=times)
        return get_body_barycentric(body=body, time=times)

    pos, vel = _horizons_barycentric_posvel(body, times, ephem_step)
    return (pos, vel) if velocity else pos


def _horizons_barycentric_posvel(body, times, ephem_step=1.0):
    """
    Fetch a Horizons state-vector table for ``body`` and interpolate it.

    The table is barycentric (``location='@0'``) and in the ICRF equatorial
    frame (``refplane='earth'``, i.e. Horizons REF_PLANE=FRAME), matching
    astropy's ``get_body_barycentric``. astroquery's default
    ``refplane='ecliptic'`` would be rotated 23.4 deg about x.
    """
    from astroquery.jplhorizons import Horizons
    from scipy.interpolate import CubicHermiteSpline

    jd = times.tdb.jd
    jd_flat = np.atleast_1d(jd).ravel()

    # Fixed grid, padded by two steps so every requested time has spline
    # support on both sides. Horizons vector tables are in TDB.
    step_min = max(1, int(round(ephem_step * 1440)))
    pad = 2 * step_min / 1440.0
    start = Time(jd_flat.min() - pad, format='jd', scale='tdb').iso
    stop = Time(jd_flat.max() + pad, format='jd', scale='tdb').iso

    obj = Horizons(id=body, location='@0',
                   epochs={'start': start, 'stop': stop, 'step': f'{step_min}m'})
    tab = obj.vectors(refplane='earth')

    t_tab = np.asarray(tab['datetime_jd'], dtype=float)
    xyz = np.column_stack([tab[c].to(u.au).value for c in ('x', 'y', 'z')])
    vxyz = np.column_stack(
        [tab[c].to(u.au / u.day).value for c in ('vx', 'vy', 'vz')])

    if jd_flat.min() < t_tab[0] or jd_flat.max() > t_tab[-1]:
        raise ValueError(f'Horizons table for {body!r} does not cover the '
                         f'requested times ({t_tab[0]}-{t_tab[-1]} JD TDB).')

    spline = CubicHermiteSpline(t_tab, xyz, vxyz, axis=0)
    pos = spline(jd_flat)                 # AU
    vel = spline.derivative()(jd_flat)    # AU/day

    # Restore the input shape (scalar or array).
    shape = np.shape(jd)
    pos_rep = CartesianRepresentation(
        *(pos[:, i].reshape(shape) for i in range(3)), unit=u.au)
    vel_rep = CartesianRepresentation(
        *(vel[:, i].reshape(shape) for i in range(3)), unit=u.au / u.day)
    return pos_rep, vel_rep


def sun_position(mjd, radians=False):
    """

    NAME:
          SUNPOS

    PURPOSE:
          To compute the RA and Dec of the Sun at a given date.

    INPUTS:
          mjd    - The modified Julian date of the day (and time), scalar or vector

    OUTPUTS:
          ra:
              | The right ascension of the sun at that date in DEGREES
              | double precision, same number of elements as jd
          dec:
              The declination of the sun at that date in DEGREES
          elong:
              Ecliptic longitude of the sun at that date in DEGREES.
          obliquity:
              the obliquity of the ecliptic, in DEGREES

    OPTIONAL INPUT KEYWORD:
          RADIAN [def=False] - If this keyword is set to True, then all output variables
          are given in Radians rather than Degrees

    NOTES:
          Patrick Wallace (Rutherford Appleton Laboratory, UK) has tested the
          accuracy of a C adaptation of the sunpos.pro code and found the
          following results.   From 1900-2100 SUNPOS  gave 7.3 arcsec maximum
          error, 2.6 arcsec RMS.  Over the shorter interval 1950-2050 the figures
          were 6.4 arcsec max, 2.2 arcsec RMS.

          The returned RA and Dec are in the given date's equinox.

          Procedure was extensively revised in May 1996, and the new calling
          sequence is incompatible with the old one.
    METHOD:
          Uses a truncated version of Newcomb's Sun.    Adapted from the IDL
          routine SUN_POS by CD Pike, which was adapted from a FORTRAN routine
          by B. Emerson (RGO).
    EXAMPLE:
          (1) Find the apparent RA and Dec of the Sun on May 1, 1982

          | IDL> jdcnv, 1982, 5, 1,0 ,jd      ;Find Julian date jd = 2445090.5
          | IDL> sunpos, jd, ra, dec
          | IDL> print,adstring(ra,dec,2)
          | 02 31 32.61  +14 54 34.9

          The Astronomical Almanac gives 02 31 32.58 +14 54 34.9 so the error
          in SUNPOS for this case is < 0.5".

          (2) Find the apparent RA and Dec of the Sun for every day in 1997

          | IDL> jdcnv, 1997,1,1,0, jd                ;Julian date on Jan 1, 1997
          | IDL> sunpos, jd+ dindgen(365), ra, dec    ;RA and Dec for each day

    MODIFICATION HISTORY:

          * Written by Michael R. Greason, STX, 28 October 1988.
          * Accept vector arguments, W. Landsman -     April,1989
          * Eliminated negative right ascensions - MRG, Hughes STX, 6 May 1992.
          * Rewritten using the 1993 Almanac.  Keywords added.  MRG, HSTX, 10 February 1994.
          * Major rewrite, improved accuracy, always return values in degrees - W. Landsman May, 1996
          * Added /RADIAN keyword; W. Landsman; August, 1997
          * Converted to IDL V5.0; W. Landsman; September 1997
          * Converted to python; J. R. Lu; August 2016
    """
    #  form time in Julian centuries from 1900.0
    t_obj = Time(mjd, format='mjd')
    t = (t_obj.jd - 2415020.0) / 36525.0

    #  form sun's mean longitude
    l = (279.696678 + ((36000.768925 * t) % 360.0)) * 3600.0

    #  allow for ellipticity of the orbit (equation of centre)
    #  using the Earth's mean anomaly ME
    me = 358.475844 + ((35999.049750 * t) % 360.0)
    ellcor = (6910.1 - 17.2 * t) * np.sin(np.radians(me)) + 72.3 * np.sin(
        np.radians(2.0 * me))
    l = l + ellcor

    # allow for the Venus perturbations using the mean anomaly of Venus MV
    mv = 212.603219 + ((58517.803875 * t) % 360.0)
    vencorr = 4.8 * np.cos(np.radians(299.1017 + mv - me)) + \
              5.5 * np.cos(np.radians(148.3133 + 2.0 * mv - 2.0 * me)) + \
              2.5 * np.cos(np.radians(315.9433 + 2.0 * mv - 3.0 * me)) + \
              1.6 * np.cos(np.radians(345.2533 + 3.0 * mv - 4.0 * me)) + \
              1.0 * np.cos(np.radians(318.1500 + 3.0 * mv - 5.0 * me))
    l += vencorr

    #  Allow for the Mars perturbations using the mean anomaly of Mars MM
    mm = 319.529425 + ((19139.858500 * t) % 360.0)
    marscorr = 2.0 * np.cos(np.radians(343.8883 - 2.0 * mm + 2.0 * me)) + \
               1.8 * np.cos(np.radians(200.4017 - 2.0 * mm + me))
    l += marscorr

    # Allow for the Jupiter perturbations using the mean anomaly of Jupiter MJ
    mj = 225.328328 + ((3034.6920239 * t) % 360.0)
    jupcorr = 7.2 * np.cos(np.radians(179.5317 - mj + me)) + \
              2.6 * np.cos(np.radians(263.2167 - mj)) + \
              2.7 * np.cos(np.radians(87.1450 - 2.0 * mj + 2.0 * me)) + \
              1.6 * np.cos(np.radians(109.4933 - 2.0 * mj + me))
    l += jupcorr

    # Allow for the Moons perturbations using the mean elongation of
    # the Moon from the Sun D
    d = 350.7376814 + ((445267.11422 * t) % 360.0)
    mooncorr = 6.5 * np.sin(np.radians(d))
    l += mooncorr

    # Allow for long period terms
    longterm = + 6.4 * np.sin(np.radians(231.19 + 20.20 * t))
    l += longterm
    l = (l + 2592000.0) % 1296000.0
    longmed = l / 3600.0

    # Allow for Aberration
    l -= 20.5

    # Allow for Nutation using the longitude of the Moons mean node OMEGA
    omega = 259.183275 - ((1934.142008 * t) % 360.0)
    l -= 17.2 * np.sin(np.radians(omega))

    # Form the True Obliquity
    oblt = 23.452294 - 0.0130125 * t + (
            9.2 * np.cos(np.radians(omega))) / 3600.0

    # Form Right Ascension and Declination
    l = l / 3600.0
    l_rad = np.radians(l)
    oblt_rad = np.radians(oblt)
    ra = np.arctan2(np.sin(l_rad) * np.cos(oblt_rad), np.cos(l_rad))

    if (len(ra) > 1):
        neg = np.where(ra < 0.0)[0]
        ra[neg] = ra[neg] + 2.0 * math.pi

    dec = np.arcsin(np.sin(l_rad) * np.sin(oblt_rad))

    if radians:
        oblt = oblt_rad
        longmed = np.radians(longmed)
    else:
        ra = np.degrees(ra)
        dec = np.degrees(dec)

    return ra, dec, longmed, oblt
