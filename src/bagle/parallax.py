import math

import numpy as np
import joblib
import os
from astropy import units, units as u
from astropy.coordinates import SkyCoord, get_body_barycentric, get_body_barycentric_posvel, solar_system_ephemeris, \
    CartesianRepresentation
from astropy.time import Time

# Use the JPL ephemerides.
solar_system_ephemeris.set('jpl')

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
    | R.A. in degrees. (J2000)
    | Dec. in degrees. (J2000)
    | MJD

    Equations following MulensModel.
    """
    #print('parallax_in_direction: len(t) = ', len(mjd))

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
    R.A. in degrees. (J2000)
    Dec. in degrees. (J2000)
    MJD

    Equations following MulensModel.
    Time derivative --> units are yr^-1

    """
    # print('parallax_in_direction: len(t) = ', len(mjd))
    # Munge inputs into astropy format.
    times = Time(mjd + 2400000.5, format='jd', scale='tdb')
    coord = SkyCoord(RA, Dec, unit=(units.deg, units.deg))

    direction = coord.cartesian.xyz.value
    import pdb
    north = np.array([0., 0., 1.])
    _east_projected = np.cross(north, direction) / np.linalg.norm(np.cross(north, direction))
    _north_projected = np.cross(direction, _east_projected) / np.linalg.norm(np.cross(direction, _east_projected))

    obs_posvel = get_observer_barycentric(obsLocation, times, velocity=True)[1]
    sun_posvel = get_body_barycentric_posvel('Sun', times)[1]
    sun_obs_vel = sun_posvel - obs_posvel
    #pdb.set_trace()

    vel = sun_obs_vel.xyz.T.to(units.au / units.year)

    e = np.dot(vel, _east_projected)
    n = np.dot(vel, _north_projected)

    dpvec_dt = np.array([e.value, n.value]).T

    return dpvec_dt


def get_observer_barycentric(body, times, min_ephem_step=1, velocity=False):
    """
    Get the barycentric position of a satellite or other Solar System body
    using JPL emphemerides through the Horizon app.

    The ephemeris is queried at a decimated time step set by min_ephem_step
    (def=1 day) that must be 1 day or larger. The positions
    (and optionally velocities) are then interpolated onto the desired
    time array.

    Inputs
    ------
    body : str
        The name of the Solar System body. Must use the JPL Horizon
        naming scheme.

    times : astropy.time.Time array
        Array of times (astropy.time.core.Time) objects at which to
        fetch the position of the specified Solar System body. Times
        should be in the TDB scale.

    Optional Inputs
    ---------------
    min_ephem_step : int
        Minimum time step to query JPL in days. Must not be <1 and must
        be in integer days.

    veloctiy : bool
        If true, return both position and velocity vectors over time.

    Return
    ------
    coord : astropy.coordinates.CartesianRepresentation
        The xyz coordinates in the plane of the Solar System at the
        input times.
    """

    if body in solar_system_ephemeris.bodies:
        if velocity:
            obs_pos, obs_vel = get_body_barycentric_posvel(body=body, time=times)
        else:
            obs_pos = get_body_barycentric(body=body, time=times)
    else:
        # Figure out a cadence for the ephemerides, not smaller than 1 day.
        dt = np.median(np.diff(times)).jd
        if dt < min_ephem_step:
            dt = min_ephem_step

        # Get the date range, add some padding on each side.
        t_min = times.min()
        t_max = times.max()
        t_min.format = 'iso'
        t_max.format = 'iso'
        t_min = str(t_min - dt*u.day).split()[0]
        t_max = str(t_max + dt*u.day).split()[0]
        step = f'{dt:.0f}d'

        # Fetch the Horizons ephemeris.
        from astroquery.jplhorizons import Horizons
        obj = Horizons(id=body, location="@0", epochs={'start':t_min, 'stop':t_max, 'step':step})
        obj_data = obj.vectors()

        ephem_jd = obj_data['datetime_jd']

        # Interpolate to the actual time array.
        obj_x_at_t = np.interp(times.jd, ephem_jd, obj_data['x'].to('km')) * u.km
        obj_y_at_t = np.interp(times.jd, ephem_jd, obj_data['y'].to('km')) * u.km
        obj_z_at_t = np.interp(times.jd, ephem_jd, obj_data['z'].to('km')) * u.km

        if velocity:
            obj_vx_at_t = np.interp(times.jd, ephem_jd, obj_data['vx'].to('km/s')) * u.km / u.s
            obj_vy_at_t = np.interp(times.jd, ephem_jd, obj_data['vy'].to('km/s')) * u.km / u.s
            obj_vz_at_t = np.interp(times.jd, ephem_jd, obj_data['vz'].to('km/s')) * u.km / u.s

            obs_vel = CartesianRepresentation(obj_vx_at_t, obj_vy_at_t, obj_vz_at_t)

        obs_pos = CartesianRepresentation(obj_x_at_t, obj_y_at_t, obj_z_at_t)

    if velocity:
        return (obs_pos, obs_vel)
    else:
        return obs_pos


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
