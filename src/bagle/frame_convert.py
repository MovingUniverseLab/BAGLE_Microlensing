from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u
import numpy as np 
from astropy.time import Time
from astropy.coordinates.builtin_frames.utils import get_jd12
from astropy.coordinates import EarthLocation
from astropy.coordinates import SkyCoord
import erfa
import matplotlib.pyplot as plt
import math

from bagle import parallax
from matplotlib.ticker import MaxNLocator


##############
#
# Converters for Different Packages
#
##############
def get_pylima_model(bagle_model, times_mjd=None, verbose=False):
    """
    Using an input BAGLE model, return a pyLIMA model. This utility function does
    all the appropriate frame conversions.

    Parameters
    ----------
    bagle_model : BAGLE model instance
        An instance of a BAGLE model. PSPL and photometry-only FSPL
        models are supported.

    times_mjd : None or array
        An array of times in MJD that is not really used.
    verbose : bool, optional
        If `True`, print conversion mode details.

    Returns
    -------
    pylima_mod : object
        pyLIMA model object configured to match the input BAGLE model.
    pylima_par : object
        pyLIMA parameter object for `pylima_mod`.
    pylima_tel : object
        pyLIMA telescope object attached to the generated event.

    """

    from pyLIMA import event
    from pyLIMA.models import generate_model
    from pyLIMA import telescopes
    from pyLIMA.toolbox import brightness_transformation
    from pyLIMA import toolbox as microltoolbox

    # Need to mock up some observation times. Not 100% sure why
    # pyLIMA needs this.
    if times_mjd is None:
        times_mjd = np.arange(bagle_model.t0 - 2 * bagle_model.tE,
                              bagle_model.t0 + 2 * bagle_model.tE)
    times_jd = times_mjd + 2400000.5

    # Set up the reference time
    t0_par_pylima = bagle_model.t0 + 2400000.5  # t0_par in JD

    #####
    # PyLIMA Data Objects for photometry and astrometry.
    #####
    pylima_data = np.zeros((len(times_jd), 3), dtype=float)
    pylima_data[:, 0] = times_jd
    pylima_data[:, 1] = np.ones(len(times_jd)) * 1  # mag
    pylima_data[:, 2] = np.ones(len(times_jd)) * 0.01  # mag err

    pylima_data_ast = np.zeros((len(times_jd), 5), dtype=float)
    pylima_data_ast[:, 0] = times_jd
    pylima_data_ast[:, 1] = np.ones(len(times_jd)) * 1.0  # RA
    pylima_data_ast[:, 2] = np.ones(len(times_jd)) * 0.001  # RA error
    pylima_data_ast[:, 3] = np.ones(len(times_jd)) * 1.0  # Dec
    pylima_data_ast[:, 4] = np.ones(len(times_jd)) * 0.001  # Dec error

    #####
    # PyLIMA Telescope Object
    #####
    if bagle_model.astrometryFlag:
        pylima_tel = telescopes.Telescope(name='OGLE', camera_filter='I',
                                          lightcurve=pylima_data,
                                          lightcurve_names=['time', 'mag', 'err_mag'],
                                          lightcurve_units=['JD', 'mag', 'mag'],
                                          astrometry=pylima_data_ast,
                                          astrometry_names=['time', 'ra', 'err_ra', 'dec', 'err_dec'],
                                          astrometry_units=['JD', 'deg', 'deg', 'deg', 'deg'],
                                          location='Earth',
                                          altitude=1000, longitude=-109.285399, latitude=-27.130)
    else:
        pylima_tel = telescopes.Telescope(name='OGLE', camera_filter='I',
                                          lightcurve=pylima_data,
                                          lightcurve_names=['time', 'mag', 'err_mag'],
                                          lightcurve_units=['JD', 'mag', 'mag'],
                                          location='Earth',
                                          altitude=1000, longitude=-109.285399, latitude=-27.130)

    #####
    # PyLIMA Event Object
    #####
    if bagle_model.parallaxFlag:
        pylima_ev = event.Event(ra=bagle_model.raL, dec=bagle_model.decL)
    else:
        pylima_ev = event.Event()
    pylima_ev.name = 'Fubar'
    pylima_ev.telescopes.append(pylima_tel)

    #####
    # PyLIMA Model Object
    #####
    if bagle_model.parallaxFlag:
        parallax_model = ['Full', t0_par_pylima]  # assumes pylima_t0_par is t0.
    else:
        parallax_model = ['None', t0_par_pylima]

    bagle_class_name = str(bagle_model.__class__)
    if 'FSPL' in bagle_class_name:
        # Finite-source point-lens (photometry); rho follows Paczynski params.
        pylima_mod = generate_model.create_model(
            'FSPL', pylima_ev, parallax=parallax_model, fancy_parameters=None)
    elif 'PSPL' in bagle_class_name:
        pylima_mod = generate_model.create_model('PSPL', pylima_ev, parallax=parallax_model, fancy_parameters=None)
    elif 'PSBL' in bagle_class_name:
        pylima_mod = generate_model.create_model('PSBL', pylima_ev, parallax=parallax_model, fancy_parameters=None)
    else:
        raise Exception('pyLIMA does not support models with BAGLE type: ' + bagle_class_name)

    pylima_mod.define_model_parameters()
    pylima_mod.blend_flux_ratio = False
    pylima_mod.astrometry = bagle_model.astrometryFlag

    #####
    # PyLIMA Parameters Object
    #####
    ### Photometry-Only and No Parallax
    if not bagle_model.astrometryFlag and not bagle_model.parallaxFlag:
        if verbose: print('PyLIMA: Photometry-only, no parallax')

        u0_pylima = bagle_model.u0_amp
        t0_pylima = bagle_model.t0 + 2400000.5
        tE_pylima = bagle_model.tE

        # Finite-source single (rho in Einstein radii).
        if 'FSPL' in bagle_class_name:
            rho_pylima = float(bagle_model.radiusS)
            tmp_params = [t0_pylima, u0_pylima, tE_pylima, rho_pylima]
        # Point-source singles
        elif 'PSPL' in bagle_class_name:
            tmp_params = [t0_pylima, u0_pylima, tE_pylima]
        # Binary Lens
        elif 'PSBL' in bagle_class_name:
            log_s_pylima = np.log10(bagle_model.sep) # sep in thetaE
            log_q_pylima = np.log10(bagle_model.q)   # mass ratio (low-mass / high-mass)
            phi_pylima = bagle_model.alpha_rad       # angle between bin axis and muRel vec in radians

            tmp_params = [t0_pylima, u0_pylima, tE_pylima,
                          log_s_pylima, log_q_pylima, phi_pylima]

        pylima_par = pylima_mod.compute_pyLIMA_parameters(tmp_params)
        pylima_par.fsource_OGLE = microltoolbox.brightness_transformation.magnitude_to_flux(bagle_model.mag_src[0])
        pylima_par.fblend_OGLE = pylima_par.fsource_OGLE * (1.0 - bagle_model.b_sff[0]) / bagle_model.b_sff[0]

    ### Photometry-Only and With Parallax
    elif not bagle_model.astrometryFlag and bagle_model.parallaxFlag:
        if verbose: print('PyLIMA: Photometry-only, with parallax')
        # Add the RA and Dec to the model
        pylima_mod.ra = bagle_model.raL
        pylima_mod.dec = bagle_model.decL

        if 'FSPL' in bagle_class_name or 'PSPL' in bagle_class_name:

            # Convert from helio to geo-tr coordinates.
            geo_params = convert_helio_geo_phot(bagle_model.raL, bagle_model.decL,
                                                bagle_model.t0, bagle_model.u0_amp, bagle_model.tE,
                                                bagle_model.piE[0], bagle_model.piE[1], bagle_model.t0,
                                                murel_in='SL', murel_out='LS',
                                                coord_in='EN', coord_out='EN', plot=False)
            t0_pylima = geo_params[0] + 2400000.5
            u0_pylima = geo_params[1]
            tE_pylima = geo_params[2]
            piEE_pylima = geo_params[3]
            piEN_pylima = geo_params[4]

            # FSPL: [t0, u0, tE, rho, piEN, piEE]; PSPL omits rho.
            if 'FSPL' in bagle_class_name:
                rho_pylima = float(bagle_model.radiusS)
                tmp_params = [t0_pylima, u0_pylima, tE_pylima, rho_pylima,
                              piEN_pylima, piEE_pylima]
            else:
                tmp_params = [t0_pylima, u0_pylima, tE_pylima, piEN_pylima, piEE_pylima]

        # Binary Lens
        elif 'PSBL' in bagle_class_name:
            geo_params = convert_bagle_pylima_psbl_phot(bagle_model.raL, bagle_model.decL,
                                                        bagle_model.t0, bagle_model.u0_amp, bagle_model.tE,
                                                        bagle_model.piEE, bagle_model.piEN, bagle_model.t0,
                                                        murel_in='SL', murel_out='LS',
                                                        coord_in='EN', coord_out='tb',
                                                        plot=False)

            t0_pylima = geo_params[0] + 2400000.5
            u0_pylima = geo_params[1]
            tE_pylima = geo_params[2]
            piEE_pylima = geo_params[3]
            piEN_pylima = geo_params[4]
            q_pylima = geo_params[5]
            phi_pylima = geo_params[6]
            s_pylima = geo_params[7]

            tmp_params = [t0_pylima, u0_pylima, tE_pylima,
                          s_pylima, q_pylima, np.deg2rad(phi_pylima),
                          piEN_pylima, piEE_pylima]

        pylima_par = pylima_mod.compute_pyLIMA_parameters(tmp_params)

        pylima_par.fsource_OGLE = microltoolbox.brightness_transformation.magnitude_to_flux(bagle_model.mag_src[0])
        pylima_par.fblend_OGLE = pylima_par.fsource_OGLE * (1.0 - bagle_model.b_sff[0]) / bagle_model.b_sff[0]

    ### Photometry and Astrometry
    else:
        if verbose: print('PyLIMA: PSPL with Astrometry and Photometry')
        # FSPL / PSBL phot+astrom conversion is not wired for pyLIMA here.
        if 'FSPL' in bagle_class_name or 'PSBL' in bagle_class_name:
            raise Exception('pyLIMA does not support models with BAGLE type: ' + bagle_class_name)

        # Convert from helio to geo-tr coordinates.
        geo_params = convert_helio_geo_phot(bagle_model.raL, bagle_model.decL,
                                            bagle_model.t0, bagle_model.u0_amp, bagle_model.tE,
                                            bagle_model.piE[0], bagle_model.piE[1], bagle_model.t0,
                                            murel_in='SL', murel_out='LS',
                                            coord_in='EN', coord_out='EN', plot=False)

        u0_pylima = geo_params[1]
        t0_pylima = geo_params[0] + 2400000.5
        tE_pylima = geo_params[2]
        piEE_pylima = geo_params[3]
        piEN_pylima = geo_params[4]
        thetaE_pylima = bagle_model.thetaE_amp
        piS_pylima = bagle_model.piS
        muSE_pylima = bagle_model.muS[0]
        muSN_pylima = bagle_model.muS[1]
        # Filter 0. xS0 is (n_filters, 2).
        xS0E_pylima = bagle_model.xS0[0, 0] / 3600.  # degrees
        xS0N_pylima = bagle_model.xS0[0, 1] / 3600.  # degrees

        tmp_params = [t0_pylima, u0_pylima, tE_pylima, thetaE_pylima, piS_pylima, muSN_pylima, muSE_pylima,
                      xS0N_pylima, xS0E_pylima, piEN_pylima, piEE_pylima]
        pylima_par = pylima_mod.compute_pyLIMA_parameters(tmp_params)

        pylima_par.fsource_OGLE = microltoolbox.brightness_transformation.magnitude_to_flux(bagle_model.mag_src[0])
        pylima_par.fblend_OGLE = pylima_par.fsource_OGLE * (1.0 - bagle_model.b_sff[0]) / bagle_model.b_sff[0]

    return pylima_mod, pylima_par, pylima_tel


def get_mulens_model(bagle_model):
    """
    Return a MuLensModel matched to the input BAGLE model.

    No astrometry supported. Photometry-only FSPL models set ``rho``
    from BAGLE ``radiusS`` (Einstein radii) and use Gould94 uniform FS.

    Parameters
    ----------
    bagle_model : BAGLE model instance

    Returns
    -------
    mulens_mod : MuLens model
        MuLensModel model instance configured with BAGLE-equivalent parameters.
    """
    import MulensModel

    c = SkyCoord(ra=bagle_model.raL * u.deg, dec=bagle_model.decL * u.deg, frame='icrs')

    geo_params = convert_helio_geo_phot(bagle_model.raL, bagle_model.decL,
                                        bagle_model.t0, bagle_model.u0_amp, bagle_model.tE,
                                        bagle_model.piE[0], bagle_model.piE[1], bagle_model.t0,
                                        murel_in='SL', murel_out='LS',
                                        coord_in='EN', coord_out='EN', plot=False)

    # Phot-only FSPL stores source radius in Einstein radii.
    bagle_class_name = str(bagle_model.__class__)
    is_fspl = 'FSPL' in bagle_class_name
    rho_mulens = float(bagle_model.radiusS) if is_fspl else None

    if bagle_model.parallaxFlag:
        t0_mulens = geo_params[0] + 2400000.5
        u0_mulens = geo_params[1]
        tE_mulens = geo_params[2]
        piEE_mulens = geo_params[3]
        piEN_mulens = geo_params[4]

        # Set up the reference time
        t0_par_mulens = bagle_model.t0 + 2400000.5  # t0_par in JD

        mulens_params = {'t_0': t0_mulens, 'u_0': u0_mulens, 't_E': tE_mulens,
                         'pi_E_N': piEN_mulens, 'pi_E_E': piEE_mulens,
                         't_0_par': t0_par_mulens}
        if is_fspl:
            mulens_params['rho'] = rho_mulens

        mulens_mod = MulensModel.Model(mulens_params, coords=c)
    else:
        t0_mulens = bagle_model.t0 + 2400000.5
        u0_mulens = bagle_model.u0_amp
        tE_mulens = bagle_model.tE

        mulens_params = {'t_0': t0_mulens, 'u_0': u0_mulens, 't_E': tE_mulens}
        if is_fspl:
            mulens_params['rho'] = rho_mulens

        mulens_mod = MulensModel.Model(mulens_params)

    mulens_mod.parallax(satellite=False,
                        earth_orbital=bagle_model.parallaxFlag,
                        topocentric=bagle_model.parallaxFlag)

    # Enable finite-source magnification when rho is present.
    if is_fspl:
        mulens_mod.set_magnification_methods(
            [0.0, 'finite_source_uniform_Gould94', 1.0e20])

    return mulens_mod


# BinaryLightCurveKepler reads the Sun table even when both parallax
# components are zero. Shift TDB MJD onto a date inside that table.
# ``t - t0`` is unchanged when the light-curve times use the same shift,
# so the magnification does not pick up the 0.5 day HJD' offset.
VBM_ORBIT_TIME_SHIFT = 50000.0

# Tiny source radius for point-source binary and triple light curves.
# rho = 1e-6 matches point-source magnification at about 1e-8. rho = 1e-10
# is noisier in VBM's contour solver.
_VBM_POINT_SOURCE_RHO = 1e-6


def _vbm_lens_family(class_name):
    """Classify a BAGLE class name for the VBM multi-lens light curves.

    Parameters
    ----------
    class_name : str
        ``bagle_model.__class__.__name__``.

    Returns
    -------
    family : str or None
        ``'triple'``, ``'binary'``, or None for a single lens.
        Binary-source classes are None: they are not one-source binaries.
    """
    # Triple before binary: the names do not overlap, but keep the
    # order obvious.
    if 'PSTL' in class_name:
        return 'triple'

    # BFSPL / BFSBL contain the substrings FSPL / FSBL.
    if ('BFSPL' in class_name or 'BFSBL' in class_name
            or 'BSPL' in class_name or 'BSBL' in class_name):
        return None

    if 'FSBL' in class_name or 'PSBL' in class_name:
        return 'binary'

    return None


def _is_finite_source_class(class_name):
    """True for a single-source finite-source BAGLE class.

    Parameters
    ----------
    class_name : str
        ``bagle_model.__class__.__name__``.

    Returns
    -------
    finite : bool
    """
    if 'BFSPL' in class_name or 'BFSBL' in class_name:
        return False

    finite = ('FSPL' in class_name or 'FSBL' in class_name
              or 'FSTL' in class_name)

    return finite


def _vbm_orbit_kind(class_name):
    """Orbit family understood by the VBM light-curve functions.

    Parameters
    ----------
    class_name : str
        ``bagle_model.__class__.__name__``.

    Returns
    -------
    kind : str
        ``'kepler'`` (EllOrbs or CircOrbs), ``'linear'`` (LinOrbs or
        AccOrbs), or ``'static'``.
    """
    if 'EllOrbs' in class_name or 'CircOrbs' in class_name:
        return 'kepler'

    if 'LinOrbs' in class_name or 'AccOrbs' in class_name:
        return 'linear'

    return 'static'


def _vbm_rho(bagle_model, tE_bagle, tE_vbm):
    """Source radius in VBM's Einstein units.

    Point-source models get :data:`_VBM_POINT_SOURCE_RHO`. Finite-source
    models use ``source_radius_thetaE_units`` when the model has it, then
    rescale from BAGLE's Einstein radius to the total-mass radius VBM uses.

    Parameters
    ----------
    bagle_model : BAGLE model instance
    tE_bagle : float
        BAGLE Einstein timescale (days).
    tE_vbm : float
        Einstein timescale in the VBM mass normalization (days).

    Returns
    -------
    rho : float
        Source radius in VBM Einstein radii. Always positive.
    """
    class_name = bagle_model.__class__.__name__

    if not _is_finite_source_class(class_name):
        return _VBM_POINT_SOURCE_RHO

    # Native BAGLE rho, in whatever Einstein radius that model uses.
    if hasattr(bagle_model, 'source_radius_thetaE_units'):
        rho = float(bagle_model.source_radius_thetaE_units())
    elif hasattr(bagle_model, 'thetaE_amp'):
        rho = float(bagle_model.radiusS) * 1e3 / float(bagle_model.thetaE_amp)
    else:
        rho = float(bagle_model.radiusS)

    # thetaE scales with tE at fixed proper motion. Triple PhotAstrom
    # defines thetaE from the 1-2 pair; VBM uses the total mass.
    rho *= float(tE_bagle) / float(tE_vbm)

    if rho <= 0.0:
        return _VBM_POINT_SOURCE_RHO

    return rho


def _mass_fractions(bagle_model, n_lens):
    """Lens masses as fractions of the total, primary first.

    PhotAstrom models also store ``m1`` as a polynomial coefficient in
    arcsec^2. Prefer the solar masses ``mLp``, ``mLs``, ``mLt`` when
    they exist.

    Parameters
    ----------
    bagle_model : BAGLE model instance
    n_lens : int
        2 or 3.

    Returns
    -------
    mass : ndarray, shape (n_lens,)
        Fractions summing to 1.
    """
    if n_lens == 3 and hasattr(bagle_model, 'mLt'):
        mass = np.array([bagle_model.mLp, bagle_model.mLs,
                         bagle_model.mLt], dtype=float)
    elif hasattr(bagle_model, 'mLp') and hasattr(bagle_model, 'mLs'):
        mass = np.array([bagle_model.mLp, bagle_model.mLs], dtype=float)
    elif n_lens == 3:
        mass = np.array([bagle_model.m1, bagle_model.m2,
                         bagle_model.m3], dtype=float)
    else:
        mass = np.array([bagle_model.m1, bagle_model.m2], dtype=float)

    mass = mass / np.sum(mass)

    return mass


def _einstein_source_lenses(bagle_model, t, n_lens):
    """Source and lenses in VBM's total-mass Einstein frame.

    Phot-only positions are already in Einstein radii of the total mass.
    PhotAstrom positions are arcseconds. A triple PhotAstrom ``thetaE``
    is the 1-2 pair, so those coordinates are divided by the total-mass
    Einstein radius and ``tE`` is stretched by the same factor.

    Parameters
    ----------
    bagle_model : BAGLE model instance
    t : array_like
        Times (MJD).
    n_lens : int
        2 or 3.

    Returns
    -------
    src : ndarray, shape (N_times, 2)
        Unlensed source, East/North, VBM Einstein radii.
    lenses : ndarray, shape (N_times, n_lens, 2)
        Lens positions in that same frame.
    tE_vbm : float
        Einstein timescale for that frame (days).
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    tE_bagle = float(bagle_model.tE)

    # Phot-only: get_u and the lens table share one Einstein origin.
    if not bagle_model.astrometryFlag:
        lenses = np.asarray(
            bagle_model.get_resolved_lens_astrometry(t), dtype=float)
        src = np.asarray(bagle_model.get_u(t), dtype=float)
        if src.ndim == 3:
            src = src[:, 0, :]

        return src, lenses, tE_bagle

    # PhotAstrom sky positions (arcsec), relative to the lens origin.
    xL = np.asarray(
        bagle_model.get_resolved_lens_astrometry(t), dtype=float)
    xS = np.asarray(
        bagle_model.get_source_astrometry_unlensed(t), dtype=float)
    x0 = np.asarray(
        bagle_model.get_lens_origin_astrometry(t), dtype=float)
    xL = xL - x0[:, None, :]
    xS = xS - x0

    # Total-mass Einstein radius. Binary thetaE is already the total.
    thetaE_mas = float(bagle_model.thetaE_amp)
    if n_lens == 3 and hasattr(bagle_model, 'mLt'):
        m_tot = float(bagle_model.mLp + bagle_model.mLs + bagle_model.mLt)
        m_pair = float(bagle_model.mLp + bagle_model.mLs)
        thetaE_vbm_mas = thetaE_mas * np.sqrt(m_tot / m_pair)
    else:
        thetaE_vbm_mas = thetaE_mas

    # arcsec per VBM Einstein radius.
    scale = thetaE_vbm_mas * 1e-3
    src = xS / scale
    lenses = xL / scale
    tE_vbm = tE_bagle * (thetaE_vbm_mas / thetaE_mas)

    return src, lenses, tE_vbm


def _static_multilens_params(bagle_model, n_lens):
    """VBM static binary or triple parameters from the on-sky geometry.

    The VBM origin is the center of mass of the primary and secondary
    (the first two lenses). ``alpha`` and signed ``u0`` come from the
    source position and its velocity in that frame, so geometric-center
    and primary-centered BAGLE parameters do not need a separate shift.

    Parameters
    ----------
    bagle_model : BAGLE model instance
        Static (no orbital motion), parallax off.
    n_lens : int
        2 or 3.

    Returns
    -------
    vbm_par : list of float
        ``BinaryLightCurve`` (7 numbers) or ``TripleLightCurve`` (10).
    """
    # Any epoch works: static, no parallax, so the track is linear.
    if hasattr(bagle_model, 't0'):
        t_ref = float(np.asarray(bagle_model.t0).reshape(-1)[0])
    else:
        t_ref = float(bagle_model.t0_com)

    dt = 1.0
    src, lenses, tE_vbm = _einstein_source_lenses(
        bagle_model, np.array([t_ref, t_ref + dt]), n_lens)
    mass = _mass_fractions(bagle_model, n_lens)

    # VBM +x runs from the primary toward the secondary.
    r1 = lenses[0, 0]
    r2 = lenses[0, 1]
    rel = r2 - r1
    s = float(np.hypot(rel[0], rel[1]))
    ex = rel / s
    ey = np.array([-ex[1], ex[0]])

    # CoM of the first two lenses. The tertiary is not in this origin.
    com = (mass[0] * r1 + mass[1] * r2) / (mass[0] + mass[1])
    y0 = np.array([(src[0] - com) @ ex, (src[0] - com) @ ey])
    y1 = np.array([(src[1] - com) @ ex, (src[1] - com) @ ey])

    # Velocity per Einstein time. Direction sets alpha; |v| is 1
    # when tE_vbm matches the position units.
    v = (y1 - y0) / dt * tE_vbm
    alpha = float(np.arctan2(-v[1], -v[0]))

    # Invert y = (u0 sin alpha - tau cos alpha,
    #             -u0 cos alpha - tau sin alpha).
    u0 = float(y0[0] * np.sin(alpha) - y0[1] * np.cos(alpha))
    tau = float(-y0[0] * np.cos(alpha) - y0[1] * np.sin(alpha))
    t0 = t_ref - tau * tE_vbm

    q = float(mass[1] / mass[0])
    rho = _vbm_rho(bagle_model, bagle_model.tE, tE_vbm)
    vbm_par = [math.log(s), math.log(q), u0, alpha, math.log(rho),
               math.log(tE_vbm), t0]

    if n_lens == 2:
        return vbm_par

    # Tertiary: separation and angle from the primary, relative to +x.
    r3 = lenses[0, 2]
    d3 = r3 - r1
    s13 = float(np.hypot(d3[0], d3[1]))
    q3 = float(mass[2] / mass[0])
    psi = float(np.arctan2(d3 @ ey, d3 @ ex))
    vbm_par.extend([math.log(s13), math.log(q3), psi])

    return vbm_par


def _relative_los_einstein(bagle_model, t):
    """Line-of-sight secondary minus primary, in Einstein radii.

    Parameters
    ----------
    bagle_model : BAGLE model instance
        Keplerian binary (phot-only or PhotAstrom).
    t : float
        Time (MJD).

    Returns
    -------
    z : float
        Line-of-sight offset (Einstein radii). Positive is the VBM sign
        carried by the Thiele-Innes ``C`` and ``H`` elements.
    """
    # Imported here: bagle.model imports this module at load time.
    from bagle.model import keplerian_lens_orbit

    # Phot-only stores (w, o). PhotAstrom stores the two node angles.
    if hasattr(bagle_model, 'w') and hasattr(bagle_model, 'o'):
        omega = bagle_model.w
        big_omega = bagle_model.o
    else:
        omega = bagle_model.omega_pri
        big_omega = bagle_model.big_omega_sec

    orb = keplerian_lens_orbit(
        omega, big_omega, bagle_model.i, bagle_model.e, bagle_model.tp,
        bagle_model.p, bagle_model.aleph, bagle_model.aleph_sec)
    orb.oal2xy(np.array([t]))

    # Orbit-plane coordinates. C and H are the line-of-sight Thiele-Innes
    # elements; the secondary minus the primary is (C2 - C, H2 - H).
    mean_anom = 2.0 * np.pi * (t - bagle_model.tp) / bagle_model.p
    ecc_anom = orb.eccen_anomaly(np.array([mean_anom]), bagle_model.e)[0]
    X = np.cos(ecc_anom) - bagle_model.e
    Y = np.sqrt(1.0 - bagle_model.e ** 2) * np.sin(ecc_anom)
    z = ((orb.conC2 - orb.conC) * X + (orb.conH2 - orb.conH) * Y)

    # PhotAstrom aleph is in mas. thetaE_amp is also mas.
    if hasattr(bagle_model, 'thetaE_amp') and not hasattr(bagle_model, 'w'):
        z = z / bagle_model.thetaE_amp

    return float(z)


def _kepler_binary_params(bagle_model):
    """``BinaryLightCurveKepler`` parameters at the lens-CoM closest approach.

    Parameter order, matching VBMicrolensing::

        [log s, log q, u0, alpha, log rho, log tE, t0,
         paiN, paiE, w1, w2, w3, z/s, a_r]

    ``t0`` is ``t0_com - VBM_ORBIT_TIME_SHIFT``. Pass light-curve times
    through that same shift. Parallax components are zero: this path is
    for no-parallax Keplerian models. ``u0`` is the absolute CoM impact
    parameter. ``alpha`` is measured from the primary-to-secondary axis.

    Parameters
    ----------
    bagle_model : BAGLE model instance
        Keplerian PSBL or FSBL, parallax off.

    Returns
    -------
    vbm_par : list of float
        Fourteen BinaryLightCurveKepler parameters.
    """
    tc = float(bagle_model.t0_com)
    dt = 1e-3
    is_pa = (hasattr(bagle_model, 'thetaE_amp')
             and not hasattr(bagle_model, 'w'))

    # Primary and secondary in Einstein radii, at three epochs.
    t_obs = np.array([tc - dt, tc, tc + dt])
    lens = np.asarray(
        bagle_model.get_resolved_lens_astrometry(t_obs), dtype=float)
    if is_pa:
        lens = lens / (bagle_model.thetaE_amp * 1e-3)

    rel = lens[:, 1] - lens[:, 0]
    rel_dot = (rel[2] - rel[0]) / (2.0 * dt)

    # Line of sight and its rate, per day, in Einstein radii.
    z = _relative_los_einstein(bagle_model, tc)
    z_dot = (_relative_los_einstein(bagle_model, tc + dt)
             - _relative_los_einstein(bagle_model, tc - dt)) / (2.0 * dt)

    # VBM axes at t0_com. +x is primary -> secondary.
    s = float(np.hypot(rel[1, 0], rel[1, 1]))
    ex = rel[1] / s
    ey = np.array([-ex[1], ex[0]])
    w1 = float(rel_dot @ ex / s)
    w2 = float(-(rel_dot @ ey) / s)

    # Semi-major axis over the current 3-D separation (VBM a_r).
    if is_pa:
        a_ein = float(bagle_model.a) / float(bagle_model.thetaE_amp)
    else:
        a_ein = float(bagle_model.a)
    ar = a_ein / (s * np.sqrt(1.0 + (z / s) ** 2))

    # Source motion in the binary frame, per Einstein time.
    v = np.array([bagle_model.thetaE_hat @ ex, bagle_model.thetaE_hat @ ey])
    alpha = float(np.arctan2(-v[1], -v[0]))
    rho = _vbm_rho(bagle_model, bagle_model.tE, bagle_model.tE)

    vbm_par = [math.log(s), math.log(float(bagle_model.q)),
               abs(float(bagle_model.u0_amp_com)), alpha, math.log(rho),
               math.log(float(bagle_model.tE)), tc - VBM_ORBIT_TIME_SHIFT,
               0.0, 0.0, w1, w2, z_dot / s, z / s, ar]

    return vbm_par


def _set_vbm_coordinates(vbm_mod, bagle_model):
    """Point a VBM model at the BAGLE lens coordinates.

    Parameters
    ----------
    vbm_mod : VBMicrolensing.VBMicrolensing
    bagle_model : BAGLE model instance
        Read for ``raL`` and ``decL`` (degrees).

    Returns
    -------
    None
    """
    # Keplerian calls require coordinates even when parallax is zero.
    if bagle_model.raL is None or bagle_model.decL is None:
        vbm_mod.SetObjectCoordinates('17:52:00 -29:00:00')
        return None

    c = SkyCoord(ra=bagle_model.raL * u.deg, dec=bagle_model.decL * u.deg,
                 frame='icrs')
    ra_string = c.ra.to_string(unit=u.hourangle, sep=':')
    dec_string = c.dec.to_string(unit=u.degree, sep=':', alwayssign=True)
    vbm_mod.SetObjectCoordinates(f'{ra_string} {dec_string}')

    return None


def _get_vbm_multilens(bagle_model, vbm_mod, verbose=False):
    """Parameter vector for a binary or triple BAGLE model.

    Parameters
    ----------
    bagle_model : BAGLE model instance
    vbm_mod : VBMicrolensing.VBMicrolensing
        Solver. Tolerances and coordinates are set here.
    verbose : bool
        Print which VBM light-curve path was used.

    Returns
    -------
    vbm_mod : VBMicrolensing.VBMicrolensing
    vbm_par : list of float

    Notes
    -----
    Static no-parallax models leave ``t0`` in TDB MJD. Call
    ``BinaryLightCurve`` or ``TripleLightCurve`` with unconverted times.
    Keplerian models put ``t0`` on the ``VBM_ORBIT_TIME_SHIFT`` clock.
    Call ``BinaryLightCurveKepler`` with times shifted the same way,
    or use :func:`vbm_light_curve`.
    """
    class_name = bagle_model.__class__.__name__
    family = _vbm_lens_family(class_name)
    orbit = _vbm_orbit_kind(class_name)

    if bagle_model.parallaxFlag:
        raise NotImplementedError(
            'get_vbm_model does not map parallax for binary, triple, '
            'or Keplerian lenses. Use a no-parallax model.')

    if orbit == 'linear':
        raise NotImplementedError(
            'get_vbm_model supports static lenses and Keplerian '
            '(EllOrbs, CircOrbs) orbits, not linear or accelerated orbits.')

    if family == 'triple' and orbit == 'kepler':
        raise NotImplementedError(
            'VBMicrolensing has no full Keplerian triple light curve.')

    # Same tolerances the comparison notebook and the VBM tests use.
    vbm_mod.Tol = 1e-6
    vbm_mod.RelTol = 1e-7

    if orbit == 'kepler':
        _set_vbm_coordinates(vbm_mod, bagle_model)
        vbm_par = _kepler_binary_params(bagle_model)
        if verbose:
            print('VBM BinaryLightCurveKepler')
        return vbm_mod, vbm_par

    n_lens = 3 if family == 'triple' else 2
    vbm_par = _static_multilens_params(bagle_model, n_lens)
    if verbose:
        method = 'TripleLightCurve' if n_lens == 3 else 'BinaryLightCurve'
        print('VBM ' + method)

    return vbm_mod, vbm_par


def _vbm_method_name(bagle_model):
    """VBM light-curve method that matches ``get_vbm_model`` parameters.

    Parameters
    ----------
    bagle_model : BAGLE model instance

    Returns
    -------
    name : str
        Method name on a ``VBMicrolensing`` instance.
    """
    class_name = bagle_model.__class__.__name__
    family = _vbm_lens_family(class_name)
    finite = _is_finite_source_class(class_name)

    if family is not None and _vbm_orbit_kind(class_name) == 'kepler':
        return 'BinaryLightCurveKepler'

    if family == 'triple':
        return 'TripleLightCurve'

    if family == 'binary':
        return 'BinaryLightCurve'

    # Single lens. Astrometry curves always carry the parallax slots.
    if bagle_model.astrometryFlag:
        if finite:
            return 'ESPLAstroLightCurve'
        return 'PSPLAstroLightCurve'

    if bagle_model.parallaxFlag:
        if finite:
            return 'ESPLLightCurveParallax'
        return 'PSPLLightCurveParallax'

    if finite:
        return 'ESPLLightCurve'

    return 'PSPLLightCurve'


def _vbm_eval_times(bagle_model, time_mjd):
    """Times on the clock expected by the matching VBM light-curve method.

    Parameters
    ----------
    bagle_model : BAGLE model instance
    time_mjd : array_like
        BAGLE times (TDB MJD).

    Returns
    -------
    t_vbm : ndarray
        Times to pass into the VBM method.
    """
    t = np.atleast_1d(np.asarray(time_mjd, dtype=float))
    class_name = bagle_model.__class__.__name__

    # Kepler t0 was stored already shifted.
    if (_vbm_lens_family(class_name) is not None
            and _vbm_orbit_kind(class_name) == 'kepler'):
        return t - VBM_ORBIT_TIME_SHIFT

    # Single-lens parallax and astrometry paths convert the clock.
    # Static binary and triple no-parallax paths keep TDB MJD.
    if _vbm_lens_family(class_name) is None and (
            bagle_model.parallaxFlag or bagle_model.astrometryFlag):
        return np.atleast_1d(mjd_prime_to_vbm_time(t))

    return t


def vbm_light_curve(bagle_model, time_mjd, verbose=False):
    """Magnification and source track from the matching VBM light curve.

    Builds parameters with :func:`get_vbm_model` and calls the VBM
    method named by :func:`_vbm_method_name`. Keplerian times are shifted
    by :data:`VBM_ORBIT_TIME_SHIFT`. Single-lens parallax and astrometry
    times go through :func:`mjd_prime_to_vbm_time`.

    Parameters
    ----------
    bagle_model : BAGLE model instance
    time_mjd : array_like
        Observation times (TDB MJD).
    verbose : bool, optional
        Passed to :func:`get_vbm_model`.

    Returns
    -------
    amp : ndarray, shape (N_times,)
        Total magnification.
    y1, y2 : ndarray, shape (N_times,)
        Source position in the VBM lens frame.
    sep : ndarray, shape (N_times,), or None
        Projected lens separation for Keplerian binaries. None otherwise.
    """
    vbm_mod, vbm_par = get_vbm_model(bagle_model, verbose=verbose)
    method_name = _vbm_method_name(bagle_model)
    t_vbm = _vbm_eval_times(bagle_model, time_mjd)
    t_list = [float(ti) for ti in np.atleast_1d(t_vbm)]

    method = getattr(vbm_mod, method_name)
    out = method(vbm_par, t_list)

    # Astro light curves return centroids before the lens-frame y.
    if 'Astro' in method_name:
        y1 = np.asarray(out[5], dtype=float)
        y2 = np.asarray(out[6], dtype=float)
        sep = None
    else:
        y1 = np.asarray(out[1], dtype=float)
        y2 = np.asarray(out[2], dtype=float)
        sep = None
        if len(out) > 3:
            sep = np.asarray(out[3], dtype=float)

    amp = np.asarray(out[0], dtype=float)

    return amp, y1, y2, sep


def get_vbm_model(bagle_model, verbose=False):
    """
    Return a VBMicrolensing matched to the input BAGLE model.

    Photometry-only FSPL uses ESPL parameter lists with ``log_rho``
    from BAGLE ``radiusS``. Call ``ESPLLightCurve`` /
    ``ESPLLightCurveParallax`` (not PSPL) for those models.
    FSPL with astrometry uses ``ESPLAstroLightCurve`` (``log_rho`` inserted
    after ``t0`` in the PSPLAstro parameter list).

    Static PSBL and FSBL use ``BinaryLightCurve``::

        [log s, log q, u0, alpha, log rho, log tE, t0]

    Static PSTL uses ``TripleLightCurve``, appending
    ``[log s13, log q3, psi]``. Separations and ``tE`` are in Einstein
    units of the total lens mass. PhotAstrom triples are rescaled from
    the 1-2-pair Einstein radius. Point sources use a source radius of
    1e-6; finite-source models pass BAGLE ``radiusS``. These paths have
    no parallax. ``t0`` stays in TDB MJD.

    Keplerian PSBL and FSBL (EllOrbs, CircOrbs) use
    ``BinaryLightCurveKepler``. ``t0`` is
    ``t0_com - VBM_ORBIT_TIME_SHIFT``. Shift the time grid the same way,
    or call :func:`vbm_light_curve`.

    Parallax paths for single lenses convert ``t0`` and ``t0_par`` with
    ``mjd_prime_to_vbm_time`` (TDB MJD to UTC ``JD - 2450000``).
    Pass the light-curve time grid through that same function.
    Single-lens paths without parallax leave ``t0`` in TDB MJD; pass
    unconverted times.

    Parameters
    ----------
    bagle_model : BAGLE model instance
    verbose : bool, optional
        Print which VBM parameter path was used.

    Returns
    -------
    vbm_mod : VBMicrolensing.VBMicrolensing
        Configured VBM instance (coords / parallax flags as needed).
    vbm_par : list of float
        Parameter vector for the matching VBM light-curve method.
    """
    import VBMicrolensing

    #####
    # Setup VBM model
    #####
    vbm_mod = VBMicrolensing.VBMicrolensing()

    # Binary and triple, including finite-source and Keplerian orbits.
    # Single lenses (PSPL, FSPL) stay on the parameter paths below.
    class_name = bagle_model.__class__.__name__
    if _vbm_lens_family(class_name) is not None:
        return _get_vbm_multilens(bagle_model, vbm_mod, verbose=verbose)

    bagle_class_name = str(bagle_model.__class__)
    is_fspl = 'FSPL' in bagle_class_name

    # Phot-only FSPL stores radiusS in Einstein radii; PhotAstrom stores
    # radiusS in arcsec. Prefer source_radius_thetaE_units() when present.
    rho_vbm = None
    if is_fspl:
        if hasattr(bagle_model, 'source_radius_thetaE_units'):
            rho_vbm = float(bagle_model.source_radius_thetaE_units())
        elif (getattr(bagle_model, 'astrometryFlag', False)
              and hasattr(bagle_model, 'thetaE_amp')):
            rho_vbm = (float(bagle_model.radiusS) * 1e3
                       / float(bagle_model.thetaE_amp))
        else:
            rho_vbm = float(bagle_model.radiusS)

    if bagle_model.parallaxFlag:
        c = SkyCoord(ra=bagle_model.raL * u.deg, dec=bagle_model.decL * u.deg, frame='icrs')

        # Convert string with colon separators
        ra_string = c.ra.to_string(unit=u.hourangle, sep=':')
        dec_string = c.dec.to_string(unit=u.degree, sep=':', alwayssign=True)
        full_coord_string = f"{ra_string} {dec_string}"
        vbm_mod.SetObjectCoordinates(full_coord_string)

    #####
    # VBM Parameters Object
    #####
    ### Photometry-Only and No Parallax
    if not bagle_model.astrometryFlag and not bagle_model.parallaxFlag:
        if verbose: print('VBM with Photometry-Only, no parallax')
        u0_vbm = float(bagle_model.u0_amp)
        # No Sun table on this path. Keep t0 in BAGLE TDB MJD and pass
        # the light-curve times in that same unconverted clock.
        t0_vbm = float(bagle_model.t0)
        tE_vbm = float(bagle_model.tE)

        # PSPLLightCurve: [log_u0, log_tE, t0]
        # ESPLLightCurve: [log_u0, log_tE, t0, log_rho]
        if is_fspl:
            vbm_par = [math.log(abs(u0_vbm)), math.log(tE_vbm), t0_vbm,
                       math.log(rho_vbm)]
        else:
            vbm_par = [math.log(u0_vbm), math.log(tE_vbm), t0_vbm]

    ### Photometry-Only and With Parallax
    elif not bagle_model.astrometryFlag and bagle_model.parallaxFlag:
        if verbose: print('VBM with Photometry-Only, with parallax')
        # Convert
        geo_params = convert_helio_geo_phot(bagle_model.raL, bagle_model.decL,
                                            bagle_model.t0, bagle_model.u0_amp, bagle_model.tE,
                                            bagle_model.piE[0], bagle_model.piE[1], bagle_model.t0,
                                            murel_in='SL', murel_out='LS',
                                            coord_in='EN', coord_out='EN', plot=False)
        u0_vbm = geo_params[1]
        tE_vbm = geo_params[2]
        piEE_vbm = geo_params[3]
        piEN_vbm = geo_params[4]

        # BAGLE times are TDB MJD. Shift onto VBM's UTC JD-2450000 clock.
        # VBM still applies its own light-travel term when t_in_HJD is set.
        t0_vbm = mjd_prime_to_vbm_time(geo_params[0])
        t0_par_vbm = mjd_prime_to_vbm_time(bagle_model.t0)

        # Update model
        vbm_mod.parallaxsystem = 1
        vbm_mod.t0_par_fixed = 1
        vbm_mod.t0_par = t0_par_vbm

        # PSPLLightCurveParallax: [u0, log_tE, t0, piEN, piEE]
        # ESPLLightCurveParallax: [u0, log_tE, t0, log_rho, piEN, piEE]
        if is_fspl:
            vbm_par = [u0_vbm, math.log(tE_vbm), t0_vbm, math.log(rho_vbm),
                       piEN_vbm, piEE_vbm]
        else:
            vbm_par = [u0_vbm, math.log(tE_vbm), t0_vbm, piEN_vbm, piEE_vbm]

    # Astrometry
    else:
        if verbose: print('VBM with Astrometry')
        # Convert
        geo_params = convert_helio_geo_phot(bagle_model.raL, bagle_model.decL,
                                            bagle_model.t0, bagle_model.u0_amp, bagle_model.tE,
                                            bagle_model.piE[0], bagle_model.piE[1], bagle_model.t0,
                                            murel_in='SL', murel_out='LS',
                                            coord_in='EN', coord_out='EN', plot=False)
        u0_vbm = geo_params[1]
        tE_vbm = geo_params[2]
        piEE_vbm = geo_params[3]
        piEN_vbm = geo_params[4]

        # Same TDB MJD -> UTC JD-2450000 shift as the photometry path.
        t0_vbm = mjd_prime_to_vbm_time(geo_params[0])
        t0_par_vbm = mjd_prime_to_vbm_time(bagle_model.t0)

        muSE_vbm = bagle_model.muS[0]
        muSN_vbm = bagle_model.muS[1]
        piS_vbm = bagle_model.piS
        thetaE_vbm = bagle_model.thetaE_amp

        # Update model
        vbm_mod.parallaxsystem = 1
        vbm_mod.t0_par_fixed = 1
        vbm_mod.t0_par = t0_par_vbm

        # PSPLAstroLightCurve: [u0, log_tE, t0, piEN, piEE, muSN, muSE, piS, thetaE]
        # ESPLAstroLightCurve: [u0, log_tE, t0, log_rho, piEN, piEE, muSN, muSE,
        #                       piS, thetaE]
        if is_fspl:
            vbm_par = [u0_vbm, math.log(tE_vbm), t0_vbm, math.log(rho_vbm),
                       piEN_vbm, piEE_vbm, muSN_vbm, muSE_vbm, piS_vbm,
                       thetaE_vbm]
        else:
            vbm_par = [u0_vbm, math.log(tE_vbm), t0_vbm, piEN_vbm, piEE_vbm,
                       muSN_vbm, muSE_vbm, piS_vbm, thetaE_vbm]

    return vbm_mod, vbm_par


# Utility to account for light travel time.
def get_mjd_prime(target_coords, time_mjd):
    """
    Convert UTC MJD times to heliocentric-corrected MJD ("MJD prime").

    Parameters
    ----------
    target_coords : astropy.coordinates.SkyCoord
        Sky position of the target used for the light-travel-time correction.
    time_mjd : float or array-like
        Observation time(s) in MJD (UTC).

    Returns
    -------
    float or np.ndarray
        Light-travel-time-corrected MJD value(s), computed for Keck Observatory.

    Notes
    -----
    VBMicrolensing comparisons use ``mjd_prime_to_vbm_time``. That
    function converts TDB MJD to UTC and does not call this one.
    """
    location = EarthLocation.of_site('Keck Observatory')
    time_to_correct = Time(time_mjd, format='mjd', scale='utc', location=location)
    ltt_helio = time_to_correct.light_travel_time(target_coords, 'heliocentric')
    time_mjd_prime = (time_to_correct + ltt_helio).value
    return time_mjd_prime


def mjd_prime_to_vbm_time(time_mjd, target_coords=None):
    """
    Convert BAGLE TDB MJD to the VBMicrolensing numeric clock.

    BAGLE model times are MJD on the TDB scale. VBMicrolensing's bundled
    Sun ephemeris (``SunEphemeris.txt``) is a Horizons table tagged in UT,
    and VBM times are ``JD - 2450000`` on that UTC clock. This function
    converts the TDB instant to UTC, then applies that zero point::

        t_vbm = MJD_UTC - 50000 + 0.5

    which is ``JD_UTC - 2450000``. At MJD 55775 the TDB-UTC offset is
    about 66 s; after the 2017 leap second it is about 69 s.

    Light-travel time is not applied. With the default ``t_in_HJD=True``,
    VBM subtracts its own heliocentric light-travel time when it indexes
    the Sun table, and it leaves ``(t - t0)`` on the input clock. A
    topocentric correction here (the old Keck "MJD prime" term) shifts
    the light-curve clock as well as the ephemeris lookup, and it does
    not match VBM's geocentric table. The name of this function is the
    old one; the light-travel step is no longer part of the conversion.

    Parameters
    ----------
    time_mjd : float or array_like
        BAGLE time(s) in TDB MJD. Float or numpy array, not ``Time`` objects.
    target_coords : astropy.coordinates.SkyCoord, optional
        Ignored. Kept so callers that used to pass a sky position for the
        light-travel correction still run. The UTC conversion does not
        depend on direction.

    Returns
    -------
    time_vbm : float or numpy array
        Time(s) on VBM's clock, ``JD_UTC - 2450000``. A scalar input
        returns a float.

    Notes
    -----
    VBM's table is geocentric, daily, and linearly interpolated, and its
    equatorial basis includes the obliquity. Those are separate from this
    clock conversion. See ``docs/frame_convert.rst``.
    """
    # Accept a scalar or an array. The UTC step is the same either way.
    time_arr = np.asarray(time_mjd, dtype=float)
    scalar_input = time_arr.ndim == 0

    # Sky position used to drive a Keck light-travel correction. The
    # TDB to UTC shift does not use it.
    del target_coords

    # Same physical instant, labeled UTC. Leap seconds make the offset
    # stepwise; astropy carries the table.
    t_tdb = Time(time_arr, format='mjd', scale='tdb')
    mjd_utc = np.asarray(t_tdb.utc.mjd, dtype=float)

    # VBM zero point: JD - 2450000 = MJD - 50000 + 0.5.
    time_vbm = mjd_utc - 50000.0 + 0.5

    if scalar_input:
        return float(time_vbm)

    return time_vbm


def convert_bagle_mulens_psbl_phot(ra, dec,
                                   t0_in, u0_in, tE_in,
                                   piEE_in, piEN_in, t0par,
                                   q_in, alpha_in, sep_in,
                                   mod_in='bagle', plot=True):
    """
    Convert between the native fit parameters of BAGLE and MulensModel
    for a point source binary lens photometry model.

    Parameters
    ----------
    t0_in, u0_in, tE_in, piEE_in, piEN_in, q_in, alpha_in, sep can be either be
    floats or arrays. If they are arrays, they must be the same length.
    
    ra, dec : str, float, or int
        Equatorial coordinates of the microlensing event.
        
        If string, needs to be of the form 
        'HH:MM:SS.SSSS', 'DD:MM:SS.SSSS'

    t0_in : float or array (MJD)
        Time at which minimum source-lens projected separation in the rectilinear
        frame occurs, in input frame.

    u0_in : float or array (thetaE)
        Minimum source-lens projected separation in the rectilinear frame,
        in units of the Einstein radius, in input frame.
    
    tE_in : float or array (days)
        Einstein crossing time, in input frame.
    
    piEE_in : float or array
        Microlensing parallax, East component, in input frame.

    piEN_in : float or array
        Microlensing parallax, North component, in input frame.

    t0par : float
        Reference time for geocentric projection conversions (MJD).

    q_in : float or array
        Binary lens mass ratio.
    
    alpha_in : float or array (deg)
        Angle between relative proper motion vector and the binary axis, in input frame.
    
    sep_in : float or array (thetaE)
        Separation between the binary lens components.
    
    mod_in : str
        'bagle' if converting from BAGLE to MulensModel parameters
        'mulens' if converting from MulensModel to BAGLE paramters

    plot : bool
        Plot the conversion figures if true.

    Returns
    -------
    t0_out : float or np.ndarray
        Converted closest-approach time (MJD).
    u0_out : float or np.ndarray
        Converted signed impact parameter (Einstein-radius units).
    tE_out : float or np.ndarray
        Converted Einstein timescale (days).
    piEE_out : float or np.ndarray
        Converted east microlensing parallax component.
    piEN_out : float or np.ndarray
        Converted north microlensing parallax component.
    q_out : float or np.ndarray
        Converted binary mass ratio.
    alpha_out : float or np.ndarray
        Converted binary-axis angle (degrees).
    sep_out : float or np.ndarray
        Converted binary separation (Einstein-radius units).
    """
    ##########
    # Convert between helio and geo projected.
    #
    # Calculate the trajectory made by the source-lens relative the
    # binary axis (this is different, because the source-lens relative motion
    # changes across the two frames).
    ##########
    if mod_in == 'bagle':
        output = convert_helio_geo_phot(ra, dec, t0_in, u0_in,
                                        tE_in, piEE_in, piEN_in,
                                        t0par, in_frame='helio',
                                        murel_in='SL', murel_out='LS',
                                        coord_in='EN', coord_out='tb',
                                        plot=plot)

        t0_out, u0_out, tE_out, piEE_out, piEN_out = output

        sep_out = sep_in
        q_out = q_in

        phi_murel_in = np.rad2deg(np.arctan2(piEN_in, piEE_in))
        phi_murel_out = np.rad2deg(np.arctan2(piEN_out, piEE_out))
        delta_alpha = phi_murel_out - phi_murel_in + 180

        # BAGLE actually gives phi_bagle as angle between binary and muRel vector
        alpha_out = -(alpha_in + delta_alpha)

        q_prime = (1.0 - q_in) / (2.0 * (1 + q_in))
        t0_out += q_prime * sep_in * tE_out * np.cos(np.deg2rad(alpha_out))
        u0_out += q_prime * sep_in * np.sin(np.deg2rad(alpha_out))

    elif mod_in == 'mulens':
        q_prime = (1.0 - q_in) / (2.0 * (1 + q_in))
        t0_in -= q_prime * sep_in * tE_in * np.cos(np.deg2rad(alpha_in))
        u0_in -= q_prime * sep_in * np.sin(np.deg2rad(alpha_in))

        output = convert_helio_geo_phot(ra, dec, t0_in, u0_in,
                                        tE_in, piEE_in, piEN_in,
                                        t0par, in_frame='geo',
                                        murel_in='LS', murel_out='SL',
                                        coord_in='tb', coord_out='EN',
                                        plot=plot)

        t0_out, u0_out, tE_out, piEE_out, piEN_out = output

        sep_out = sep_in
        q_out = q_in

        phi_murel_in = np.rad2deg(np.arctan2(piEN_in, piEE_in))
        phi_murel_out = np.rad2deg(np.arctan2(piEN_out, piEE_out))
        delta_alpha = phi_murel_in - phi_murel_out + 180  # note the sign flip compared to above.

        alpha_out = -(alpha_in + delta_alpha)

    else:
        raise Exception("mod_in must be 'bagle' or 'mulens'")



    return t0_out, u0_out, tE_out, piEE_out, piEN_out, q_out, alpha_out, sep_out


def convert_bagle_pylima_psbl_phot(ra, dec,
                                   t0_in, u0_in, tE_in,
                                   piEE_in, piEN_in, t0par,
                                   q_in, alpha_in, sep_in,
                                   mod_in='bagle', plot=True):
    """
    Convert between the native fit parameters of BAGLE and pyLIMA
    for a point source binary lens photometry model.

    Parameters
    ----------
    t0_in, u0_in, tE_in, piEE_in, piEN_in, q_in, alpha_in, sep can be either be
    floats or arrays. If they are arrays, they must be the same length.

    ra, dec : str, float, or int
        Equatorial coordinates of the microlensing event.

        If string, needs to be of the form
        'HH:MM:SS.SSSS', 'DD:MM:SS.SSSS'

    t0_in : float or array (MJD)
        Time at which minimum source-lens projected separation in the rectilinear
        frame occurs, in input frame.

    u0_in : float or array (thetaE)
        Minimum source-lens projected separation in the rectilinear frame,
        in units of the Einstein radius, in input frame.

    tE_in : float or array (days)
        Einstein crossing time, in input frame.

    piEE_in : float or array
        Microlensing parallax, East component, in input frame.

    piEN_in : float or array
        Microlensing parallax, North component, in input frame.

    t0par : float
        Reference time for geocentric projection conversions (MJD).

    q_in : float or array
        Binary lens mass ratio.

    alpha_in : float or array (deg)
        Angle between relative proper motion vector and the binary axis, in input frame.

    sep_in : float or array (thetaE)
        Separation between the binary lens components.

    mod_in : str
        'bagle' if converting from BAGLE to pyLIMA parameters
        'pylima' if converting from pyLIMA to BAGLE paramters

    plot : bool
        Plot the conversion figures if true.

    Returns
    -------
    t0_out : float or np.ndarray
        Converted closest-approach time (MJD).
    u0_out : float or np.ndarray
        Converted signed impact parameter (Einstein-radius units).
    tE_out : float or np.ndarray
        Converted Einstein timescale (days).
    piEE_out : float or np.ndarray
        Converted east microlensing parallax component.
    piEN_out : float or np.ndarray
        Converted north microlensing parallax component.
    q_out : float or np.ndarray
        Converted binary mass ratio.
    alpha_out : float or np.ndarray
        Converted binary-axis angle (degrees).
    sep_out : float or np.ndarray
        Converted binary separation (Einstein-radius units).
    """
    ##########
    # Convert between helio and geo projected.
    ##########
    if mod_in == 'bagle':
        output = convert_helio_geo_phot(ra, dec, t0_in, u0_in,
                                        tE_in, piEE_in, piEN_in,
                                        t0par, in_frame='helio',
                                        murel_in='SL', murel_out='LS',
                                        coord_in='EN', coord_out='tb',
                                        plot=plot)

        t0_out, u0_out, tE_out, piEE_out, piEN_out = output

        sep_out = sep_in
        q_out = q_in

        phi_murel_in = np.rad2deg(np.arctan2(piEN_in, piEE_in))
        phi_murel_out = np.rad2deg(np.arctan2(piEN_out, piEE_out))
        delta_alpha = phi_murel_out - phi_murel_in + 180

        # BAGLE actually gives phi_bagle as angle between binary and muRel vector
        alpha_out = -(alpha_in + delta_alpha)

        q_prime = (1.0 - q_in) / (2.0 * (1 + q_in))
        t0_out += q_prime * sep_in * tE_out * np.cos(np.deg2rad(alpha_out))
        u0_out += q_prime * sep_in * np.sin(np.deg2rad(alpha_out))

    elif mod_in == 'pylima':
        q_prime = (1.0 - q_in) / (2.0 * (1 + q_in))
        t0_in -= q_prime * sep_in * tE_in * np.cos(np.deg2rad(alpha_in))
        u0_in -= q_prime * sep_in * np.sin(np.deg2rad(alpha_in))

        output = convert_helio_geo_phot(ra, dec, t0_in, u0_in,
                                        tE_in, piEE_in, piEN_in,
                                        t0par, in_frame='geo',
                                        murel_in='LS', murel_out='SL',
                                        coord_in='tb', coord_out='EN',
                                        plot=plot)

        t0_out, u0_out, tE_out, piEE_out, piEN_out = output

        sep_out = sep_in
        q_out = q_in

        phi_murel_in = np.rad2deg(np.arctan2(piEN_in, piEE_in))
        phi_murel_out = np.rad2deg(np.arctan2(piEN_out, piEE_out))
        delta_alpha = phi_murel_in - phi_murel_out + 180  # note the sign flip compared to above.

        alpha_out = -(alpha_in + delta_alpha)


    else:
        raise Exception("mod_in must be 'bagle' or 'pylima'")

    return t0_out, u0_out, tE_out, piEE_out, piEN_out, q_out, alpha_out, sep_out


##############
#
# Converters for Reference Frames
#
##############
def convert_helio_geo_ast(ra, dec,
                          piS, xS0E_in, xS0N_in,
                          muSE_in, muSN_in, 
                          t0_in, u0_in, tE_in, 
                          piEE_in, piEN_in, t0par,
                          in_frame='helio',
                          murel_in='SL', murel_out='LS', 
                          coord_in='EN', coord_out='tb', plot=True):
    """
    Convert between Solar System barycentric and geocentric-projected parameters.
    This converts only the subset of parameters in astrometry fits
    (xS0E, xS0N, muSE, muSN).

    The default conversion assumes that inputs are in
    source-lens (S - L) and EN coordinate conventions.

    Note that "helio" (h) here, as elsewhere in BAGLE, means BAGLE's
    internal Solar System barycentric frame, not a true heliocentric one.

    Units and definitions
    ---------------------
    BAGLE's source astrometry, in either frame, is

        xS(t) [arcsec] = xS0 + 1e-3 * [muS * (t - t0) / 365.25
                                       + piS * P(t)]

    with ``xS0`` in arcsec, ``muS`` in mas/yr, ``piS`` in mas, ``t`` and
    ``t0`` in MJD (days), and ``P(t)`` the East/North parallax vector in AU
    from :func:`bagle.parallax.parallax_in_direction`. ``xS0`` is the source
    position at that frame's own ``t0``.

    In the geocentric-projected frame, Earth's parallax motion is
    linearized about ``t0par`` and moved into the rectilinear part:

        P_geo(t) = P(t) - P0 - dP0 * (t - t0par) / 365.25

    where ``P0`` (AU) and ``dP0`` (AU/yr, from
    :func:`bagle.parallax.dparallax_dt_in_direction`) are evaluated at
    ``t0par`` for Earth. Both frames then describe the same source path,
    which gives, for helio (h) -> geo (g):

        muS_g = muS_h + piS * dP0                                   [mas/yr]
        xS0_g = xS0_h + 1e-3 * [muS_h * (t0_g - t0_h) / 365.25
                                + piS * P0
                                + piS * dP0 * (t0_g - t0par) / 365.25]  [arcsec]

    and the algebraic inverse for geo -> helio. ``t0_g`` and ``t0_h`` come
    from :func:`convert_helio_geo_phot`.

    Parameters
    ----------
    ra, dec : str, float
        Equatorial coordinates of the microlensing event.
        If float, in degrees. If string, needs to be of the form
        'HH:MM:SS.SSSS', 'DD:MM:SS.SSSS'
    piS : float or array
        Source parallax (mas).
    xS0E_in : float or array
        Source position relative to RA/Dec, East component, at the input
        frame's t0, in the input frame (arcsec).
    xS0N_in : float or array
        Source position relative to RA/Dec, North component, at the input
        frame's t0, in the input frame (arcsec).
    muSE_in : float or array
        Source proper motion, East component, in input frame (mas/yr).
    muSN_in : float or array
        Source proper motion, North component, in input frame (mas/yr).
    t0_in : float or array
        Time at which minimum source-lens projected separation in the rectilinear
        frame occurs, in input frame (MJD, days).
    u0_in : float or array
        Minimum source-lens projected separation in the rectilinear frame,
        in units of the Einstein radius, in input frame (dimensionless).
    tE_in : float or array
        Einstein crossing time, in input frame (days).
    piEE_in : float or array
        Microlensing parallax, East component, in input frame (dimensionless).
    piEN_in : float or array
        Microlensing parallax, North component, in input frame (dimensionless).
    t0par : float
        Reference time for the geocentric projected coordinate system
        (MJD, days).
    in_frame : str
        'helio' if converting from Solar System barycentric to geocentric projected frame.
        'geo' if converting from geocentric projected to Solar System barycentric frame.
    murel_in : str
        Definition of "relative" for the input relative proper motion.
        'SL' if relative proper motion in the input parameters is defined as source-lens.
        'LS' if relative proper motion in the input parameters is defined as lens-source.
    murel_out : str
        Definition of "relative" for the output relative proper motion.
        'SL' if relative proper motion in the output parameters is defined as source-lens.
        'LS' if relative proper motion in the output parameters is defined as lens-source.
    coord_in : str
        Definition of coordinate system used to define input parameters.
        'EN' if using fixed on-sky East-North coordinate system (Lu)
        'tb' if using right-handed tau-beta system based on murel and minimum separation (Gould)
    coord_out : str
        Definition of coordinate system used to define output parameters.
        'EN' if using fixed on-sky East-North coordinate system (Lu)
        'tb' if using right-handed tau-beta system based on murel and minimum separation (Gould)
    plot : bool

    Returns
    -------
    xS0E_out : float or array
        East component of the source position relative to RA and Dec,
        at the output frame's t0, in the new frame (arcsec).
    xS0N_out : float or array
        North component of the source position relative to RA and Dec,
        at the output frame's t0, in the new frame (arcsec).
    muSE_out : float or array
        East component of the source proper motion in the new frame (mas/yr).
    muSN_out : float or array
        North component of the source proper motion in the new frame (mas/yr).
    """
    days_per_year = 365.25
    mas_to_arcsec = 1e-3

    if in_frame not in ('helio', 'geo'):
        raise Exception('in_frame can only be "helio" or "geo"!')

    if type(ra) == str:
        ra = str(str(Angle(ra, unit = u.hourangle)))

    # Earth's parallax vector (AU) and its rate (AU/yr) at t0par.
    par_t0par = parallax.parallax_in_direction(ra, dec, t0par)
    dp_dt_t0par = parallax.dparallax_dt_in_direction(ra, dec, t0par)

    t0_out, _, _, _, _ = convert_helio_geo_phot(ra, dec,
                                                t0_in, u0_in, tE_in,
                                                piEE_in, piEN_in, t0par,
                                                in_frame,
                                                murel_in, murel_out,
                                                coord_in, coord_out,
                                                plot=plot)

    # +1 adds Earth's linearized parallax motion (helio -> geo);
    # -1 removes it (geo -> helio).
    sign = 1.0 if in_frame == 'helio' else -1.0

    # Elapsed years from the input t0 to the output t0, and from t0par to
    # the heliocentric t0 (the epoch where the parallax offset is applied).
    dt_out_in = (t0_out - t0_in) / days_per_year
    t0_helio = t0_in if in_frame == 'helio' else t0_out
    t0_geo = t0_out if in_frame == 'helio' else t0_in

    xS0_in = np.array([xS0E_in, xS0N_in], dtype=float)
    muS_in = np.array([muSE_in, muSN_in], dtype=float)
    out = []
    for i in range(2):
        # Proper motion: add/remove piS * dP0 (mas * AU/yr = mas/yr).
        muS_out_i = muS_in[i] + sign * piS * dp_dt_t0par[i]

        # Position at the output t0: carry the input motion to t0_out, then
        # add/remove Earth's linearized parallax offset at that epoch.
        if in_frame == 'helio':
            offset = (piS * par_t0par[i] +
                      piS * dp_dt_t0par[i] * (t0_geo - t0par) / days_per_year)
        else:
            offset = (piS * par_t0par[i] +
                      piS * dp_dt_t0par[i] * (t0_helio - t0par) / days_per_year)
        xS0_out_i = xS0_in[i] + mas_to_arcsec * (muS_in[i] * dt_out_in +
                                                 sign * offset)
        out.append((xS0_out_i, muS_out_i))

    (xS0E_out, muSE_out), (xS0N_out, muSN_out) = out

    return xS0E_out, xS0N_out, muSE_out, muSN_out


def _check_input_convert_helio_geo_phot(ra, dec, 
                                        t0_in, u0_in, tE_in, 
                                        piEE_in, piEN_in, t0par,
                                        in_frame,
                                        murel_in, murel_out, 
                                        coord_in, coord_out,
                                        plot):
    """
    Check the inputs to convert_helio_geo_phot.
    This is an internal code and not to be called except by convert_helio_geo_phot.

    Parameters
    ----------
    ra, dec : str, float, or int
        Equatorial coordinates of the microlensing event.
    t0_in : float or array-like
        Input closest-approach time (MJD).
    u0_in : float or array-like
        Input signed impact parameter (Einstein-radius units).
    tE_in : float or array-like
        Input Einstein timescale (days).
    piEE_in : float or array-like
        Input east microlensing parallax component.
    piEN_in : float or array-like
        Input north microlensing parallax component.
    t0par : float
        Reference time for geocentric-projected quantities (MJD).
    in_frame : str
        Input frame selector: `'helio'` or `'geo'`.
    murel_in : str
        Input relative-motion convention: `'SL'` or `'LS'`.
    murel_out : str
        Output relative-motion convention: `'SL'` or `'LS'`.
    coord_in : str
        Input coordinate convention: `'EN'` or `'tb'`.
    coord_out : str
        Output coordinate convention: `'EN'` or `'tb'`.
    plot : bool
        Plotting flag passed through to conversion utilities.

    Returns
    -------
    None
        This function validates inputs and raises an exception on invalid values.
    """

    var_str = ['ra', 'dec']
    for vv, var in enumerate([ra, dec]):
        if not isinstance(var, (int, float, str)):
            raise Exception('{0} ({1}) must be an integer, float, or string.'.format(var_str[vv], var))

    var_str = ['t0_in', 'u0_in', 'tE_in', 'piEE_in', 'piEN_in']
    array_like = 5 * [True]
    for vv, var in enumerate([t0_in, u0_in, tE_in, piEE_in, piEN_in]):
        if not hasattr(var, '__len__'):
            array_like[vv] = False
            if not isinstance(var, (int, float)):
                raise Exception('{0} ({1}) must be an integer, float, or array-like.'.format(var_str[vv], var))

    if np.sum(array_like) > 1:
        lens = [len([t0_in, u0_in, tE_in, piEE_in, piEN_in][i]) for i, x in enumerate(array_like) if x]
        if len(np.unique(lens)) > 1:
            raise Exception('t0_in, u0_in, tE_in, piEE_in, piEN_in must all be the same length, floats, or integers.')

    if not isinstance(t0par, (int, float)):
        raise Exception('t0par ({0}) must be either an integer or float.'.format(in_frame))
    
    if in_frame != 'helio' and in_frame != 'geo':
        raise Exception('in_frame ({0}) must be either "helio" or "geo".'.format(in_frame))

    var_str = ['murel_in', 'murel_out']
    for vv, var in enumerate([murel_in, murel_out]):
        if var != 'SL' and var != 'LS':
            raise Exception('{0} ({1}) must be "SL" or "LS".'.format(var_str[vv], in_frame))

    var_str = ['coord_in', 'coord_out']
    for vv, var in enumerate([coord_in, coord_out]):
        if var != 'EN' and var != 'tb':
            raise Exception('{0} ({1}) must be "EN" or "tb".'.format(var_str[vv], in_frame))

    if not isinstance(plot, bool):
        raise Exception('plot ({0}) must be a boolean.'.format(plot))


def convert_helio_geo_phot(ra, dec, 
                           t0_in, u0_in, tE_in, 
                           piEE_in, piEN_in, t0par,
                           in_frame='helio',
                           murel_in='SL', murel_out='LS', 
                           coord_in='EN', coord_out='tb',
                           plot=True):
    """
    Convert between Solar System barycentric and geocentric-projected parameters.
    This converts only the subset of parameters in photometry fits
    (t0, u0, tE, piEE, piEN).

    The default conversion assumes that inputs are in
    source-lens (S - L) and EN coordinate conventions.

    Parameters
    ----------
    ra, dec : str, float, or int
        Equatorial coordinates of the microlensing event.
        
        If string, needs to be of the form 
        'HH:MM:SS.SSSS', 'DD:MM:SS.SSSS'

    t0_in : float or array (MJD)
        Time at which minimum source-lens projected separation in the rectilinear
        frame occurs, in input frame.

    u0_in : float or array (thetaE)
        Minimum source-lens projected separation in the rectilinear frame,
        in units of the Einstein radius, in input frame.
    
    tE_in : float or array (days)
        Einstein crossing time, in input frame.
    
    piEE_in : float or array
        Microlensing parallax, East component, in input frame.

    piEN_in : float or array
        Microlensing parallax, North component, in input frame.

    t0par : float (MJD)
        Reference time for the geocentric projected coordinate system.

    in_frame : str
        'helio' if converting from Solar System barycentric to geocentric projected frame.
        'geo' if converting from geocentric projected to Solar System barycentric frame.

    murel_in : str
        Definition of "relative" for the input relative proper motion.
        'SL' if relative proper motion in the input parameters is defined as source-lens.
        'LS' if relative proper motion in the input parameters is defined as lens-source.

    murel_out : str
        Definition of "relative" for the output relative proper motion.
        'SL' if relative proper motion in the output parameters is defined as source-lens.
        'LS' if relative proper motion in the output parameters is defined as lens-source.
    
    coord_in : str
        Definition of coordinate system used to define input parameters.
        'EN' if using fixed on-sky East-North coordinate system (Lu)
        'tb' if using right-handed tau-beta system based on murel and minimum separation (Gould)

    coord_out : str
        Definition of coordinate system used to define output parameters.
        'EN' if using fixed on-sky East-North coordinate system (Lu)
        'tb' if using right-handed tau-beta system based on murel and minimum separation (Gould)
    plot : bool
        If `True`, show conversion diagnostic plots.

    Returns
    -------
    t0_out : float or array
        Time of closest approach in the new frame. (MJD)
    u0_out : float or array
        Closest approach distance in the new frame. (thetaE)
    tE_out : float or array
        Eisntein crossing time in the new frame (MJD)
    piEE_out : float or array
        Microlensing parallax in the East direction in the new frame.
    piEN_out : float or array
        Microlensing parallax in the North direction in the new frame.
    """

    # Check inputs.
    _check_input_convert_helio_geo_phot(ra, dec, 
                                        t0_in, u0_in, tE_in, 
                                        piEE_in, piEN_in, t0par,
                                        in_frame,
                                        murel_in, murel_out, 
                                        coord_in, coord_out,
                                        plot)
    
    if type(ra) == str:
        ra = str(str(Angle(ra, unit = u.hourangle)))

    # Flip from LS to SL as needed (conversion equations assume SL)
    if murel_in=='LS':
        piEE_in *= -1
        piEN_in *= -1

    #####
    # Calculate piEE, piEN, and tE. 
    #####
    piEE_out, piEN_out, tE_out = convert_piEvec_tE(ra, dec, t0par, 
                                                   piEE_in, piEN_in, tE_in, 
                                                   in_frame=in_frame)

    piE = np.hypot(piEE_in, piEN_in)

    #####
    # Calculate u0 and t0.
    #####
    # Define tauhat vector (same direction as piE vector).
    tauhatE_in = piEE_in/piE
    tauhatN_in = piEN_in/piE
    tauhatE_out = piEE_out/piE
    tauhatN_out = piEN_out/piE

    # Define the u0hat vector, which is orthogonal to the tauhat vector.
    # It is the direction of source-lens separation. Note that this is NOT 
    # always the same direction as Gould's beta vector (they are sometimes
    # antiparallel).
    if coord_in=='EN':
        try:
            # Handle conversion of single values.
            if np.sign(u0_in * piEN_in) < 0:
                u0hatE_in = -tauhatN_in
                u0hatN_in = tauhatE_in
            elif np.sign(u0_in * piEN_in) > 0:
                u0hatE_in = tauhatN_in
                u0hatN_in = -tauhatE_in
            else:
                if np.sign(u0_in * piEE_in) > 0:
                    u0hatE_in = -tauhatN_in
                    u0hatN_in = tauhatE_in
                else:
                    u0hatE_in = tauhatN_in
                    u0hatN_in = -tauhatE_in
                
        except:
            cond1 = np.sign(u0_in * piEN_in) < 0
            cond2 = np.sign(u0_in * piEN_in) > 0
            cond3 = (np.sign(u0_in * piEN_in) == 0) & (np.sign(u0_in * piEE_in) > 0)
            cond4 = (np.sign(u0_in * piEN_in) == 0) & (np.sign(u0_in * piEE_in) <= 0)
        
            u0hatE_in = np.where(cond1, -tauhatN_in, np.where(cond2,  tauhatN_in, np.where(cond3, -tauhatN_in, tauhatN_in)))
            u0hatN_in = np.where(cond1,  tauhatE_in, np.where(cond2, -tauhatE_in, np.where(cond3,  tauhatE_in, -tauhatE_in)))
            
    elif coord_in=='tb':
        try:
            # Handle conversion of single values.
            if np.sign(u0_in) > 0:
                u0hatE_in = tauhatN_in
                u0hatN_in = -tauhatE_in
            else:
                u0hatE_in = -tauhatN_in
                u0hatN_in = tauhatE_in
        except:
            # Handle conversions with array-like inputs.
            _u0hatE_in = tauhatN_in
            _u0hatN_in = -tauhatE_in
            u0hatE_in = np.where(np.sign(u0_in) > 0, _u0hatE_in, -_u0hatE_in)
            u0hatN_in = np.where(np.sign(u0_in) > 0, _u0hatN_in, -_u0hatN_in)

    else:
        raise Exception('coord_in can only be "EN" or "tb"!')

    # Calculate t0 and u0 vector.
    t0_out, u0vec_out = convert_u0vec_t0(ra, dec, t0par, t0_in, u0_in, tE_in, tE_out, piE, 
                                         tauhatE_in, tauhatN_in, u0hatE_in, u0hatN_in, 
                                         tauhatE_out, tauhatN_out,
                                         in_frame=in_frame)

    # Now get u0 (the scalar) and its associated sign from u0 vector.
    try:
        # Handle conversion of single values.
        if u0vec_out[0] > 0:
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])
        else:
            u0_out = -np.hypot(u0vec_out[0], u0vec_out[1])
    except:
        # Handle conversions with array-like inputs.
        u0_out = np.zeros(len(t0_in))
        _u0_out = np.hypot(u0vec_out[:,0], u0vec_out[:,1])
        u0_out = np.where(u0vec_out[:,0] > 0, _u0_out, -_u0_out)

    if plot:
        #####
        # Plot conversion diagrams. 
        #####
        # Parallax vector (Sun-Earth projected separation vector in AU) at t0par.
        par_t0par = parallax.parallax_in_direction(ra, dec, np.array([t0par])).reshape(2, )
        
        tau_in = (t0par - t0_in)/tE_in
        tau_out = (t0par - t0_out)/tE_out
        
        vec_u0_in = [np.abs(u0_in)*u0hatE_in, 
                     np.abs(u0_in)*u0hatN_in]
        vec_tau_in = [tau_in*tauhatE_in,
                      tau_in*tauhatN_in]
        vec_u0_out = u0vec_out
        vec_tau_out = [tau_out*tauhatE_out,
                       tau_out*tauhatN_out]
        vec_par = [par_t0par[0],
                   par_t0par[1]]
        
        plot_conversion_diagram(vec_u0_in, vec_tau_in, vec_u0_out, vec_tau_out, piE, vec_par, in_frame,
                                t0par, t0_in, u0_in, tE_in, piEE_in, piEN_in,
                                t0_out, u0_out, tE_out, piEE_out, piEN_out)

    # Transform from tb to EN as needed (so user gets back what they expect).
    if coord_out=='tb':
        # 2D cross product. NumPy 2 rejects np.cross on length-2 vectors.
        # Array events store East on axis 0 and North on axis 1.
        u0_arr = np.asarray(u0vec_out)
        x = np.sign(tauhatE_out * u0_arr[1] - tauhatN_out * u0_arr[0])
        hyp = np.hypot(u0_arr[0], u0_arr[1])
        if np.ndim(x) == 0:
            u0_out = -hyp if x > 0 else hyp
        else:
            u0_out = np.where(x < 0, hyp, -hyp)

    # Flip from LS to SL as needed (so user gets back what they expect).
    if murel_out=='LS':
        piEE_out *= -1
        piEN_out *= -1

    return t0_out, u0_out, tE_out, piEE_out, piEN_out


def convert_u0vec_t0(ra, dec, t0par, t0_in, u0_in, tE_in, tE_out, piE,
                     tauhatE_in, tauhatN_in, u0hatE_in, u0hatN_in, 
                     tauhatE_out, tauhatN_out,
                     in_frame='helio'):
    # FIXME: Fix the broadcasting stuff using the check above for the lenght
    # don't hardcode as t0_in.
    """
    Convert the values of u0 vector and t0 between the Solar System barycentric and geocentric projected frames.

    Note: 
    *** PROPER MOTIONS ARE DEFINED AS SOURCE - LENS ***
    *** COORDINATE SYSTEM IS EAST-NORTH ON-SKY (NOT TAU-BETA) ***
    *** VECTORS ARE ARRAYS DEFINED AS [E, N] ***

    ra, dec : str, float, or int
        Equatorial coordinates of the microlensing event.
        
        If string, needs to be of the form 
        'HH:MM:SS.SSSS', 'DD:MM:SS.SSSS'

    t0par : float (MJD)
        Reference time for the geocentric projected coordinate system.

    t0_in : float or array (MJD)
        Time at which minimum source-lens projected separation in the rectilinear
        frame occurs, in input frame.

    u0_in : float or array (thetaE)
        Minimum source-lens projected separation in the rectilinear frame,
        in units of the Einstein radius, in input frame.
    
    tE_in : float or array (days)
        Einstein crossing time, in input frame.

    tE_out : float or array (days)
        Einstein crossing time, in output frame.

    piE : float or array
        Microlensing parallax.
        Note that this is the same in both the input and output frame (invariant).
    
    in_frame : str
        'helio' if converting from Solar System barycentric to geocentric projected frame.
        'geo' if converting from geocentric projected to Solar System barycentric frame.

    tauhatE_in : float or array
        East component of the input proper-motion unit vector.
    tauhatN_in : float or array
        North component of the input proper-motion unit vector.
    u0hatE_in : float or array
        East component of the input impact-parameter unit vector.
    u0hatN_in : float or array
        North component of the input impact-parameter unit vector.
    tauhatE_out : float or array
        East component of the output proper-motion unit vector.
    tauhatN_out : float or array
        North component of the output proper-motion unit vector.

    Returns
    -------
    t0_out : float or np.ndarray
        Converted closest-approach time in the output frame (MJD).
    u0vec_out : np.ndarray
        Converted impact-parameter vector(s) in East/North coordinates.

    """
    # Parallax vector (Sun-Earth projected separation vector in AU) at t0par.
    # Get dp_dt_t0par in 1/days.
    par_t0par = parallax.parallax_in_direction(ra, dec, np.array([t0par])).reshape(2, )

    # NOTE: dp_dt_t0par doesn't seem quite as good as calculating it from tauhat/tE/piE...
    # not sure why this is....
    # dp_dt_t0par = parallax.dparallax_dt_in_direction(ra, dec, np.array([t0par])).reshape(2,)/365.25
        
    # Calculate a bunch of values we need to get u0 and t0.
    tau_in = (t0par - t0_in)/tE_in
    u0vec_in = np.abs(u0_in) * np.array([u0hatE_in, u0hatN_in])
    try:
        tauvec_in = tau_in * np.array([tauhatE_in, tauhatN_in])
    except:
        tauvec_in = tau_in[:, np.newaxis] * np.array([tauhatE_in, tauhatN_in])
    uvec_in = u0vec_in + tauvec_in

    # Get direction of tauhat_out and u0hat_out.
    tauhat_out = np.array([tauhatE_out, tauhatN_out])
    tauhat_in = np.array([tauhatE_in, tauhatN_in])
    
    # Actually get u0 and t0.
    if in_frame=='helio':
        try:
            dp_dt_t0par = ((tauhat_in/tE_in)  - (tauhat_out/tE_out))/piE
            t0_out = t0_in - tE_out * np.dot(tauhat_out, u0vec_in - piE*par_t0par - (t0_in - t0par)*piE*dp_dt_t0par)
#            u0vec_out = u0vec_in + ((t0_out - t0_in)/tE_out)*tauhat_out - piE*par_t0par - (t0_in - t0par)*piE*dp_dt_t0par
            u0vec_out = u0vec_in + tauhat_in * (t0par - t0_in)/tE_in - tauhat_out * (t0par - t0_out)/tE_out - piE*par_t0par
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1]) 

        except:
            par_t0par = np.tile(par_t0par,(len(t0_in),1)).T
            dp_dt_t0par = ((tauhat_in/tE_in)  - (tauhat_out/tE_out))/piE
            _vec = u0vec_in - piE*par_t0par - (t0_in - t0par)*piE*dp_dt_t0par
            t0_out = t0_in - tE_out * np.sum(tauhat_out* _vec, axis=0)
            u0vec_out = u0vec_in + tauhat_in * (t0par - t0_in)/tE_in - tauhat_out * (t0par - t0_out)/tE_out - piE*par_t0par
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])

    elif in_frame=='geo':
        try:
            dp_dt_t0par = -1*((tauhat_in/tE_in) - (tauhat_out/tE_out))/piE
            t0_out = t0_in - tE_out * np.dot(tauhat_out, u0vec_in + piE*par_t0par + (t0_in - t0par)*piE*dp_dt_t0par)
        #        u0vec_out = u0vec_in + ((t0_out - t0_in)/tE_out)*tauhat_out + piE*par_t0par + (t0_in - t0par)*piE*dp_dt_t0par
            u0vec_out = u0vec_in + tauhat_in * (t0par - t0_in)/tE_in - tauhat_out * (t0par - t0_out)/tE_out + piE*par_t0par
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])
        except:
            par_t0par = np.tile(par_t0par,(len(t0_in),1)).T
            dp_dt_t0par = ((tauhat_in/tE_in)  - (tauhat_out/tE_out))/piE
            _vec = u0vec_in + piE*par_t0par + (t0_in - t0par)*piE*dp_dt_t0par
            t0_out = t0_in - tE_out * np.sum(tauhat_out* _vec, axis=0)
            u0vec_out = u0vec_in + tauhat_in * (t0par - t0_in)/tE_in - tauhat_out * (t0par - t0_out)/tE_out + piE*par_t0par
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])

    else:
        raise Exception('in_frame can only be "helio" or "geo"!')

    try:
        if u0vec_out[0] > 0:
            u0_out = np.hypot(u0vec_out[0], u0vec_out[1])
        else:
            u0_out = -np.hypot(u0vec_out[0], u0vec_out[1])
    except:
        u0_out = np.zeros(len(t0_out))
        _u0_out = np.hypot(u0vec_out[:,0], u0vec_out[:,1])
        u0_out = np.where(u0vec_out[:,0] > 0, _u0_out, -_u0_out)

    return t0_out, u0vec_out


def convert_piEvec_tE(ra, dec, t0par, 
                      piEE_in, piEN_in, tE_in, 
                      in_frame='helio'):     
    """
    Convert the values of piE vector and tE between the
    Solar System barycentric and geoprojected frame.
    
    !!! NOTE: INPUT AND OUTPUT RELATIVE PROPER MOTION
    (AND HENCE piE VECTOR) ARE DEFINED TO BE SOURCE - LENS !!!

    If you want lens-source output, flip piEE and piEN by -1.

    Parameters
    ----------
    in_frame : str
        'helio' or 'geo'
        If 'helio', convert to helio to geo.
        If 'geo', convert to geo to helio.

    ra, dec : str, float, or int
        Equatorial coordinates.
        
        If string, needs to be of the form 
        'HH:MM:SS.SSSS', 'DD:MM:SS.SSSS'

    t0par : float
        Reference time for the geocentric-projected frame value in MJD.

    piEE_in, piEN_in, tE_in : float
        piEE, piEN, and tE of the event in the frame passed 
        in for in_frame (i.e. either helio or geo)

    Returns
    -------
    piEE_out : float or np.ndarray
        Microlensing parallax east component in the output frame.
    piEN_out : float or np.ndarray
        Microlensing parallax north component in the output frame.
    tE_out : float or np.ndarray
        Einstein timescale in the output frame (days).
    """
    if in_frame not in ['helio', 'geo']:
        raise Exception('in_frame must be either helio or geo.')

    # Convert from AU/day to km/s.
    au_day_to_km_s = 1731.45683

    # Magnitude of the microlensing parallax piE
    # and magnitude squared piE2.
    piE = np.hypot(piEE_in, piEN_in)
    piE2 = piEE_in**2 + piEN_in**2

    # vtilde (projected velocity) in units of AU/day.
    vtildeN_in = piEN_in/(tE_in * piE2)
    vtildeE_in = piEE_in/(tE_in * piE2) 
    
    # Convert vtilde_in from AU/day to km/s.
    vtildeN_in *= au_day_to_km_s
    vtildeE_in *= au_day_to_km_s

    # Get Earth's instantaneous velocity at t0par 
    # to get Solar System barycentric velocity as projected on
    # sky (in km/s)
    v_Earth_perp_E, v_Earth_perp_N = v_Earth_proj(ra, dec, t0par)

    # Convert vtilde.
    if in_frame=='helio':
        vtildeN_out = -vtildeN_in - v_Earth_perp_N
        vtildeE_out = -vtildeE_in - v_Earth_perp_E
    elif in_frame=='geo':
        vtildeN_out = -vtildeN_in + v_Earth_perp_N
        vtildeE_out = -vtildeE_in + v_Earth_perp_E
    else:
        raise Exception('in_frame can only be "helio" or "geo"!')

    # Convert piEE, piEN, and tE
    vtilde_in = np.hypot(vtildeE_in, vtildeN_in)
    vtilde_out = np.hypot(vtildeE_out, vtildeN_out)
    e_out = -vtildeE_out/vtilde_out
    n_out = -vtildeN_out/vtilde_out

    piEE_out = piE*e_out 
    piEN_out = piE*n_out 
    tE_out = (vtilde_in/vtilde_out) * tE_in
    import pdb
#    pdb.set_trace()
    
    return piEE_out, piEN_out, tE_out


def v_Earth_proj(ra, dec, mjd):
    """
    !!! This is the Earth-Sun vector direction. 
    (This is the opposite of how we define the parallax vector.)
    Returns in units of km/s.

    Based on MulensModel code calculation.
    (I just took out all the object oriented stuff.)
    
    ra, dec : str, float, or int
        Equatorial coordinates.
        
        If string, needs to be of the form 
        'HH:MM:SS.SSSS', 'DD:MM:SS.SSSS'

        If float or int, needs to be in degrees.
        
    mjd : float
        Time in MJD.

    Returns
    -------
    v_Earth_perp_E : float or np.ndarray
        East component of Earth's projected barycentric velocity (km/s).
    v_Earth_perp_N : float or np.ndarray
        North component of Earth's projected barycentric velocity (km/s).
    """
    # Convert from AU/day to km/s.
    au_day_to_km_s = 1731.45683

    # Make this check dec too.
    if type(ra) == str:
        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    elif (isinstance(ra, float) or isinstance(ra, int)):
        coord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    else:
        raise Exception('ra and dec must be either strings or int/floats.')

    direction = coord.cartesian.xyz.value    
    north = np.array([0., 0., 1.])
    _east_projected = np.cross(north, direction)/np.linalg.norm(np.cross(north, direction))
    _north_projected = np.cross(direction, _east_projected)/np.linalg.norm(np.cross(direction, _east_projected))

    jd = mjd+2400000.5
    time = Time(jd, format='jd', scale='tdb')
    (jd1, jd2) = get_jd12(time, 'tdb')
    (earth_pv_helio, earth_pv_bary) = erfa.epv00(jd1, jd2) # this is earth-sun. or is it earth - SSB?
    # erfa returns things in a weird format... 
    # recast results into numpy array so we can use broadcasting.
    try:
        nn = earth_pv_bary.shape[0]
        earth_pv_bary_arr = np.empty((nn, 3))
        for ii in np.arange(nn):
            earth_pv_bary_arr[ii] = earth_pv_bary[ii][1]
        velocity = earth_pv_bary_arr * au_day_to_km_s
    except:
        velocity = np.asarray(earth_pv_bary[1]) * au_day_to_km_s

    v_Earth_perp_N = np.dot(velocity, _north_projected)
    v_Earth_perp_E = np.dot(velocity, _east_projected)

    return v_Earth_perp_E, v_Earth_perp_N


def plot_conversion_diagram(vec_u0_in, vec_tau_in, vec_u0_out, vec_tau_out, 
                            piE, vec_par, in_frame,
                            t0par, t0_in, u0_in, tE_in, piEE_in, piEN_in,
                            t0_out, u0_out, tE_out, piEE_out, piEN_out):
    """
    The parallax vector direction is always Earth to Sun (blue to red).
    All the input values are in the Solar System barycentric frame, following the
    source-lens and East-North coordinate conventions.

    FIXME: There seems to be some bugs in the reported values in the side panel still.

    Parameters
    ----------
    vec_u0_in : array-like
        Input-frame impact-parameter vector in East/North components.
    vec_tau_in : array-like
        Input-frame trajectory (`tau`) vector in East/North components.
    vec_u0_out : array-like
        Output-frame impact-parameter vector in East/North components.
    vec_tau_out : array-like
        Output-frame trajectory (`tau`) vector in East/North components.
    piE : float
        Microlensing parallax magnitude.
    vec_par : array-like
        Projected Earth-Sun parallax vector components at `t0par`.
    in_frame : str
        Frame of the supplied input parameters (`'helio'` or `'geo'`).
    t0par : float
        Geocentric reference time (MJD).
    t0_in : float
        Input-frame closest-approach time (MJD).
    u0_in : float
        Input-frame signed impact parameter.
    tE_in : float
        Input-frame Einstein timescale (days).
    piEE_in : float
        Input-frame east parallax component.
    piEN_in : float
        Input-frame north parallax component.
    t0_out : float
        Output-frame closest-approach time (MJD).
    u0_out : float
        Output-frame signed impact parameter.
    tE_out : float
        Output-frame Einstein timescale (days).
    piEE_out : float
        Output-frame east parallax component.
    piEN_out : float
        Output-frame north parallax component.

    Returns
    -------
    None
        Displays diagnostic plots for the frame conversion and pauses briefly.
    """
    #####
    # Figure in Lu convention (S-L frame, E-N coord)
    ####
    fig, ax = plt.subplots(1, 1, num=2, figsize=(9,6))
    plt.clf()
    fig, ax = plt.subplots(1, 1, num=2, figsize=(9,6))
    plt.subplots_adjust(left=0.15, right=0.7)

    if in_frame == 'helio':
        label_in = 'Src (Hel)'
        label_out = 'Src (Geo $t_r$)'
        color_in = 'red'
        color_out = 'blue'
    if in_frame == 'geo':
        label_in = 'Src (Geo $t_r$)'
        label_out = 'Src (Hel)'
        color_in = 'blue'
        color_out = 'red'

    # u0 vector
    ax.annotate('', xy=(0,0), xycoords='data',
                xytext=(vec_u0_out[0], vec_u0_out[1]), textcoords='data',
                ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_out, lw=2, mutation_scale=15))

    ax.annotate('', xy=(0,0), xycoords='data',
                xytext=(vec_u0_in[0], vec_u0_in[1]), textcoords='data',
                ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_in, lw=2, mutation_scale=15))

    # tau vector (starting at u0 vector)
    ax.annotate('', xy=(vec_u0_out[0], vec_u0_out[1]), xycoords='data',
                xytext=(vec_u0_out[0] + vec_tau_out[0], vec_u0_out[1] + vec_tau_out[1]), textcoords='data',
                ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_out, lw=2, mutation_scale=15, shrinkA=0))

    ax.annotate('', xy=(vec_u0_in[0], vec_u0_in[1]), xycoords='data',
                xytext=(vec_u0_in[0] + vec_tau_in[0], vec_u0_in[1] + vec_tau_in[1]), textcoords='data',
                ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_in, lw=2, mutation_scale=15, shrinkA=0))

    # Plot the lens (i.e. origin)
    ax.plot(0, 0, 'o', ms=8, mec='k', color='k', label='Lens')

    # Plot the source (both helio and geo proj positions at t0)
    ax.plot(vec_u0_in[0], vec_u0_in[1], 'o', ms=8, mec=color_in, color='yellow', mew=2, label=label_in)
    ax.plot(vec_u0_out[0], vec_u0_out[1], 'o', ms=8, mec=color_out, color='yellow', mew=2, label=label_out)

    # Parallax vector
    if in_frame=='helio':
        ax.annotate('', xy=(vec_u0_out[0] + vec_tau_out[0], vec_u0_out[1] + vec_tau_out[1]), xycoords='data',
                    xytext=(vec_u0_out[0] + vec_tau_out[0] + vec_par[0]*piE, vec_u0_out[1] + vec_tau_out[1] + vec_par[1]*piE),
                    textcoords='data', ha='right', 
                    arrowprops=dict(arrowstyle= '<|-', color='gray', lw=2, mutation_scale=15, shrinkA=0, shrinkB=0))
    else:
        ax.annotate('', xy=(vec_u0_out[0] + vec_tau_out[0], vec_u0_out[1] + vec_tau_out[1]), xycoords='data',
                    xytext=(vec_u0_out[0] + vec_tau_out[0] - vec_par[0]*piE, vec_u0_out[1] + vec_tau_out[1] - vec_par[1]*piE),
                    textcoords='data', ha='right', 
                    arrowprops=dict(arrowstyle= '<|-', color='gray', lw=2, mutation_scale=15, shrinkA=0, shrinkB=0))

#    # Not sure why these next two lines are totally necessary...
    ax.plot(vec_u0_in[0] + vec_tau_in[0], vec_u0_in[1] + vec_tau_in[1], 'o', ms=0.001, color=color_in)
    ax.plot(vec_u0_out[0] + vec_tau_out[0], vec_u0_out[1] + vec_tau_out[1], 'o', ms=0.001, color=color_out)

    ax.legend()
    ax.axhline(y=0, ls=':', color='gray')
    ax.axvline(x=0, ls=':', color='gray')
    ax.set_xlabel('$u_E$')
    ax.set_ylabel('$u_N$')
    ax.axis('equal')
    ax.invert_xaxis()
    ax.set_title('Lu convention (S-L frame, E-N coord)')
    
    # Text panel listing parameters.
    tleft = 0.75
    ttop = 0.8
    ttstep = 0.05

    fig.text(tleft, ttop - -1*ttstep, '$t_r$ = {0:.1f} MJD'.format(t0par), fontsize=12)

    # Input and output are helio, Lu convention. So parameters are as reported.
    if in_frame == 'helio':
        fig.text(tleft, ttop - 1*ttstep, 'Helio', weight='bold', fontsize=14)
        fig.text(tleft, ttop - 9*ttstep, 'Geo $t_r$', weight='bold', fontsize=14)

        fig.text(tleft, ttop - 2*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_in), fontsize=12)
        fig.text(tleft, ttop - 3*ttstep, '$u_0$ = {0:.2f}'.format(u0_in), fontsize=12)
        fig.text(tleft, ttop - 4*ttstep, '$t_E$ = {0:.1f} days'.format(tE_in), fontsize=12)
        fig.text(tleft, ttop - 5*ttstep, '$\\pi_{{E,E}}$ = {0:.2f}'.format(piEE_in), fontsize=12)
        fig.text(tleft, ttop - 6*ttstep, '$\\pi_{{E,N}}$ = {0:.2f}'.format(piEN_in), fontsize=12)

        fig.text(tleft, ttop - 10*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_out), fontsize=12)
        fig.text(tleft, ttop - 11*ttstep, '$u_0$ = {0:.2f}'.format(u0_out), fontsize=12)
        fig.text(tleft, ttop - 12*ttstep, '$t_E$ = {0:.1f} days'.format(tE_out), fontsize=12)
        fig.text(tleft, ttop - 13*ttstep, '$\\pi_{{E,E}}$ = {0:.2f}'.format(piEE_out), fontsize=12)
        fig.text(tleft, ttop - 14*ttstep, '$\\pi_{{E,N}}$ = {0:.2f}'.format(piEN_out), fontsize=12)

    if in_frame == 'geo':
        fig.text(tleft, ttop - 1*ttstep, 'Geo $t_r$', weight='bold', fontsize=14)
        fig.text(tleft, ttop - 9*ttstep, 'Helio', weight='bold', fontsize=14)

        fig.text(tleft, ttop - 2*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_in), fontsize=12)
        # Input is Gould geo, so need to fix those to play nice with our vectors in Lu helio. 
        if np.sign(u0_in * piEN_in) > 0:
            fig.text(tleft, ttop - 3*ttstep, '$u_0$ = {0:.2f}'.format(np.abs(u0_in)), fontsize=12)
        else:
            fig.text(tleft, ttop - 3*ttstep, '$u_0$ = {0:.2f}'.format(-np.abs(u0_in)), fontsize=12)

        fig.text(tleft, ttop - 4*ttstep, '$t_E$ = {0:.1f} days'.format(tE_in), fontsize=12)
        fig.text(tleft, ttop - 5*ttstep, '$\\pi_{{E,E}}$ = {0:.2f}'.format(piEE_in), fontsize=12)
        fig.text(tleft, ttop - 6*ttstep, '$\\pi_{{E,N}}$ = {0:.2f}'.format(piEN_in), fontsize=12)

        fig.text(tleft, ttop - 10*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_out), fontsize=12)
        fig.text(tleft, ttop - 11*ttstep, '$u_0$ = {0:.2f}'.format(u0_out), fontsize=12)
        fig.text(tleft, ttop - 12*ttstep, '$t_E$ = {0:.1f} days'.format(tE_out), fontsize=12)
        fig.text(tleft, ttop - 13*ttstep, '$\\pi_{{E,E}}$ = {0:.2f}'.format(piEE_out), fontsize=12)
        fig.text(tleft, ttop - 14*ttstep, '$\\pi_{{E,N}}$ = {0:.2f}'.format(piEN_out), fontsize=12)

    plt.show()
    plt.pause(1)

    #####
    # Figure in Gould convention (L-S frame, tau-beta coord)
    ####
    fig, ax = plt.subplots(1, 2, num=4, figsize=(12,6), gridspec_kw={'width_ratios': [2.5, 1]})
    plt.clf()
    fig, ax = plt.subplots(1, 2, num=4, figsize=(12,6), gridspec_kw={'width_ratios': [2.5, 1]})
    plt.subplots_adjust(left=0.15*3/4, right=0.78, wspace=0.3)

    # FIXME: is the parallax vector supposed to be the same direction
    # in Gould frame or antiparallel to the one in Lu???
    if in_frame == 'helio':
        label_in = 'Lens (Hel)'
        label_out = 'Lens (Geo $t_r$)'
        color_in = 'red'
        color_out = 'blue'
    if in_frame == 'geo':
        label_in = 'Lens (Geo $t_r$)'
        label_out = 'Lens (Hel)'
        color_in = 'blue'
        color_out = 'red'

    # u0 vector
    ax[0].annotate('', xy=(0,0), xycoords='data',
                xytext=(-vec_u0_out[0], -vec_u0_out[1]), textcoords='data',
                ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_out, lw=2, mutation_scale=15))

    ax[0].annotate('', xy=(0,0), xycoords='data',
                xytext=(-vec_u0_in[0], -vec_u0_in[1]), textcoords='data',
                ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_in, lw=2, mutation_scale=15))

    # tau vector (starting at u0 vector)
    ax[0].annotate('', xy=(-vec_u0_out[0], -vec_u0_out[1]), xycoords='data',
                   xytext=(-vec_u0_out[0] - vec_tau_out[0], -vec_u0_out[1] - vec_tau_out[1]), textcoords='data',
                   ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_out, lw=2, mutation_scale=15, shrinkA=0))

    ax[0].annotate('', xy=(-vec_u0_in[0], -vec_u0_in[1]), xycoords='data',
                   xytext=(-vec_u0_in[0] - vec_tau_in[0], -vec_u0_in[1] - vec_tau_in[1]), textcoords='data',
                   ha='right', arrowprops=dict(arrowstyle= '<|-', color=color_in, lw=2, mutation_scale=15, shrinkA=0))

    # Plot the source (i.e. origin)
    ax[0].plot(0, 0, 'o', ms=8, mec='k', color='yellow', label='Source')

    # Plot the lens (both helio and geo proj positions at t0)
    ax[0].plot(-vec_u0_in[0], -vec_u0_in[1], 'o', ms=8, mec=color_in, mew=2, color='k', label=label_in)
    ax[0].plot(-vec_u0_out[0], -vec_u0_out[1], 'o', ms=8, mec=color_out, mew=2, color='k', label=label_out)

    # Parallax vector
    if in_frame=='helio':
        ax[0].annotate('', xy=(-vec_u0_out[0] - vec_tau_out[0], -vec_u0_out[1] - vec_tau_out[1]), xycoords='data',
                    xytext=(-vec_u0_out[0] - vec_tau_out[0] - vec_par[0]*piE, -vec_u0_out[1] - vec_tau_out[1] - vec_par[1]*piE),
                    textcoords='data', ha='right', 
                    arrowprops=dict(arrowstyle= '<|-', color='gray', lw=2, mutation_scale=15, shrinkA=0, shrinkB=0))
    else:
        ax[0].annotate('', xy=(-vec_u0_out[0] - vec_tau_out[0], -vec_u0_out[1] - vec_tau_out[1]), xycoords='data',
                    xytext=(-vec_u0_out[0] - vec_tau_out[0] + vec_par[0]*piE, vec_u0_out[1] - vec_tau_out[1] + vec_par[1]*piE),
                    textcoords='data', ha='right', 
                    arrowprops=dict(arrowstyle= '<|-', color='gray', lw=2, mutation_scale=15, shrinkA=0, shrinkB=0))

    # Not sure why these next two lines are totally necessary...
    ax[0].plot(-vec_u0_in[0] - vec_tau_in[0], -vec_u0_in[1] - vec_tau_in[1], 'o', ms=0.001, color=color_in)
    ax[0].plot(-vec_u0_out[0] - vec_tau_out[0], -vec_u0_out[1] - vec_tau_out[1], 'o', ms=0.001, color=color_out)

    ax[0].legend()
    ax[0].axhline(y=0, ls=':', color='gray')
    ax[0].axvline(x=0, ls=':', color='gray')
    ax[0].set_xlabel('$u_E$')
    ax[0].set_ylabel('$u_N$')
    ax[0].axis('equal')
    ax[0].invert_xaxis()
    ax[0].set_title(r'Gould convention (L-S frame, $\tau$-$\beta$ coord)')

    q_tau_out = ax[1].quiver(0, 0, -piEE_out/piE, -piEN_out/piE, 
                           color=color_out, scale_units='xy', angles='xy', scale=1)
    
    q_tau_in = ax[1].quiver(0, 0, -piEE_in/piE, -piEN_in/piE, 
                          color=color_in, scale_units='xy', angles='xy', scale=1)

    q_u0_out = ax[1].quiver(0, 0, -piEN_out/piE, piEE_out/piE, 
                          color=color_out, scale_units='xy', angles='xy', scale=1, alpha=0.5, width=0.02)
    
    q_u0_in = ax[1].quiver(0, 0, -piEN_in/piE, piEE_in/piE, 
                         color=color_in, scale_units='xy', angles='xy', scale=1, alpha=0.5, width=0.02)
    
    if in_frame == 'helio':
        ax[1].quiverkey(q_tau_out, 1, -2.0, 0.3, r'$\hat{\tau}_G$', coordinates='data', labelpos='E', labelsep=0.5)
        ax[1].quiverkey(q_tau_in, 1, -2.4, 0.3, r'$\hat{\tau}_H$', coordinates='data', labelpos='E', labelsep=0.5)
        ax[1].quiverkey(q_u0_out, 0, -2.0, 0.3, r'$\hat{\beta}_G$', coordinates='data', labelpos='E', labelsep=0.5)
        ax[1].quiverkey(q_u0_in, 0, -2.4, 0.3, r'$\hat{\beta}_H$', coordinates='data', labelpos='E', labelsep=0.5)
    else:
        ax[1].quiverkey(q_tau_out, 1, -2.0, 0.3, r'$\hat{\tau}_H$', coordinates='data', labelpos='E', labelsep=0.5)
        ax[1].quiverkey(q_tau_in, 1, -2.4, 0.3, r'$\hat{\tau}_G$', coordinates='data', labelpos='E', labelsep=0.5)
        ax[1].quiverkey(q_u0_out, 0, -2.0, 0.3, r'$\hat{\beta}_H$', coordinates='data', labelpos='E', labelsep=0.5)
        ax[1].quiverkey(q_u0_in, 0, -2.4, 0.3, r'$\hat{\beta}_G$', coordinates='data', labelpos='E', labelsep=0.5)

    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[1].set_xlabel('$E$')
    ax[1].set_ylabel('$N$')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].invert_xaxis()
    ax[1].set_aspect('equal', 'box')

    # Text panel listing parameters.
    tleft = 0.75 * 1.08
    ttop = 0.8
    ttstep = 0.05

    fig.text(tleft, ttop - -1*ttstep, '$t_r$ = {0:.1f} MJD'.format(t0par), fontsize=12)

    if in_frame == 'helio':
        fig.text(tleft, ttop - 1*ttstep, 'Helio', weight='bold', fontsize=14)
        fig.text(tleft, ttop - 9*ttstep, 'Geo $t_r$', weight='bold', fontsize=14)

        fig.text(tleft, ttop - 2*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_in), fontsize=12)
        # Input is Lu helio, so need to fix those to be in Gould geo.
        if np.sign(u0_in * piEN_in) > 0:
            fig.text(tleft, ttop - 3*ttstep, '$u_0$ = {0:.2f}'.format(np.abs(u0_in)), fontsize=12)
        else:
            fig.text(tleft, ttop - 3*ttstep, '$u_0$ = {0:.2f}'.format(-np.abs(u0_in)), fontsize=12)
            
        fig.text(tleft, ttop - 4*ttstep, '$t_E$ = {0:.1f} days'.format(tE_in), fontsize=12)
        fig.text(tleft, ttop - 5*ttstep, '$\\pi_{{E,E}}$ = {0:.2f}'.format(-piEE_in), fontsize=12)
        fig.text(tleft, ttop - 6*ttstep, '$\\pi_{{E,N}}$ = {0:.2f}'.format(-piEN_in), fontsize=12)

        fig.text(tleft, ttop - 10*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_out), fontsize=12)
        # Output is Lu helio, so need to fix those to be in Gould geo.
        if np.sign(u0_out * piEN_out) > 0:
            fig.text(tleft, ttop - 11*ttstep, '$u_0$ = {0:.2f}'.format(np.abs(u0_out)), fontsize=12)
        else:
            fig.text(tleft, ttop - 11*ttstep, '$u_0$ = {0:.2f}'.format(-np.abs(u0_out)), fontsize=12)

        fig.text(tleft, ttop - 12*ttstep, '$t_E$ = {0:.1f} days'.format(tE_out), fontsize=12)
        fig.text(tleft, ttop - 13*ttstep, '$\\pi_{{E,E}}$ = {0:.2f}'.format(-piEE_out), fontsize=12)
        fig.text(tleft, ttop - 14*ttstep, '$\\pi_{{E,N}}$ = {0:.2f}'.format(-piEN_out), fontsize=12)

    if in_frame == 'geo':
        fig.text(tleft, ttop - 1*ttstep, 'Geo $t_r$', weight='bold', fontsize=14)
        fig.text(tleft, ttop - 9*ttstep, 'Helio', weight='bold', fontsize=14)

        fig.text(tleft, ttop - 2*ttstep, '$t_0$ = {0:.1f} MJD'.format(t0_in), fontsize=12)
        fig.text(tleft, ttop - 3*ttstep, '$u_0$ = {0:.2f}'.format(u0_in), fontsize=12)           
        fig.text(tleft, ttop - 4*ttstep, '$t_E$ = {0:.1f} days'.format(tE_in), fontsize=12)
        fig.text(tleft, ttop - 5*ttstep, '$\\pi_{{E,E}}$ = {0:.2f}'.format(-piEE_in), fontsize=12)
        fig.text(tleft, ttop - 6*ttstep, '$\\pi_{{E,N}}$ = {0:.2f}'.format(-piEN_in), fontsize=12)

        fig.text(tleft, ttop - 10*ttstep, '$t_0$ = {0:.1f}'.format(t0_out), fontsize=12)
        # Output is Lu helio, so need to fix those to be in Gould geo.
        if np.sign(u0_out * piEN_out) > 0:
            fig.text(tleft, ttop - 11*ttstep, '$u_0$ = {0:.2f}'.format(np.abs(u0_out)), fontsize=12)
        else:
            fig.text(tleft, ttop - 11*ttstep, '$u_0$ = {0:.2f}'.format(-np.abs(u0_out)), fontsize=12)
        fig.text(tleft, ttop - 12*ttstep, '$t_E$ = {0:.1f}'.format(tE_out), fontsize=12)
        fig.text(tleft, ttop - 13*ttstep, '$\\pi_{{E,E}}$ = {0:.2f}'.format(-piEE_out), fontsize=12)
        fig.text(tleft, ttop - 14*ttstep, '$\\pi_{{E,N}}$ = {0:.2f}'.format(-piEN_out), fontsize=12)

    plt.show()
    plt.pause(1)


def convert_u0_t0_psbl(t0_in, u0_x_in, u0_y_in, tE, theta_E, 
                       q, phi, sep, mu_rel_x, mu_rel_y,
                       coords_in=None, coords_out=None, d_mas = None):
    """
    Converts time of closest approach (t0) and 
    vector distance of closest approach (u0) for PSBL events.
    The default coordinate system transformations supported are:
    Geometric Midpoint <--> Primary Center
    Geometric Midpoint <--> Center of Mass Center
    
    You can also specify your own transformation by inputing a distance
    along binary axis d_mas. BE AWARE OF THE SIGN OF d_mas.
    If output coord center is closer to primary than input d > 0.
    If output coord center is closer to secondary than input d < 0.
    
    Parameters
    -----------
    t0_in : float
        Time of closest approach between source and lens specified in coords_in in days.
    u0_x_in : float
        u0 in specified coordinate frame in coords_in in x direction.
        x direction is coordinate independent, but standard is East.
    u0_y_in : float
        u0 in specified coordinate frame in coords_in in y direction.
        y direction is coordinate independent, but standard is North.
    tE : float
        Characteristic timescale of microlensing event in days.
    theta_E : float
        Characterisitc lengthscale of microlensing event in mas.
    q : float
        Ratio between M_2/M_1.
    phi : float
        Angle between mu_rel and the binary axis in radians: the position
        angle (East of North) of the secondary -> primary direction minus
        that of mu_rel (BAGLE's ``phi_rad = alpha_rad - PA(thetaE_hat)``).
    sep : float
        Separation between primary and secondary of binary in mas.
    mu_rel_x : float
        Relative proper motion between lens and source in mas/yr in x direction.
        x direction is coordinate independent, but standard is East.
    mu_rel_y : float
        Relative proper motion between lens and source in mas/yr in y direction.
        y direction is coordinate independent, but standard is North.
    
    Optional Parameters
    --------------------
    coords_in : string or None
        Input coordinate system. 
        Must be 'geom_mid', 'prim_center', or 'COM'.
        Default is None.
    coords_out : string or None
        Output coordinate system. 
        Must be 'geom_mid', 'prim_center', or 'COM'.
        Default is None.
    d_mas : float or None
        Distance along binary axis in units of mas.
        BE AWARE OF SIGN! 
        If output coord center is closer to primary than input d > 0.
        If output coord center is closer to secondary than input d < 0.
    
    Returns
    -------
    u0_x_out : float
        u0 in specified coordinate frame in coords_out.
        x direction is coordinate independent, but standard is East.
    u0_y_out : float
        u0 in specified coordinate frame in coords_in.
        y direction is coordinate independent, but standard is North.
    t0_out : float
        Time of closest approach between source and lens specified in coords_out in days.
    """
    
    if coords_in is None and d_mas is None:
        raise Exception('Must specify coord system or distance along binary axis')

    if coords_in is not None and d_mas is not None:
        raise Exception('Can only specify default coordinate transform or distance along binary axis, not both')

    if d_mas is not None:
        # User-supplied shift toward the primary (Einstein units).
        d = d_mas / theta_E
    else:
        valid_coords = ['geom_mid', 'prim_center', 'COM']
        if coords_in not in valid_coords or coords_out not in valid_coords:
            raise Exception('coord_in and coord_out must be one of: {}'.format(valid_coords))

        # Each reference point's offset from the geometric midpoint toward
        # the primary (Einstein units). For q > 1 the CoM offset is negative.
        offset = {'geom_mid': 0.0,
                  'prim_center': 0.5 * sep / theta_E,
                  'COM': 0.5 * (1 - q) / (1 + q) * sep / theta_E}
        d = offset[coords_out] - offset[coords_in]

    # Unit vectors along mu_rel and 90 deg East of it (position angle + 90).
    # The binary axis (toward the primary) is cos(phi) mu_hat + sin(phi) perp_hat.
    mu_amp = np.hypot(mu_rel_x, mu_rel_y)
    mu_hat = np.array([mu_rel_x, mu_rel_y]) / mu_amp
    perp_hat = np.array([mu_hat[1], -mu_hat[0]])

    # The source passes the output point d*cos(phi)*tE later, and its
    # closest-approach vector loses the perpendicular part of the shift.
    t0_out = t0_in + d * np.cos(phi) * tE
    u0_x_out = u0_x_in - d * np.sin(phi) * perp_hat[0]
    u0_y_out = u0_y_in - d * np.sin(phi) * perp_hat[1]

    return u0_x_out, u0_y_out, t0_out

def keplerian_point_offset(orb, t, coords):
    """
    Offset of a binary-lens reference point from the lens CoM at time t.

    Parameters
    ----------
    orb : bagle.orbits.Orbit
        Lens orbit (see :func:`bagle.model.keplerian_lens_orbit`).
    t : float
        Time (MJD).
    coords : {'geom_mid', 'prim_center', 'COM'}
        Geometric midpoint of the two lenses, the primary, or the CoM.

    Returns
    -------
    offset : numpy array, shape = [2]
        (E, N) offset of the point from the CoM, in the units of
        ``orb.aleph``.
    """
    # Primary (x, y) and secondary (x2, y2) offsets from the CoM.
    x, y, x2, y2 = orb.oal2xy(np.array([t]))

    if coords == 'prim_center':
        offset = np.array([x[0], y[0]])
    elif coords == 'geom_mid':
        offset = 0.5 * np.array([x[0] + x2[0], y[0] + y2[0]])
    elif coords == 'COM':
        offset = np.zeros(2)
    else:
        raise ValueError("coords must be 'geom_mid', 'prim_center' or 'COM', "
                         f"not {coords!r}")

    return offset


def convert_u0_amp_t0_psbl_orbit(t0_in, u0_amp_in, tE, thetaE_hat, orb, thetaE_orb,
                                 coords_in, coords_out):
    """
    Move a closest approach between reference points of an orbiting binary lens.

    Parameters
    ----------
    t0_in : float
        Reference time of the input point (MJD); see Notes.
    u0_amp_in : float
        Signed impact parameter about the input point at ``t0_in`` (Einstein
        units), with the :func:`bagle.model.u0_hat_from_thetaE_hat` sign
        convention.
    tE : float
        Einstein crossing time (days).
    thetaE_hat : array_like, shape (2,)
        Unit vector along the source-lens relative proper motion (E, N).
    orb : bagle.orbits.Orbit
        Lens orbit (see :func:`bagle.model.keplerian_lens_orbit`).
    thetaE_orb : float
        The Einstein radius in the units of ``orb.aleph`` (e.g.
        ``thetaE_amp * 1e-3`` for orbits in arcsec, 1.0 for orbits in
        Einstein radii).
    coords_in, coords_out : {'geom_mid', 'prim_center', 'COM'}
        Input and output lens reference points.

    Returns
    -------
    t0_out : float
        Reference time of the output point (MJD).
    u0_amp_out : float
        Signed impact parameter about the output point at ``t0_out``
        (Einstein units).

    Notes
    -----
    The CoM moves on a straight line, so ``(t0_com, u0_com)`` is a true
    closest approach. The primary and the geometric midpoint orbit the CoM,
    so for them ``(t0, u0)`` is defined at an instant: at time ``t0`` the
    source sits at ``u0 * thetaE`` from the point's position *at that time*,
    and ``u0`` is perpendicular to the relative proper motion. That ``t0`` is
    a reference epoch, not a closest-approach time, and in general differs
    from ``t0_com``. Parallax is not included (it moves the source relative
    to every lens point equally).

    With ``d(t)`` the point's offset from the CoM (Einstein units) and the
    source track about the CoM ``u0_com + (t - t0_com) / tE * thetaE_hat``:

    * point -> CoM is explicit: ``t0_com = t0 - tE * d(t0) . thetaE_hat``
      and ``u0_com = u0 + d_perp(t0)``.
    * CoM -> point solves ``(t0 - t0_com) / tE = d(t0) . thetaE_hat`` for
      ``t0``; the root is unique when the point's orbital speed along
      ``thetaE_hat`` is below the relative proper motion.

    Without orbital motion this reduces to
    :func:`bagle.model.convert_u0_amp_t0_psbl`. This is
    :func:`convert_u0_amp_t0_bsbl_orbit` with a single (static) source.
    """
    t0_out, u0_amp_out = convert_u0_amp_t0_bsbl_orbit(
        t0_in, u0_amp_in, tE, thetaE_hat, orb, None, thetaE_orb,
        lens_in=coords_in, lens_out=coords_out)

    return t0_out, u0_amp_out


def convert_u0_amp_t0_bsbl_orbit(t0_in, u0_amp_in, tE, thetaE_hat, orbL, orbS,
                                 thetaE_orb, lens_in, lens_out, src_in='COM',
                                 src_out='COM'):
    """
    Move a closest approach between reference points of orbiting binaries.

    Generalizes :func:`convert_u0_amp_t0_psbl_orbit` to a binary source:
    both the lens point and the source point may orbit their own CoM.

    Parameters
    ----------
    t0_in : float
        Reference time of the input pair (MJD); see Notes.
    u0_amp_in : float
        Signed impact parameter of the input source point about the input
        lens point at ``t0_in`` (Einstein units), with the
        :func:`bagle.model.u0_hat_from_thetaE_hat` sign convention.
    tE : float
        Einstein crossing time (days).
    thetaE_hat : array_like, shape (2,)
        Unit vector along the source-lens relative proper motion (E, N).
    orbL : bagle.orbits.Orbit or None
        Lens orbit about the lens CoM; ``None`` for a single lens.
    orbS : bagle.orbits.Orbit or None
        Source orbit about the source CoM; ``None`` for a single source.
    thetaE_orb : float
        The Einstein radius in the units of ``orb.aleph`` (e.g.
        ``thetaE_amp * 1e-3`` for orbits in arcsec). Both orbits must use
        the same units.
    lens_in, lens_out : {'geom_mid', 'prim_center', 'COM'}
        Input and output lens reference points.
    src_in, src_out : {'prim_center', 'COM'}, optional
        Input and output source reference points (default ``'COM'``).

    Returns
    -------
    t0_out : float
        Reference time of the output pair (MJD).
    u0_amp_out : float
        Signed impact parameter of the output pair at ``t0_out`` (Einstein
        units).

    Notes
    -----
    With ``d_L(t)`` and ``d_S(t)`` the lens and source points' offsets from
    their CoMs (Einstein units), the source point relative to the lens point
    is ``u(t) = u_com(t) + d_S(t) - d_L(t)``, where the CoM-CoM separation
    ``u_com(t) = u0_com + (t - t0_com) / tE * thetaE_hat`` is a straight
    line. A pair's ``(t0, u0)`` is defined at an instant: ``u(t0)`` is
    perpendicular to ``thetaE_hat`` and ``u0_amp = u(t0) . u0_hat_+``.

    * Input pair -> CoM pair is explicit.
    * CoM pair -> output pair solves
      ``(t0 - t0_com) / tE + (d_S(t0) - d_L(t0)) . thetaE_hat = 0``. The
      root is unique when the points' orbital speeds along ``thetaE_hat``
      stay below the relative proper motion. Otherwise the sign change
      nearest ``t0_in`` is used.

    Raises
    ------
    ValueError
        If no root exists in the bracket.
    """
    from scipy import optimize

    # bagle.model imports this module at load time, so import it lazily.
    from bagle.model import u0_hat_from_thetaE_hat

    thetaE_hat = np.asarray(thetaE_hat, dtype=float)

    # Direction of positive u0_amp.
    u0_hat_pos = u0_hat_from_thetaE_hat(thetaE_hat, 1.0)

    def point_offset(orb, t, coords):
        """Offset of a reference point from its CoM at t (Einstein units)."""
        if orb is None or coords == 'COM':
            return np.zeros(2)
        return keplerian_point_offset(orb, t, coords) / thetaE_orb

    def rel_offset(t, lens, src):
        """Source point minus lens point, relative to the CoM-CoM line."""
        return point_offset(orbS, t, src) - point_offset(orbL, t, lens)

    # Input pair -> CoM pair. At t0_in, u_com = u_in - (d_S - d_L).
    u_com = u0_amp_in * u0_hat_pos - rel_offset(t0_in, lens_in, src_in)
    t0_com = t0_in - tE * np.dot(u_com, thetaE_hat)
    u0_com = u_com - np.dot(u_com, thetaE_hat) * thetaE_hat

    if lens_out == 'COM' and src_out == 'COM':
        t0_out = t0_com
        u_out = u0_com
    else:
        # CoM pair -> output pair: when u(t) is perpendicular to the motion.
        def along(t):
            return ((t - t0_com) / tE
                    + np.dot(rel_offset(t, lens_out, src_out), thetaE_hat))

        # |d| <= (aleph + aleph_sec)(1 + e) for each orbit brackets the root.
        d_max = 0.0
        for orb in (orbL, orbS):
            if orb is not None:
                d_max += (orb.aleph + orb.aleph2) * (1 + orb.e) / thetaE_orb
        half_width = tE * d_max + 1e-6

        # Sample the bracket; take the sign change nearest t0_in.
        grid = np.linspace(t0_com - half_width, t0_com + half_width, 257)
        vals = np.array([along(t) for t in grid])
        idx = np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) <= 0)[0]
        if len(idx) == 0:
            raise ValueError('No (t0, u0) for the requested reference points: '
                             'the orbit never makes the source - point vector '
                             'perpendicular to the relative motion.')
        k = idx[np.argmin(np.abs(grid[idx] - t0_in))]
        if vals[k] == 0:
            t0_out = grid[k]
        else:
            t0_out = optimize.brentq(along, grid[k], grid[k + 1],
                                     xtol=1e-10, rtol=1e-14)

        # Source point relative to the lens point at t0_out.
        u_out = (u0_com + (t0_out - t0_com) / tE * thetaE_hat
                 + rel_offset(t0_out, lens_out, src_out))
        u_out = u_out - np.dot(u_out, thetaE_hat) * thetaE_hat

    # Signed amplitude along the positive-u0_amp direction.
    u0_amp_out = np.dot(u_out, u0_hat_pos)

    return t0_out, u0_amp_out

def convert_astrometry_to_scan_frame(x, y, scan_angle):
    """
    Convert RA/Dec frame astrometry to alternative frame based on
    a scan angle.
    
    Parameters
    ----------
    x, y : array_like
        Astrometry time series data in delta RA/Dec (arcsec)
    scan_angle: array_like
        Scan angle (theta as defined by Gaia) time series
    """
    scan_angle_rad = np.deg2rad(scan_angle)
    w = x*np.sin(scan_angle_rad) + y*np.cos(scan_angle_rad)
    z = x*np.cos(scan_angle_rad) + y*np.sin(scan_angle_rad)
    return w,z

