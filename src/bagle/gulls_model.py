"""BAGLE finite-source models in the RMDC26 Beginner Gulls parameterization.

This module targets BAGLE's ``fsbl_ld_jax`` branch. Its models inherit BAGLE's
finite-source contour integration, blending, and photometry methods. Gulls
contributes parameter initializers and a historical observer/parallax family.
The Parquet convenience loader is :func:`bagle.gulls.load_model`.
"""
import inspect
import numpy as np

from bagle import model_jax as model
from bagle.gulls import (
    JD_ZERO, JD_TO_MJD, NATIVE_ECLIPTIC_OBLIQUITY_DEG,
    _galactic_to_event_radec, _sky_rotation, _unit_from_radec,
    catalog_to_event_radec,
)
from bagle.gulls_orbit import roman_orbit5


def _times(t):
    times = np.asarray(t, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError('model times must be a finite one-dimensional MJD array')
    return times


def _geometry(t0lens1, u0lens1, tE_ref, piEE, piEN, Planet_s, Planet_q,
              alpha, ra_deg, dec_deg, tcroin, rho, galactic_l, galactic_b):
    values = np.array([t0lens1, u0lens1, tE_ref, piEE, piEN, Planet_s,
                       Planet_q, alpha, ra_deg, dec_deg, tcroin, rho], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('Gulls trajectory, radius, coordinates, and reference must be finite')
    if min(tE_ref, Planet_s, Planet_q, rho) <= 0:
        raise ValueError('tE_ref, Planet_s, Planet_q, and rho must be positive')
    _unit_from_radec(ra_deg, dec_deg)
    if (galactic_l is None) != (galactic_b is None):
        raise ValueError('galactic_l and galactic_b must be supplied together')
    event_radec = (catalog_to_event_radec(ra_deg, dec_deg)
                   if galactic_l is None else
                   _galactic_to_event_radec(galactic_l, galactic_b))
    pi = -_sky_rotation(*event_radec, NATIVE_ECLIPTIC_OBLIQUITY_DEG) @ [piEE, piEN]
    if np.linalg.norm(pi) == 0:
        raise ValueError('nonzero parallax is required to define the sky orientation')
    motion = pi / np.linalg.norm(pi)
    left = np.array([-motion[1], motion[0]])
    angle = np.deg2rad(alpha)
    impact = u0lens1 + Planet_s / 2 * np.sin(angle)
    sign = np.dot(left, model.u0_hat_from_thetaE_hat(motion, 1.0))
    return dict(t0=(JD_ZERO - JD_TO_MJD) + t0lens1
                + Planet_s / 2 * tE_ref * np.cos(angle),
                u0_amp=float(impact * sign), tE=tE_ref,
                piE_E=float(pi[0]), piE_N=float(pi[1]), q=Planet_q,
                sep=Planet_s, phi=float((180 + alpha) % 360),
                raL=event_radec[0], decL=event_radec[1])


def _set_native(obj, values, ra_deg, dec_deg, tcroin, galactic_l, galactic_b):
    """Retain native scalar inputs alongside BAGLE's derived variables."""
    for name, value in values.items():
        setattr(obj, name, float(value))
    obj.catalog_radec = (float(ra_deg), float(dec_deg))
    obj.event_radec = (obj.raL, obj.decL)
    obj.tcroin = float(tcroin)
    obj.galactic_l, obj.galactic_b = galactic_l, galactic_b
    obj.reference_mjd = (JD_ZERO - JD_TO_MJD) + obj.tcroin
    obj.piE_E, obj.piE_N = obj.piE


class GullsBeginner_Parallax(model.ParallaxClassABC):
    """Historical Beginner parallax with the analytic Roman ORBIT5 observer.

    The observer displacement subtracts its position and velocity at
    ``tcroin``. The native transverse sign differs from physical BAGLE
    parallax. This family preserves that sign; standard Earth frame-conversion
    methods therefore do not apply. Returned trajectories describe the
    historical photometric model, not a certified physical sky orbit.
    """
    parallaxFlag = True
    parallax_convention = "gulls_beginner"
    ref_frame_parallax_flag = False
    fixed_param_names = ['ra_deg', 'dec_deg', 'tcroin', 'galactic_l', 'galactic_b']
    fixed_phot_param_names = []

    def start(self):
        self._reference_bjd = JD_ZERO + self.tcroin
        _, self._observer_origin, self._observer_velocity = roman_orbit5(
            np.array([self._reference_bjd]), self._reference_bjd)

    def get_observer_displacement(self, t):
        """Return reference-subtracted Cartesian equatorial positions in AU.

        Times are MJD. These are effective observer displacements, not absolute
        satellite coordinates. They include the historical ORBIT5 model and
        the position/velocity subtraction at ``tcroin``, independently of piE.
        Outward converters use them with the converted parallax convention.
        """
        bjd = _times(t) + JD_TO_MJD
        xyz, _, _ = roman_orbit5(bjd, self._reference_bjd)
        d = (xyz - self._observer_origin
             - (bjd - self._reference_bjd)[:, None]*self._observer_velocity)
        eps = np.deg2rad(NATIVE_ECLIPTIC_OBLIQUITY_DEG)
        return np.column_stack((d[:, 0], np.cos(eps)*d[:, 1]-np.sin(eps)*d[:, 2],
                                np.sin(eps)*d[:, 1]+np.cos(eps)*d[:, 2]))

    def get_source_parallax(self, t, filt_idx=0):
        """Additive source displacement in the model's East/North frame.

        ``t`` is a one-dimensional MJD array. All Beginner filters share the
        same observer; ``filt_idx`` is accepted for the BAGLE interface.
        """
        displacement = self.get_observer_displacement(t)
        ra, dec = np.deg2rad(self.event_radec)
        east = displacement @ [-np.sin(ra), np.cos(ra), 0.]
        north = displacement @ [-np.sin(dec)*np.cos(ra),
                                 -np.sin(dec)*np.sin(ra), np.cos(dec)]
        # BAGLE piE points source-minus-lens; the native convention gives
        # these signs when expressed in that equatorial tangent basis.
        pe, pn = self.piE
        along = pe*east + pn*north
        across = pn*east - pe*north
        left = np.array([-self.muRel_hat[1], self.muRel_hat[0]])
        return along[:, None]*self.muRel_hat + across[:, None]*left


class GullsFSBL_Phot(model.FSBL_Phot):
    """BAGLE FSBL photometry with a selectable historical parallax family.

    Only trajectory dispatch differs from ``FSBL_Phot``. Magnification,
    source integration, image arrays, and photometry are inherited unchanged.
    """
    def get_u(self, t, filt_idx=0):
        times = _times(t)
        u = self.u0 + ((times - self.t0)/self.tE)[:, None]*self.muRel_hat
        if self.parallaxFlag:
            u += self.get_source_parallax(times, filt_idx=filt_idx)
        return u

    def get_complex_pos(self, t, filt_idx=0):
        u = self.get_u(t, filt_idx=filt_idx)
        lenses = self.get_resolved_lens_astrometry(_times(t), filt_idx=filt_idx)
        return (u[:, 0] + 1j*u[:, 1],
                lenses[:, 0, 0] + 1j*lenses[:, 0, 1],
                lenses[:, 1, 0] + 1j*lenses[:, 1, 1])


class GullsBFSBL_Phot(model.BFSBL_Phot):
    """BAGLE two-finite-source photometry with the Beginner trajectory family."""
    def get_u(self, t, filt_idx=0):
        times = _times(t)
        travel = ((times - self.t0_pri)/self.tE)[:, None]*self.muRel_hat
        u = np.stack((self.u0_pri + travel, self.u0_sec + travel), axis=1)
        if self.parallaxFlag:
            u += self.get_source_parallax(times, filt_idx=filt_idx)[:, None, :]
        return u

class FSBL_PhotParamGullsBeginner(model.FSBL_PhotParam1):
    """Convert a single-source Beginner parameter set into BAGLE variables.

    Parameters
    ----------
    t0lens1 : float
        Lens-1 closest-approach time, in simulation days since JD 2461406.5.
    u0lens1 : float
        Signed lens-1 impact, positive to the left of source travel.
    tE_ref : float
        Saved reference crossing time in days; no CROIN draw is recomputed.
    piEE, piEN : float
        Native ecliptic tangent components, lens minus source.
    Planet_s, Planet_q : float
        Lens separation and secondary/primary mass ratio.
    alpha : float
        Counterclockwise source travel angle from the lens-1 to lens-2 axis,
        in degrees.
    rho : float
        Source radius in total-lens Einstein radii. No mass rescaling applies.
    b_sff, mag_src : array_like
        Source fraction of total baseline and source magnitude per filter.
    ra_deg, dec_deg : float
        Saved catalog direction in degrees. ``catalog_radec`` retains it;
        ``event_radec`` is the corrected direction for native parallax.
    tcroin : float
        Parallax reference time in simulation days, distinct from t0lens1.
    galactic_l, galactic_b : float, optional
        Saved Galactic degrees. When present, these determine event_radec
        directly, avoiding inversion of the saved catalog matrix.
    n_outline : int, optional
        Initial BAGLE source-limb sampling (default 64).
    root_tol : float, optional
        Lens-root acceptance tolerance (default 1e-8).

    Notes
    -----
    Query times and the derived ``t0`` use MJD notation. The native input
    times above use simulation days. BAGLE places the origin at the geometric
    lens midpoint, while the inputs use lens 1. The observer family retains
    the reference-subtracted historical trajectory, not standard Earth
    barycentric parallax. Physical astrometry is not established.

    ``get_amplification`` and ``get_photometry`` use BAGLE's own finite-source
    integrator. ``amg_tol`` and ``amg_max_pts`` control that calculation;
    they are not VBM's ``Tol``/``RelTol`` and do not promise identical output.
    At a sampled RMDC26_000027 caustic peak with ``amg_tol=1e-7``, increasing
    the contour point limit from 4096 to 8192 to 16384 gave baseline-normalized
    fluxes of 3.739593, 3.738810, and 3.744634, respectively. These results
    did not show that increasing the limit made the flux more accurate.
    It is not yet validated for precise reproduction of every Beginner curve.
    Precise VBM comparisons use ``BinaryMag2`` with ``Tol=1e-4`` (absolute
    magnification-error target), ``RelTol=1e-6`` (relative target),
    ``minannuli=1``, and ``a1=0`` (uniform brightness). The tolerance pair
    matches related Gulls source; minannuli=1 is the VBM 5.3.5 default.
    These settings control VBM's integration, not any fitted event parameter.
    The VBM adapter sets them explicitly; MulensModel passes the tolerance
    pair to VBM, whose default supplies minannuli=1. pyLIMA 1.9.8 instead
    defaults to ``Tol=RelTol=1e-3, minannuli=2`` in a shared engine, which the
    adapter leaves unchanged. In focused examples, pyLIMA differed from the
    direct VBM reference by up to 68 ppm with those defaults and below
    0.013 ppm with ``Tol=1e-4, RelTol=1e-6, minannuli=1``. The event parameters
    were identical; the numerical integration settings changed.
    Querying Simulation_time
    avoids the printed BJD grid of 8.64 ms, which produced up to 2.6 ppm
    residuals where the light curve changes rapidly. Tiny differences from
    the historical generator remain even with matching VBM settings.

    This parameterization implements the host BAGLE model interface. Its
    native parameter names do not use the parent class's JAX likelihood
    kernel. The existing ``frame_convert.get_vbm_model``, ``get_mulens_model``,
    and ``get_pylima_model`` accept these Gulls models, including two sources.
    They derive destination parameters from BAGLE's source/lens geometry and
    carry the effective historical observer. A parameter dictionary alone
    omits that observer and is insufficient. VBM and pyLIMA conversions require
    ``times_mjd``; VBM samples its ephemeris at those times, while pyLIMA attaches
    them to its telescope. MulensModel carries an analytic observer adapter.
    ``filt_idx`` chooses the flux ratio for binary sources. pyLIMA retains its
    own magnification tolerances; the VBM and MulensModel adapters use the
    matching Gulls controls above. See the Gulls guide for destination calls,
    time units, and flux normalization. These are photometric conversions;
    physical sky astrometry is not certified.
    """
    fitter_param_names = ['t0lens1', 'u0lens1', 'tE_ref', 'piEE', 'piEN',
                          'Planet_s', 'Planet_q', 'alpha', 'rho', 'b_sff', 'mag_src']
    filt_param_names = ['b_sff', 'mag_src']
    filt_param_usage = ['both', 'phot']
    fixed_param_names = ['ra_deg', 'dec_deg', 'tcroin', 'galactic_l', 'galactic_b']
    fixed_phot_param_names = []
    jax_log_likely_photometry = None
    jax_loglik_mode = None

    def __init__(self, t0lens1, u0lens1, tE_ref, piEE, piEN, Planet_s, Planet_q,
                 alpha, rho, b_sff, mag_src, ra_deg, dec_deg, tcroin,
                 galactic_l=None, galactic_b=None, n_outline=64, root_tol=1e-8):
        pars = _geometry(t0lens1, u0lens1, tE_ref, piEE, piEN, Planet_s,
                         Planet_q, alpha, ra_deg, dec_deg, tcroin, rho,
                         galactic_l, galactic_b)
        if not isinstance(n_outline, (int, np.integer)) or n_outline < 16:
            raise ValueError('n_outline must be an integer of at least 16')
        if not np.isfinite(root_tol) or root_tol <= 0:
            raise ValueError('root_tol must be finite and positive')
        super().__init__(**pars, radiusS=rho, b_sff=b_sff, mag_src=mag_src,
                         n_outline=n_outline, root_tol=root_tol)
        if (not np.all(np.isfinite(self.mag_src))
                or not np.all(np.isfinite(self.b_sff)) or np.any(self.b_sff <= 0)):
            raise ValueError('source magnitudes must be finite and source fractions positive')
        _set_native(self, dict(t0lens1=t0lens1, u0lens1=u0lens1, tE_ref=tE_ref,
                              piEE=piEE, piEN=piEN, Planet_s=Planet_s,
                              Planet_q=Planet_q, alpha=alpha, rho=rho,
                              ra_deg=ra_deg, dec_deg=dec_deg), ra_deg, dec_deg,
                    tcroin, galactic_l, galactic_b)
        self.mag_base = self.mag_src + 2.5*np.log10(self.b_sff)


class BFSBL_PhotParamGullsBeginner(model.BFSBL_PhotParam1):
    """Convert the two-source Beginner truth into BAGLE binary-source variables.

    Common trajectory inputs follow ``FSBL_PhotParamGullsBeginner``.
    ``Source2_rho`` uses total-lens Einstein radii. ``Source2_s`` is the
    projected-orbit scale; ``Source2_alpha``, ``Source2_inc``, and
    ``Source2_phase`` are degrees. They determine the static projected source
    separation used by the Beginner light curves. No new source orbit is fit.

    ``mag_src_pri`` and ``mag_src_sec`` contain each source's magnitudes;
    ``b_sff`` is their *combined* fraction of baseline flux. In the native
    answer row, both ``Obs_i_fs`` and the misleadingly named
    ``Obs_i_fs2ofs1`` are fractions of total baseline. The loader converts
    these to magnitudes and adds them for b_sff, without multiplying twice.

    BAGLE's BFSBL data class integrates each source with its own radius and
    combines their light using their filter magnitudes. Source positions are
    not read from the saved light curve. The finite-source accuracy, observer,
    time-precision, and physical-astrometry limits of the single-source
    parameterization apply here too. VBM comparisons use ``BinaryMag2``,
    ``Tol=1e-4, RelTol=1e-6, minannuli=1, a1=0`` for both sources. pyLIMA's
    defaults differ as described above. The JAX likelihood kernel is disabled.
    """
    fitter_param_names = ['t0lens1', 'u0lens1', 'tE_ref', 'piEE', 'piEN',
                          'Planet_s', 'Planet_q', 'alpha', 'rho', 'Source2_rho',
                          'Source2_s', 'Source2_alpha', 'Source2_inc', 'Source2_phase',
                          'mag_src_pri', 'mag_src_sec', 'b_sff']
    filt_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']
    filt_param_usage = ['phot', 'phot', 'both']
    fixed_param_names = ['ra_deg', 'dec_deg', 'tcroin', 'galactic_l', 'galactic_b']
    fixed_phot_param_names = []
    jax_log_likely_photometry = None
    jax_loglik_mode = None

    def __init__(self, t0lens1, u0lens1, tE_ref, piEE, piEN, Planet_s, Planet_q,
                 alpha, rho, Source2_rho, Source2_s, Source2_alpha,
                 Source2_inc, Source2_phase, mag_src_pri, mag_src_sec,
                 b_sff, ra_deg, dec_deg, tcroin, galactic_l=None,
                 galactic_b=None, n_outline=64, root_tol=1e-8):
        pars = _geometry(t0lens1, u0lens1, tE_ref, piEE, piEN, Planet_s,
                         Planet_q, alpha, ra_deg, dec_deg, tcroin, rho,
                         galactic_l, galactic_b)
        secondary = dict(Source2_rho=Source2_rho, Source2_s=Source2_s,
                         Source2_alpha=Source2_alpha, Source2_inc=Source2_inc,
                         Source2_phase=Source2_phase)
        if not np.all(np.isfinite(list(secondary.values()))) or min(Source2_rho, Source2_s) <= 0:
            raise ValueError('secondary geometry must be finite with positive radius and separation')
        if not isinstance(n_outline, (int, np.integer)) or n_outline < 16:
            raise ValueError('n_outline must be an integer of at least 16')
        if not np.isfinite(root_tol) or root_tol <= 0:
            raise ValueError('root_tol must be finite and positive')
        phase, inc, node = np.deg2rad([Source2_phase, Source2_inc, Source2_alpha])
        along, across = Source2_s*np.cos(phase), Source2_s*np.sin(phase)*np.cos(inc)
        dx = along*np.cos(node) - across*np.sin(node)
        dy = along*np.sin(node) + across*np.cos(node)
        angle = np.deg2rad(alpha)
        dtau = dx*np.cos(angle) + dy*np.sin(angle)
        du = -dx*np.sin(angle) + dy*np.cos(angle)
        sep = pars.pop('sep')
        phi = pars.pop('phi')
        super().__init__(**pars, sep_SL=0., sep_S=np.hypot(dx, dy),
                         phi_S=float(-np.rad2deg(np.arctan2(du, dtau))),
                         sep_L=sep, phi_L=phi, mag_src_pri=mag_src_pri,
                         mag_src_sec=mag_src_sec, b_sff=b_sff, dmag_Lp_Ls=0.,
                         radiusS_pri=rho, radiusS_sec=Source2_rho)
        if (not all(np.all(np.isfinite(v)) for v in
                    (self.mag_src_pri, self.mag_src_sec, self.b_sff))
                or np.any(self.b_sff <= 0)):
            raise ValueError('source magnitudes must be finite and source fractions positive')
        self.root_tol = root_tol
        self.n_outline = n_outline
        left = np.array([-self.muRel_hat[1], self.muRel_hat[0]])
        # Retain both components; the secondary has its own closest-approach time.
        self.u0_sec = self.u0_pri + dtau*self.muRel_hat + du*left
        self.u0_amp_sec = float(self.u0_sec @ model.u0_hat_from_thetaE_hat(self.muRel_hat, 1.))
        self.t0_sec = self.t0_pri - dtau*self.tE
        _set_native(self, dict(t0lens1=t0lens1, u0lens1=u0lens1, tE_ref=tE_ref,
                              piEE=piEE, piEN=piEN, Planet_s=Planet_s,
                              Planet_q=Planet_q, alpha=alpha, rho=rho,
                              ra_deg=ra_deg, dec_deg=dec_deg, **secondary),
                    ra_deg, dec_deg, tcroin, galactic_l, galactic_b)


@model.inheritdocstring
class FSBL_Phot_Par_GullsBeginner(model.ModelClassABC, GullsFSBL_Phot,
                                 GullsBeginner_Parallax, FSBL_PhotParamGullsBeginner):
    """Single-source Beginner model using BAGLE finite-source photometry."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model.startbases(self)
        model.checkconflicts(self)


@model.inheritdocstring
class BFSBL_Phot_Par_GullsBeginner(model.ModelClassABC, GullsBFSBL_Phot,
                                  GullsBeginner_Parallax, BFSBL_PhotParamGullsBeginner):
    """Binary-source Beginner model using BAGLE finite-source photometry."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model.startbases(self)
        model.checkconflicts(self)


# The upstream decorator selects the most distant documented ancestor. Here
# that would describe an older point-source parameterization, not the native
# Gulls constructor. Keep the required decorator, but expose the correct
# parameter description and signature on the concrete classes.
for _concrete, _parameters in (
    (FSBL_Phot_Par_GullsBeginner, FSBL_PhotParamGullsBeginner),
    (BFSBL_Phot_Par_GullsBeginner, BFSBL_PhotParamGullsBeginner),
):
    _concrete.__doc__ = inspect.cleandoc(_parameters.__doc__)
    _concrete.__signature__ = inspect.signature(_parameters)
