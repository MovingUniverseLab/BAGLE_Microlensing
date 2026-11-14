"""RMDC26 Beginner Gulls truth as inspectable BAGLE parameters.

The native and physical parallax conventions differ; see docs/gulls.rst.
Parquet support requires the optional ``pyarrow`` package.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

JD_ZERO = 2461406.5
JD_TO_MJD = 2400000.5
NATIVE_ECLIPTIC_OBLIQUITY_DEG = 23.439279444444445

# The first matrix is the rounded Galactic-to-equatorial matrix used when
# Gulls wrote the answer-file RA/Dec fields. The second is the equatorial-to-
# Galactic matrix used by its coords::lb2ad parallax path.
_ANSWER_GALACTIC_TO_EQUATORIAL = np.array([
    [-0.0548755604, 0.4941094279, -0.8676661490],
    [-0.8734370902, -0.4448296300, -0.1980763734],
    [-0.4838350155, 0.7469822445, 0.4559837762],
])
_NATIVE_EQUATORIAL_TO_GALACTIC = np.array([
    [-0.0548765273295172, -0.873436658110768, -0.483835685968435],
    [0.49411066572488, -0.444830355306367, 0.746980993744103],
    [-0.867665382947348, -0.198076649977489, 0.455985113757595],
])
_NATIVE_GALACTIC_TO_EQUATORIAL = _NATIVE_EQUATORIAL_TO_GALACTIC.T


def _unit_from_radec(ra_deg, dec_deg):
    try:
        ra, dec = np.asarray([ra_deg, dec_deg], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('RA and Dec must be finite numeric degrees') from exc
    if not np.isfinite(ra) or not np.isfinite(dec) or not -90 <= dec <= 90:
        raise ValueError('RA must be finite and Dec must be finite within [-90, 90] degrees')
    if abs(dec) == 90:
        ra = 0.0  # RA is undefined at the celestial poles.
    ra, dec = np.deg2rad([ra % 360.0, dec])
    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])


def _radec_from_unit(vector):
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError('coordinate direction must be a finite three-vector')
    norm = np.linalg.norm(value)
    if norm == 0:
        raise ValueError('coordinate direction must be nonzero')
    x, y, z = value / norm
    horizontal = np.hypot(x, y)
    ra = 0.0 if horizontal < 1e-15 else np.rad2deg(np.arctan2(y, x)) % 360.0
    dec = np.rad2deg(np.arctan2(z, horizontal))
    return float(ra), float(dec)


def catalog_to_event_radec(ra_deg, dec_deg):
    """Convert answer-file RA/Dec degrees to Gulls' effective event direction.

    Gulls wrote the answer-file coordinates with a rounded Galactic-to-
    equatorial matrix, then used a different matrix for its parallax path.
    This function reverses the saved-coordinate matrix with a linear solve and
    applies the parallax-path matrix. It is a reproduction of that Gulls
    convention, not a general-purpose reference-frame transformation.
    """
    equatorial = _unit_from_radec(ra_deg, dec_deg)
    galactic = np.linalg.solve(_ANSWER_GALACTIC_TO_EQUATORIAL, equatorial)
    galactic /= np.linalg.norm(galactic)
    return _radec_from_unit(_NATIVE_GALACTIC_TO_EQUATORIAL @ galactic)


def _galactic_to_event_radec(longitude_deg, latitude_deg):
    try:
        longitude, latitude = np.asarray([longitude_deg, latitude_deg], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Galactic longitude and latitude must be finite numeric degrees') from exc
    if (not np.isfinite(longitude) or not np.isfinite(latitude)
            or not -90 <= latitude <= 90):
        raise ValueError('Galactic longitude must be finite and latitude within [-90, 90] degrees')
    longitude, latitude = np.deg2rad([longitude % 360.0, latitude])
    galactic = np.array([
        np.cos(latitude)*np.cos(longitude),
        np.cos(latitude)*np.sin(longitude),
        np.sin(latitude),
    ])
    return _radec_from_unit(_NATIVE_GALACTIC_TO_EQUATORIAL @ galactic)


class UnsupportedGullsEvent(ValueError):
    """The event is outside the supported single-source binary-lens profile."""


def _sky_rotation(ra, dec, obliquity_deg=23.439291111):
    ra, dec, eps = np.deg2rad([ra, dec, obliquity_deg])
    rot = np.array([[1, 0, 0], [0, np.cos(eps), np.sin(eps)],
                    [0, -np.sin(eps), np.cos(eps)]])
    position = rot @ np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    lam = np.arctan2(position[1], position[0])
    beta = np.arcsin(np.clip(position[2], -1, 1))
    e = rot.T @ np.array([-np.sin(lam), np.cos(lam), 0])
    n = rot.T @ np.array([-np.sin(beta)*np.cos(lam), -np.sin(beta)*np.sin(lam), np.cos(beta)])
    east = np.array([-np.sin(ra), np.cos(ra), 0])
    north = np.array([-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)])
    return np.array([[east @ e, east @ n], [north @ e, north @ n]])


@dataclass
class GullsBeginner:
    """Load Beginner answers and convert their Gulls model parameters.

    The original answer row remains in ``event.raw``. The conversion methods
    return parameters for the selected modeling package; the caller constructs
    that package's model and chooses how to use its outputs.

    On BAGLE's ``fsbl_ld_jax`` branch, ``to_bagle_model()`` constructs an
    actual finite-source BAGLE model with the historical Beginner observer
    and parallax convention, for either one or two sources. The shorter
    ``load_model(name, truth_path)`` loads that model directly from Parquet.
    Its inherited ``get_photometry`` and ``get_amplification`` methods use
    BAGLE's own integration. Model query times are MJD. See the returned
    class description for the complete conventions and numerical limits.

    ``to_bagle(mag_src)`` returns keyword arguments for BAGLE's
    ``PSBL_Phot_noPar_Param1``, a single-source, point-source binary-lens
    class. ``mag_src`` contains source magnitudes in F146/F087/F213 order.
    ``to_mulensmodel()`` returns a single-source binary-lens parameter
    dictionary including rho. Neither export alone encodes the Beginner
    Roman observer or guarantees the native parallax behavior at new times.
    Binary-source events are rejected by these two export methods.
    A pyLIMA parameter export has not yet been implemented.

    ``to_vbm(...)`` returns and writes high-level VBM model parameters and
    observer configuration, supporting one or two sources. Its docstring
    describes the required observer inputs. ``predict(...)`` is an optional
    convenience for generating baseline-normalized VBM flux; it reconstructs
    the observer internally. The caller can instead use the exported VBM
    parameters with VBM directly. ``render_native`` is a separate check at
    saved epochs using the recorded observer shifts.

    Precision when matching the archived Gulls flux with VBM
    ------------------------------------------------------
    Use ``Simulation_time`` (days since JD 2461406.5) rather than printed
    BJD, whose seven decimal places give 8.64 ms steps. BJD rounding produced
    sawtooth residuals up to 2.6 ppm in examined events, amplified where flux
    changes rapidly. Reduced-JD VBM times should be formed as
    ``(JD_ZERO - 2450000) + Simulation_time`` to retain precision.

    Related Gulls source uses ``Tol=1e-4, RelTol=1e-6``, uniform source
    brightness, and no lens light. ``predict`` uses these tolerances by
    default; the lower-level ``to_vbm`` bundle retains RelTol=1e-10, so set
    it to 1e-6 in VBM when matching the source settings. Smaller tolerances
    need not reproduce the archived numerical result more closely.

    The VBM export uses ``event.event_radec`` for the historical parallax
    calculation and preserves the answer-file pair as ``catalog_radec``.
    It includes the historical transverse parallax convention, reconstructed
    ORBIT5 observer, and total-lens-mass normalization: rho, s and tE need
    no further mass rescaling. ``to_bagle`` and ``to_mulensmodel`` retain
    their separately documented coordinate and parallax conventions.

    With simulation times and matching tolerances, all 188 events were
    evaluated with a largest fractional-flux difference of about 1.31 ppm.
    The exact historical executable remains unidentified. These photometric
    checks do not establish physical astrometry or equivalent behavior of
    the other packages. VBM-specific precision settings are not a general
    accuracy guarantee for every exported model.
    """
    raw: dict
    curve: object = None

    def __post_init__(self):
        self.raw = dict(self.raw)
        if 'Source2ID' not in self.raw:
            raise UnsupportedGullsEvent('Source2ID is missing; source multiplicity is unknown')
        second = self.raw.get('Source2ID')
        self.has_secondary_source = not (
            second is None
            or (isinstance(second, (float, np.floating)) and np.isnan(second))
        )
        secondary_flux = [self.raw.get(f'Obs_{i}_fs2ofs1') for i in range(3)]
        if not self.has_secondary_source and any(
            value not in (None, 0, 0.0) for value in secondary_flux
        ):
            raise UnsupportedGullsEvent(
                'Secondary-source flux is nonzero but Source2ID is missing'
            )
        if self.has_secondary_source:
            required = (
                'Source2_rho', 'Source2_s', 'Source2_alpha',
                'Source2_inc', 'Source2_phase',
            )
            missing = [key for key in required if key not in self.raw or self.raw[key] is None]
            if missing:
                raise UnsupportedGullsEvent(
                    f'Secondary-source event lacks required fields: {missing}'
                )
            values = [float(self.raw[key]) for key in required]
            if not np.all(np.isfinite(values)):
                raise UnsupportedGullsEvent('Secondary-source fields must be finite')
            if values[0] <= 0 or values[1] <= 0:
                raise UnsupportedGullsEvent(
                    'Source2_rho and Source2_s must be positive'
                )
            if any(
                value is None or not np.isfinite(float(value)) or float(value) < 0
                for value in secondary_flux
            ):
                raise UnsupportedGullsEvent(
                    'Secondary baseline flux fractions must be finite and nonnegative'
                )
        if self.raw.get('LDgamma') != 0:
            raise UnsupportedGullsEvent('Only the uniform-source Beginner profile is supported')
        if not np.isclose(self.raw['tref'], self.raw['tcroin'], rtol=0, atol=1e-9):
            raise UnsupportedGullsEvent('tref differs from tcroin; parallax rebasing is unsupported')

    @property
    def name(self):
        return self.raw['name']

    @property
    def catalog_radec(self):
        """The answer-file ``ra_deg``/``dec_deg`` input pair, in degrees.

        These fields describe the coordinates supplied to the Beginner
        answer file. They are distinct from the stellar-source fields named
        ``Source_RA20000`` and ``Source_DEC20000`` when those are present.
        """
        try:
            return float(self.raw['ra_deg']), float(self.raw['dec_deg'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('answer-file ra_deg and dec_deg fields are required') from exc

    @property
    def event_radec(self):
        """Gulls' effective event direction as an (RA, Dec) degree pair.

        When both saved Galactic fields are available, this uses them
        directly. Otherwise it derives Galactic direction from
        ``catalog_radec`` using the inverse of Gulls' saved-coordinate matrix.
        """
        longitude = self.raw.get('galactic_l')
        latitude = self.raw.get('galactic_b')
        if longitude is not None and latitude is not None:
            return _galactic_to_event_radec(longitude, latitude)
        return catalog_to_event_radec(*self.catalog_radec)

    @property
    def event_radec_source(self):
        """Describe which saved fields supplied :attr:`event_radec`."""
        if self.raw.get('galactic_l') is not None and self.raw.get('galactic_b') is not None:
            return 'galactic_l_b'
        return 'catalog_radec_inverse_matrix'

    @property
    def reference_bjd(self):
        return JD_ZERO + float(self.raw['tcroin'])

    def physical_parallax(self):
        """Equatorial parallax vectors in lens-source and source-lens order.

        The first vector is the General geoproj input; the second is the
        source-minus-lens direction used by ``PSBL_Phot_noPar_Param1``.
        Neither vector supplies the reference observer or its velocity.
        """
        r = self.raw
        ls = _sky_rotation(r['ra_deg'], r['dec_deg']) @ np.array(
            [r['piEE'], r['piEN']], dtype=float)
        if not np.all(np.isfinite(ls)) or np.linalg.norm(ls) == 0:
            raise ValueError('A finite nonzero parallax vector is required')
        return {'lens_minus_source': ls, 'source_minus_lens': -ls}

    def parallax_shifts(self, observer_delta_en, *, convention, pi_ecliptic=None):
        """Trajectory shifts for reference-subtracted ecliptic observer EN (AU).

        ``native`` is the historical Beginner sign; ``physical`` is the
        General astrometry-corrected sign. Both use the supplied lens-minus-
        source ecliptic piE, or the unchanged injected values by default.
        The observer displacement must already be relative to tcroin's
        position and velocity, using the simulated reference observer.
        """
        d = np.asarray(observer_delta_en, dtype=float)
        p = np.asarray([self.raw['piEE'], self.raw['piEN']] if pi_ecliptic is None
                       else pi_ecliptic, dtype=float)
        if d.ndim == 0 or d.shape[-1] != 2 or not np.all(np.isfinite(d)):
            raise ValueError('observer_delta_en must contain finite EN pairs')
        if p.shape != (2,) or not np.all(np.isfinite(p)):
            raise ValueError('pi_ecliptic must contain two finite components')
        if convention not in ('native', 'physical'):
            raise ValueError("convention must be 'native' or 'physical'")
        e, n = d[..., 0], d[..., 1]
        tau = -p[0]*e - p[1]*n
        u_native = p[0]*n - p[1]*e
        return np.stack((tau, u_native if convention == 'native' else -u_native), axis=-1)

    def to_bagle(self, mag_src):
        """Return keyword arguments for BAGLE PSBL_Phot_noPar_Param1.

        ``mag_src`` supplies three source magnitudes in observatory order. The
        returned point-source parameters do not encode rho or the Roman orbit.
        They describe a single source. Construct the named BAGLE class with
        ``PSBL_Phot_noPar_Param1(**parameters)`` in a full BAGLE installation.
        The separate ``predict`` method provides VBM finite-source photometry,
        including binary-source events.
        """
        if self.has_secondary_source:
            raise UnsupportedGullsEvent(
                'to_bagle exports BAGLE PSBL single-source parameters; '
                'use render_native for this binary-source event'
            )
        from bagle.model import u0_hat_from_thetaE_hat
        r = self.raw
        s, q, te = (float(r[k]) for k in ('Planet_s', 'Planet_q', 'tE_ref'))
        if not np.all(np.isfinite([s, q, te])) or min(s, q, te) <= 0:
            raise ValueError('Planet_s, Planet_q and tE_ref must be finite and positive')
        mags = list(mag_src)
        if len(mags) != 3 or not np.all(np.isfinite(mags)):
            raise ValueError('mag_src must contain three finite magnitudes')
        pi = self.physical_parallax()['source_minus_lens']
        norm = np.linalg.norm(pi)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('A finite nonzero parallax vector is required to define the BAGLE direction')
        motion = pi / norm
        a = np.deg2rad(r['alpha'])
        u = r['u0lens1'] + s/2 * np.sin(a)
        left = np.array([-motion[1], motion[0]])
        positive = u0_hat_from_thetaE_hat(motion, 1.0)
        signed_u = float(u * np.dot(left, positive))
        return dict(t0=float(JD_ZERO + r['t0lens1'] + s/2*te*np.cos(a) - JD_TO_MJD),
                    u0_amp=signed_u, tE=te, piE_E=float(pi[0]), piE_N=float(pi[1]),
                    q=q, sep=s, phi=float((180+r['alpha']) % 360),
                    b_sff=[float(r[f'Obs_{i}_fs']) for i in range(3)], mag_src=mags)

    def to_bagle_model(self, *, mag_base=0., n_outline=64, root_tol=1e-8):
        """Construct the finite-source BAGLE model for this Beginner event.

        Requires BAGLE's ``fsbl_ld_jax`` branch. The result is an actual
        BAGLE ``ModelClassABC`` instance, with inherited ``get_photometry``
        and ``get_amplification`` methods. Model query times use MJD, formed
        as ``(JD_ZERO - JD_TO_MJD) + Simulation_time``. No saved light-curve
        trajectory or VBM routine participates in the calculation.

        ``mag_base`` sets the baseline magnitude, once or per filter in
        F146/F087/F213 order. Its default zero makes
        ``10**(-0.4 * model.get_photometry(times, filt_idx=i))`` directly
        comparable to the saved baseline-normalized flux. A two-source row
        selects BAGLE's BFSBL family automatically.

        The original truth row remains available as ``model.raw``. Native
        input parameters and BAGLE's derived variables are model attributes;
        their conventions are described by ``help(type(model))``.
        """
        from .gulls_model import (
            FSBL_Phot_Par_GullsBeginner, BFSBL_Phot_Par_GullsBeginner,
        )
        r = self.raw
        base = np.asarray(mag_base, dtype=float)
        try:
            base = np.broadcast_to(base, (3,)).copy()
        except ValueError as exc:
            raise ValueError('mag_base must be scalar or contain three magnitudes') from exc
        primary = np.array([r[f'Obs_{i}_fs'] for i in range(3)], dtype=float)
        if (not np.all(np.isfinite(base)) or not np.all(np.isfinite(primary))
                or np.any(primary <= 0)):
            raise ValueError('baseline magnitudes must be finite and primary flux fractions positive')
        parameters = {key: r[key] for key in (
            't0lens1', 'u0lens1', 'tE_ref', 'piEE', 'piEN', 'Planet_s',
            'Planet_q', 'alpha', 'rho', 'ra_deg', 'dec_deg', 'tcroin',
        )}
        for key in ('galactic_l', 'galactic_b'):
            if r.get(key) is not None:
                parameters[key] = r[key]
        parameters.update(n_outline=n_outline, root_tol=root_tol)
        if self.has_secondary_source:
            secondary = np.array([r[f'Obs_{i}_fs2ofs1'] for i in range(3)], dtype=float)
            if not np.all(np.isfinite(secondary)) or np.any(secondary <= 0):
                raise ValueError('BAGLE binary-source magnitudes require positive secondary flux fractions')
            parameters.update({key: r[key] for key in (
                'Source2_rho', 'Source2_s', 'Source2_alpha', 'Source2_inc', 'Source2_phase',
            )})
            parameters.update(mag_src_pri=base - 2.5*np.log10(primary),
                              mag_src_sec=base - 2.5*np.log10(secondary),
                              b_sff=primary + secondary)
            model = BFSBL_Phot_Par_GullsBeginner(**parameters)
        else:
            parameters.update(mag_src=base - 2.5*np.log10(primary), b_sff=primary)
            model = FSBL_Phot_Par_GullsBeginner(**parameters)
        model.raw = dict(r)
        model.name = self.name
        return model

    def to_mulensmodel(self):
        """Return COM-origin finite-source parameters in absolute native BJD.

        This dictionary alone does not encode the recorded Roman observer.
        """
        if self.has_secondary_source:
            raise UnsupportedGullsEvent(
                'to_mulensmodel exports a single-source parameter set; '
                'use render_native for this binary-source event'
            )
        r = self.raw
        s, q, te = (float(r[k]) for k in ('Planet_s', 'Planet_q', 'tE_ref'))
        if not np.all(np.isfinite([s, q, te])) or min(s, q, te) <= 0:
            raise ValueError('Planet_s, Planet_q and tE_ref must be finite and positive')
        a = np.deg2rad(r['alpha'])
        c = s*q/(1+q)
        pi = self.physical_parallax()['source_minus_lens']
        return dict(pi_E_E=float(pi[0]), pi_E_N=float(pi[1]),
                    t_0=float(JD_ZERO+r['t0lens1']+c*te*np.cos(a)),
                    u_0=float(r['u0lens1']+c*np.sin(a)), t_E=te, s=s, q=q,
                    rho=float(r['rho']), alpha=float((r['alpha']+180)%360),
                    t_0_par=self.reference_bjd)

    def to_vbm(self, directory, observer_bjd, observer_xyz, reference_xyz, reference_velocity):
        """Export a high-level single- or binary-source VBM bundle.

        See :func:`bagle.gulls_vbm.export_vbm` for the model profile and
        observer-frame requirements.
        """
        from .gulls_vbm import export_vbm
        return export_vbm(self, directory, observer_bjd, observer_xyz, reference_xyz, reference_velocity)

    def predict(self, simulation_time=None, *, band=None,
                accuracy=1e-4, relative_accuracy=1e-6):
        """Return VBM model flux, with the total unlensed baseline equal to 1.

        With no arguments, return one value per loaded observation, in its
        original order, using its ``Simulation_time`` and ``observatory_code``.
        The observed noiseless comparison column is ``true_relative_flux``;
        ``measured_relative_flux`` includes noise. Both have the same baseline.

        For a finer model line, pass a one-dimensional array of simulation
        times (days since JD 2461406.5) and a band, for example::

            times = np.linspace(267.0, 268.0, 4001)
            flux = event.predict(times, band="F146")

        For a whole event, divide the intervals between observations within
        each campaign into smaller steps. A fixed-size grid spanning years
        can be coarser than the data and miss sharp features. Model and data
        should use the same band, with residuals at observation times only.

        ``band`` is 0/F146, 1/F087 or 2/F213, either a scalar or one per
        time. Legacy labels W146, Z087 and K213 are also accepted. When
        explicit times are supplied, a band is required. Times must be finite;
        they may be unsorted or repeated. An empty input returns an empty array.

        VBM calculates source motion and magnification from fixed truth. The
        method reconstructs the Beginner ORBIT5 observer and applies the event
        coordinate correction automatically. It works with either one or two
        sources and does not need saved source positions or parallax shifts.
        For comparisons, Simulation_time avoids the coarser printed BJD.

        ``accuracy`` sets VBM's absolute magnification tolerance (Tol), and
        ``relative_accuracy`` sets RelTol. Defaults match related Gulls source;
        the original executable is not authenticated. These settings do not
        promise bitwise agreement with the archived flux. Predictions at new
        times use the reconstructed orbit, not a measured Roman ephemeris.
        This method provides photometry, not physical astrometry.
        """
        from .gulls_vbm import predict_flux
        return predict_flux(self, simulation_time, band=band,
                            accuracy=accuracy, relative_accuracy=relative_accuracy)

    def native_trajectory(self):
        """Return center-of-mass x, y at the loaded curve's recorded epochs.

        The saved observer shifts are used directly; this does not construct
        an observer or predict its position at other times.
        """
        if self.curve is None:
            raise ValueError('Load a curve with BJD and parallax shift columns first')
        needed = ('BJD', 'parallax_shift_t', 'parallax_shift_u')
        missing = [key for key in needed if key not in self.curve.column_names]
        if missing:
            raise ValueError(f'Missing curve columns: {missing}')
        r = self.raw
        s, q, te, rho = (float(r[k]) for k in ('Planet_s', 'Planet_q', 'tE_ref', 'rho'))
        if not np.all(np.isfinite([s, q, te, rho])) or min(s, q, te, rho) <= 0:
            raise ValueError('Planet_s, Planet_q, tE_ref and rho must be finite and positive')
        bjd, dt, du = (np.asarray(self.curve[k].to_numpy(), dtype=float) for k in needed)
        if not all(np.all(np.isfinite(v)) for v in (bjd, dt, du)):
            raise ValueError('Curve epochs and observer shifts must be finite')
        a = np.deg2rad(float(r['alpha']))
        tau = (bjd - (JD_ZERO + float(r['t0lens1']))) / te + dt
        u = float(r['u0lens1']) + du
        return tau*np.cos(a) - u*np.sin(a) - s*q/(1+q), tau*np.sin(a) + u*np.cos(a)

    def native_source_trajectories(self):
        """Return source positions at recorded epochs in lens-axis Einstein units.

        The tuple contains one ``(N, 2)`` array for a single-source event and
        two arrays for a binary-source event. The primary uses the truth
        trajectory and saved observer shifts; the secondary uses its saved
        per-epoch offset from the primary. This retains source motion while
        avoiding rounded absolute coordinates from the curve.
        """
        if not self.has_secondary_source:
            x, y = self.native_trajectory()
            return (np.column_stack((x, y)),)
        if self.curve is None:
            raise ValueError('Load the native curve with source positions first')
        needed = ('source_x', 'source_y', 'source2_x', 'source2_y')
        missing = [key for key in needed if key not in self.curve.column_names]
        if missing:
            raise ValueError(f'Missing curve columns: {missing}')
        curve_source1 = np.column_stack(tuple(
            np.asarray(self.curve[key].to_numpy(), dtype=float)
            for key in ('source_x', 'source_y')
        ))
        curve_source2 = np.column_stack(tuple(
            np.asarray(self.curve[key].to_numpy(), dtype=float)
            for key in ('source2_x', 'source2_y')
        ))
        if not np.all(np.isfinite(curve_source1)) or not np.all(np.isfinite(curve_source2)):
            raise ValueError('Saved source positions must be finite')
        # The primary position can be recovered at full precision from truth
        # and the recorded shifts. Keep the curve's relative source offset so
        # secondary-source motion is retained without carrying the rounded
        # absolute-coordinate error shared by the saved positions.
        x1, y1 = self.native_trajectory()
        source1 = np.column_stack((x1, y1))
        source2 = source1 + (curve_source2 - curve_source1)
        return source1, source2

    def render_native_sources(self, accuracy=1e-4, relative_accuracy=1e-10):
        """Return per-source VBM magnifications and blended relative flux.

        This method supports binary-source events and evaluates each uniform
        disk separately with ``VBMicrolensing.BinaryMag2``. The saved
        ``Obs_i_fs2ofs1`` value is the secondary source's fraction of the total
        baseline flux, so it is used directly in the flux sum.

        The loaded curve must include BJD, the saved parallax shifts, the
        observatory code, and both saved source positions. Predictions are
        defined at those recorded epochs.
        """
        if not self.has_secondary_source:
            raise ValueError('render_native_sources requires a binary-source event')
        import VBMicrolensing
        if self.curve is None:
            raise ValueError('Load the native curve with source positions first')
        if 'observatory_code' not in self.curve.column_names:
            raise ValueError('Missing curve column: observatory_code')
        codes = np.asarray(self.curve['observatory_code'].to_numpy())
        if (not np.all(np.isfinite(codes)) or np.any(codes != np.floor(codes))
                or np.any((codes < 0) | (codes > 2))):
            raise ValueError('Observatory codes must be 0, 1 or 2')
        if (not np.isfinite(accuracy) or not np.isfinite(relative_accuracy)
                or min(accuracy, relative_accuracy) <= 0):
            raise ValueError('VBM tolerances must be finite and positive')

        source1, source2 = self.native_source_trajectories()
        r = self.raw
        vbm = VBMicrolensing.VBMicrolensing()
        vbm.Tol, vbm.RelTol, vbm.a1 = accuracy, relative_accuracy, 0.0
        magnifications = np.empty((2, len(source1)), dtype=float)
        for source_index, (positions, rho) in enumerate((
            (source1, float(r['rho'])),
            (source2, float(r['Source2_rho'])),
        )):
            magnifications[source_index] = np.fromiter((
                vbm.BinaryMag2(
                    float(r['Planet_s']), float(r['Planet_q']),
                    float(position[0]), float(position[1]), rho,
                )
                for position in positions
            ), dtype=float, count=len(positions))
        invalid = ~np.isfinite(magnifications) | (magnifications <= 0)
        if np.any(invalid):
            bad_sources = np.flatnonzero(np.any(invalid, axis=1)).tolist()
            raise ValueError(
                'VBM returned invalid magnification for source indices '
                f'{bad_sources}; invalid values={int(np.count_nonzero(invalid))}'
            )

        f1_by_obs = np.array([float(r[f'Obs_{i}_fs']) for i in range(3)])
        f2_by_obs = np.array([float(r[f'Obs_{i}_fs2ofs1']) for i in range(3)])
        if not np.all(np.isfinite(f1_by_obs)) or np.any(f1_by_obs < 0):
            raise ValueError('Primary baseline flux fractions must be finite and nonnegative')
        if not np.all(np.isfinite(f2_by_obs)) or np.any(f2_by_obs < 0):
            raise ValueError('Secondary baseline flux fractions must be finite and nonnegative')
        f1, f2 = f1_by_obs[codes.astype(int)], f2_by_obs[codes.astype(int)]
        source_flux = f1 + f2
        flux = 1.0 + f1*(magnifications[0] - 1.0) + f2*(magnifications[1] - 1.0)
        combined_magnification = np.divide(
            f1*magnifications[0] + f2*magnifications[1], source_flux,
            out=np.ones_like(flux), where=source_flux > 0,
        )
        return magnifications, combined_magnification, flux

    def render_native(self, accuracy=1e-4, relative_accuracy=1e-10):
        """Return source magnification and baseline-normalized relative flux.

        Requires VBMicrolensing and the loaded curve's BJD, parallax shifts,
        and ``observatory_code`` columns. Binary-source events also require
        the saved positions for both sources. Predictions are defined at the
        recorded epochs. A nonpositive or nonfinite VBM result raises an error
        before it can be converted to flux.

        For a single source, the first result is its raw magnification, as
        before. For a binary source, it is the source-flux-weighted combined
        magnification; ``render_native_sources`` also returns both component
        magnifications separately.
        """
        if self.has_secondary_source:
            _, combined, flux = self.render_native_sources(
                accuracy=accuracy, relative_accuracy=relative_accuracy
            )
            return combined, flux
        import VBMicrolensing
        x, y = self.native_trajectory()
        if 'observatory_code' not in self.curve.column_names:
            raise ValueError('Missing curve column: observatory_code')
        codes = np.asarray(self.curve['observatory_code'].to_numpy())
        if not np.all(np.isfinite(codes)) or np.any(codes != np.floor(codes)) or np.any((codes < 0) | (codes > 2)):
            raise ValueError('Observatory codes must be 0, 1 or 2')
        fs = np.array([self.raw[f'Obs_{i}_fs'] for i in range(3)], dtype=float)
        if not np.all(np.isfinite(fs)):
            raise ValueError('Source fractions must be finite')
        if not np.isfinite(accuracy) or not np.isfinite(relative_accuracy) or min(accuracy, relative_accuracy) <= 0:
            raise ValueError('VBM tolerances must be finite and positive')
        vbm = VBMicrolensing.VBMicrolensing()
        vbm.Tol, vbm.RelTol, vbm.a1 = accuracy, relative_accuracy, 0.0
        r = self.raw
        mag = np.fromiter((vbm.BinaryMag2(float(r['Planet_s']), float(r['Planet_q']),
                                        float(xi), float(yi), float(r['rho']))
                           for xi, yi in zip(x, y)), dtype=float, count=len(x))
        invalid = ~np.isfinite(mag) | (mag <= 0)
        if np.any(invalid):
            raise ValueError(f'VBM returned invalid magnification at {np.count_nonzero(invalid)} epochs; indices {np.flatnonzero(invalid)[:10].tolist()}')
        return mag, 1.0 + fs[codes.astype(int)]*(mag - 1.0)


def load_event(name, truth_path, curve_path=None, columns=None):
    """Load an event by name from existing truth and optional curve Parquets.

    ``columns`` selects curve columns; the name column is always included.
    Returns a :class:`GullsBeginner` with the unmodified truth row and an
    optional pyarrow Table. Curve row groups are selected by name statistics,
    falling back to reading just the name column when statistics are absent.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    truth = pq.read_table(Path(truth_path), filters=[('name', '=', name)]).to_pylist()
    if len(truth) != 1:
        raise ValueError(f'Expected one truth row for {name!r}; found {len(truth)}')
    event = GullsBeginner(truth[0])
    if curve_path is None:
        return event
    pf = pq.ParquetFile(Path(curve_path))
    names_index = pf.schema_arrow.get_field_index('name')
    if names_index < 0:
        raise ValueError('Curve parquet has no name column')
    groups = []
    for i in range(pf.num_row_groups):
        stats = pf.metadata.row_group(i).column(names_index).statistics
        if stats is not None and stats.has_min_max:
            matches = stats.min <= name <= stats.max
        else:
            matches = name in pf.read_row_group(i, columns=['name']).column('name').to_pylist()
        if matches:
            groups.append(i)
    if not groups:
        raise ValueError(f'No curve rows for {name!r}')
    selected = None if columns is None else list(dict.fromkeys(['name', *columns]))
    table = pf.read_row_groups(groups, columns=selected)
    event.curve = table.filter(pc.equal(table['name'], name))
    if not len(event.curve):
        raise ValueError(f'No curve rows for {name!r}')
    return event


def load_model(name, truth_path, *, mag_base=0., n_outline=64, root_tol=1e-8):
    """Load Beginner truth directly into a BAGLE finite-source model.

    Requires the ``fsbl_ld_jax`` branch and the truth Parquet only. The
    returned class is ``FSBL_Phot_Par_GullsBeginner`` or
    ``BFSBL_Phot_Par_GullsBeginner`` according to source multiplicity.
    Its inherited BAGLE methods accept MJD arrays. For simulation-day
    observations, ``mjd = (JD_ZERO - JD_TO_MJD) + Simulation_time``.

    With the default zero baseline magnitude,
    ``flux = 10**(-0.4 * model.get_photometry(mjd, filt_idx=0))`` is the
    baseline-normalized F146 flux. Indices 1 and 2 select F087 and F213.
    ``mag_base`` may instead contain a scalar or three baseline magnitudes.
    The original row is preserved in ``model.raw``; the native and canonical
    BAGLE parameters are model attributes. See the class description for
    the historical observer/sign convention and precision limits.

    Examples
    --------
    >>> from bagle.gulls import load_model, JD_ZERO, JD_TO_MJD
    >>> event = load_model('RMDC26_000028', 'truth.parquet')  # doctest: +SKIP
    >>> mjd = (JD_ZERO - JD_TO_MJD) + simulation_time  # doctest: +SKIP
    >>> magnification = event.get_amplification(mjd)  # doctest: +SKIP
    """
    return load_event(name, truth_path).to_bagle_model(
        mag_base=mag_base, n_outline=n_outline, root_tol=root_tol)
