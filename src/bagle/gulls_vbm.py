"""Export single-source Gulls models for stock VBMicrolensing 5.3.5.

The satellite table contains observer positions in the injected reference
frame. It is an ephemeris, not a table of source or lens positions.
"""
import json
from pathlib import Path

import numpy as np

from .gulls import JD_ZERO, NATIVE_ECLIPTIC_OBLIQUITY_DEG, _sky_rotation


from .gulls_orbit import roman_orbit5


def predict_flux(event, simulation_time=None, *, band=None,
                 accuracy=1e-4, relative_accuracy=1e-6):
    """Generate relative flux through VBM's high-level trajectory routines.

    See ``GullsBeginner.predict`` for the public input and time conventions.
    Temporary observer files are removed before this function returns.
    """
    import tempfile
    import VBMicrolensing

    if simulation_time is None:
        if event.curve is None or 'Simulation_time' not in event.curve.column_names:
            raise ValueError('predict() needs loaded Simulation_time, or explicit simulation times and band')
        simulation_time = event.curve['Simulation_time'].to_numpy()
        if band is None:
            if 'observatory_code' not in event.curve.column_names:
                raise ValueError('predict() needs loaded observatory_code, or an explicit band')
            band = event.curve['observatory_code'].to_numpy()
    elif band is None:
        raise ValueError('an explicit band is required with explicit simulation times')
    times = np.asarray(simulation_time, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError('simulation times must be a finite one-dimensional array in days since JD 2461406.5')
    labels = {'0': 0, '1': 1, '2': 2, 'F146': 0, 'F087': 1, 'F213': 2,
              'W146': 0, 'Z087': 1, 'K213': 2}
    raw_band = np.asarray(band)
    if raw_band.ndim == 0:
        raw_band = np.full(times.shape, raw_band.item())
    if raw_band.shape != times.shape:
        raise ValueError('band must be a scalar or have one value per simulation time')
    try:
        codes = np.array([labels[str(value)] for value in raw_band], dtype=int)
    except KeyError as exc:
        raise ValueError('band must be 0/F146, 1/F087 or 2/F213 (W146/Z087/K213 also accepted)') from exc
    if not np.all(np.isfinite([accuracy, relative_accuracy])) or min(accuracy, relative_accuracy) <= 0:
        raise ValueError('VBM tolerances must be finite and positive')
    if not len(times):
        return np.empty(0, dtype=float)

    r = event.raw
    alpha = np.deg2rad(float(r['alpha']))
    te = float(r['tE_ref'])
    center = float(r['Planet_s'])*float(r['Planet_q'])/(1+float(r['Planet_q']))
    t01 = JD_ZERO + float(r['t0lens1']) + center*te*np.cos(alpha)
    anchors = [t01, event.reference_bjd]
    if event.has_secondary_source:
        phase, inc, node = np.deg2rad([r['Source2_phase'], r['Source2_inc'], r['Source2_alpha']])
        along = float(r['Source2_s'])*np.cos(phase)
        across = float(r['Source2_s'])*np.sin(phase)*np.cos(inc)
        dx = along*np.cos(node)-across*np.sin(node)
        dy = along*np.sin(node)+across*np.cos(node)
        anchors.append(t01-(dx*np.cos(alpha)+dy*np.sin(alpha))*te)
    # Include every requested time in the table, avoiding interpolation across
    # gaps between campaigns. Native epochs stabilize the exported static-source
    # limit across calls that request different subwindows of the same event.
    native_times = []
    if event.curve is not None and 'Simulation_time' in event.curve.column_names:
        native_times = JD_ZERO + event.curve['Simulation_time'].to_numpy()
        if not np.all(np.isfinite(native_times)):
            raise ValueError('loaded Simulation_time must be finite')
    grid = np.unique(np.r_[JD_ZERO+times, native_times, anchors])
    grid = np.r_[grid[0]-.01, grid, grid[-1]+.01]
    xyz, origin, velocity = roman_orbit5(grid, event.reference_bjd)
    # Reduced-JD arithmetic preserves more of Simulation_time's precision.
    query = (JD_ZERO-2450000)+times
    flux = np.empty_like(times)
    with tempfile.TemporaryDirectory(prefix='gulls_vbm_') as temporary:
        directory = Path(temporary)
        config = export_vbm(event, directory, grid, xyz, origin, velocity)
        for code in np.unique(codes):
            selected = codes == code
            vbm = VBMicrolensing.VBMicrolensing()
            vbm.LoadSunTable(str(directory/config['files']['sun']))
            vbm.SetObjectCoordinates(str(directory/config['files']['coordinates']), str(directory))
            if vbm.nsat != 1:
                raise ValueError('VBM did not load the reconstructed Roman observer')
            for key, value in config['vbm'].items():
                setattr(vbm, key, value)
            vbm.Tol, vbm.RelTol = float(accuracy), float(relative_accuracy)
            if event.has_secondary_source:
                vbm.mass_luminosity_exponent = config['mass_luminosity_exponents'][code]
                vbm.mass_radius_exponent = config['mass_radius_exponents'][code]
                result = np.asarray(vbm.BinSourceBinLensLightCurve(
                    config['params_by_band'][code], query[selected].tolist()))
                source_flux = config['source1_flux_fractions'][code]+config['source2_flux_fractions'][code]
                expected_rows = 6
            else:
                result = np.asarray(vbm.BinaryLightCurveParallax(config['params'], query[selected].tolist()))
                source_flux = config['source_flux_fractions'][code]
                expected_rows = 3
            if (result.shape != (expected_rows, np.count_nonzero(selected))
                    or not np.all(np.isfinite(result)) or np.any(result[0] <= 0)
                    or vbm.parallaxextrapolation):
                raise ValueError(f'VBM returned invalid or extrapolated predictions for band {code}')
            flux[selected] = 1+source_flux*(result[0]-1)
    if not np.all(np.isfinite(flux)):
        raise ValueError('VBM returned nonfinite relative flux')
    return flux




def _coordinate_string(ra, dec):
    h = (ra % 360) / 15
    hh = int(h)
    mm = int((h - hh) * 60)
    ss = (h - hh - mm / 60) * 3600
    d = abs(dec)
    dd = int(d)
    dm = int((d - dd) * 60)
    ds = (d - dd - dm / 60) * 3600
    # VBM's parser loses the sign of -0 degrees.
    if dec < 0 and dd == 0:
        raise ValueError('VBM cannot parse negative declinations between -1 and 0 degrees')
    return f'{hh:02d}:{mm:02d}:{ss:.12f} {"-" if dec < 0 else "+"}{dd:02d}:{dm:02d}:{ds:.12f}'


def export_vbm(event, directory, observer_bjd, observer_xyz, reference_xyz, reference_velocity):
    """Write a high-level VBM parameter and observer bundle.

    ``observer_bjd`` is a strictly increasing grid of at least three absolute
    native BJD epochs. It must span the primary source reference time and all
    prediction epochs; ``observer_xyz`` are the corresponding heliocentric
    ecliptic Cartesian positions in AU. ``reference_xyz`` and
    ``reference_velocity`` are the same observer's position (AU) and velocity
    (AU/day) at the event's tcroin. VBM linearly interpolates the supplied
    observer grid. The caller must choose a grid dense enough for the desired
    accuracy and keep prediction epochs inside it.

    This uses a stationary reference ephemeris plus the observer displacement
    from the tcroin position and velocity, rather than a physical Earth/Sun
    table. The frame conversion is event-specific but independent of the lens
    and source trajectory. It avoids VBM 5.3.5's use of its parallax reference
    epoch in the linear trajectory when that epoch is fixed. The unmodified
    input truth remains in ``event.raw``. Binary-source bundles use VBM's
    high-level ``BinSourceBinLensLightCurve`` routine once per band so its
    separate source radii and baseline flux ratios can match the truth.
    """
    r = event.raw
    t = np.asarray(observer_bjd, dtype=float)
    xyz = np.asarray(observer_xyz, dtype=float)
    ref = np.asarray(reference_xyz, dtype=float)
    vel = np.asarray(reference_velocity, dtype=float)
    if t.ndim != 1 or len(t) < 3 or xyz.shape != (len(t), 3):
        raise ValueError('observer grid must contain at least three times and XYZ positions')
    if ref.shape != (3,) or vel.shape != (3,) or not all(
        np.all(np.isfinite(v)) for v in (t, xyz, ref, vel)
    ) or np.any(np.diff(t) <= 0):
        raise ValueError('observer samples must be finite and times strictly increasing')
    s, q, rho, te = (float(r[k]) for k in ('Planet_s', 'Planet_q', 'rho', 'tE_ref'))
    catalog_ra, catalog_dec = event.catalog_radec
    ra, dec = event.event_radec
    alpha, u = (float(r[k]) for k in ('alpha', 'u0lens1'))
    if (not np.all(np.isfinite([s, q, rho, te, catalog_ra, catalog_dec, ra, dec, alpha, u]))
            or min(s, q, rho, te) <= 0 or not -90 < catalog_dec < 90
            or not -90 < dec < 90):
        raise ValueError('invalid lens, source, or sky parameters')
    tref = JD_ZERO + float(r['tcroin'])
    a = np.deg2rad(alpha)
    c = s*q/(1+q)
    t0 = JD_ZERO + float(r['t0lens1']) + c*te*np.cos(a)
    pi = _sky_rotation(ra, dec, NATIVE_ECLIPTIC_OBLIQUITY_DEG) @ np.array(
        [r['piEE'], r['piEN']], dtype=float
    )
    if not np.all(np.isfinite(np.r_[tref, t0, pi])) or np.linalg.norm(pi) == 0:
        raise ValueError('finite reference times and nonzero parallax are required')
    u0 = u+c*np.sin(a)
    alpha_vbm = a+np.pi
    params = [float(x) for x in (np.log(s), np.log(q), u0,
                                   alpha_vbm, np.log(rho), np.log(te),
                                   t0-2450000, pi[1], pi[0])]
    fs1 = [float(r[f'Obs_{i}_fs']) for i in range(3)]
    if not np.all(np.isfinite(fs1)) or np.any(np.asarray(fs1) < 0):
        raise ValueError('source flux fractions must be finite')

    binary = None
    if event.has_secondary_source:
        r2 = float(r['Source2_rho'])
        sep = float(r['Source2_s'])
        phase = np.deg2rad(float(r['Source2_phase']))
        inclination = np.deg2rad(float(r['Source2_inc']))
        node = np.deg2rad(float(r['Source2_alpha']))
        masses = np.array([r.get('Source_Mass'), r.get('Source2_Mass')], dtype=float)
        if (not np.all(np.isfinite([r2, sep, phase, inclination, node]))
                or r2 <= 0 or sep <= 0 or not np.all(np.isfinite(masses))
                or np.any(masses <= 0)):
            raise ValueError('binary-source radii, orbit fields and masses must be finite and positive')
        # The truth stores the projected separation in lens-axis Einstein units.
        along = sep*np.cos(phase)
        across = sep*np.sin(phase)*np.cos(inclination)
        dx = along*np.cos(node)-across*np.sin(node)
        dy = along*np.sin(node)+across*np.cos(node)
        dtau = dx*np.cos(a)+dy*np.sin(a)
        du = -dx*np.sin(a)+dy*np.cos(a)
        u02 = u0+du
        t02 = t0-dtau*te
        if not (t[0] < t0 < t[-1]):
            raise ValueError('observer grid must span the primary source reference time')

        fs2 = [float(r[f'Obs_{i}_fs2ofs1']) for i in range(3)]
        if not np.all(np.isfinite(fs2)) or np.any(np.asarray(fs2) <= 0) or np.any(np.asarray(fs1) <= 0):
            raise ValueError('binary-source VBM export requires positive finite flux fractions for both sources')
        source_mass_ratio = float(masses[1]/masses[0])
        if not np.isfinite(source_mass_ratio) or source_mass_ratio <= 0:
            raise ValueError('source masses must define a positive mass ratio')
        log_lens_mass_ratio = float(np.log(q))
        log_radius_ratio = float(np.log(r2)-np.log(rho))
        if log_lens_mass_ratio == 0 and log_radius_ratio != 0:
            raise ValueError(
                'VBM 5.3.5 cannot represent unequal source radii when the lens mass ratio is 1'
            )

        max_dt = float(np.max(np.abs(t-t0)))
        source_speed = 0.5e-9/max(max_dt, 1.0)
        params_by_band = []
        lum_exponents = []
        radius_exponents = []
        flux_ratios = []
        for f_primary, f_secondary in zip(fs1, fs2):
            flux_ratio = f_secondary/f_primary
            log_flux_ratio = float(np.log(flux_ratio))
            log_mass_ratio = float(np.log(source_mass_ratio))
            # VBM derives its source mass ratio from the flux ratio. A
            # per-band exponent preserves the truth mass ratio while allowing
            # the truth's band-dependent source colors when both ratios differ
            # from unity. In the degenerate unity case, use a finite exponent;
            # the exported orbit speed is too small for that ratio to affect
            # the light curve materially.
            if abs(log_mass_ratio) > 1e-12 and abs(log_flux_ratio) > 1e-12:
                lum_exponent = log_flux_ratio/log_mass_ratio
            else:
                # The exported orbit is the static-source limit, so this
                # degenerate case does not move either component trajectory.
                lum_exponent = 4.0
            radius_exponent = (
                0.0 if log_lens_mass_ratio == 0
                else log_radius_ratio*lum_exponent/log_lens_mass_ratio
            )
            band_params = [
                np.log(s), np.log(q), u0, alpha_vbm, np.log(rho), np.log(te),
                t0-2450000, pi[1], pi[0], 0.0, 0.0, 0.0,
                u02, t02-2450000, log_flux_ratio, 0.0, 0.0, source_speed,
            ]
            params_by_band.append([float(x) for x in band_params])
            lum_exponents.append(float(lum_exponent))
            radius_exponents.append(float(radius_exponent))
            flux_ratios.append(float(flux_ratio))
        binary = {
            'source2_rho': r2,
            'source2_separation': [float(dx), float(dy)],
            'source_mass_ratio': source_mass_ratio,
            'source2_flux_fractions': fs2,
            'source2_to_source1_flux_ratios': flux_ratios,
            'source_orbit_speed_einstein_per_day': float(source_speed),
            'source_orbit_excursion_bound_einstein': float(source_speed*max_dt),
            'params_by_band': params_by_band,
            'mass_luminosity_exponents': lum_exponents,
            'mass_radius_exponents': radius_exponents,
        }

    coordinates = _coordinate_string(ra, dec)
    directory = Path(directory).resolve()
    # VBM 5.3.5 uses a 256-byte C buffer for the satellite directory.
    if len(str(directory).encode()) > 220:
        raise ValueError('VBM 5.3.5 satellite directory path is too long')
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob('satellite?.txt')) or any((directory / x).exists() for x in ('coordinates.txt', 'stationary_sun.txt', 'vbm.json')):
        raise FileExistsError('output directory already contains VBM bundle files')
    shift = xyz-ref-(t-tref)[:, None]*vel
    # Ecliptic-to-equatorial rotation used by Gulls' coordinate transform.
    co, si = .9174821430652419, .39777696911260596
    eq = np.column_stack((shift[:, 0], co*shift[:, 1]-si*shift[:, 2],
                          si*shift[:, 1]+co*shift[:, 2]))
    dist = np.linalg.norm(eq, axis=1)
    sat_ra = np.degrees(np.arctan2(eq[:, 1], eq[:, 0])) % 360
    sat_dec = np.degrees(np.arctan2(eq[:, 2], np.hypot(eq[:, 0], eq[:, 1])))
    with (directory/'satellite1.txt').open('x') as f:
        f.write('$$SOE\n')
        np.savetxt(f, np.column_stack((t, sat_ra, sat_dec, dist, np.zeros(len(t)))), fmt='%.15f')
        f.write('$$EOE\n')
    (directory/'coordinates.txt').write_text(coordinates+'\n')
    start = np.floor(min(t[0], t0)/1000)*1000-1000
    end = max(t[-1], t0)+2000
    with (directory/'stationary_sun.txt').open('x') as f:
        f.write('$$SOE\n')
        for jd in np.arange(start, end+1000, 1000):
            f.write(f'{jd:.1f} 0 0 0 0\n')
        f.write('$$EOE\n')
    common = {
        'time_offset_jd': 2450000,
        'reference_bjd': tref,
        'observer_grid_bjd': [float(t[0]), float(t[-1])],
        'observer_frame': 'position minus position and linear velocity at reference_bjd; ecliptic AU',
        'coordinates': {
            'catalog_radec_deg': [catalog_ra, catalog_dec],
            'event_radec_deg': [ra, dec],
            'event_direction_source': event.event_radec_source,
            'event_direction_method': (
                'saved galactic_l/galactic_b transformed with the transpose of '
                'Gulls coords::lb2ad equatorial-to-Galactic matrix'
                if event.event_radec_source == 'galactic_l_b'
                else 'answer-file RA/Dec inverted through the rounded astroFns.cpp '
                     'eq2gal matrix, then transformed with the transpose of Gulls '
                     'coords::lb2ad equatorial-to-Galactic matrix'
            ),
            'ecliptic_to_equatorial_parallax_obliquity_deg': (
                NATIVE_ECLIPTIC_OBLIQUITY_DEG
            ),
        },
        'files': {'coordinates': 'coordinates.txt', 'sun': 'stationary_sun.txt', 'satellite': 'satellite1.txt'},
    }
    vbm_config = {
        'satellite': 1, 'parallaxsystem': 1, 'parallaxephemeris': True,
        't_in_HJD': False, 't0_par_fixed': 1, 't0_par': float(t0-2450000),
        'Tol': 1e-4, 'RelTol': 1e-10, 'a1': 0.0,
    }
    if binary is None:
        config = {
            **common,
            'schema': 'gulls_vbm_5.3.5_single_source_1',
            'event': event.name,
            'params': params,
            'params_order': ['ln_s', 'ln_q', 'u0', 'alpha_rad', 'ln_rho', 'ln_tE_days', 't0_jd_minus_2450000', 'piE_N', 'piE_E'],
            'source_flux_fractions': fs1,
            'vbm': vbm_config,
        }
    else:
        config = {
            **common,
            'schema': 'gulls_vbm_5.3.5_binary_source_1',
            'event': event.name,
            'routine': 'BinSourceBinLensLightCurve',
            'params_order': [
                'ln_s', 'ln_q', 'u0_source1', 'alpha_rad', 'ln_rho_source1',
                'ln_tE_days', 't0_source1_jd_minus_2450000', 'piE_N', 'piE_E',
                'lens_w1', 'lens_w2', 'lens_w3', 'u0_source2',
                't0_source2_jd_minus_2450000', 'ln_flux_ratio_source2_to_source1',
                'source_w1', 'source_w2', 'source_w3',
            ],
            'source1_flux_fractions': fs1,
            **binary,
            'vbm': vbm_config,
        }
    (directory/'vbm.json').write_text(json.dumps(config, indent=2)+'\n')
    return config
