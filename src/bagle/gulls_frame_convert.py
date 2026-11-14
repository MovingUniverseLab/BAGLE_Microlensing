"""Destination adapters used by BAGLE's existing ``frame_convert`` functions.

Conversions start from BAGLE's derived source/lens geometry, radii, fluxes,
and observer interface. They never read ``raw`` or the saved light curves.
Optional destination packages are imported only when their converter is used.
"""
import copy
import tempfile
from pathlib import Path

import numpy as np

from bagle.gulls import JD_TO_MJD


def _parameters(model, filt_idx):
    """Express canonical BAGLE midpoint geometry about the lens barycenter."""
    binary = hasattr(model, 'radiusS_sec')
    mag1 = np.asarray(model.mag_src_pri if binary else model.mag_src)
    if not isinstance(filt_idx, (int, np.integer)) or not 0 <= filt_idx < len(mag1):
        raise ValueError('filt_idx must select an existing model filter')
    lens1, lens2 = np.asarray(model.xL1_over_theta), np.asarray(model.xL2_over_theta)
    axis = lens2-lens1
    s = np.linalg.norm(axis)
    axis /= s
    perpendicular = np.array([-axis[1], axis[0]])
    q = float(model.m2/model.m1)
    center = (lens1+q*lens2)/(1+q)
    motion = np.asarray(model.muRel_hat)
    left = np.array([-motion[1], motion[0]])
    angle = np.arctan2(motion @ perpendicular, motion @ axis)
    origin = model.t0_pri if binary else model.t0
    positions = [model.u0_pri, model.u0_sec] if binary else [model.u0]
    t0, u0 = [], []
    for position in positions:
        relative = np.asarray(position)-center
        t0.append(float(origin-(relative @ motion)*model.tE))
        u0.append(float(relative @ left))
    f1 = float(10**(-.4*mag1[filt_idx]))
    f2 = float(10**(-.4*model.mag_src_sec[filt_idx])) if binary else 0.
    total = (f1+f2)/float(model.b_sff[filt_idx])
    radii = ([float(model.radiusS_pri), float(model.radiusS_sec)] if binary
             else [float(model.radiusS)])
    return dict(s=float(s), q=q, alpha=float(angle+np.pi), t0=t0, u0=u0,
                tE=float(model.tE), rho=radii,
                pi=np.asarray(model.piE, dtype=float).copy(),
                reference=float(model.reference_mjd), binary=binary,
                source_fluxes=[f1, f2] if binary else [f1],
                blend_flux=float(total-f1-f2))


def _epochs(times_mjd):
    if times_mjd is None:
        raise ValueError('times_mjd is required to sample the Gulls observer; '
                         'use (2461406.5 - 2400000.5) + Simulation_time')
    t = np.asarray(times_mjd, dtype=float)
    if t.ndim != 1 or not len(t) or not np.all(np.isfinite(t)):
        raise ValueError('times_mjd must be a nonempty finite one-dimensional array')
    return t


def to_mulens(model, *, filt_idx=0):
    """Return a MulensModel model with an analytic effective Roman observer."""
    from bagle.gulls_mulens import RomanModel
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    p = _parameters(model, filt_idx)
    params = dict(t_E=p['tE'], s=p['s'], q=p['q'],
                  alpha=np.rad2deg(p['alpha']) % 360,
                  pi_E_E=p['pi'][0], pi_E_N=p['pi'][1],
                  t_0_par=p['reference']+JD_TO_MJD)
    for i, (t0, u0, rho) in enumerate(zip(p['t0'], p['u0'], p['rho'])):
        suffix = f'_{i+1}' if p['binary'] else ''
        params.update({f't_0{suffix}': t0+JD_TO_MJD,
                       f'u_0{suffix}': u0, f'rho{suffix}': rho})
    target = RomanModel(params,
                        coords=SkyCoord(model.raL*u.deg, model.decL*u.deg),
                        observer=copy.copy(model))
    target.source_fluxes = np.array(p['source_fluxes'])
    target.blend_flux = p['blend_flux']
    target.parallax(earth_orbital=False, satellite=True, topocentric=False)
    target.default_magnification_method = 'VBM'
    target.set_magnification_methods_parameters(
        {'VBM': {'accuracy': 1e-4, 'relative_accuracy': 1e-6}})
    return target


def to_vbm(model, *, times_mjd=None, filt_idx=0):
    """Return stock VBM and its high-level parameter vector for one filter.

    The observer table includes every requested epoch and both source closest
    approaches. VBM interpolates between samples; a different query grid needs
    another conversion. The tables are loaded into VBM before removal.
    """
    import VBMicrolensing
    from bagle.gulls_vbm import _coordinate_string
    p = _parameters(model, filt_idx)
    times = _epochs(times_mjd)
    grid = np.unique(np.r_[times, p['reference'], p['t0']])
    grid = np.r_[grid[0]-.01, grid, grid[-1]+.01]
    jd = grid+JD_TO_MJD
    xyz = model.get_observer_displacement(grid)
    distance = np.linalg.norm(xyz, axis=1)
    ra = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360
    dec = np.rad2deg(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
    vbm = VBMicrolensing.VBMicrolensing()
    # VBM uses lens-minus-source piE. BAGLE stores source-minus-lens.
    pe, pn = -p['pi']
    pars = [np.log(p['s']), np.log(p['q']), p['u0'][0], p['alpha'],
            np.log(p['rho'][0]), np.log(p['tE']),
            p['t0'][0] + (JD_TO_MJD-2450000), pn, pe]
    if p['binary']:
        if p['q'] == 1. and p['rho'][0] != p['rho'][1]:
            raise ValueError('VBM cannot encode unequal source radii at lens q=1; '
                             'MulensModel and pyLIMA support independent radii')
        ratio = p['source_fluxes'][1]/p['source_fluxes'][0]
        vbm.mass_luminosity_exponent = 4.
        vbm.mass_radius_exponent = (4*np.log(p['rho'][1]/p['rho'][0])
                                    / np.log(p['q']) if p['q'] != 1. else 0.)
        # VBM's combined routine requires nonzero orbital speed. This bounded
        # static limit moves the source by less than 5e-10 Einstein radii.
        speed = .5e-9/max(np.max(np.abs(grid-p['t0'][0])), 1.)
        pars += [0., 0., 0., p['u0'][1],
                 p['t0'][1]+(JD_TO_MJD-2450000), np.log(ratio), 0., 0., speed]
    with tempfile.TemporaryDirectory(prefix='bagle_vbm_') as directory:
        path = Path(directory)
        # VBM 5.3.5 uses a fixed-size C buffer for this directory path.
        if len(str(path).encode()) > 220:
            raise ValueError('VBM satellite directory path exceeds its C buffer')
        with (path/'satellite1.txt').open('w') as stream:
            stream.write('$$SOE\n')
            np.savetxt(stream, np.c_[jd, ra, dec, distance, np.zeros(len(jd))],
                       fmt='%.15f')
            stream.write('$$EOE\n')
        (path/'coordinates.txt').write_text(
            _coordinate_string(model.raL, model.decL)+'\n')
        with (path/'sun.txt').open('w') as stream:
            stream.write('$$SOE\n')
            for epoch in np.arange(np.floor(jd[0]/1000)*1000-1000,
                                   jd[-1]+3000, 1000):
                stream.write(f'{epoch:.1f} 0 0 0 0\n')
            stream.write('$$EOE\n')
        vbm.LoadSunTable(str(path/'sun.txt'))
        vbm.SetObjectCoordinates(str(path/'coordinates.txt'), str(path))
        if vbm.nsat != 1:
            raise ValueError('VBM failed to load the Gulls observer table')
    vbm.satellite, vbm.parallaxsystem = 1, 1
    vbm.parallaxephemeris, vbm.t_in_HJD = True, False
    vbm.t0_par_fixed, vbm.t0_par = 1, float(pars[6])
    vbm.Tol, vbm.RelTol, vbm.minannuli, vbm.a1 = 1e-4, 1e-6, 1, 0.
    return vbm, [float(value) for value in pars]


def to_pylima(model, *, times_mjd=None, filt_idx=0):
    """Return pyLIMA's uniform-source binary-lens model, params, telescope."""
    from pyLIMA.event import Event
    from pyLIMA.models.USBL_model import USBLmodel
    from bagle.gulls_pylima import RomanTelescope
    p = _parameters(model, filt_idx)
    times = _epochs(times_mjd)
    # pyLIMA sorts and deduplicates telescope data. Its returned telescope is
    # the authority for output order, so construct the observer on that grid.
    tel = RomanTelescope(
        observer=copy.copy(model), name='Roman', camera_filter=f'Gulls{filt_idx}',
        lightcurve=np.c_[times+JD_TO_MJD, np.ones(len(times)),
                         np.full(len(times), .01)],
        lightcurve_names=['time', 'flux', 'err_flux'],
        lightcurve_units=['JD', 'W/m^2', 'W/m^2'])
    event = Event(ra=model.raL, dec=model.decL)
    event.name = 'Gulls Beginner'
    event.telescopes.append(tel)
    target = USBLmodel(event, parallax=['Full', p['reference']+JD_TO_MJD],
                       double_source=['Static', 0.] if p['binary']
                       else ['None', 0.])
    pars = dict(t0=p['t0'][0]+JD_TO_MJD, u0=p['u0'][0], tE=p['tE'],
                rho=p['rho'][0], separation=p['s'], mass_ratio=p['q'],
                alpha=p['alpha'], piEE=p['pi'][0], piEN=p['pi'][1],
                fsource_Roman=p['source_fluxes'][0], fblend_Roman=p['blend_flux'])
    if p['binary']:
        pars.update(delta_t0=p['t0'][1]-p['t0'][0],
                    delta_u0=p['u0'][1]-p['u0'][0], rho_2=p['rho'][1])
        pars[f'q_flux_{tel.filter}'] = p['source_fluxes'][1]/p['source_fluxes'][0]
    values = [pars[name] for name in target.model_dictionnary]
    return target, target.compute_pyLIMA_parameters(values), tel
