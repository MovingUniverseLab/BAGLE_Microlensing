"""Tests for bagle.parallax observer ephemerides and parallax vectors.

The Horizons path (any observer not in astropy's ephemeris, e.g. a
spacecraft) is tested offline with a fake ``astroquery.jplhorizons`` module
that serves astropy's Earth ephemeris. One slow test queries live Horizons.
"""
import sys
import types

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import (CartesianRepresentation, SkyCoord,
                                 get_body_barycentric_posvel)
from astropy.table import Table
from astropy.time import Time

from bagle import parallax

# Sgr A*, near the ecliptic, where an ecliptic/equatorial mix-up inflates
# the North parallax from ~0.1 AU to ~0.5 AU.
RA_GC, DEC_GC = 266.41683, -29.00781

# Sparse, unevenly spaced epochs (median gap ~110 d) that used to set a
# ~110-day Horizons step and linear chords across the orbit.
SPARSE_JD = 2460310.5 + np.array([0.0, 37.3, 151.7, 260.2, 361.9])


class _FakeHorizons:
    """
    Stand-in for ``astroquery.jplhorizons.Horizons`` that serves Earth.

    Records the constructor and ``vectors`` arguments in ``calls`` and
    returns astropy's barycentric ICRS Earth state on the requested grid.
    """
    calls = []

    def __init__(self, id=None, location=None, epochs=None):
        self.kw = dict(id=id, location=location, epochs=epochs)

    def vectors(self, refplane='ecliptic', **kwargs):
        """Return an Earth state-vector table on the requested grid."""
        self.kw['refplane'] = refplane
        _FakeHorizons.calls.append(self.kw)

        # Rebuild the fixed grid from start/stop/step ('<N>m').
        ep = self.kw['epochs']
        start = Time(ep['start'], scale='tdb').jd
        stop = Time(ep['stop'], scale='tdb').jd
        step = float(ep['step'].rstrip('m')) / 1440.0
        jd = np.arange(start, stop + 0.5 * step, step)

        pos, vel = get_body_barycentric_posvel(
            'earth', Time(jd, format='jd', scale='tdb'))
        tab = Table()
        tab['datetime_jd'] = jd
        for c, q in zip('xyz', pos.xyz.to(u.au)):
            tab[c] = q
        for c, q in zip(('vx', 'vy', 'vz'), vel.xyz.to(u.au / u.day)):
            tab[c] = q
        return tab


@pytest.fixture
def fake_horizons(monkeypatch):
    """Install the fake Horizons module and return its class."""
    _FakeHorizons.calls = []
    mod = types.ModuleType('astroquery.jplhorizons')
    mod.Horizons = _FakeHorizons
    monkeypatch.setitem(sys.modules, 'astroquery.jplhorizons', mod)
    return _FakeHorizons


def _max_dist_km(a, b):
    """Largest |a - b| over time for two CartesianRepresentations, in km."""
    d = (a - b).xyz.to(u.km).value
    return np.max(np.linalg.norm(np.atleast_2d(d.T), axis=-1))


@pytest.mark.parametrize(('text', 'scale'), [
    ('prior to A.D. 2013-DEC-19 09:54:19.5\n774 UT', 'utc'),
    ('after A.D. 2025-JAN-25 12:00:00.1234 TDB', 'tdb'),
])
def test_parse_horizons_observer_range_boundary(text, scale):
    parsed = parallax._parse_to_mjd(text)
    expected = Time(
        '2013-12-19 09:54:19.5774' if scale == 'utc'
        else '2025-01-25 12:00:00.1234',
        format='iso', scale=scale,
    ).tdb.mjd
    assert parsed == pytest.approx(expected, abs=1e-10)


def test_horizons_query_is_barycentric_equatorial(fake_horizons):
    """The Horizons query uses the SSB origin and the ICRF equatorial plane."""
    t = Time(SPARSE_JD, format='jd', scale='tdb')
    parallax.get_observer_barycentric('fake', t)

    (kw,) = fake_horizons.calls
    assert kw['location'] == '@0'
    assert kw['refplane'] == 'earth'


def test_horizons_grid_ignores_data_cadence(fake_horizons):
    """The Horizons step is ephem_step, not the spacing of the input times."""
    t = Time(SPARSE_JD, format='jd', scale='tdb')
    parallax.get_observer_barycentric('fake', t)
    parallax.get_observer_barycentric('fake', t, ephem_step=0.25)

    steps = [kw['epochs']['step'] for kw in fake_horizons.calls]
    assert steps == ['1440m', '360m']


@pytest.mark.parametrize('jd', [
    SPARSE_JD,                                       # sparse epochs
    np.arange(2460310.5, 2460676.5, 1.0) + 0.37,     # daily, off-grid
    np.float64(2460400.123),                         # scalar
])
def test_horizons_path_matches_astropy(fake_horizons, jd):
    """Hermite interpolation of the Horizons grid reproduces the orbit."""
    t = Time(jd, format='jd', scale='tdb')
    pos, vel = parallax.get_observer_barycentric('fake', t, velocity=True)
    ref_pos, ref_vel = get_body_barycentric_posvel('earth', t)

    assert pos.shape == t.shape
    # Cubic Hermite on a 1-day grid: Earth to well under 1 km and 1 cm/s.
    assert _max_dist_km(pos, ref_pos) < 1.0
    dv = (vel - ref_vel).xyz.to(u.m / u.s).value
    assert np.max(np.abs(dv)) < 0.01


def test_horizons_parallax_matches_earth(fake_horizons):
    """Spacecraft-path parallax toward the GC matches the astropy path."""
    mjd = SPARSE_JD - 2400000.5
    p_fake = parallax.parallax_in_direction.func(RA_GC, DEC_GC, mjd,
                                                 obsLocation='fake')
    p_earth = parallax.parallax_in_direction.func(RA_GC, DEC_GC, mjd,
                                                  obsLocation='earth')
    np.testing.assert_allclose(p_fake, p_earth, atol=1e-8)
    # Sanity: GC is near the ecliptic, so the North excursion is small.
    assert np.max(np.abs(p_earth[:, 1])) < 0.11


@pytest.mark.parametrize('obs', ['earth', 'fake'])
def test_dparallax_dt_is_derivative_of_parallax(fake_horizons, obs):
    """dparallax_dt_in_direction is d/dt of parallax_in_direction (AU/yr)."""
    mjd = np.array([60320.3, 60400.7, 60512.1, 60633.9])
    h = 0.01  # days
    p_plus = parallax.parallax_in_direction.func(RA_GC, DEC_GC, mjd + h,
                                                 obsLocation=obs)
    p_minus = parallax.parallax_in_direction.func(RA_GC, DEC_GC, mjd - h,
                                                  obsLocation=obs)
    numeric = (p_plus - p_minus) / (2 * h) * 365.25   # AU/day -> AU/yr

    analytic = parallax.dparallax_dt_in_direction(RA_GC, DEC_GC, mjd,
                                                  obsLocation=obs)
    # The old heliocentric form was off by the Sun's barycentric velocity,
    # ~2e-3 AU/yr; this tolerance is well below that.
    np.testing.assert_allclose(analytic, numeric, atol=2e-5)


def test_horizons_coverage_error(monkeypatch):
    """A Horizons table that misses the requested times raises."""

    class _Short(_FakeHorizons):
        def vectors(self, refplane='ecliptic', **kwargs):
            tab = super().vectors(refplane=refplane, **kwargs)
            return tab[3:-3]   # drop the 2-step padding and one more row

    mod = types.ModuleType('astroquery.jplhorizons')
    mod.Horizons = _Short
    monkeypatch.setitem(sys.modules, 'astroquery.jplhorizons', mod)

    t = Time(SPARSE_JD, format='jd', scale='tdb')
    with pytest.raises(ValueError, match='does not cover'):
        parallax.get_observer_barycentric('fake', t)


@pytest.mark.slow
def test_jwst_live_horizons_sparse_epochs():
    """Live Horizons: JWST at sparse epochs matches an exact-epoch query.

    Needs network and astroquery.
    """
    jplhorizons = pytest.importorskip('astroquery.jplhorizons')
    t = Time(SPARSE_JD, format='jd', scale='tdb')

    tab = jplhorizons.Horizons(id='-170', location='@0',
                               epochs=list(t.jd)).vectors(refplane='earth')
    exact = CartesianRepresentation(tab['x'].to(u.km), tab['y'].to(u.km),
                                    tab['z'].to(u.km))
    pos = parallax.get_observer_barycentric('-170', t)

    # Old code (linear interp on a ~110-day step) was off by ~7e7 km.
    assert _max_dist_km(pos, exact) < 1.0
