"""JAX Keplerian orbit utilities (from bagle.orbits.Orbit)."""
from __future__ import annotations

import math

import jax.numpy as jnp


def eccen_anomaly(M, e):
    """Solve Kepler equation M = E - e sin(E) (Newton)."""
    E = M
    for _ in range(30):
        E = E - (E - e * jnp.sin(E) - M) / (1.0 - e * jnp.cos(E))
    return E


def oal2xy(t, w, o, i, e, p, tp, aleph, aleph2, accel=False, ax=0.0, ay=0.0):
    """
    Port of :meth:`bagle.orbits.Orbit.oal2xy` for primary/secondary positions.

    Positions are in the same units as ``aleph`` / ``aleph2`` (arcsec for
    PhotAstrom, Einstein-radius units for PSBL phot-only orbit models).
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    mean_motion = 2.0 * jnp.pi / p
    M = mean_motion * (t - tp)
    E = eccen_anomaly(M, e)
    ecc_sqrt = jnp.sqrt(1.0 - e**2)
    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)
    X = cos_E - e
    Y = ecc_sqrt * sin_E
    cos_bigOm = jnp.cos(jnp.deg2rad(o))
    sin_bigOm = jnp.sin(jnp.deg2rad(o))
    cos_i = jnp.cos(jnp.deg2rad(i))
    sin_i = jnp.sin(jnp.deg2rad(i))
    cos_om = jnp.cos(jnp.deg2rad(w))
    sin_om = jnp.sin(jnp.deg2rad(w))
    con_a = aleph * (cos_om * cos_bigOm - sin_om * sin_bigOm * cos_i)
    con_b = aleph * (cos_om * sin_bigOm + sin_om * cos_bigOm * cos_i)
    con_f = aleph * (-sin_om * cos_bigOm - cos_om * sin_bigOm * cos_i)
    con_g = aleph * (-sin_om * sin_bigOm + cos_om * cos_bigOm * cos_i)
    cos_om2 = jnp.cos(jnp.deg2rad(w + 180.0))
    sin_om2 = jnp.sin(jnp.deg2rad(w + 180.0))
    con_a2 = aleph2 * (cos_om2 * cos_bigOm - sin_om2 * sin_bigOm * cos_i)
    con_b2 = aleph2 * (cos_om2 * sin_bigOm + sin_om2 * cos_bigOm * cos_i)
    con_f2 = aleph2 * (-sin_om2 * cos_bigOm - cos_om2 * sin_bigOm * cos_i)
    con_g2 = aleph2 * (-sin_om2 * sin_bigOm + cos_om2 * cos_bigOm * cos_i)
    x = con_b * X + con_g * Y
    y = con_a * X + con_f * Y
    x2 = con_b2 * X + con_g2 * Y
    y2 = con_a2 * X + con_f2 * Y
    if accel:
        dt = (t - tp) / 365.25
        x2 = x2 + ax * dt**2
        y2 = y2 + ay * dt**2
    return x, y, x2, y2


_ORBIT_KINDS = frozenset(
    {
        "linear",
        "accelerated",
        "keplerian",
    }
)


def supports_orbit_kind(orbit: str) -> bool:
    return orbit in _ORBIT_KINDS


def supports_orbit_layout(layout_or_orbit) -> bool:
    """Back-compat: accept an orbit string or an object with ``.orbit``."""
    if isinstance(layout_or_orbit, str):
        return supports_orbit_kind(layout_or_orbit)
    orbit = getattr(layout_or_orbit, "orbit", None)
    return orbit in _ORBIT_KINDS if orbit is not None else False
