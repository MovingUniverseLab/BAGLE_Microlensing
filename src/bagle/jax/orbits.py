"""JAX Keplerian orbit utilities (from bagle.orbits.Orbit)."""
from __future__ import annotations

import math

import jax.numpy as jnp

from bagle.jax.layout_registry import LayoutSpec


def eccen_anomaly(M, e):
    """Solve Kepler equation M = E - e sin(E) (Newton)."""
    E = M
    for _ in range(30):
        E = E - (E - e * jnp.sin(E) - M) / (1.0 - e * jnp.cos(E))
    return E


def oal2xy(t, w, o, i, e, p, tp, aleph, aleph2, accel=False, ax=0.0, ay=0.0):
    """
  Port of :meth:`bagle.orbits.Orbit.oal2xy` for primary/secondary positions (arcsec).
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
    A = aleph * (cos_om * cos_bigOm - sin_om * sin_bigOm * cos_i)
    B = aleph * (cos_om * sin_bigOm + sin_om * cos_bigOm * cos_i)
    F = aleph * (-sin_om * sin_i)
    G = aleph * (cos_om * cos_bigOm - sin_om * sin_bigOm * cos_i)
    H = aleph * (cos_om * sin_bigOm + sin_om * cos_bigOm * cos_i)
    C = aleph * (-sin_om * sin_i)
    x = A * X + B * Y
    y = F * X + G * Y
    x2 = -A * X - B * Y
    y2 = -F * X - G * Y
    if accel:
        dt = (t - tp) / 365.25
        x2 = x2 + ax * dt**2
        y2 = y2 + ay * dt**2
    return x, y, x2, y2


_ORBIT_LAYOUTS = frozenset(
    {
        "linear",
        "accelerated",
        "keplerian",
    }
)


def supports_orbit_layout(layout: LayoutSpec) -> bool:
    return layout.orbit in _ORBIT_LAYOUTS
