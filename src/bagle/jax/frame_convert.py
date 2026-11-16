"""JAX-friendly geocentric-projected ↔ heliocentric photometry parameter conversion."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

_AU_DAY_TO_KM_S = 1731.45683


def _geo_phot_callback(x, ra, dec):
    """Run the NumPy geocentric conversion outside a JAX trace.

    Parameters
    ----------
    x : array_like, shape (6,)
        ``[t0, u0, tE, piE_E, piE_N, t0par]`` in the geocentric frame.
    ra, dec : float
        Lens coordinates in degrees.

    Returns
    -------
    helio : ndarray, shape (5,)
        ``[t0, u0, tE, piE_E, piE_N]`` in the heliocentric frame.
    """
    # Concrete values: this callback is invoked by pure_callback, not traced.
    helio = _geo_phot_to_helio_np(
        np.asarray(x, dtype=np.float64), float(ra), float(dec))

    return helio


def _geo_phot_to_helio_np(x: np.ndarray, ra: float, dec: float) -> np.ndarray:
    """Convert one geocentric photometry vector with the NumPy frame code.

    Parameters
    ----------
    x : ndarray, shape (6,)
        ``[t0, u0, tE, piE_E, piE_N, t0par]``.
    ra, dec : float
        Lens coordinates in degrees.

    Returns
    -------
    helio : ndarray, shape (5,)
        ``[t0, u0, tE, piE_E, piE_N]``.
    """
    from bagle import frame_convert as fc

    ra = float(ra)
    dec = float(dec)
    t0_g, u0_g, tE_g, piE_E_g, piE_N_g, t0par = (float(v) for v in x)
    t0, u0, tE, piE_E, piE_N = fc.convert_helio_geo_phot(
        ra,
        dec,
        t0_g,
        u0_g,
        tE_g,
        piE_E_g,
        piE_N_g,
        t0par,
        in_frame="geo",
        murel_in="LS",
        murel_out="SL",
        coord_in="tb",
        coord_out="EN",
        plot=False,
    )
    return np.array([t0, u0, tE, piE_E, piE_N], dtype=np.float64)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def geo_phot_to_helio_jax(x, ra: float, dec: float):
    """Convert geo-projected photometry params to heliocentric (5 outputs).

    The NumPy frame transform is not traceable. ``jax.pure_callback``
    runs it from a jitted likelihood, which is how the fitter calls this.

    Parameters
    ----------
    x : array_like, shape (6,)
        ``[t0, u0, tE, piE_E, piE_N, t0par]`` in the geocentric frame.
    ra, dec : float
        Lens coordinates in degrees. Not differentiated.

    Returns
    -------
    helio : jax.Array, shape (5,)
        ``[t0, u0, tE, piE_E, piE_N]`` in the heliocentric frame.
    """
    # ra and dec stay concrete (nondiff). x may be a tracer under jit.
    return jax.pure_callback(
        _geo_phot_callback,
        jax.ShapeDtypeStruct((5,), jnp.float64),
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(ra, dtype=jnp.float64),
        jnp.asarray(dec, dtype=jnp.float64),
    )


def _geo_phot_fwd(x, ra, dec):
    """Forward rule for ``geo_phot_to_helio_jax``.

    Parameters
    ----------
    x : array_like, shape (6,)
        Geocentric parameter vector. Kept as a residual tracer.
    ra, dec : float
        Lens coordinates in degrees.

    Returns
    -------
    y : jax.Array, shape (5,)
        Heliocentric parameters.
    residuals : tuple
        ``(x, ra, dec)`` for the finite-difference backward pass.
    """
    y = geo_phot_to_helio_jax(x, ra, dec)

    return y, (x, ra, dec)


def _geo_phot_bwd(ra, dec, res, g):
    x, _, _ = res
    ra = float(ra)
    dec = float(dec)
    x0 = np.asarray(x, dtype=np.float64)
    g_np = np.asarray(g, dtype=np.float64)
    jac = np.zeros((5, x0.size), dtype=np.float64)
    eps = 1e-7
    f0 = _geo_phot_to_helio_np(x0, ra, dec)
    for i in range(x0.size):
        xp = x0.copy()
        xp[i] += eps
        jac[:, i] = (_geo_phot_to_helio_np(xp, ra, dec) - f0) / eps
    return (jnp.asarray(jac.T @ g_np, dtype=jnp.float64),)


geo_phot_to_helio_jax.defvjp(_geo_phot_fwd, _geo_phot_bwd)
