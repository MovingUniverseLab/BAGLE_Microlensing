"""JAX-friendly geocentric-projected ↔ heliocentric photometry parameter conversion."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

_AU_DAY_TO_KM_S = 1731.45683


def _geo_phot_to_helio_np(x: np.ndarray, ra: float, dec: float) -> np.ndarray:
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
    """Convert geo-projected photometry params to heliocentric (5 outputs)."""
    return jnp.asarray(_geo_phot_to_helio_np(np.asarray(x, dtype=np.float64), ra, dec))


def _geo_phot_fwd(x, ra, dec):
    y = geo_phot_to_helio_jax(x, ra, dec)
    return y, (np.asarray(x, dtype=np.float64), ra, dec)


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
