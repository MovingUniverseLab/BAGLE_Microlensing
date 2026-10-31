"""Traced PSTL (point-source triple-lens) photometry.

Geometry and the degree-10 lens polynomial follow :class:`bagle.model.PSTL`.
Companion-matrix roots use :func:`jax.numpy.linalg.eigvals`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from bagle.jax_physics import (
    _DAYS_PER_YEAR,
    _DEG2RAD,
    einstein_source_position,
    flux2mag_jax,
    gaussian_astrometry_log_likelihood_sum,
    gaussian_log_likelihood_sum,
    mag2flux_jax,
    u0_hat_from_thetaE_hat_jax,
)


def derive_pstl_static_geometry(u0_amp, piE_E, piE_N, q_12, q_13,
                                sep12, sep13, phi12, phi13):
    """
    Derive static PSTL phot-only geometry from fitter parameters.

    Parameters
    ----------
    u0_amp : float
        Signed impact parameter in Einstein radii.
    piE_E, piE_N : float
        Microlensing parallax components in Einstein radii.
    q_12, q_13 : float
        Mass ratios M2/M1 and M3/M1.
    sep12, sep13 : float
        Projected 1-2 and 1-3 separations in Einstein radii.
    phi12, phi13 : float
        1-2 and 1-3 axis angles relative to muRel (degrees).

    Returns
    -------
    m1, m2, m3, u0, thetaE_hat, xL1, xL2, xL3, piE_amp
        Mass fractions, source u0, muRel hat, and lens positions.
    """
    piE = jnp.stack([piE_E, piE_N])
    piE_amp = jnp.linalg.norm(piE)
    thetaE_hat = piE / piE_amp
    u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, u0_amp)
    u0 = jnp.abs(u0_amp) * u0_hat

    phi_piE_rad = jnp.arctan2(piE_E, piE_N)
    phi12_rho1 = phi_piE_rad + phi12 * _DEG2RAD
    phi13_rho1 = phi_piE_rad + phi13 * _DEG2RAD

    xL1 = jnp.stack(
        [0.5 * sep12 * jnp.sin(phi12_rho1),
         0.5 * sep12 * jnp.cos(phi12_rho1)])
    xL2 = -xL1
    xL3 = xL1 + jnp.stack(
        [sep13 * jnp.sin(phi13_rho1),
         sep13 * jnp.cos(phi13_rho1)])

    m1 = 1.0 / (1.0 + q_12 + q_13)
    m2 = q_12 * m1
    m3 = q_13 * m1

    return m1, m2, m3, u0, thetaE_hat, xL1, xL2, xL3, piE_amp


def triple_lens_poly_coeffs(w, wbar, a, r3, r3bar, e1, e2):
    """Degree-10 lens-equation coefficients (caustics frame).

    Parameters
    ----------
    w, wbar, a, r3, r3bar, e1, e2
        Same as :meth:`bagle.model.PSTL._triple_lens_poly_coeffs`.

    Returns
    -------
    p_0 ... p_10
        Polynomial coefficients, highest degree first.
    """
    p_0 = -(a**2) * wbar + a**2 * r3bar + wbar**3 - wbar**2 * r3bar

    p_1 = (
        a**2 * w * wbar
        - a**2 * w * r3bar
        + 3 * a**2 * wbar * r3
        - 3 * a**2 * r3bar * r3
        - a**2 * e1
        - a**2 * e2
        - a * wbar * e1
        + a * wbar * e2
        + a * r3bar * e1
        - a * r3bar * e2
        - w * wbar**3
        + w * wbar**2 * r3bar
        - 3 * wbar**3 * r3
        + 3 * wbar**2 * r3bar * r3
        + 2 * wbar**2
        + wbar * r3bar * e1
        + wbar * r3bar * e2
        - 2 * wbar * r3bar
    )

    p_2 = (
        3 * a**4 * wbar
        - 3 * a**4 * r3bar
        - a**3 * e1
        + a**3 * e2
        - 3 * a**2 * w * wbar * r3
        + 3 * a**2 * w * r3bar * r3
        + a**2 * w
        - 3 * a**2 * wbar**3
        + 3 * a**2 * wbar**2 * r3bar
        - 3 * a**2 * wbar * r3**2
        + 3 * a**2 * r3bar * r3**2
        + 4 * a**2 * e1 * r3
        + 4 * a**2 * e2 * r3
        - a**2 * r3
        + 3 * a * wbar**2 * e1
        - 3 * a * wbar**2 * e2
        - 2 * a * wbar * r3bar * e1
        + 2 * a * wbar * r3bar * e2
        + 3 * a * wbar * e1 * r3
        - 3 * a * wbar * e2 * r3
        - 3 * a * r3bar * e1 * r3
        + 3 * a * r3bar * e2 * r3
        - a * e1
        + a * e2
        + 3 * w * wbar**3 * r3
        - 3 * w * wbar**2 * r3bar * r3
        - 3 * w * wbar**2
        + 2 * w * wbar * r3bar
        + 3 * wbar**3 * r3**2
        - 3 * wbar**2 * r3bar * r3**2
        - 3 * wbar**2 * e1 * r3
        - 3 * wbar**2 * e2 * r3
        - 3 * wbar**2 * r3
        - wbar * r3bar * e1 * r3
        - wbar * r3bar * e2 * r3
        + 4 * wbar * r3bar * r3
        + wbar
        + r3bar * e1
        + r3bar * e2
        - r3bar
    )

    p_3 = (
        -3 * a**4 * w * wbar
        + 3 * a**4 * w * r3bar
        - 9 * a**4 * wbar * r3
        + 9 * a**4 * r3bar * r3
        + 2 * a**4 * e1
        + 2 * a**4 * e2
        + a**3 * w * e1
        - a**3 * w * e2
        + 3 * a**3 * wbar * e1
        - 3 * a**3 * wbar * e2
        - 3 * a**3 * r3bar * e1
        + 3 * a**3 * r3bar * e2
        + 3 * a**3 * e1 * r3
        - 3 * a**3 * e2 * r3
        + 3 * a**2 * w * wbar**3
        - 3 * a**2 * w * wbar**2 * r3bar
        + 3 * a**2 * w * wbar * r3**2
        - 3 * a**2 * w * r3bar * r3**2
        - a**2 * w * e1 * r3
        - a**2 * w * e2 * r3
        - 2 * a**2 * w * r3
        + 9 * a**2 * wbar**3 * r3
        - 9 * a**2 * wbar**2 * r3bar * r3
        + 3 * a**2 * wbar**2 * e1
        + 3 * a**2 * wbar**2 * e2
        - 6 * a**2 * wbar**2
        - 5 * a**2 * wbar * r3bar * e1
        - 5 * a**2 * wbar * r3bar * e2
        + 6 * a**2 * wbar * r3bar
        + a**2 * wbar * r3**3
        - a**2 * r3bar * r3**3
        - a**2 * e1**2
        + 2 * a**2 * e1 * e2
        - 5 * a**2 * e1 * r3**2
        - a**2 * e2**2
        - 5 * a**2 * e2 * r3**2
        + 2 * a**2 * r3**2
        - 3 * a * w * wbar**2 * e1
        + 3 * a * w * wbar**2 * e2
        + 2 * a * w * wbar * r3bar * e1
        - 2 * a * w * wbar * r3bar * e2
        - 9 * a * wbar**2 * e1 * r3
        + 9 * a * wbar**2 * e2 * r3
        + 6 * a * wbar * r3bar * e1 * r3
        - 6 * a * wbar * r3bar * e2 * r3
        - 3 * a * wbar * e1 * r3**2
        + 4 * a * wbar * e1
        + 3 * a * wbar * e2 * r3**2
        - 4 * a * wbar * e2
        + a * r3bar * e1**2
        + 3 * a * r3bar * e1 * r3**2
        - 2 * a * r3bar * e1
        - a * r3bar * e2**2
        - 3 * a * r3bar * e2 * r3**2
        + 2 * a * r3bar * e2
        + a * e1**2 * r3
        + 2 * a * e1 * r3
        - a * e2**2 * r3
        - 2 * a * e2 * r3
        - 3 * w * wbar**3 * r3**2
        + 3 * w * wbar**2 * r3bar * r3**2
        + 3 * w * wbar**2 * e1 * r3
        + 3 * w * wbar**2 * e2 * r3
        + 6 * w * wbar**2 * r3
        - 2 * w * wbar * r3bar * e1 * r3
        - 2 * w * wbar * r3bar * e2 * r3
        - 4 * w * wbar * r3bar * r3
        - 3 * w * wbar
        + w * r3bar
        - wbar**3 * r3**3
        + wbar**2 * r3bar * r3**3
        + 6 * wbar**2 * e1 * r3**2
        + 6 * wbar**2 * e2 * r3**2
        - wbar * r3bar * e1 * r3**2
        - wbar * r3bar * e2 * r3**2
        - 2 * wbar * r3bar * r3**2
        - 4 * wbar * e1 * r3
        - 4 * wbar * e2 * r3
        + wbar * r3
        - r3bar * e1**2 * r3
        - 2 * r3bar * e1 * e2 * r3
        - r3bar * e2**2 * r3
        + r3bar * r3
    )

    p_4 = (
        -3 * a**6 * wbar
        + 3 * a**6 * r3bar
        + 2 * a**5 * e1
        - 2 * a**5 * e2
        + 9 * a**4 * w * wbar * r3
        - 9 * a**4 * w * r3bar * r3
        + a**4 * w * e1
        + a**4 * w * e2
        - 3 * a**4 * w
        + 3 * a**4 * wbar**3
        - 3 * a**4 * wbar**2 * r3bar
        + 9 * a**4 * wbar * r3**2
        - 9 * a**4 * r3bar * r3**2
        - 9 * a**4 * e1 * r3
        - 9 * a**4 * e2 * r3
        + 3 * a**4 * r3
        - 3 * a**3 * w * e1 * r3
        + 3 * a**3 * w * e2 * r3
        - 6 * a**3 * wbar**2 * e1
        + 6 * a**3 * wbar**2 * e2
        + 4 * a**3 * wbar * r3bar * e1
        - 4 * a**3 * wbar * r3bar * e2
        - 9 * a**3 * wbar * e1 * r3
        + 9 * a**3 * wbar * e2 * r3
        + 9 * a**3 * r3bar * e1 * r3
        - 9 * a**3 * r3bar * e2 * r3
        - a**3 * e1**2
        - 3 * a**3 * e1 * r3**2
        + 3 * a**3 * e1
        + a**3 * e2**2
        + 3 * a**3 * e2 * r3**2
        - 3 * a**3 * e2
        - 9 * a**2 * w * wbar**3 * r3
        + 9 * a**2 * w * wbar**2 * r3bar * r3
        - 3 * a**2 * w * wbar**2 * e1
        - 3 * a**2 * w * wbar**2 * e2
        + 9 * a**2 * w * wbar**2
        + 2 * a**2 * w * wbar * r3bar * e1
        + 2 * a**2 * w * wbar * r3bar * e2
        - 6 * a**2 * w * wbar * r3bar
        - a**2 * w * wbar * r3**3
        + a**2 * w * r3bar * r3**3
        + 2 * a**2 * w * e1 * r3**2
        + 2 * a**2 * w * e2 * r3**2
        + a**2 * w * r3**2
        - 9 * a**2 * wbar**3 * r3**2
        + 9 * a**2 * wbar**2 * r3bar * r3**2
        + 9 * a**2 * wbar**2 * r3
        + 9 * a**2 * wbar * r3bar * e1 * r3
        + 9 * a**2 * wbar * r3bar * e2 * r3
        - 12 * a**2 * wbar * r3bar * r3
        + 3 * a**2 * wbar * e1**2
        - 6 * a**2 * wbar * e1 * e2
        + 4 * a**2 * wbar * e1
        + 3 * a**2 * wbar * e2**2
        + 4 * a**2 * wbar * e2
        - 3 * a**2 * wbar
        + 4 * a**2 * r3bar * e1 * e2
        - 5 * a**2 * r3bar * e1
        - 5 * a**2 * r3bar * e2
        + 3 * a**2 * r3bar
        + 3 * a**2 * e1**2 * r3
        - 6 * a**2 * e1 * e2 * r3
        + 2 * a**2 * e1 * r3**3
        + 3 * a**2 * e2**2 * r3
        + 2 * a**2 * e2 * r3**3
        - a**2 * r3**3
        + 9 * a * w * wbar**2 * e1 * r3
        - 9 * a * w * wbar**2 * e2 * r3
        - 6 * a * w * wbar * r3bar * e1 * r3
        + 6 * a * w * wbar * r3bar * e2 * r3
        - 6 * a * w * wbar * e1
        + 6 * a * w * wbar * e2
        + 2 * a * w * r3bar * e1
        - 2 * a * w * r3bar * e2
        + 9 * a * wbar**2 * e1 * r3**2
        - 9 * a * wbar**2 * e2 * r3**2
        - 6 * a * wbar * r3bar * e1 * r3**2
        + 6 * a * wbar * r3bar * e2 * r3**2
        - 6 * a * wbar * e1**2 * r3
        + a * wbar * e1 * r3**3
        - 6 * a * wbar * e1 * r3
        + 6 * a * wbar * e2**2 * r3
        - a * wbar * e2 * r3**3
        + 6 * a * wbar * e2 * r3
        - a * r3bar * e1**2 * r3
        - a * r3bar * e1 * r3**3
        + 4 * a * r3bar * e1 * r3
        + a * r3bar * e2**2 * r3
        + a * r3bar * e2 * r3**3
        - 4 * a * r3bar * e2 * r3
        - 2 * a * e1**2 * r3**2
        - a * e1 * r3**2
        + a * e1
        + 2 * a * e2**2 * r3**2
        + a * e2 * r3**2
        - a * e2
        + w * wbar**3 * r3**3
        - w * wbar**2 * r3bar * r3**3
        - 6 * w * wbar**2 * e1 * r3**2
        - 6 * w * wbar**2 * e2 * r3**2
        - 3 * w * wbar**2 * r3**2
        + 4 * w * wbar * r3bar * e1 * r3**2
        + 4 * w * wbar * r3bar * e2 * r3**2
        + 2 * w * wbar * r3bar * r3**2
        + 6 * w * wbar * e1 * r3
        + 6 * w * wbar * e2 * r3
        + 3 * w * wbar * r3
        - 2 * w * r3bar * e1 * r3
        - 2 * w * r3bar * e2 * r3
        - w * r3bar * r3
        - w
        - 3 * wbar**2 * e1 * r3**3
        - 3 * wbar**2 * e2 * r3**3
        + wbar**2 * r3**3
        + wbar * r3bar * e1 * r3**3
        + wbar * r3bar * e2 * r3**3
        + 3 * wbar * e1**2 * r3**2
        + 6 * wbar * e1 * e2 * r3**2
        + 2 * wbar * e1 * r3**2
        + 3 * wbar * e2**2 * r3**2
        + 2 * wbar * e2 * r3**2
        - 2 * wbar * r3**2
        + r3bar * e1**2 * r3**2
        + 2 * r3bar * e1 * e2 * r3**2
        - r3bar * e1 * r3**2
        + r3bar * e2**2 * r3**2
        - r3bar * e2 * r3**2
        - e1 * r3
        - e2 * r3
        + r3
    )

    p_5 = (
        3 * a**6 * w * wbar
        - 3 * a**6 * w * r3bar
        + 9 * a**6 * wbar * r3
        - 9 * a**6 * r3bar * r3
        - a**6 * e1
        - a**6 * e2
        - 2 * a**5 * w * e1
        + 2 * a**5 * w * e2
        - 3 * a**5 * wbar * e1
        + 3 * a**5 * wbar * e2
        + 3 * a**5 * r3bar * e1
        - 3 * a**5 * r3bar * e2
        - 6 * a**5 * e1 * r3
        + 6 * a**5 * e2 * r3
        - 3 * a**4 * w * wbar**3
        + 3 * a**4 * w * wbar**2 * r3bar
        - 9 * a**4 * w * wbar * r3**2
        + 9 * a**4 * w * r3bar * r3**2
        + 6 * a**4 * w * r3
        - 9 * a**4 * wbar**3 * r3
        + 9 * a**4 * wbar**2 * r3bar * r3
        - 6 * a**4 * wbar**2 * e1
        - 6 * a**4 * wbar**2 * e2
        + 6 * a**4 * wbar**2
        + 7 * a**4 * wbar * r3bar * e1
        + 7 * a**4 * wbar * r3bar * e2
        - 6 * a**4 * wbar * r3bar
        - 3 * a**4 * wbar * r3**3
        + 3 * a**4 * r3bar * r3**3
        + 2 * a**4 * e1**2
        - 4 * a**4 * e1 * e2
        + 12 * a**4 * e1 * r3**2
        + 2 * a**4 * e2**2
        + 12 * a**4 * e2 * r3**2
        - 6 * a**4 * r3**2
        + 6 * a**3 * w * wbar**2 * e1
        - 6 * a**3 * w * wbar**2 * e2
        - 4 * a**3 * w * wbar * r3bar * e1
        + 4 * a**3 * w * wbar * r3bar * e2
        + 3 * a**3 * w * e1 * r3**2
        - 3 * a**3 * w * e2 * r3**2
        + 18 * a**3 * wbar**2 * e1 * r3
        - 18 * a**3 * wbar**2 * e2 * r3
        - 12 * a**3 * wbar * r3bar * e1 * r3
        + 12 * a**3 * wbar * r3bar * e2 * r3
        + 6 * a**3 * wbar * e1**2
        + 9 * a**3 * wbar * e1 * r3**2
        - 8 * a**3 * wbar * e1
        - 6 * a**3 * wbar * e2**2
        - 9 * a**3 * wbar * e2 * r3**2
        + 8 * a**3 * wbar * e2
        - 4 * a**3 * r3bar * e1**2
        - 9 * a**3 * r3bar * e1 * r3**2
        + 4 * a**3 * r3bar * e1
        + 4 * a**3 * r3bar * e2**2
        + 9 * a**3 * r3bar * e2 * r3**2
        - 4 * a**3 * r3bar * e2
        + a**3 * e1 * r3**3
        - 6 * a**3 * e1 * r3
        - a**3 * e2 * r3**3
        + 6 * a**3 * e2 * r3
        + 9 * a**2 * w * wbar**3 * r3**2
        - 9 * a**2 * w * wbar**2 * r3bar * r3**2
        - 18 * a**2 * w * wbar**2 * r3
        + 12 * a**2 * w * wbar * r3bar * r3
        - 3 * a**2 * w * wbar * e1**2
        + 6 * a**2 * w * wbar * e1 * e2
        - 6 * a**2 * w * wbar * e1
        - 3 * a**2 * w * wbar * e2**2
        - 6 * a**2 * w * wbar * e2
        + 9 * a**2 * w * wbar
        + a**2 * w * r3bar * e1**2
        - 2 * a**2 * w * r3bar * e1 * e2
        + 2 * a**2 * w * r3bar * e1
        + a**2 * w * r3bar * e2**2
        + 2 * a**2 * w * r3bar * e2
        - 3 * a**2 * w * r3bar
        - a**2 * w * e1 * r3**3
        - a**2 * w * e2 * r3**3
        + 3 * a**2 * wbar**3 * r3**3
        - 3 * a**2 * wbar**2 * r3bar * r3**3
        - 9 * a**2 * wbar**2 * e1 * r3**2
        - 9 * a**2 * wbar**2 * e2 * r3**2
        - 3 * a**2 * wbar * r3bar * e1 * r3**2
        - 3 * a**2 * wbar * r3bar * e2 * r3**2
        + 6 * a**2 * wbar * r3bar * r3**2
        - 15 * a**2 * wbar * e1**2 * r3
        + 6 * a**2 * wbar * e1 * e2 * r3
        + 6 * a**2 * wbar * e1 * r3
        - 15 * a**2 * wbar * e2**2 * r3
        + 6 * a**2 * wbar * e2 * r3
        - 3 * a**2 * wbar * r3
        + 5 * a**2 * r3bar * e1**2 * r3
        - 2 * a**2 * r3bar * e1 * e2 * r3
        + 4 * a**2 * r3bar * e1 * r3
        + 5 * a**2 * r3bar * e2**2 * r3
        + 4 * a**2 * r3bar * e2 * r3
        - 3 * a**2 * r3bar * r3
        - 3 * a**2 * e1**2 * r3**2
        + 2 * a**2 * e1**2
        + 6 * a**2 * e1 * e2 * r3**2
        - 4 * a**2 * e1 * e2
        + a**2 * e1
        - 3 * a**2 * e2**2 * r3**2
        + 2 * a**2 * e2**2
        + a**2 * e2
        - 9 * a * w * wbar**2 * e1 * r3**2
        + 9 * a * w * wbar**2 * e2 * r3**2
        + 6 * a * w * wbar * r3bar * e1 * r3**2
        - 6 * a * w * wbar * r3bar * e2 * r3**2
        + 6 * a * w * wbar * e1**2 * r3
        + 12 * a * w * wbar * e1 * r3
        - 6 * a * w * wbar * e2**2 * r3
        - 12 * a * w * wbar * e2 * r3
        - 2 * a * w * r3bar * e1**2 * r3
        - 4 * a * w * r3bar * e1 * r3
        + 2 * a * w * r3bar * e2**2 * r3
        + 4 * a * w * r3bar * e2 * r3
        - 3 * a * w * e1
        + 3 * a * w * e2
        - 3 * a * wbar**2 * e1 * r3**3
        + 3 * a * wbar**2 * e2 * r3**3
        + 2 * a * wbar * r3bar * e1 * r3**3
        - 2 * a * wbar * r3bar * e2 * r3**3
        + 12 * a * wbar * e1**2 * r3**2
        - 12 * a * wbar * e2**2 * r3**2
        - a * r3bar * e1**2 * r3**2
        - 2 * a * r3bar * e1 * r3**2
        + a * r3bar * e2**2 * r3**2
        + 2 * a * r3bar * e2 * r3**2
        + a * e1**2 * r3**3
        - 4 * a * e1**2 * r3
        + a * e1 * r3
        - a * e2**2 * r3**3
        + 4 * a * e2**2 * r3
        - a * e2 * r3
        + 3 * w * wbar**2 * e1 * r3**3
        + 3 * w * wbar**2 * e2 * r3**3
        - 2 * w * wbar * r3bar * e1 * r3**3
        - 2 * w * wbar * r3bar * e2 * r3**3
        - 3 * w * wbar * e1**2 * r3**2
        - 6 * w * wbar * e1 * e2 * r3**2
        - 6 * w * wbar * e1 * r3**2
        - 3 * w * wbar * e2**2 * r3**2
        - 6 * w * wbar * e2 * r3**2
        + w * r3bar * e1**2 * r3**2
        + 2 * w * r3bar * e1 * e2 * r3**2
        + 2 * w * r3bar * e1 * r3**2
        + w * r3bar * e2**2 * r3**2
        + 2 * w * r3bar * e2 * r3**2
        + 3 * w * e1 * r3
        + 3 * w * e2 * r3
        - 3 * wbar * e1**2 * r3**3
        - 6 * wbar * e1 * e2 * r3**3
        + 2 * wbar * e1 * r3**3
        - 3 * wbar * e2**2 * r3**3
        + 2 * wbar * e2 * r3**3
        + 2 * e1**2 * r3**2
        + 4 * e1 * e2 * r3**2
        - 2 * e1 * r3**2
        + 2 * e2**2 * r3**2
        - 2 * e2 * r3**2
    )

    p_6 = (
        a**8 * wbar
        - a**8 * r3bar
        - a**7 * e1
        + a**7 * e2
        - 9 * a**6 * w * wbar * r3
        + 9 * a**6 * w * r3bar * r3
        - 2 * a**6 * w * e1
        - 2 * a**6 * w * e2
        + 3 * a**6 * w
        - a**6 * wbar**3
        + a**6 * wbar**2 * r3bar
        - 9 * a**6 * wbar * r3**2
        + 9 * a**6 * r3bar * r3**2
        + 6 * a**6 * e1 * r3
        + 6 * a**6 * e2 * r3
        - 3 * a**6 * r3
        + 6 * a**5 * w * e1 * r3
        - 6 * a**5 * w * e2 * r3
        + 3 * a**5 * wbar**2 * e1
        - 3 * a**5 * wbar**2 * e2
        - 2 * a**5 * wbar * r3bar * e1
        + 2 * a**5 * wbar * r3bar * e2
        + 9 * a**5 * wbar * e1 * r3
        - 9 * a**5 * wbar * e2 * r3
        - 9 * a**5 * r3bar * e1 * r3
        + 9 * a**5 * r3bar * e2 * r3
        + 2 * a**5 * e1**2
        + 6 * a**5 * e1 * r3**2
        - 3 * a**5 * e1
        - 2 * a**5 * e2**2
        - 6 * a**5 * e2 * r3**2
        + 3 * a**5 * e2
        + 9 * a**4 * w * wbar**3 * r3
        - 9 * a**4 * w * wbar**2 * r3bar * r3
        + 6 * a**4 * w * wbar**2 * e1
        + 6 * a**4 * w * wbar**2 * e2
        - 9 * a**4 * w * wbar**2
        - 4 * a**4 * w * wbar * r3bar * e1
        - 4 * a**4 * w * wbar * r3bar * e2
        + 6 * a**4 * w * wbar * r3bar
        + 3 * a**4 * w * wbar * r3**3
        - 3 * a**4 * w * r3bar * r3**3
        - 3 * a**4 * w * e1 * r3**2
        - 3 * a**4 * w * e2 * r3**2
        - 3 * a**4 * w * r3**2
        + 9 * a**4 * wbar**3 * r3**2
        - 9 * a**4 * wbar**2 * r3bar * r3**2
        + 9 * a**4 * wbar**2 * e1 * r3
        + 9 * a**4 * wbar**2 * e2 * r3
        - 9 * a**4 * wbar**2 * r3
        - 15 * a**4 * wbar * r3bar * e1 * r3
        - 15 * a**4 * wbar * r3bar * e2 * r3
        + 12 * a**4 * wbar * r3bar * r3
        + 12 * a**4 * wbar * e1 * e2
        - 8 * a**4 * wbar * e1
        - 8 * a**4 * wbar * e2
        + 3 * a**4 * wbar
        - 2 * a**4 * r3bar * e1**2
        - 8 * a**4 * r3bar * e1 * e2
        + 7 * a**4 * r3bar * e1
        - 2 * a**4 * r3bar * e2**2
        + 7 * a**4 * r3bar * e2
        - 3 * a**4 * r3bar
        - 6 * a**4 * e1**2 * r3
        + 12 * a**4 * e1 * e2 * r3
        - 5 * a**4 * e1 * r3**3
        - 6 * a**4 * e2**2 * r3
        - 5 * a**4 * e2 * r3**3
        + 3 * a**4 * r3**3
        - 18 * a**3 * w * wbar**2 * e1 * r3
        + 18 * a**3 * w * wbar**2 * e2 * r3
        + 12 * a**3 * w * wbar * r3bar * e1 * r3
        - 12 * a**3 * w * wbar * r3bar * e2 * r3
        - 6 * a**3 * w * wbar * e1**2
        + 12 * a**3 * w * wbar * e1
        + 6 * a**3 * w * wbar * e2**2
        - 12 * a**3 * w * wbar * e2
        + 2 * a**3 * w * r3bar * e1**2
        - 4 * a**3 * w * r3bar * e1
        - 2 * a**3 * w * r3bar * e2**2
        + 4 * a**3 * w * r3bar * e2
        - a**3 * w * e1 * r3**3
        + a**3 * w * e2 * r3**3
        - 18 * a**3 * wbar**2 * e1 * r3**2
        + 18 * a**3 * wbar**2 * e2 * r3**2
        + 12 * a**3 * wbar * r3bar * e1 * r3**2
        - 12 * a**3 * wbar * r3bar * e2 * r3**2
        - 6 * a**3 * wbar * e1**2 * r3
        - 3 * a**3 * wbar * e1 * r3**3
        + 12 * a**3 * wbar * e1 * r3
        + 6 * a**3 * wbar * e2**2 * r3
        + 3 * a**3 * wbar * e2 * r3**3
        - 12 * a**3 * wbar * e2 * r3
        + 8 * a**3 * r3bar * e1**2 * r3
        + 3 * a**3 * r3bar * e1 * r3**3
        - 8 * a**3 * r3bar * e1 * r3
        - 8 * a**3 * r3bar * e2**2 * r3
        - 3 * a**3 * r3bar * e2 * r3**3
        + 8 * a**3 * r3bar * e2 * r3
        + a**3 * e1**3
        - 3 * a**3 * e1**2 * e2
        + 3 * a**3 * e1**2 * r3**2
        + 4 * a**3 * e1**2
        + 3 * a**3 * e1 * e2**2
        + 3 * a**3 * e1 * r3**2
        - 2 * a**3 * e1
        - a**3 * e2**3
        - 3 * a**3 * e2**2 * r3**2
        - 4 * a**3 * e2**2
        - 3 * a**3 * e2 * r3**2
        + 2 * a**3 * e2
        - 3 * a**2 * w * wbar**3 * r3**3
        + 3 * a**2 * w * wbar**2 * r3bar * r3**3
        + 9 * a**2 * w * wbar**2 * e1 * r3**2
        + 9 * a**2 * w * wbar**2 * e2 * r3**2
        + 9 * a**2 * w * wbar**2 * r3**2
        - 6 * a**2 * w * wbar * r3bar * e1 * r3**2
        - 6 * a**2 * w * wbar * r3bar * e2 * r3**2
        - 6 * a**2 * w * wbar * r3bar * r3**2
        + 15 * a**2 * w * wbar * e1**2 * r3
        - 6 * a**2 * w * wbar * e1 * e2 * r3
        - 6 * a**2 * w * wbar * e1 * r3
        + 15 * a**2 * w * wbar * e2**2 * r3
        - 6 * a**2 * w * wbar * e2 * r3
        - 9 * a**2 * w * wbar * r3
        - 5 * a**2 * w * r3bar * e1**2 * r3
        + 2 * a**2 * w * r3bar * e1 * e2 * r3
        + 2 * a**2 * w * r3bar * e1 * r3
        - 5 * a**2 * w * r3bar * e2**2 * r3
        + 2 * a**2 * w * r3bar * e2 * r3
        + 3 * a**2 * w * r3bar * r3
        - 3 * a**2 * w * e1**2
        + 6 * a**2 * w * e1 * e2
        - 3 * a**2 * w * e1
        - 3 * a**2 * w * e2**2
        - 3 * a**2 * w * e2
        + 3 * a**2 * w
        + 6 * a**2 * wbar**2 * e1 * r3**3
        + 6 * a**2 * wbar**2 * e2 * r3**3
        - 3 * a**2 * wbar**2 * r3**3
        - a**2 * wbar * r3bar * e1 * r3**3
        - a**2 * wbar * r3bar * e2 * r3**3
        + 12 * a**2 * wbar * e1**2 * r3**2
        - 12 * a**2 * wbar * e1 * e2 * r3**2
        - 6 * a**2 * wbar * e1 * r3**2
        + 12 * a**2 * wbar * e2**2 * r3**2
        - 6 * a**2 * wbar * e2 * r3**2
        + 6 * a**2 * wbar * r3**2
        - 7 * a**2 * r3bar * e1**2 * r3**2
        - 2 * a**2 * r3bar * e1 * e2 * r3**2
        + a**2 * r3bar * e1 * r3**2
        - 7 * a**2 * r3bar * e2**2 * r3**2
        + a**2 * r3bar * e2 * r3**2
        - 3 * a**2 * e1**3 * r3
        + 3 * a**2 * e1**2 * e2 * r3
        + a**2 * e1**2 * r3**3
        - 7 * a**2 * e1**2 * r3
        + 3 * a**2 * e1 * e2**2 * r3
        - 2 * a**2 * e1 * e2 * r3**3
        - 2 * a**2 * e1 * e2 * r3
        + 4 * a**2 * e1 * r3
        - 3 * a**2 * e2**3 * r3
        + a**2 * e2**2 * r3**3
        - 7 * a**2 * e2**2 * r3
        + 4 * a**2 * e2 * r3
        - 3 * a**2 * r3
        + 3 * a * w * wbar**2 * e1 * r3**3
        - 3 * a * w * wbar**2 * e2 * r3**3
        - 2 * a * w * wbar * r3bar * e1 * r3**3
        + 2 * a * w * wbar * r3bar * e2 * r3**3
        - 12 * a * w * wbar * e1**2 * r3**2
        - 6 * a * w * wbar * e1 * r3**2
        + 12 * a * w * wbar * e2**2 * r3**2
        + 6 * a * w * wbar * e2 * r3**2
        + 4 * a * w * r3bar * e1**2 * r3**2
        + 2 * a * w * r3bar * e1 * r3**2
        - 4 * a * w * r3bar * e2**2 * r3**2
        - 2 * a * w * r3bar * e2 * r3**2
        + 6 * a * w * e1**2 * r3
        + 3 * a * w * e1 * r3
        - 6 * a * w * e2**2 * r3
        - 3 * a * w * e2 * r3
        - 6 * a * wbar * e1**2 * r3**3
        + 2 * a * wbar * e1 * r3**3
        + 6 * a * wbar * e2**2 * r3**3
        - 2 * a * wbar * e2 * r3**3
        + a * r3bar * e1**2 * r3**3
        - a * r3bar * e2**2 * r3**3
        + 3 * a * e1**3 * r3**2
        + 3 * a * e1**2 * e2 * r3**2
        + 2 * a * e1**2 * r3**2
        - 3 * a * e1 * e2**2 * r3**2
        - 2 * a * e1 * r3**2
        - 3 * a * e2**3 * r3**2
        - 2 * a * e2**2 * r3**2
        + 2 * a * e2 * r3**2
        + 3 * w * wbar * e1**2 * r3**3
        + 6 * w * wbar * e1 * e2 * r3**3
        + 3 * w * wbar * e2**2 * r3**3
        - w * r3bar * e1**2 * r3**3
        - 2 * w * r3bar * e1 * e2 * r3**3
        - w * r3bar * e2**2 * r3**3
        - 3 * w * e1**2 * r3**2
        - 6 * w * e1 * e2 * r3**2
        - 3 * w * e2**2 * r3**2
        - e1**3 * r3**3
        - 3 * e1**2 * e2 * r3**3
        + e1**2 * r3**3
        - 3 * e1 * e2**2 * r3**3
        + 2 * e1 * e2 * r3**3
        - e2**3 * r3**3
        + e2**2 * r3**3
    )

    p_7 = (
        -(a**8) * w * wbar
        + a**8 * w * r3bar
        - 3 * a**8 * wbar * r3
        + 3 * a**8 * r3bar * r3
        + a**7 * w * e1
        - a**7 * w * e2
        + a**7 * wbar * e1
        - a**7 * wbar * e2
        - a**7 * r3bar * e1
        + a**7 * r3bar * e2
        + 3 * a**7 * e1 * r3
        - 3 * a**7 * e2 * r3
        + a**6 * w * wbar**3
        - a**6 * w * wbar**2 * r3bar
        + 9 * a**6 * w * wbar * r3**2
        - 9 * a**6 * w * r3bar * r3**2
        + 3 * a**6 * w * e1 * r3
        + 3 * a**6 * w * e2 * r3
        - 6 * a**6 * w * r3
        + 3 * a**6 * wbar**3 * r3
        - 3 * a**6 * wbar**2 * r3bar * r3
        + 3 * a**6 * wbar**2 * e1
        + 3 * a**6 * wbar**2 * e2
        - 2 * a**6 * wbar**2
        - 3 * a**6 * wbar * r3bar * e1
        - 3 * a**6 * wbar * r3bar * e2
        + 2 * a**6 * wbar * r3bar
        + 3 * a**6 * wbar * r3**3
        - 3 * a**6 * r3bar * r3**3
        - a**6 * e1**2
        + 2 * a**6 * e1 * e2
        - 9 * a**6 * e1 * r3**2
        - a**6 * e2**2
        - 9 * a**6 * e2 * r3**2
        + 6 * a**6 * r3**2
        - 3 * a**5 * w * wbar**2 * e1
        + 3 * a**5 * w * wbar**2 * e2
        + 2 * a**5 * w * wbar * r3bar * e1
        - 2 * a**5 * w * wbar * r3bar * e2
        - 6 * a**5 * w * e1 * r3**2
        + 6 * a**5 * w * e2 * r3**2
        - 9 * a**5 * wbar**2 * e1 * r3
        + 9 * a**5 * wbar**2 * e2 * r3
        + 6 * a**5 * wbar * r3bar * e1 * r3
        - 6 * a**5 * wbar * r3bar * e2 * r3
        - 6 * a**5 * wbar * e1**2
        - 9 * a**5 * wbar * e1 * r3**2
        + 4 * a**5 * wbar * e1
        + 6 * a**5 * wbar * e2**2
        + 9 * a**5 * wbar * e2 * r3**2
        - 4 * a**5 * wbar * e2
        + 3 * a**5 * r3bar * e1**2
        + 9 * a**5 * r3bar * e1 * r3**2
        - 2 * a**5 * r3bar * e1
        - 3 * a**5 * r3bar * e2**2
        - 9 * a**5 * r3bar * e2 * r3**2
        + 2 * a**5 * r3bar * e2
        - 3 * a**5 * e1**2 * r3
        - 2 * a**5 * e1 * r3**3
        + 6 * a**5 * e1 * r3
        + 3 * a**5 * e2**2 * r3
        + 2 * a**5 * e2 * r3**3
        - 6 * a**5 * e2 * r3
        - 9 * a**4 * w * wbar**3 * r3**2
        + 9 * a**4 * w * wbar**2 * r3bar * r3**2
        - 9 * a**4 * w * wbar**2 * e1 * r3
        - 9 * a**4 * w * wbar**2 * e2 * r3
        + 18 * a**4 * w * wbar**2 * r3
        + 6 * a**4 * w * wbar * r3bar * e1 * r3
        + 6 * a**4 * w * wbar * r3bar * e2 * r3
        - 12 * a**4 * w * wbar * r3bar * r3
        - 12 * a**4 * w * wbar * e1 * e2
        + 12 * a**4 * w * wbar * e1
        + 12 * a**4 * w * wbar * e2
        - 9 * a**4 * w * wbar
        + 4 * a**4 * w * r3bar * e1 * e2
        - 4 * a**4 * w * r3bar * e1
        - 4 * a**4 * w * r3bar * e2
        + 3 * a**4 * w * r3bar
        + 2 * a**4 * w * e1 * r3**3
        + 2 * a**4 * w * e2 * r3**3
        - 3 * a**4 * wbar**3 * r3**3
        + 3 * a**4 * wbar**2 * r3bar * r3**3
        + 9 * a**4 * wbar * r3bar * e1 * r3**2
        + 9 * a**4 * wbar * r3bar * e2 * r3**2
        - 6 * a**4 * wbar * r3bar * r3**2
        + 12 * a**4 * wbar * e1**2 * r3
        - 12 * a**4 * wbar * e1 * e2 * r3
        + 12 * a**4 * wbar * e2**2 * r3
        + 3 * a**4 * wbar * r3
        - a**4 * r3bar * e1**2 * r3
        + 10 * a**4 * r3bar * e1 * e2 * r3
        - 8 * a**4 * r3bar * e1 * r3
        - a**4 * r3bar * e2**2 * r3
        - 8 * a**4 * r3bar * e2 * r3
        + 3 * a**4 * r3bar * r3
        + 3 * a**4 * e1**3
        - 3 * a**4 * e1**2 * e2
        + 6 * a**4 * e1**2 * r3**2
        - 3 * a**4 * e1 * e2**2
        - 12 * a**4 * e1 * e2 * r3**2
        + 8 * a**4 * e1 * e2
        - 2 * a**4 * e1
        + 3 * a**4 * e2**3
        + 6 * a**4 * e2**2 * r3**2
        - 2 * a**4 * e2
        + 18 * a**3 * w * wbar**2 * e1 * r3**2
        - 18 * a**3 * w * wbar**2 * e2 * r3**2
        - 12 * a**3 * w * wbar * r3bar * e1 * r3**2
        + 12 * a**3 * w * wbar * r3bar * e2 * r3**2
        + 6 * a**3 * w * wbar * e1**2 * r3
        - 24 * a**3 * w * wbar * e1 * r3
        - 6 * a**3 * w * wbar * e2**2 * r3
        + 24 * a**3 * w * wbar * e2 * r3
        - 2 * a**3 * w * r3bar * e1**2 * r3
        + 8 * a**3 * w * r3bar * e1 * r3
        + 2 * a**3 * w * r3bar * e2**2 * r3
        - 8 * a**3 * w * r3bar * e2 * r3
        - a**3 * w * e1**3
        + 3 * a**3 * w * e1**2 * e2
        - 6 * a**3 * w * e1**2
        - 3 * a**3 * w * e1 * e2**2
        + 6 * a**3 * w * e1
        + a**3 * w * e2**3
        + 6 * a**3 * w * e2**2
        - 6 * a**3 * w * e2
        + 6 * a**3 * wbar**2 * e1 * r3**3
        - 6 * a**3 * wbar**2 * e2 * r3**3
        - 4 * a**3 * wbar * r3bar * e1 * r3**3
        + 4 * a**3 * wbar * r3bar * e2 * r3**3
        - 6 * a**3 * wbar * e1**2 * r3**2
        + 6 * a**3 * wbar * e2**2 * r3**2
        - 4 * a**3 * r3bar * e1**2 * r3**2
        + 4 * a**3 * r3bar * e1 * r3**2
        + 4 * a**3 * r3bar * e2**2 * r3**2
        - 4 * a**3 * r3bar * e2 * r3**2
        - 9 * a**3 * e1**3 * r3
        + 3 * a**3 * e1**2 * e2 * r3
        - 2 * a**3 * e1**2 * r3**3
        + 2 * a**3 * e1**2 * r3
        - 3 * a**3 * e1 * e2**2 * r3
        - 2 * a**3 * e1 * r3
        + 9 * a**3 * e2**3 * r3
        + 2 * a**3 * e2**2 * r3**3
        - 2 * a**3 * e2**2 * r3
        + 2 * a**3 * e2 * r3
        - 6 * a**2 * w * wbar**2 * e1 * r3**3
        - 6 * a**2 * w * wbar**2 * e2 * r3**3
        + 4 * a**2 * w * wbar * r3bar * e1 * r3**3
        + 4 * a**2 * w * wbar * r3bar * e2 * r3**3
        - 12 * a**2 * w * wbar * e1**2 * r3**2
        + 12 * a**2 * w * wbar * e1 * e2 * r3**2
        + 12 * a**2 * w * wbar * e1 * r3**2
        - 12 * a**2 * w * wbar * e2**2 * r3**2
        + 12 * a**2 * w * wbar * e2 * r3**2
        + 4 * a**2 * w * r3bar * e1**2 * r3**2
        - 4 * a**2 * w * r3bar * e1 * e2 * r3**2
        - 4 * a**2 * w * r3bar * e1 * r3**2
        + 4 * a**2 * w * r3bar * e2**2 * r3**2
        - 4 * a**2 * w * r3bar * e2 * r3**2
        + 3 * a**2 * w * e1**3 * r3
        - 3 * a**2 * w * e1**2 * e2 * r3
        + 12 * a**2 * w * e1**2 * r3
        - 3 * a**2 * w * e1 * e2**2 * r3
        - 6 * a**2 * w * e1 * r3
        + 3 * a**2 * w * e2**3 * r3
        + 12 * a**2 * w * e2**2 * r3
        - 6 * a**2 * w * e2 * r3
        + 12 * a**2 * wbar * e1 * e2 * r3**3
        - 4 * a**2 * wbar * e1 * r3**3
        - 4 * a**2 * wbar * e2 * r3**3
        + 2 * a**2 * r3bar * e1**2 * r3**3
        + 2 * a**2 * r3bar * e2**2 * r3**3
        + 9 * a**2 * e1**3 * r3**2
        + 3 * a**2 * e1**2 * e2 * r3**2
        - 4 * a**2 * e1**2 * r3**2
        + 3 * a**2 * e1 * e2**2 * r3**2
        - 8 * a**2 * e1 * e2 * r3**2
        + 4 * a**2 * e1 * r3**2
        + 9 * a**2 * e2**3 * r3**2
        - 4 * a**2 * e2**2 * r3**2
        + 4 * a**2 * e2 * r3**2
        + 6 * a * w * wbar * e1**2 * r3**3
        - 6 * a * w * wbar * e2**2 * r3**3
        - 2 * a * w * r3bar * e1**2 * r3**3
        + 2 * a * w * r3bar * e2**2 * r3**3
        - 3 * a * w * e1**3 * r3**2
        - 3 * a * w * e1**2 * e2 * r3**2
        - 6 * a * w * e1**2 * r3**2
        + 3 * a * w * e1 * e2**2 * r3**2
        + 3 * a * w * e2**3 * r3**2
        + 6 * a * w * e2**2 * r3**2
        - 3 * a * e1**3 * r3**3
        - 3 * a * e1**2 * e2 * r3**3
        + 2 * a * e1**2 * r3**3
        + 3 * a * e1 * e2**2 * r3**3
        + 3 * a * e2**3 * r3**3
        - 2 * a * e2**2 * r3**3
        + w * e1**3 * r3**3
        + 3 * w * e1**2 * e2 * r3**3
        + 3 * w * e1 * e2**2 * r3**3
        + w * e2**3 * r3**3
    )

    p_8 = (
        3 * a**8 * w * wbar * r3
        - 3 * a**8 * w * r3bar * r3
        + a**8 * w * e1
        + a**8 * w * e2
        - a**8 * w
        + 3 * a**8 * wbar * r3**2
        - 3 * a**8 * r3bar * r3**2
        - a**8 * e1 * r3
        - a**8 * e2 * r3
        + a**8 * r3
        - 3 * a**7 * w * e1 * r3
        + 3 * a**7 * w * e2 * r3
        - 3 * a**7 * wbar * e1 * r3
        + 3 * a**7 * wbar * e2 * r3
        + 3 * a**7 * r3bar * e1 * r3
        - 3 * a**7 * r3bar * e2 * r3
        - a**7 * e1**2
        - 3 * a**7 * e1 * r3**2
        + a**7 * e1
        + a**7 * e2**2
        + 3 * a**7 * e2 * r3**2
        - a**7 * e2
        - 3 * a**6 * w * wbar**3 * r3
        + 3 * a**6 * w * wbar**2 * r3bar * r3
        - 3 * a**6 * w * wbar**2 * e1
        - 3 * a**6 * w * wbar**2 * e2
        + 3 * a**6 * w * wbar**2
        + 2 * a**6 * w * wbar * r3bar * e1
        + 2 * a**6 * w * wbar * r3bar * e2
        - 2 * a**6 * w * wbar * r3bar
        - 3 * a**6 * w * wbar * r3**3
        + 3 * a**6 * w * r3bar * r3**3
        + 3 * a**6 * w * r3**2
        - 3 * a**6 * wbar**3 * r3**2
        + 3 * a**6 * wbar**2 * r3bar * r3**2
        - 6 * a**6 * wbar**2 * e1 * r3
        - 6 * a**6 * wbar**2 * e2 * r3
        + 3 * a**6 * wbar**2 * r3
        + 7 * a**6 * wbar * r3bar * e1 * r3
        + 7 * a**6 * wbar * r3bar * e2 * r3
        - 4 * a**6 * wbar * r3bar * r3
        - 3 * a**6 * wbar * e1**2
        - 6 * a**6 * wbar * e1 * e2
        + 4 * a**6 * wbar * e1
        - 3 * a**6 * wbar * e2**2
        + 4 * a**6 * wbar * e2
        - a**6 * wbar
        + 2 * a**6 * r3bar * e1**2
        + 4 * a**6 * r3bar * e1 * e2
        - 3 * a**6 * r3bar * e1
        + 2 * a**6 * r3bar * e2**2
        - 3 * a**6 * r3bar * e2
        + a**6 * r3bar
        + 3 * a**6 * e1**2 * r3
        - 6 * a**6 * e1 * e2 * r3
        + 4 * a**6 * e1 * r3**3
        + 3 * a**6 * e2**2 * r3
        + 4 * a**6 * e2 * r3**3
        - 3 * a**6 * r3**3
        + 9 * a**5 * w * wbar**2 * e1 * r3
        - 9 * a**5 * w * wbar**2 * e2 * r3
        - 6 * a**5 * w * wbar * r3bar * e1 * r3
        + 6 * a**5 * w * wbar * r3bar * e2 * r3
        + 6 * a**5 * w * wbar * e1**2
        - 6 * a**5 * w * wbar * e1
        - 6 * a**5 * w * wbar * e2**2
        + 6 * a**5 * w * wbar * e2
        - 2 * a**5 * w * r3bar * e1**2
        + 2 * a**5 * w * r3bar * e1
        + 2 * a**5 * w * r3bar * e2**2
        - 2 * a**5 * w * r3bar * e2
        + 2 * a**5 * w * e1 * r3**3
        - 2 * a**5 * w * e2 * r3**3
        + 9 * a**5 * wbar**2 * e1 * r3**2
        - 9 * a**5 * wbar**2 * e2 * r3**2
        - 6 * a**5 * wbar * r3bar * e1 * r3**2
        + 6 * a**5 * wbar * r3bar * e2 * r3**2
        + 12 * a**5 * wbar * e1**2 * r3
        + 3 * a**5 * wbar * e1 * r3**3
        - 6 * a**5 * wbar * e1 * r3
        - 12 * a**5 * wbar * e2**2 * r3
        - 3 * a**5 * wbar * e2 * r3**3
        + 6 * a**5 * wbar * e2 * r3
        - 7 * a**5 * r3bar * e1**2 * r3
        - 3 * a**5 * r3bar * e1 * r3**3
        + 4 * a**5 * r3bar * e1 * r3
        + 7 * a**5 * r3bar * e2**2 * r3
        + 3 * a**5 * r3bar * e2 * r3**3
        - 4 * a**5 * r3bar * e2 * r3
        + 3 * a**5 * e1**3
        + 3 * a**5 * e1**2 * e2
        - 4 * a**5 * e1**2
        - 3 * a**5 * e1 * e2**2
        - 3 * a**5 * e1 * r3**2
        + a**5 * e1
        - 3 * a**5 * e2**3
        + 4 * a**5 * e2**2
        + 3 * a**5 * e2 * r3**2
        - a**5 * e2
        + 3 * a**4 * w * wbar**3 * r3**3
        - 3 * a**4 * w * wbar**2 * r3bar * r3**3
        - 9 * a**4 * w * wbar**2 * r3**2
        + 6 * a**4 * w * wbar * r3bar * r3**2
        - 12 * a**4 * w * wbar * e1**2 * r3
        + 12 * a**4 * w * wbar * e1 * e2 * r3
        - 6 * a**4 * w * wbar * e1 * r3
        - 12 * a**4 * w * wbar * e2**2 * r3
        - 6 * a**4 * w * wbar * e2 * r3
        + 9 * a**4 * w * wbar * r3
        + 4 * a**4 * w * r3bar * e1**2 * r3
        - 4 * a**4 * w * r3bar * e1 * e2 * r3
        + 2 * a**4 * w * r3bar * e1 * r3
        + 4 * a**4 * w * r3bar * e2**2 * r3
        + 2 * a**4 * w * r3bar * e2 * r3
        - 3 * a**4 * w * r3bar * r3
        - 3 * a**4 * w * e1**3
        + 3 * a**4 * w * e1**2 * e2
        + 3 * a**4 * w * e1 * e2**2
        - 12 * a**4 * w * e1 * e2
        + 6 * a**4 * w * e1
        - 3 * a**4 * w * e2**3
        + 6 * a**4 * w * e2
        - 3 * a**4 * w
        - 3 * a**4 * wbar**2 * e1 * r3**3
        - 3 * a**4 * wbar**2 * e2 * r3**3
        + 3 * a**4 * wbar**2 * r3**3
        - a**4 * wbar * r3bar * e1 * r3**3
        - a**4 * wbar * r3bar * e2 * r3**3
        - 15 * a**4 * wbar * e1**2 * r3**2
        + 6 * a**4 * wbar * e1 * e2 * r3**2
        + 6 * a**4 * wbar * e1 * r3**2
        - 15 * a**4 * wbar * e2**2 * r3**2
        + 6 * a**4 * wbar * e2 * r3**2
        - 6 * a**4 * wbar * r3**2
        + 5 * a**4 * r3bar * e1**2 * r3**2
        - 2 * a**4 * r3bar * e1 * e2 * r3**2
        + a**4 * r3bar * e1 * r3**2
        + 5 * a**4 * r3bar * e2**2 * r3**2
        + a**4 * r3bar * e2 * r3**2
        - 9 * a**4 * e1**3 * r3
        - 3 * a**4 * e1**2 * e2 * r3
        - 2 * a**4 * e1**2 * r3**3
        + 8 * a**4 * e1**2 * r3
        - 3 * a**4 * e1 * e2**2 * r3
        + 4 * a**4 * e1 * e2 * r3**3
        + 4 * a**4 * e1 * e2 * r3
        - 5 * a**4 * e1 * r3
        - 9 * a**4 * e2**3 * r3
        - 2 * a**4 * e2**2 * r3**3
        + 8 * a**4 * e2**2 * r3
        - 5 * a**4 * e2 * r3
        + 3 * a**4 * r3
        - 6 * a**3 * w * wbar**2 * e1 * r3**3
        + 6 * a**3 * w * wbar**2 * e2 * r3**3
        + 4 * a**3 * w * wbar * r3bar * e1 * r3**3
        - 4 * a**3 * w * wbar * r3bar * e2 * r3**3
        + 6 * a**3 * w * wbar * e1**2 * r3**2
        + 12 * a**3 * w * wbar * e1 * r3**2
        - 6 * a**3 * w * wbar * e2**2 * r3**2
        - 12 * a**3 * w * wbar * e2 * r3**2
        - 2 * a**3 * w * r3bar * e1**2 * r3**2
        - 4 * a**3 * w * r3bar * e1 * r3**2
        + 2 * a**3 * w * r3bar * e2**2 * r3**2
        + 4 * a**3 * w * r3bar * e2 * r3**2
        + 9 * a**3 * w * e1**3 * r3
        - 3 * a**3 * w * e1**2 * e2 * r3
        + 3 * a**3 * w * e1 * e2**2 * r3
        - 6 * a**3 * w * e1 * r3
        - 9 * a**3 * w * e2**3 * r3
        + 6 * a**3 * w * e2 * r3
        + 6 * a**3 * wbar * e1**2 * r3**3
        - 4 * a**3 * wbar * e1 * r3**3
        - 6 * a**3 * wbar * e2**2 * r3**3
        + 4 * a**3 * wbar * e2 * r3**3
        + 9 * a**3 * e1**3 * r3**2
        - 3 * a**3 * e1**2 * e2 * r3**2
        - 4 * a**3 * e1**2 * r3**2
        + 3 * a**3 * e1 * e2**2 * r3**2
        + 4 * a**3 * e1 * r3**2
        - 9 * a**3 * e2**3 * r3**2
        + 4 * a**3 * e2**2 * r3**2
        - 4 * a**3 * e2 * r3**2
        - 12 * a**2 * w * wbar * e1 * e2 * r3**3
        + 4 * a**2 * w * r3bar * e1 * e2 * r3**3
        - 9 * a**2 * w * e1**3 * r3**2
        - 3 * a**2 * w * e1**2 * e2 * r3**2
        - 3 * a**2 * w * e1 * e2**2 * r3**2
        + 12 * a**2 * w * e1 * e2 * r3**2
        - 9 * a**2 * w * e2**3 * r3**2
        - 3 * a**2 * e1**3 * r3**3
        + 3 * a**2 * e1**2 * e2 * r3**3
        + 3 * a**2 * e1 * e2**2 * r3**3
        - 4 * a**2 * e1 * e2 * r3**3
        - 3 * a**2 * e2**3 * r3**3
        + 3 * a * w * e1**3 * r3**3
        + 3 * a * w * e1**2 * e2 * r3**3
        - 3 * a * w * e1 * e2**2 * r3**3
        - 3 * a * w * e2**3 * r3**3
    )

    p_9 = (
        -3 * a**8 * w * wbar * r3**2
        + 3 * a**8 * w * r3bar * r3**2
        - 2 * a**8 * w * e1 * r3
        - 2 * a**8 * w * e2 * r3
        + 2 * a**8 * w * r3
        - a**8 * wbar * r3**3
        + a**8 * r3bar * r3**3
        + 2 * a**8 * e1 * r3**2
        + 2 * a**8 * e2 * r3**2
        - 2 * a**8 * r3**2
        + 3 * a**7 * w * e1 * r3**2
        - 3 * a**7 * w * e2 * r3**2
        + 3 * a**7 * wbar * e1 * r3**2
        - 3 * a**7 * wbar * e2 * r3**2
        - 3 * a**7 * r3bar * e1 * r3**2
        + 3 * a**7 * r3bar * e2 * r3**2
        + 2 * a**7 * e1**2 * r3
        + a**7 * e1 * r3**3
        - 2 * a**7 * e1 * r3
        - 2 * a**7 * e2**2 * r3
        - a**7 * e2 * r3**3
        + 2 * a**7 * e2 * r3
        + 3 * a**6 * w * wbar**3 * r3**2
        - 3 * a**6 * w * wbar**2 * r3bar * r3**2
        + 6 * a**6 * w * wbar**2 * e1 * r3
        + 6 * a**6 * w * wbar**2 * e2 * r3
        - 6 * a**6 * w * wbar**2 * r3
        - 4 * a**6 * w * wbar * r3bar * e1 * r3
        - 4 * a**6 * w * wbar * r3bar * e2 * r3
        + 4 * a**6 * w * wbar * r3bar * r3
        + 3 * a**6 * w * wbar * e1**2
        + 6 * a**6 * w * wbar * e1 * e2
        - 6 * a**6 * w * wbar * e1
        + 3 * a**6 * w * wbar * e2**2
        - 6 * a**6 * w * wbar * e2
        + 3 * a**6 * w * wbar
        - a**6 * w * r3bar * e1**2
        - 2 * a**6 * w * r3bar * e1 * e2
        + 2 * a**6 * w * r3bar * e1
        - a**6 * w * r3bar * e2**2
        + 2 * a**6 * w * r3bar * e2
        - a**6 * w * r3bar
        - a**6 * w * e1 * r3**3
        - a**6 * w * e2 * r3**3
        + a**6 * wbar**3 * r3**3
        - a**6 * wbar**2 * r3bar * r3**3
        + 3 * a**6 * wbar**2 * e1 * r3**2
        + 3 * a**6 * wbar**2 * e2 * r3**2
        - 5 * a**6 * wbar * r3bar * e1 * r3**2
        - 5 * a**6 * wbar * r3bar * e2 * r3**2
        + 2 * a**6 * wbar * r3bar * r3**2
        + 3 * a**6 * wbar * e1**2 * r3
        + 6 * a**6 * wbar * e1 * e2 * r3
        - 2 * a**6 * wbar * e1 * r3
        + 3 * a**6 * wbar * e2**2 * r3
        - 2 * a**6 * wbar * e2 * r3
        - a**6 * wbar * r3
        - 3 * a**6 * r3bar * e1**2 * r3
        - 6 * a**6 * r3bar * e1 * e2 * r3
        + 4 * a**6 * r3bar * e1 * r3
        - 3 * a**6 * r3bar * e2**2 * r3
        + 4 * a**6 * r3bar * e2 * r3
        - a**6 * r3bar * r3
        + a**6 * e1**3
        + 3 * a**6 * e1**2 * e2
        - 3 * a**6 * e1**2 * r3**2
        - 2 * a**6 * e1**2
        + 3 * a**6 * e1 * e2**2
        + 6 * a**6 * e1 * e2 * r3**2
        - 4 * a**6 * e1 * e2
        + a**6 * e1
        + a**6 * e2**3
        - 3 * a**6 * e2**2 * r3**2
        - 2 * a**6 * e2**2
        + a**6 * e2
        - 9 * a**5 * w * wbar**2 * e1 * r3**2
        + 9 * a**5 * w * wbar**2 * e2 * r3**2
        + 6 * a**5 * w * wbar * r3bar * e1 * r3**2
        - 6 * a**5 * w * wbar * r3bar * e2 * r3**2
        - 12 * a**5 * w * wbar * e1**2 * r3
        + 12 * a**5 * w * wbar * e1 * r3
        + 12 * a**5 * w * wbar * e2**2 * r3
        - 12 * a**5 * w * wbar * e2 * r3
        + 4 * a**5 * w * r3bar * e1**2 * r3
        - 4 * a**5 * w * r3bar * e1 * r3
        - 4 * a**5 * w * r3bar * e2**2 * r3
        + 4 * a**5 * w * r3bar * e2 * r3
        - 3 * a**5 * w * e1**3
        - 3 * a**5 * w * e1**2 * e2
        + 6 * a**5 * w * e1**2
        + 3 * a**5 * w * e1 * e2**2
        - 3 * a**5 * w * e1
        + 3 * a**5 * w * e2**3
        - 6 * a**5 * w * e2**2
        + 3 * a**5 * w * e2
        - 3 * a**5 * wbar**2 * e1 * r3**3
        + 3 * a**5 * wbar**2 * e2 * r3**3
        + 2 * a**5 * wbar * r3bar * e1 * r3**3
        - 2 * a**5 * wbar * r3bar * e2 * r3**3
        - 6 * a**5 * wbar * e1**2 * r3**2
        + 6 * a**5 * wbar * e2**2 * r3**2
        + 5 * a**5 * r3bar * e1**2 * r3**2
        - 2 * a**5 * r3bar * e1 * r3**2
        - 5 * a**5 * r3bar * e2**2 * r3**2
        + 2 * a**5 * r3bar * e2 * r3**2
        - 3 * a**5 * e1**3 * r3
        - 3 * a**5 * e1**2 * e2 * r3
        + a**5 * e1**2 * r3**3
        + 2 * a**5 * e1**2 * r3
        + 3 * a**5 * e1 * e2**2 * r3
        + a**5 * e1 * r3
        + 3 * a**5 * e2**3 * r3
        - a**5 * e2**2 * r3**3
        - 2 * a**5 * e2**2 * r3
        - a**5 * e2 * r3
        + 3 * a**4 * w * wbar**2 * e1 * r3**3
        + 3 * a**4 * w * wbar**2 * e2 * r3**3
        - 2 * a**4 * w * wbar * r3bar * e1 * r3**3
        - 2 * a**4 * w * wbar * r3bar * e2 * r3**3
        + 15 * a**4 * w * wbar * e1**2 * r3**2
        - 6 * a**4 * w * wbar * e1 * e2 * r3**2
        - 6 * a**4 * w * wbar * e1 * r3**2
        + 15 * a**4 * w * wbar * e2**2 * r3**2
        - 6 * a**4 * w * wbar * e2 * r3**2
        - 5 * a**4 * w * r3bar * e1**2 * r3**2
        + 2 * a**4 * w * r3bar * e1 * e2 * r3**2
        + 2 * a**4 * w * r3bar * e1 * r3**2
        - 5 * a**4 * w * r3bar * e2**2 * r3**2
        + 2 * a**4 * w * r3bar * e2 * r3**2
        + 9 * a**4 * w * e1**3 * r3
        + 3 * a**4 * w * e1**2 * e2 * r3
        - 12 * a**4 * w * e1**2 * r3
        + 3 * a**4 * w * e1 * e2**2 * r3
        + 3 * a**4 * w * e1 * r3
        + 9 * a**4 * w * e2**3 * r3
        - 12 * a**4 * w * e2**2 * r3
        + 3 * a**4 * w * e2 * r3
        + 3 * a**4 * wbar * e1**2 * r3**3
        - 6 * a**4 * wbar * e1 * e2 * r3**3
        + 2 * a**4 * wbar * e1 * r3**3
        + 3 * a**4 * wbar * e2**2 * r3**3
        + 2 * a**4 * wbar * e2 * r3**3
        - 2 * a**4 * r3bar * e1**2 * r3**3
        - 2 * a**4 * r3bar * e2**2 * r3**3
        + 3 * a**4 * e1**3 * r3**2
        - 3 * a**4 * e1**2 * e2 * r3**2
        + 2 * a**4 * e1**2 * r3**2
        - 3 * a**4 * e1 * e2**2 * r3**2
        + 4 * a**4 * e1 * e2 * r3**2
        - 2 * a**4 * e1 * r3**2
        + 3 * a**4 * e2**3 * r3**2
        + 2 * a**4 * e2**2 * r3**2
        - 2 * a**4 * e2 * r3**2
        - 6 * a**3 * w * wbar * e1**2 * r3**3
        + 6 * a**3 * w * wbar * e2**2 * r3**3
        + 2 * a**3 * w * r3bar * e1**2 * r3**3
        - 2 * a**3 * w * r3bar * e2**2 * r3**3
        - 9 * a**3 * w * e1**3 * r3**2
        + 3 * a**3 * w * e1**2 * e2 * r3**2
        + 6 * a**3 * w * e1**2 * r3**2
        - 3 * a**3 * w * e1 * e2**2 * r3**2
        + 9 * a**3 * w * e2**3 * r3**2
        - 6 * a**3 * w * e2**2 * r3**2
        - a**3 * e1**3 * r3**3
        + 3 * a**3 * e1**2 * e2 * r3**3
        - 2 * a**3 * e1**2 * r3**3
        - 3 * a**3 * e1 * e2**2 * r3**3
        + a**3 * e2**3 * r3**3
        + 2 * a**3 * e2**2 * r3**3
        + 3 * a**2 * w * e1**3 * r3**3
        - 3 * a**2 * w * e1**2 * e2 * r3**3
        - 3 * a**2 * w * e1 * e2**2 * r3**3
        + 3 * a**2 * w * e2**3 * r3**3
    )

    p_10 = (
        a**8 * w * wbar * r3**3
        - a**8 * w * r3bar * r3**3
        + a**8 * w * e1 * r3**2
        + a**8 * w * e2 * r3**2
        - a**8 * w * r3**2
        - a**8 * e1 * r3**3
        - a**8 * e2 * r3**3
        + a**8 * r3**3
        - a**7 * w * e1 * r3**3
        + a**7 * w * e2 * r3**3
        - a**7 * wbar * e1 * r3**3
        + a**7 * wbar * e2 * r3**3
        + a**7 * r3bar * e1 * r3**3
        - a**7 * r3bar * e2 * r3**3
        - a**7 * e1**2 * r3**2
        + a**7 * e1 * r3**2
        + a**7 * e2**2 * r3**2
        - a**7 * e2 * r3**2
        - a**6 * w * wbar**3 * r3**3
        + a**6 * w * wbar**2 * r3bar * r3**3
        - 3 * a**6 * w * wbar**2 * e1 * r3**2
        - 3 * a**6 * w * wbar**2 * e2 * r3**2
        + 3 * a**6 * w * wbar**2 * r3**2
        + 2 * a**6 * w * wbar * r3bar * e1 * r3**2
        + 2 * a**6 * w * wbar * r3bar * e2 * r3**2
        - 2 * a**6 * w * wbar * r3bar * r3**2
        - 3 * a**6 * w * wbar * e1**2 * r3
        - 6 * a**6 * w * wbar * e1 * e2 * r3
        + 6 * a**6 * w * wbar * e1 * r3
        - 3 * a**6 * w * wbar * e2**2 * r3
        + 6 * a**6 * w * wbar * e2 * r3
        - 3 * a**6 * w * wbar * r3
        + a**6 * w * r3bar * e1**2 * r3
        + 2 * a**6 * w * r3bar * e1 * e2 * r3
        - 2 * a**6 * w * r3bar * e1 * r3
        + a**6 * w * r3bar * e2**2 * r3
        - 2 * a**6 * w * r3bar * e2 * r3
        + a**6 * w * r3bar * r3
        - a**6 * w * e1**3
        - 3 * a**6 * w * e1**2 * e2
        + 3 * a**6 * w * e1**2
        - 3 * a**6 * w * e1 * e2**2
        + 6 * a**6 * w * e1 * e2
        - 3 * a**6 * w * e1
        - a**6 * w * e2**3
        + 3 * a**6 * w * e2**2
        - 3 * a**6 * w * e2
        + a**6 * w
        - a**6 * wbar**2 * r3**3
        + a**6 * wbar * r3bar * e1 * r3**3
        + a**6 * wbar * r3bar * e2 * r3**3
        - 2 * a**6 * wbar * e1 * r3**2
        - 2 * a**6 * wbar * e2 * r3**2
        + 2 * a**6 * wbar * r3**2
        + a**6 * r3bar * e1**2 * r3**2
        + 2 * a**6 * r3bar * e1 * e2 * r3**2
        - a**6 * r3bar * e1 * r3**2
        + a**6 * r3bar * e2**2 * r3**2
        - a**6 * r3bar * e2 * r3**2
        + a**6 * e1**2 * r3**3
        - a**6 * e1**2 * r3
        - 2 * a**6 * e1 * e2 * r3**3
        - 2 * a**6 * e1 * e2 * r3
        + 2 * a**6 * e1 * r3
        + a**6 * e2**2 * r3**3
        - a**6 * e2**2 * r3
        + 2 * a**6 * e2 * r3
        - a**6 * r3
        + 3 * a**5 * w * wbar**2 * e1 * r3**3
        - 3 * a**5 * w * wbar**2 * e2 * r3**3
        - 2 * a**5 * w * wbar * r3bar * e1 * r3**3
        + 2 * a**5 * w * wbar * r3bar * e2 * r3**3
        + 6 * a**5 * w * wbar * e1**2 * r3**2
        - 6 * a**5 * w * wbar * e1 * r3**2
        - 6 * a**5 * w * wbar * e2**2 * r3**2
        + 6 * a**5 * w * wbar * e2 * r3**2
        - 2 * a**5 * w * r3bar * e1**2 * r3**2
        + 2 * a**5 * w * r3bar * e1 * r3**2
        + 2 * a**5 * w * r3bar * e2**2 * r3**2
        - 2 * a**5 * w * r3bar * e2 * r3**2
        + 3 * a**5 * w * e1**3 * r3
        + 3 * a**5 * w * e1**2 * e2 * r3
        - 6 * a**5 * w * e1**2 * r3
        - 3 * a**5 * w * e1 * e2**2 * r3
        + 3 * a**5 * w * e1 * r3
        - 3 * a**5 * w * e2**3 * r3
        + 6 * a**5 * w * e2**2 * r3
        - 3 * a**5 * w * e2 * r3
        + 2 * a**5 * wbar * e1 * r3**3
        - 2 * a**5 * wbar * e2 * r3**3
        - a**5 * r3bar * e1**2 * r3**3
        + a**5 * r3bar * e2**2 * r3**3
        + 2 * a**5 * e1**2 * r3**2
        - 2 * a**5 * e1 * r3**2
        - 2 * a**5 * e2**2 * r3**2
        + 2 * a**5 * e2 * r3**2
        - 3 * a**4 * w * wbar * e1**2 * r3**3
        + 6 * a**4 * w * wbar * e1 * e2 * r3**3
        - 3 * a**4 * w * wbar * e2**2 * r3**3
        + a**4 * w * r3bar * e1**2 * r3**3
        - 2 * a**4 * w * r3bar * e1 * e2 * r3**3
        + a**4 * w * r3bar * e2**2 * r3**3
        - 3 * a**4 * w * e1**3 * r3**2
        + 3 * a**4 * w * e1**2 * e2 * r3**2
        + 3 * a**4 * w * e1**2 * r3**2
        + 3 * a**4 * w * e1 * e2**2 * r3**2
        - 6 * a**4 * w * e1 * e2 * r3**2
        - 3 * a**4 * w * e2**3 * r3**2
        + 3 * a**4 * w * e2**2 * r3**2
        - a**4 * e1**2 * r3**3
        + 2 * a**4 * e1 * e2 * r3**3
        - a**4 * e2**2 * r3**3
        + a**3 * w * e1**3 * r3**3
        - 3 * a**3 * w * e1**2 * e2 * r3**3
        + 3 * a**3 * w * e1 * e2**2 * r3**3
        - a**3 * w * e2**3 * r3**3
    )


    return p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9, p_10


def decic_roots(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
    """Solve one degree-10 polynomial via companion-matrix eigenvalues.

    Parameters
    ----------
    p0, ..., p10 : complex
        Coefficients, highest degree first.

    Returns
    -------
    roots : jnp.ndarray, complex, shape (10,)
    """
    # Companion matrix of the monic degree-10 polynomial.
    C = jnp.array(
        [
            [-p1 / p0, -p2 / p0, -p3 / p0, -p4 / p0, -p5 / p0,
             -p6 / p0, -p7 / p0, -p8 / p0, -p9 / p0, -p10 / p0],
            [1.0 + 0j, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1.0 + 0j, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1.0 + 0j, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1.0 + 0j, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1.0 + 0j, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1.0 + 0j, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1.0 + 0j, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1.0 + 0j, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1.0 + 0j, 0],
        ],
        dtype=jnp.complex128,
    )
    return jnp.linalg.eigvals(C)


_vmap_decic_roots = jax.vmap(
    decic_roots, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
)


def triple_lens_solve(w, z1, z2, z3, m1, m2, m3, check_sols, root_tol):
    """Frame transform, degree-10 roots, optional solution masking.

    Parameters
    ----------
    w, z1, z2, z3 : array_like, complex
        Source and lens positions (same angular units).
    m1, m2, m3 : float or array_like
        Component Einstein radii squared in those units.
    check_sols : bool
        If True, mask roots that fail the lens equation.
    root_tol : float or array_like
        Absolute root tolerance (may already be rescaled).

    Returns
    -------
    z_arr : jnp.ndarray, complex, shape (N_times, 10)
        Image positions in the original frame.
    """
    w_src = jnp.asarray(w, dtype=jnp.complex128)
    z1 = jnp.asarray(z1, dtype=jnp.complex128)
    z2 = jnp.asarray(z2, dtype=jnp.complex128)
    z3 = jnp.asarray(z3, dtype=jnp.complex128)
    n = w_src.shape[0]

    # Pair-12 midpoint frame with optional rotation.
    z_cm = (z1 + z2) / 2.0
    a_half = (z1 - z2) / 2.0
    a_abs = jnp.abs(a_half)
    safe_a = jnp.where(a_abs > 0, a_abs, jnp.ones_like(a_abs))
    rot = jnp.where(a_abs > 0, a_half / safe_a, jnp.ones_like(a_half))
    rot_inv = jnp.conj(rot)

    w_work = (w_src - z_cm) * rot_inv
    r3 = (z3 - z_cm) * rot_inv
    a = a_abs

    m_sum = m1 + m2 + m3
    e1 = m1 / m_sum
    e2 = m2 / m_sum

    # Normalize by the local Einstein radius sqrt(m1+m2+m3).
    thetaE = jnp.sqrt(jnp.asarray(m_sum, dtype=jnp.float64))
    inv_thetaE = 1.0 / thetaE
    w_work = w_work * inv_thetaE
    a = a * inv_thetaE
    r3 = r3 * inv_thetaE
    wbar = jnp.conj(w_work)
    r3bar = jnp.conj(r3)

    coeffs = triple_lens_poly_coeffs(w_work, wbar, a, r3, r3bar, e1, e2)
    roots_dim = _vmap_decic_roots(*coeffs)

    thetaE_arr = jnp.broadcast_to(jnp.asarray(thetaE, dtype=jnp.float64), (n,))
    z_arr = (
        roots_dim * thetaE_arr[:, None] * rot[:, None]
        + z_cm[:, None]
    )
    z_arr = jnp.where(thetaE_arr[:, None] > 0, z_arr, jnp.nan + 0j)

    def _mask(z_in):
        m1_arr = jnp.full((n,), m1) if jnp.ndim(m1) == 0 else jnp.asarray(m1)
        m2_arr = jnp.full((n,), m2) if jnp.ndim(m2) == 0 else jnp.asarray(m2)
        m3_arr = jnp.full((n,), m3) if jnp.ndim(m3) == 0 else jnp.asarray(m3)
        tol = (jnp.full((n,), root_tol) if jnp.ndim(root_tol) == 0
               else jnp.asarray(root_tol))
        diff = w_src[:, None] - (
            z_in
            - m1_arr[:, None] / jnp.conj(z_in - z1[:, None])
            - m2_arr[:, None] / jnp.conj(z_in - z2[:, None])
            - m3_arr[:, None] / jnp.conj(z_in - z3[:, None])
        )
        scale = jnp.maximum(1.0, jnp.abs(w_src))
        bad = (
            ~jnp.isfinite(z_in)
            | ~jnp.isfinite(diff)
            | (jnp.abs(diff) > tol[:, None] * scale[:, None])
        )
        return jnp.where(bad, jnp.nan + 0j, z_in)

    z_arr = jax.lax.cond(check_sols, _mask, lambda x: x, z_arr)
    return z_arr


def pstl_amp_arr(z_arr, z1, z2, z3, m1, m2, m3):
    """Per-image amplifications for a triple lens.

    Parameters
    ----------
    z_arr : array_like, complex, shape (N_times, 10)
        Image positions.
    z1, z2, z3 : array_like, complex, shape (N_times,)
        Lens positions.
    m1, m2, m3 : float
        Mass fractions (or Einstein radii squared).

    Returns
    -------
    amp : jnp.ndarray, shape (N_times, 10)
    """
    z_arr = jnp.asarray(z_arr)
    z1 = jnp.asarray(z1)
    z2 = jnp.asarray(z2)
    z3 = jnp.asarray(z3)
    n = z1.shape[0]

    dwbardz = m1 / (z_arr - z1.reshape((n, 1))) ** 2
    dwbardz = dwbardz + m2 / (z_arr - z2.reshape((n, 1))) ** 2
    dwbardz = dwbardz + m3 / (z_arr - z3.reshape((n, 1))) ** 2
    jacobian = 1.0 - jnp.abs(dwbardz) ** 2
    return 1.0 / jnp.abs(jacobian)


def pstl_photometry(t, t0, tE, u0, thetaE_hat, xL1, xL2, xL3,
                    m1, m2, m3, mag_src, b_sff=None,
                    parallax_vectors=None, piE_E=None, piE_N=None,
                    root_tol=1e-8, check_sols=True):
    """Unresolved PSTL magnitude at times ``t``.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0, tE : float
        Closest-approach time and Einstein crossing time.
    u0, thetaE_hat : array_like, shape (2,)
        Source u0 and muRel unit vector.
    xL1, xL2, xL3 : array_like, shape (2,)
        Static lens positions in Einstein radii.
    m1, m2, m3 : float
        Mass fractions of the total Einstein radius.
    mag_src : float
        Unlensed source magnitude.
    b_sff : float, optional
        Source flux fraction.
    parallax_vectors : array_like, optional
        Precomputed parallax table, shape (N_times, 2).
    piE_E, piE_N : float, optional
        Parallax components used with ``parallax_vectors``.
    root_tol : float
        Lens-equation root tolerance.
    check_sols : bool
        Mask unphysical polynomial roots.

    Returns
    -------
    mag : jnp.ndarray, shape (N_times,)
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    u = einstein_source_position(
        t, t0, tE, u0, thetaE_hat,
        parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N)
    w = u[:, 0] + 1j * u[:, 1]
    n = w.shape[0]
    z1 = jnp.full((n,), xL1[0] + 1j * xL1[1])
    z2 = jnp.full((n,), xL2[0] + 1j * xL2[1])
    z3 = jnp.full((n,), xL3[0] + 1j * xL3[1])

    images = triple_lens_solve(
        w, z1, z2, z3, m1, m2, m3, check_sols, root_tol)
    amps = pstl_amp_arr(images, z1, z2, z3, m1, m2, m3)
    amp = jnp.nansum(amps, axis=1)

    flux_src = mag2flux_jax(mag_src)
    flux_model = flux_src * amp
    if b_sff is not None:
        flux_model = flux_model + flux_src * (1.0 - b_sff) / b_sff
    return flux2mag_jax(flux_model)


def pstl_log_likely_photometry(t, t0, tE, u0, thetaE_hat, xL1, xL2, xL3,
                               m1, m2, m3, mag_src, b_sff, mag_obs, mag_err,
                               parallax_vectors=None, piE_E=None, piE_N=None,
                               root_tol=1e-8):
    """Gaussian photometric log-likelihood for static PSTL.

    Parameters
    ----------
    t, mag_obs, mag_err : array_like
        Times and observed photometry.
    Remaining parameters
        Same as :func:`pstl_photometry`.

    Returns
    -------
    lnL : jnp.ndarray
        Summed Gaussian log-likelihood.
    """
    mag_model = pstl_photometry(
        t, t0, tE, u0, thetaE_hat, xL1, xL2, xL3, m1, m2, m3,
        mag_src, b_sff=b_sff, parallax_vectors=parallax_vectors,
        piE_E=piE_E, piE_N=piE_N, root_tol=root_tol)
    return gaussian_log_likelihood_sum(mag_model, mag_obs, mag_err)


def pstl_sky_images(t, t0, xS0, xL0, muS, muL, thetaE_amp,
                    xL1_over_theta, xL2_over_theta, xL3_over_theta,
                    m1, m2, m3, parallax_vectors=None, piS=None, piL=None,
                    root_tol=1e-8, check_sols=True):
    """
    PSTL lens tracks, image sky positions, and magnifications.

    Parameters
    ----------
    t : array_like
        Times in MJD.
    t0 : float
        Reference time (MJD).
    xS0, xL0 : array_like, shape (2,)
        Source and 1-2 geometric-center positions at ``t0`` (arcsec).
    muS, muL : array_like, shape (2,)
        Proper motions (mas/yr).
    thetaE_amp : float
        Einstein radius from ``mLp + mLs`` (mas).
    xL1_over_theta, xL2_over_theta, xL3_over_theta : array_like
        Lens offsets from the 1-2 origin in Einstein radii.
    m1, m2, m3 : float
        Masses in units of ``thetaE(mLp+mLs)^2``.
    parallax_vectors : array_like, optional
        Shape ``(N_times, 2)``.
    piS, piL : float, optional
        Source and lens parallax (mas).
    root_tol : float
        Root tolerance.
    check_sols : bool
        Mask unphysical roots.

    Returns
    -------
    xL1, xL2, xL3 : jnp.ndarray, shape (N_times, 2)
        Lens sky positions (arcsec).
    xS_img : jnp.ndarray, shape (N_times, 10, 2)
        Image sky positions (arcsec).
    amp : jnp.ndarray, shape (N_times, 10)
        Image magnifications.
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    dt = ((t - t0) / _DAYS_PER_YEAR).reshape(-1, 1)

    # Unlensed source and 1-2 origin tracks (arcsec).
    xS = xS0.reshape(1, 2) + dt * muS.reshape(1, 2) * 1e-3
    xL = xL0.reshape(1, 2) + dt * muL.reshape(1, 2) * 1e-3
    if parallax_vectors is not None:
        pvec = jnp.asarray(parallax_vectors, dtype=jnp.float64)
        xS = xS + piS * pvec * 1e-3
        xL = xL + piL * pvec * 1e-3

    thetaE_as = thetaE_amp * 1e-3
    off1 = jnp.atleast_2d(jnp.asarray(xL1_over_theta, dtype=jnp.float64))
    off2 = jnp.atleast_2d(jnp.asarray(xL2_over_theta, dtype=jnp.float64))
    off3 = jnp.atleast_2d(jnp.asarray(xL3_over_theta, dtype=jnp.float64))
    xL1 = xL + off1 * thetaE_as
    xL2 = xL + off2 * thetaE_as
    xL3 = xL + off3 * thetaE_as

    # Host PhotAstrom stores m1/m2/m3 in arcsec^2.
    m1_as = jnp.asarray(m1, dtype=jnp.float64) * thetaE_as ** 2
    m2_as = jnp.asarray(m2, dtype=jnp.float64) * thetaE_as ** 2
    m3_as = jnp.asarray(m3, dtype=jnp.float64) * thetaE_as ** 2

    w = xS[:, 0] + 1j * xS[:, 1]
    z1 = xL1[:, 0] + 1j * xL1[:, 1]
    z2 = xL2[:, 0] + 1j * xL2[:, 1]
    z3 = xL3[:, 0] + 1j * xL3[:, 1]
    images = triple_lens_solve(
        w, z1, z2, z3, m1_as, m2_as, m3_as, check_sols, root_tol)
    amps = pstl_amp_arr(images, z1, z2, z3, m1_as, m2_as, m3_as)

    xS_img = jnp.stack([jnp.real(images), jnp.imag(images)], axis=-1)
    amp_f = jnp.where(jnp.isfinite(amps), amps, 0.0)
    xS_f = jnp.where(jnp.isfinite(xS_img), xS_img, 0.0)

    return xL1, xL2, xL3, xS_f, amp_f


def pstl_photometry_tracks(t, t0, xS0, xL0, muS, muL, thetaE_amp,
                           xL1, xL2, xL3, m1, m2, m3, mag_src, b_sff,
                           parallax_vectors=None, piS=None, piL=None,
                           root_tol=1e-8):
    """Unresolved PSTL magnitude from sky tracks.

    Parameters
    ----------
    Same as :func:`pstl_sky_images`, plus ``mag_src`` and ``b_sff``.

    Returns
    -------
    mag : jnp.ndarray, shape (N_times,)
    """
    _x1, _x2, _x3, _img, amp = pstl_sky_images(
        t, t0, xS0, xL0, muS, muL, thetaE_amp, xL1, xL2, xL3,
        m1, m2, m3, parallax_vectors=parallax_vectors, piS=piS, piL=piL,
        root_tol=root_tol)
    amp_tot = jnp.sum(amp, axis=1)
    flux_src = mag2flux_jax(mag_src)
    flux_model = flux_src * amp_tot
    if b_sff is not None:
        flux_model = flux_model + flux_src * (1.0 - b_sff) / b_sff
    return flux2mag_jax(flux_model)


def pstl_log_likely_photometry_tracks(
        t, t0, xS0, xL0, muS, muL, thetaE_amp, xL1, xL2, xL3, m1, m2, m3,
        mag_src, b_sff, mag_obs, mag_err, parallax_vectors=None,
        piS=None, piL=None, root_tol=1e-8):
    """Gaussian photometric log-likelihood from PSTL sky tracks.

    Parameters
    ----------
    mag_obs, mag_err : array_like
        Observed photometry.
    Remaining parameters
        Same as :func:`pstl_photometry_tracks`.

    Returns
    -------
    lnL : jnp.ndarray
    """
    mag_model = pstl_photometry_tracks(
        t, t0, xS0, xL0, muS, muL, thetaE_amp, xL1, xL2, xL3,
        m1, m2, m3, mag_src, b_sff, parallax_vectors=parallax_vectors,
        piS=piS, piL=piL, root_tol=root_tol)
    return gaussian_log_likelihood_sum(mag_model, mag_obs, mag_err)


def pstl_lens_fluxes(mag_src, b_sff, dmag_Lp_Ls=20.0, dmag_Lp_Lt=-20.0):
    """Split the non-source flux among the three PSTL lenses.

    Parameters
    ----------
    mag_src : float
        Unlensed source magnitude.
    b_sff : float
        Source flux fraction.
    dmag_Lp_Ls : float, optional
        ``mag_Lp - mag_Ls``. Default ``20`` makes the secondary dark.
    dmag_Lp_Lt : float, optional
        ``mag_Lp - mag_Lt``. Default ``-20`` makes the tertiary dark.

    Returns
    -------
    fL1, fL2, fL3 : jnp.ndarray
        Primary, secondary, and tertiary lens fluxes.
    """
    flux_src = mag2flux_jax(mag_src)
    flux_non = flux_src * (1.0 - b_sff) / jnp.maximum(b_sff, 1e-12)
    flux_non = jnp.maximum(jnp.nan_to_num(flux_non, nan=0.0), 0.0)

    # dmag = mag_p - mag_c  ->  f_p / f_c = 10^(dmag / -2.5)
    fr12 = jnp.nan_to_num(10.0 ** (dmag_Lp_Ls / -2.5), nan=0.0)
    fr13 = jnp.nan_to_num(10.0 ** (dmag_Lp_Lt / -2.5), nan=0.0)
    w1 = 1.0
    w2 = 1.0 / jnp.maximum(fr12, 1e-30)
    w3 = 1.0 / jnp.maximum(fr13, 1e-30)
    tot = w1 + w2 + w3

    return (flux_non * w1 / tot, flux_non * w2 / tot,
            flux_non * w3 / tot)


def pstl_astrometry(t, t0, xS0, xL0, muS, muL, thetaE_amp,
                    xL1, xL2, xL3, m1, m2, m3, mag_src, b_sff,
                    dmag_Lp_Ls=20.0, dmag_Lp_Lt=-20.0,
                    parallax_vectors=None, piS=None,
                    piL=None, root_tol=1e-8):
    """Unresolved PSTL centroid including luminous lenses (arcsec).

    Matches :meth:`bagle.model.PSTL_PhotAstrom.get_astrometry`: the
    flux-weighted mean of the images plus the three lens fluxes.

    Parameters
    ----------
    Same as :func:`pstl_sky_images`, plus ``mag_src``, ``b_sff``,
    ``dmag_Lp_Ls``, and ``dmag_Lp_Lt``.

    Returns
    -------
    pos : jnp.ndarray, shape (N_times, 2)
    """
    x1, x2, x3, xS_f, amp_f = pstl_sky_images(
        t, t0, xS0, xL0, muS, muL, thetaE_amp, xL1, xL2, xL3,
        m1, m2, m3, parallax_vectors=parallax_vectors, piS=piS, piL=piL,
        root_tol=root_tol)
    amp_3 = amp_f.reshape((amp_f.shape[0], amp_f.shape[1], 1))
    fS = mag2flux_jax(mag_src)
    fL1, fL2, fL3 = pstl_lens_fluxes(
        mag_src, b_sff, dmag_Lp_Ls=dmag_Lp_Ls, dmag_Lp_Lt=dmag_Lp_Lt)

    # centroid = [sum(A_i fS xS_i) + fL1 xL1 + fL2 xL2 + fL3 xL3]
    #            / [sum(A_i fS) + fL1 + fL2 + fL3]
    numer = (jnp.sum(xS_f * amp_3 * fS, axis=1)
             + x1 * fL1 + x2 * fL2 + x3 * fL3)
    denom = jnp.sum(amp_3 * fS, axis=1) + fL1 + fL2 + fL3
    return numer / denom


def pstl_log_likely_astrometry(
        t, t0, xS0, xL0, muS, muL, thetaE_amp, xL1, xL2, xL3, m1, m2, m3,
        mag_src, b_sff, x_obs, y_obs, x_err, y_err, dmag_Lp_Ls=20.0,
        dmag_Lp_Lt=-20.0, parallax_vectors=None, piS=None, piL=None,
        root_tol=1e-8):
    """Gaussian astrometric log-likelihood for static PSTL.

    Parameters
    ----------
    x_obs, y_obs, x_err, y_err : array_like
        Observed East/North positions and uncertainties (arcsec).
    Remaining parameters
        Same as :func:`pstl_astrometry`.

    Returns
    -------
    lnL : jnp.ndarray
    """
    pos = pstl_astrometry(
        t, t0, xS0, xL0, muS, muL, thetaE_amp, xL1, xL2, xL3,
        m1, m2, m3, mag_src, b_sff, dmag_Lp_Ls=dmag_Lp_Ls,
        dmag_Lp_Lt=dmag_Lp_Lt, parallax_vectors=parallax_vectors,
        piS=piS, piL=piL, root_tol=root_tol)
    return gaussian_astrometry_log_likelihood_sum(
        pos, x_obs, y_obs, x_err, y_err)
