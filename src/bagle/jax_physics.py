"""
Stateless JAX physics for PSBL models.

Pure functions suitable for ``jax.jit`` and ``jax.grad``.  Model classes in
``bagle.model`` remain thin adapters that pack instance state into arrays and
call these kernels.

Gaussian photometry likelihoods are implemented here for PSPL/PSBL models
without Gaussian-process residuals.  GP-enabled models keep using celerite on
the host (see :func:`supports_jax_phot_loglik`).

Set ``JAX_PLATFORMS=cpu`` for reproducible CPU runs in CI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Parameter layout (mirrors fitter_param_names in bagle.model)
# ---------------------------------------------------------------------------

PSPL_PHOT_PARAM1_FITTER_NAMES: tuple[str, ...] = (
    "t0",
    "u0_amp",
    "tE",
    "piE_E",
    "piE_N",
)

PSPL_PHOT_PARAM1_PHOT_NAMES: tuple[str, ...] = ("b_sff", "mag_src")

PSBL_PHOT_PARAM1_FITTER_NAMES: tuple[str, ...] = (
    "t0",
    "u0_amp",
    "tE",
    "piE_E",
    "piE_N",
    "q",
    "sep",
    "phi",
)

PSBL_PHOT_PARAM1_PHOT_NAMES: tuple[str, ...] = ("b_sff", "mag_src")

PSPL_PHOTASTROM_PARAM1_FITTER_NAMES: tuple[str, ...] = (
    "mL",
    "t0",
    "beta",
    "dL",
    "dL_dS",
    "xS0_E",
    "xS0_N",
    "muL_E",
    "muL_N",
    "muS_E",
    "muS_N",
)

_JAX_PHOT_MODEL_KIND: dict[str, str] = {
    "PSPL_Phot_noPar_Param1": "pspl",
    "PSPL_Phot_Par_Param1": "pspl",
    "PSBL_Phot_noPar_Param1": "psbl",
    "PSBL_Phot_Par_Param1": "psbl",
}

_JAX_JOINT_MODEL_KIND: dict[str, str] = {
    "PSPL_PhotAstrom_noPar_Param1": "pspl_photastrom_param1",
    "PSPL_PhotAstrom_Par_Param1": "pspl_photastrom_param1",
}

_DEG2RAD = jnp.pi / 180.0
_MAG_ZP = 30.0
_FLUX_ZP = 1.0
_DAYS_PER_YEAR = 365.25

# Astropy-derived constants (mas, Msun, pc) for physical PSPL_PhotAstromParam1.
from astropy import constants as _const
from astropy import units as _u

# theta_E [mas] = sqrt(_EINSTEIN_M_PER_MSUN * mL * inv_dist_diff / _PC_M)
# with inv_dist_diff = 1/dL - 1/dS in 1/pc and mL in Msun.
_EINSTEIN_M_PER_MSUN = float((4.0 * _const.G * _u.Msun / _const.c**2).to(_u.m).value)
_PC_M = float(_u.pc.to(_u.m))
_RAD_TO_MAS = float(_u.rad.to(_u.mas))
_PI_MAS_PER_PC = float((_u.rad * _u.au / _u.pc).to(_u.mas))


def pack_fitter_params(
    names: Sequence[str], params: Mapping[str, float]
) -> jnp.ndarray:
    """Pack named fitter parameters into a 1-D float64 vector."""
    return jnp.array([float(params[name]) for name in names], dtype=jnp.float64)


def unpack_fitter_params(
    names: Sequence[str], vec
) -> dict[str, jnp.ndarray]:
    """Unpack a fitter parameter vector into a name -> scalar mapping."""
    vec = jnp.asarray(vec, dtype=jnp.float64).reshape(-1)
    if vec.shape[0] != len(names):
        raise ValueError(
            f"Expected vector of length {len(names)}, got {vec.shape[0]}"
        )
    return {name: vec[i] for i, name in enumerate(names)}


def pack_psbl_phot_param1(params: Mapping[str, float]) -> jnp.ndarray:
    """Pack PSBL_PhotParam1 fitter parameters."""
    return pack_fitter_params(PSBL_PHOT_PARAM1_FITTER_NAMES, params)


def unpack_psbl_phot_param1(vec) -> dict[str, jnp.ndarray]:
    """Unpack PSBL_PhotParam1 fitter parameters."""
    return unpack_fitter_params(PSBL_PHOT_PARAM1_FITTER_NAMES, vec)


def pack_psbl_phot_param1_phot(params: Mapping[str, float]) -> jnp.ndarray:
    """Pack per-filter photometry parameters ``(b_sff, mag_src)``."""
    return pack_fitter_params(PSBL_PHOT_PARAM1_PHOT_NAMES, params)


def unpack_psbl_phot_param1_phot(vec) -> dict[str, jnp.ndarray]:
    """Unpack per-filter photometry parameters."""
    return unpack_fitter_params(PSBL_PHOT_PARAM1_PHOT_NAMES, vec)


def pack_pspl_phot_param1(params: Mapping[str, float]) -> jnp.ndarray:
    """Pack PSPL_PhotParam1 fitter parameters."""
    return pack_fitter_params(PSPL_PHOT_PARAM1_FITTER_NAMES, params)


def unpack_pspl_phot_param1(vec) -> dict[str, jnp.ndarray]:
    """Unpack PSPL_PhotParam1 fitter parameters."""
    return unpack_fitter_params(PSPL_PHOT_PARAM1_FITTER_NAMES, vec)


def pack_pspl_phot_param1_phot(params: Mapping[str, float]) -> jnp.ndarray:
    """Pack per-filter photometry parameters ``(b_sff, mag_src)``."""
    return pack_fitter_params(PSPL_PHOT_PARAM1_PHOT_NAMES, params)


def unpack_pspl_phot_param1_phot(vec) -> dict[str, jnp.ndarray]:
    """Unpack per-filter photometry parameters."""
    return unpack_fitter_params(PSPL_PHOT_PARAM1_PHOT_NAMES, vec)


# ---------------------------------------------------------------------------
# Parallax tables (host-side; Astropy ephemerides)
# ---------------------------------------------------------------------------


def precompute_parallax_vectors(
    ra_l: float,
    dec_l: float,
    t,
    obs_location: str = "earth",
) -> np.ndarray:
    """
    Precompute parallax direction vectors on the host.

    Uses :func:`bagle.parallax.parallax_in_direction` (Astropy + JPL
    ephemerides).  The returned array has shape ``(N_times, 2)`` with
    columns ``[East, North]`` in AU, suitable as ``parallax_vectors`` in
    the jitted trajectory kernels.

    Parameters
    ----------
    ra_l, dec_l : float
        Lens right ascension and declination in degrees (J2000).
    t : array_like
        Observation times in MJD.
    obs_location : str
        Observer location passed to the parallax module (e.g. ``'earth'``,
        ``'spitzer'``, ``'jwst'``).
    """
    from bagle import parallax

    return np.asarray(
        parallax.parallax_in_direction(ra_l, dec_l, t, obsLocation=obs_location),
        dtype=np.float64,
    )


def compute_parallax_offset(
    parallax_vectors,
    piE_E,
    piE_N,
):
    """
    Microlensing parallax offset in Einstein-radius units.

    Returns ``piE_amp * parallax_vectors`` with shape ``(N_times, 2)``.
    """
    parallax_vectors = jnp.asarray(parallax_vectors, dtype=jnp.float64)
    piE_amp = jnp.sqrt(piE_E**2 + piE_N**2)
    return piE_amp * parallax_vectors


# ---------------------------------------------------------------------------
# Flux helpers (JIT-safe; no host-side prints)
# ---------------------------------------------------------------------------


def mag2flux_jax(mag):
    flux = _FLUX_ZP * 10.0 ** ((mag - _MAG_ZP) / -2.5)
    flux = jnp.nan_to_num(flux, nan=0.0)
    return jnp.where(flux < 0, jnp.nan, flux)


def flux2mag_jax(flux):
    flux = jnp.asarray(flux, dtype=jnp.float64)
    mag = -2.5 * jnp.log10(flux / _FLUX_ZP) + _MAG_ZP
    return mag


# ---------------------------------------------------------------------------
# PSBL geometry derived from fitter parameters
# ---------------------------------------------------------------------------


def u0_hat_from_thetaE_hat_jax(thetaE_hat, beta):
    """JAX version of :func:`bagle.model.u0_hat_from_thetaE_hat`."""
    thetaE_hat = jnp.asarray(thetaE_hat, dtype=jnp.float64).reshape(2)
    beta = jnp.asarray(beta, dtype=jnp.float64)
    sign_prod_pos = jnp.sign(thetaE_hat[0]) * jnp.sign(thetaE_hat[1]) > 0

    u0_pos = jnp.stack(
        [
            jnp.abs(thetaE_hat[1]),
            jnp.where(sign_prod_pos, -jnp.abs(thetaE_hat[0]), jnp.abs(thetaE_hat[0])),
        ]
    )
    u0_neg = jnp.stack(
        [
            -jnp.abs(thetaE_hat[1]),
            jnp.where(sign_prod_pos, jnp.abs(thetaE_hat[0]), -jnp.abs(thetaE_hat[0])),
        ]
    )
    return jnp.where(beta > 0, u0_pos, u0_neg)


def derive_pspl_static_geometry(u0_amp, piE_E, piE_N):
    """
    Derive static PSPL geometry from fitter parameters.

    Returns ``(u0, thetaE_hat, piE_amp)``.
    """
    piE = jnp.stack([piE_E, piE_N])
    piE_amp = jnp.linalg.norm(piE)
    thetaE_hat = piE / piE_amp
    u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, u0_amp)
    u0 = jnp.abs(u0_amp) * u0_hat
    return u0, thetaE_hat, piE_amp


def derive_psbl_static_geometry(u0_amp, piE_E, piE_N, q, sep, phi):
    """
    Derive static PSBL geometry from fitter parameters.

    Returns ``(m1, m2, u0, thetaE_hat, xL1_over_theta, xL2_over_theta, piE_amp)``.
    """
    piE = jnp.stack([piE_E, piE_N])
    piE_amp = jnp.linalg.norm(piE)
    thetaE_hat = piE / piE_amp
    u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, u0_amp)
    u0 = jnp.abs(u0_amp) * u0_hat

    phi_rad = phi * _DEG2RAD
    phi_piE_rad = jnp.arctan2(piE_E, piE_N)
    phi_rho1_rad = phi_piE_rad + phi_rad

    xL1_over_theta = jnp.stack(
        [
            0.5 * sep * jnp.sin(phi_rho1_rad),
            0.5 * sep * jnp.cos(phi_rho1_rad),
        ]
    )
    xL2_over_theta = -xL1_over_theta

    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)

    return m1, m2, u0, thetaE_hat, xL1_over_theta, xL2_over_theta, piE_amp


def einstein_source_position(
    t,
    t0,
    tE,
    u0,
    thetaE_hat,
    parallax_vectors=None,
    piE_E=None,
    piE_N=None,
    parallax_correction=None,
):
    """
    Unlensed source–lens separation in Einstein-radius units.

    Parameters
    ----------
    parallax_vectors : array_like, optional
        Host-precomputed parallax table from :func:`precompute_parallax_vectors`,
        shape ``(N_times, 2)``.  Combined with ``piE_E`` and ``piE_N`` inside
        the jitted kernel so gradients w.r.t. parallax parameters are available.
    piE_E, piE_N : float, optional
        Microlensing parallax components in Einstein-radius units.  Required
        when ``parallax_vectors`` is provided.
    parallax_correction : array_like, optional
        Legacy alias for a pre-multiplied table ``piE_amp * parallax_vectors``.
        When provided, ``parallax_vectors`` / ``piE_*`` are ignored.
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    u0 = jnp.asarray(u0, dtype=jnp.float64).reshape(2)
    thetaE_hat = jnp.asarray(thetaE_hat, dtype=jnp.float64).reshape(2)

    tau = ((t - t0) / tE).reshape(-1, 1)
    u = u0.reshape(1, 2) + tau * thetaE_hat.reshape(1, 2)

    if parallax_correction is not None:
        u = u - jnp.asarray(parallax_correction, dtype=jnp.float64)
    elif parallax_vectors is not None:
        u = u - compute_parallax_offset(parallax_vectors, piE_E, piE_N)

    return u


def pspl_u(
    t,
    t0,
    tE,
    u0,
    thetaE_hat,
    parallax_vectors=None,
    piE_E=None,
    piE_N=None,
    parallax_correction=None,
):
    """PSPL separation vector ``u(t)`` with shape ``(N_times, 2)``."""
    return einstein_source_position(
        t,
        t0,
        tE,
        u0,
        thetaE_hat,
        parallax_vectors=parallax_vectors,
        piE_E=piE_E,
        piE_N=piE_N,
        parallax_correction=parallax_correction,
    )


def psbl_source_position(
    t,
    t0,
    tE,
    u0,
    thetaE_hat,
    parallax_vectors=None,
    piE_E=None,
    piE_N=None,
    parallax_correction=None,
):
    """Unlensed source position as a complex array (East + i North)."""
    u = einstein_source_position(
        t,
        t0,
        tE,
        u0,
        thetaE_hat,
        parallax_vectors=parallax_vectors,
        piE_E=piE_E,
        piE_N=piE_N,
        parallax_correction=parallax_correction,
    )
    return u[:, 0] + 1j * u[:, 1]


def psbl_static_lens_positions(xL1_over_theta, xL2_over_theta, n_times):
    """Broadcast static lens positions to ``n_times`` complex arrays."""
    xL1 = jnp.asarray(xL1_over_theta, dtype=jnp.float64).reshape(2)
    xL2 = jnp.asarray(xL2_over_theta, dtype=jnp.float64).reshape(2)
    z1 = jnp.full(n_times, xL1[0] + 1j * xL1[1], dtype=jnp.complex128)
    z2 = jnp.full(n_times, xL2[0] + 1j * xL2[1], dtype=jnp.complex128)
    return z1, z2


def psbl_complex_pos_static(
    t,
    t0,
    tE,
    u0,
    thetaE_hat,
    xL1_over_theta,
    xL2_over_theta,
    parallax_vectors=None,
    piE_E=None,
    piE_N=None,
    parallax_correction=None,
):
    """Source and static binary-lens positions as complex arrays."""
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    w = psbl_source_position(
        t,
        t0,
        tE,
        u0,
        thetaE_hat,
        parallax_vectors=parallax_vectors,
        piE_E=piE_E,
        piE_N=piE_N,
        parallax_correction=parallax_correction,
    )
    z1, z2 = psbl_static_lens_positions(xL1_over_theta, xL2_over_theta, w.shape[0])
    return w, z1, z2


# ---------------------------------------------------------------------------
# PSPL amplification and photometry
# ---------------------------------------------------------------------------


def pspl_amplification_from_u(u):
    """Total PSPL amplification from separation vectors."""
    u = jnp.asarray(u, dtype=jnp.float64)
    u_amp = jnp.linalg.norm(u, axis=1)
    return (u_amp**2 + 2) / (u_amp * jnp.sqrt(u_amp**2 + 4))


def pspl_amplification(
    t,
    t0,
    tE,
    u0,
    thetaE_hat,
    parallax_vectors=None,
    piE_E=None,
    piE_N=None,
    parallax_correction=None,
):
    """Total PSPL amplification at times ``t``."""
    u = pspl_u(
        t,
        t0,
        tE,
        u0,
        thetaE_hat,
        parallax_vectors=parallax_vectors,
        piE_E=piE_E,
        piE_N=piE_N,
        parallax_correction=parallax_correction,
    )
    return pspl_amplification_from_u(u)


def pspl_photometry_from_amp(amp, mag_src, b_sff=None):
    """Unresolved PSPL magnitude from total amplification."""
    flux_src = mag2flux_jax(mag_src)
    flux_model = flux_src * amp
    if b_sff is not None:
        flux_model = flux_model + flux_src * (1.0 - b_sff) / b_sff
    return flux2mag_jax(flux_model)


def pspl_photometry(
    t,
    t0,
    tE,
    u0,
    thetaE_hat,
    mag_src,
    b_sff=None,
    parallax_vectors=None,
    piE_E=None,
    piE_N=None,
    parallax_correction=None,
):
    """PSPL unresolved photometry at times ``t``."""
    amp = pspl_amplification(
        t,
        t0,
        tE,
        u0,
        thetaE_hat,
        parallax_vectors=parallax_vectors,
        piE_E=piE_E,
        piE_N=piE_N,
        parallax_correction=parallax_correction,
    )
    return pspl_photometry_from_amp(amp, mag_src, b_sff=b_sff)


def pspl_photometry_from_fitter_vec(
    t,
    fitter_vec,
    mag_src,
    b_sff=None,
    parallax_vectors=None,
):
    """PSPL photometry from a packed PSPL_PhotParam1 fitter vector."""
    params = unpack_pspl_phot_param1(fitter_vec)
    u0, thetaE_hat, _ = derive_pspl_static_geometry(
        params["u0_amp"],
        params["piE_E"],
        params["piE_N"],
    )
    return pspl_photometry(
        t,
        params["t0"],
        params["tE"],
        u0,
        thetaE_hat,
        mag_src,
        b_sff=b_sff,
        parallax_vectors=parallax_vectors,
        piE_E=params["piE_E"],
        piE_N=params["piE_N"],
    )


# ---------------------------------------------------------------------------
# Quintic root solver (Witt / BAGLE coefficients)
# ---------------------------------------------------------------------------


def quintic_coefficients(w, z1, z2, m1, m2):
    """Return quintic coefficients ``(a5, a4, a3, a2, a1, a0)`` high-to-low."""
    w = jnp.asarray(w, dtype=jnp.complex128).reshape(-1)
    z1 = jnp.asarray(z1, dtype=jnp.complex128).reshape(-1)
    z2 = jnp.asarray(z2, dtype=jnp.complex128).reshape(-1)
    m1 = jnp.asarray(m1, dtype=jnp.float64)
    m2 = jnp.asarray(m2, dtype=jnp.float64)

    wbar = jnp.conj(w)
    z1bar = jnp.conj(z1)
    z2bar = jnp.conj(z2)

    a5 = (wbar - z1bar) * (wbar - z2bar)
    a4 = (
        -((w + 2 * (z1 + z2)) * wbar**2)
        - m2 * z2bar
        - z1bar * (m1 + (w + 2 * (z1 + z2)) * z2bar)
        + wbar * (m1 + m2 + (w + 2 * (z1 + z2)) * (z1bar + z2bar))
    )
    a3 = (
        (z1**2 + 4 * z1 * z2 + z2**2 + 2 * w * (z1 + z2)) * wbar**2
        + (m1 * (w - z1) + m2 * (w + 2 * z1 + z2)) * z2bar
        + z1bar
        * (
            m2 * (w - z2)
            + m1 * (w + z1 + 2 * z2)
            + (z1**2 + 4 * z1 * z2 + z2**2 + 2 * w * (z1 + z2)) * z2bar
        )
        - wbar
        * (
            2 * (m2 * (w + z1) + m1 * (w + z2))
            + (z1**2 + 4 * z1 * z2 + z2**2 + 2 * w * (z1 + z2)) * (z1bar + z2bar)
        )
    )
    a2 = (
        -((m1 + m2) * (m1 * (w - z1) + m2 * (w - z2)))
        - (2 * z1 * z2 * (z1 + z2) + w * (z1**2 + 4 * z1 * z2 + z2**2)) * wbar**2
        - (m2 * (w - z2) * (2 * z1 + z2) + m1 * (w * z1 + 2 * (w + z1) * z2 + z2**2))
        * z1bar
        - (
            m2 * w * (2 * z1 + z2)
            + m1 * (w - z1) * (z1 + 2 * z2)
            + m2 * z1 * (z1 + 2 * z2)
            + 2 * z1 * z2 * (z1 + z2) * z1bar
            + w * (z1**2 + 4 * z1 * z2 + z2**2) * z1bar
        )
        * z2bar
        + wbar
        * (
            z1 * (2 * m1 * w + 4 * m2 * w - m1 * z1 + m2 * z1)
            + 2 * (2 * m1 + m2) * w * z2
            + (m1 - m2) * z2**2
            + (2 * z1 * z2 * (z1 + z2) + w * (z1**2 + 4 * z1 * z2 + z2**2))
            * (z1bar + z2bar)
        )
    )
    a1 = (
        2 * m1**2 * w * z2
        + 2 * m1 * m2 * w * z2
        - m1 * m2 * z2**2
        - 2 * m1 * w * z2**2 * wbar
        + m1 * w * z2**2 * z1bar
        + m1 * w * z2**2 * z2bar
        + z1**2
        * (
            -(m1 * m2)
            - 2 * m2 * w * wbar
            + 2 * m1 * z2 * wbar
            + 2 * w * z2 * wbar**2
            + z2**2 * wbar**2
            + m2 * (w - z2) * z1bar
            - 2 * w * z2 * wbar * z1bar
            - z2**2 * wbar * z1bar
            + m2 * w * z2bar
            - 2 * m1 * z2 * z2bar
            + m2 * z2 * z2bar
            - 2 * w * z2 * wbar * z2bar
            - z2**2 * wbar * z2bar
            + 2 * w * z2 * z1bar * z2bar
            + z2**2 * z1bar * z2bar
        )
        + z1
        * (
            2 * m1 * m2 * w
            + 2 * m2**2 * (w - z2)
            - 2 * m1**2 * z2
            - 2 * m1 * m2 * z2
            - 4 * m1 * w * z2 * wbar
            - 4 * m2 * w * z2 * wbar
            + 2 * m2 * z2**2 * wbar
            + 2 * w * z2**2 * wbar**2
            + 2 * m1 * w * z2 * z1bar
            + 2 * m2 * (w - z2) * z2 * z1bar
            + m1 * z2**2 * z1bar
            - 2 * w * z2**2 * wbar * z1bar
            + 2 * m1 * w * z2 * z2bar
            + 2 * m2 * w * z2 * z2bar
            - m1 * z2**2 * z2bar
            - 2 * w * z2**2 * wbar * z2bar
            + 2 * w * z2**2 * z1bar * z2bar
        )
    )
    a0 = (m2 * z1 + m1 * z2) * (m1 * (-w + z1) * z2 + m2 * z1 * (-w + z2)) + z1 * z2 * (
        -(w * z1 * z2 * wbar**2)
        - (m2 * z1 * (w - z2) + m1 * w * z2) * z1bar
        - (m2 * w * z1 + m1 * (w - z1) * z2 + w * z1 * z2 * z1bar) * z2bar
        + wbar
        * (
            2 * m2 * w * z1
            + 2 * m1 * w * z2
            - (m1 + m2) * z1 * z2
            + w * z1 * z2 * (z1bar + z2bar)
        )
    )
    return a5, a4, a3, a2, a1, a0


def quintic_roots_companion(a5, a4, a3, a2, a1, a0):
    """Solve a single quintic via the companion matrix (scalar coefficients)."""
    C = jnp.complex128(
        [
            [-a4 / a5, -a3 / a5, -a2 / a5, -a1 / a5, -a0 / a5],
            [1.0 + 0j, 0, 0, 0, 0],
            [0, 1.0 + 0j, 0, 0, 0],
            [0, 0, 1.0 + 0j, 0, 0],
            [0, 0, 0, 1.0 + 0j, 0],
        ]
    )
    return jnp.linalg.eigvals(C)


_vmap_quintic_roots = jax.vmap(
    quintic_roots_companion, in_axes=(0, 0, 0, 0, 0, 0)
)


def _mask_psbl_roots(z_arr, w, z1, z2, m1, m2, root_tol):
    n = w.shape[0]
    m1_arr = m1 if jnp.ndim(m1) else jnp.full((n,), m1)
    m2_arr = m2 if jnp.ndim(m2) else jnp.full((n,), m2)
    tol = root_tol if jnp.ndim(root_tol) else jnp.full((n,), root_tol)
    diff = w[:, jnp.newaxis] - (
        z_arr
        - m1_arr[:, jnp.newaxis] / jnp.conj(z_arr - z1[:, jnp.newaxis])
        - m2_arr[:, jnp.newaxis] / jnp.conj(z_arr - z2[:, jnp.newaxis])
    )
    bad = jnp.abs(diff) > tol[:, jnp.newaxis]
    return jnp.where(bad, jnp.nan + 0j, z_arr)


def psbl_image_positions(w, z1, z2, m1, m2, root_tol, check_sols: bool):
    """
    Binary-lens image positions from the Witt quintic (companion-matrix roots).

    Parameters
    ----------
    root_tol : float or array
        Lens-equation tolerance; may be per-epoch when rescaling is used.
    check_sols : bool
        When ``True``, mask roots that fail the lens equation.
    """
    a5, a4, a3, a2, a1, a0 = quintic_coefficients(w, z1, z2, m1, m2)
    z_arr = jnp.asarray(_vmap_quintic_roots(a5, a4, a3, a2, a1, a0))

    def _mask(z_arr):
        return _mask_psbl_roots(z_arr, w, z1, z2, m1, m2, root_tol)

    return jax.lax.cond(check_sols, _mask, lambda x: x, z_arr)


psbl_image_positions_jit = jax.jit(psbl_image_positions, static_argnames=("check_sols",))


# ---------------------------------------------------------------------------
# Rescaling, amplification, photometry
# ---------------------------------------------------------------------------

def rescale_complex_pos(w, z1, z2, m1, m2):
    """
    Center and scale complex positions into roughly a 1 x 1 box.

    Returns ``(w, z1, z2, m1, m2, scale, shift)``.
    """
    w = jnp.asarray(w, dtype=jnp.complex128)
    z1 = jnp.asarray(z1, dtype=jnp.complex128)
    z2 = jnp.asarray(z2, dtype=jnp.complex128)
    m1 = jnp.asarray(m1, dtype=jnp.float64)
    m2 = jnp.asarray(m2, dtype=jnp.float64)

    pos = jnp.vstack([w, z1, z2]).T
    shift = jnp.average(pos, axis=1)
    s = shift[:, jnp.newaxis] if w.ndim > 1 else shift
    w = w - s
    z1 = z1 - s
    z2 = z2 - s

    pr, pi = jnp.real(pos), jnp.imag(pos)
    xscale = jnp.max(pr, axis=1) - jnp.min(pr, axis=1)
    yscale = jnp.max(pi, axis=1) - jnp.min(pi, axis=1)
    xyscale = jnp.stack([xscale, yscale], axis=1)
    scale = 1.0 / jnp.max(xyscale, axis=1)
    sc = scale[:, jnp.newaxis] if w.ndim > 1 else scale
    w = w * sc
    z1 = z1 * sc
    z2 = z2 * sc
    m1 = m1 * (scale**2)
    m2 = m2 * (scale**2)

    return w, z1, z2, m1, m2, scale, shift


def psbl_amp_arr(z_arr, z1, z2, m1, m2):
    """Magnification of each image from the binary-lens Jacobian."""
    n_times = z1.shape[0]
    m1 = jnp.asarray(m1, dtype=jnp.float64)
    m2 = jnp.asarray(m2, dtype=jnp.float64)
    dwbardz = m1 / (z_arr - z1.reshape((n_times, 1))) ** 2
    dwbardz += m2 / (z_arr - z2.reshape((n_times, 1))) ** 2
    jacobian = 1.0 - jnp.abs(dwbardz) ** 2
    return 1.0 / jnp.abs(jacobian)


def psbl_all_arrays(w, z1, z2, m1, m2, root_tol, check_sols: bool = True, rescale: bool = True):
    """
    Image positions and per-image amplifications.

    When ``rescale=True``, uses the same rescale-then-unscale strategy as
    :meth:`bagle.model.PSBL.get_all_arrays`.
    """
    w = jnp.asarray(w, dtype=jnp.complex128)
    z1 = jnp.asarray(z1, dtype=jnp.complex128)
    z2 = jnp.asarray(z2, dtype=jnp.complex128)
    m1_phys = jnp.asarray(m1, dtype=jnp.float64)
    m2_phys = jnp.asarray(m2, dtype=jnp.float64)

    if rescale:
        rw, rz1, rz2, rm1, rm2, scale, shift = rescale_complex_pos(
            w, z1, z2, m1_phys, m2_phys
        )
        rt = root_tol * scale if jnp.ndim(root_tol) else root_tol * scale
        rimages = psbl_image_positions_jit(rw, rz1, rz2, rm1, rm2, rt, check_sols)
        images = (rimages / scale.reshape(-1, 1)) + shift.reshape(-1, 1)
        amps = psbl_amp_arr(images, z1, z2, m1_phys, m2_phys)
    else:
        images = psbl_image_positions_jit(w, z1, z2, m1_phys, m2_phys, root_tol, check_sols)
        amps = psbl_amp_arr(images, z1, z2, m1_phys, m2_phys)

    return images, amps


def psbl_total_amplification(amp_arr):
    """Sum finite per-image amplifications."""
    amp_arr = jnp.asarray(amp_arr)
    return jnp.sum(jnp.where(jnp.isfinite(amp_arr), amp_arr, 0.0), axis=1)


def psbl_photometry_from_amp(amp, mag_src, b_sff=None):
    """
    Unresolved PSBL magnitude from total amplification.

    Parameters
    ----------
    mag_src : float
        Unlensed source magnitude.
    b_sff : float, optional
        Source flux fraction; when provided, neighbor/lens flux is added.
    """
    flux_src = mag2flux_jax(mag_src)
    flux_model = flux_src * amp
    if b_sff is not None:
        flux_model = flux_model + flux_src * (1.0 - b_sff) / b_sff
    return flux2mag_jax(flux_model)


def psbl_photometry(t, t0, tE, u0, thetaE_hat, xL1_over_theta, xL2_over_theta, m1, m2, 
                    mag_src, b_sff=None, 
                    piE_E=None, piE_N=None, 
                    root_tol=1e-8, 
                    parallax_vectors=None, parallax_correction=None, 
                    check_sols: bool = True, rescale: bool = True):
    """
    PSBL unresolved photometry at times ``t``.

    Fitter parameters may be passed via :func:`unpack_psbl_phot_param1` and
    :func:`derive_psbl_static_geometry`; held-fixed quantities (parallax
    tables, blend parameters) are explicit arguments.
    """
    w, z1, z2 = psbl_complex_pos_static(
        t,
        t0,
        tE,
        u0,
        thetaE_hat,
        xL1_over_theta,
        xL2_over_theta,
        parallax_vectors=parallax_vectors,
        piE_E=piE_E,
        piE_N=piE_N,
        parallax_correction=parallax_correction,
    )
    _, amp_arr = psbl_all_arrays(
        w, z1, z2, m1, m2, root_tol, check_sols=check_sols, rescale=rescale
    )
    amp = psbl_total_amplification(amp_arr)

    return psbl_photometry_from_amp(amp, mag_src, b_sff=b_sff)


def psbl_photometry_from_fitter_vec(
    t,
    fitter_vec,
    mag_src,
    b_sff=None,
    root_tol=1e-8,
    parallax_vectors=None,
    check_sols: bool = True,
    rescale: bool = True,
):
    """
    PSBL photometry from a packed PSBL_PhotParam1 fitter vector.

    ``mag_src`` and ``b_sff`` are per-filter photometry parameters and are
    kept separate from the fitter vector (matching ``phot_param_names``).
    """
    params = unpack_psbl_phot_param1(fitter_vec)
    m1, m2, u0, thetaE_hat, xL1, xL2, _ = derive_psbl_static_geometry(
        params["u0_amp"],
        params["piE_E"],
        params["piE_N"],
        params["q"],
        params["sep"],
        params["phi"],
    )

    return psbl_photometry(
        t,
        params["t0"],
        params["tE"],
        u0,
        thetaE_hat,
        xL1,
        xL2,
        m1,
        m2,
        mag_src,
        b_sff=b_sff,
        root_tol=root_tol,
        parallax_vectors=parallax_vectors,
        piE_E=params["piE_E"],
        piE_N=params["piE_N"],
        check_sols=check_sols,
        rescale=rescale,
    )


# ---------------------------------------------------------------------------
# Photometry log-likelihood (non-GP models)
# ---------------------------------------------------------------------------

_GP_PARAM_PREFIXES = (
    "gp_log_sigma",
    "gp_log_rho",
    "gp_log_S0",
    "gp_log_omega0",
    "gp_rho",
    "gp_log_omega0_S0",
    "gp_log_omega04_S0",
    "gp_log_jit_sigma",
)


def _fitter_has_blocked_params(fitter) -> bool:
    for name in fitter.additional_param_names:
        if any(name.startswith(prefix) for prefix in _GP_PARAM_PREFIXES):
            return True
    for name in fitter.fitter_param_names:
        if any(x in name for x in ("add_err", "mult_err", "weights")):
            return True
    return False


def derive_pspl_photastrom_param1_geometry(
    mL,
    t0,
    beta,
    dL,
    dL_dS,
    xS0_E,
    xS0_N,
    muL_E,
    muL_N,
    muS_E,
    muS_N,
):
    """
    Physical-parameter geometry for :class:`~bagle.model.PSPL_PhotAstromParam1`.

    Returns quantities needed for JAX photometry and astrometry (mas, arcsec).
    """
    dS = dL / dL_dS
    xS0 = jnp.stack([xS0_E, xS0_N])
    muL = jnp.stack([muL_E, muL_N])
    muS = jnp.stack([muS_E, muS_N])
    inv_dist_diff = 1.0 / dL - 1.0 / dS
    piRel = _PI_MAS_PER_PC * inv_dist_diff
    piS = _PI_MAS_PER_PC / dS
    piL = _PI_MAS_PER_PC / dL
    muRel = muS - muL
    muRel_amp = jnp.linalg.norm(muRel)
    thetaE_amp = jnp.sqrt(
        _EINSTEIN_M_PER_MSUN * mL * inv_dist_diff / _PC_M
    ) * _RAD_TO_MAS
    thetaE_hat = muRel / muRel_amp
    u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, beta)
    u0_amp = beta / thetaE_amp
    u0 = jnp.abs(u0_amp) * u0_hat
    piE_amp = piRel / thetaE_amp
    piE = piE_amp * thetaE_hat
    tE = (thetaE_amp / muRel_amp) * _DAYS_PER_YEAR
    thetaS0 = u0 * thetaE_amp
    xL0 = xS0 - thetaS0 * 1e-3
    return (
        u0,
        thetaE_hat,
        tE,
        piE[0],
        piE[1],
        xS0,
        xL0,
        muS,
        muL,
        thetaE_amp,
        piS,
        piL,
    )


def pspl_astrometry_param1(
    t,
    t0,
    xS0,
    xL0,
    muS,
    muL,
    thetaE_amp,
    b_sff,
    parallax_vectors=None,
    piS=None,
    piL=None,
):
    """
    PSPL flux-weighted centroid astrometry (arcsec), matching
    :meth:`bagle.model.PSPL.get_astrometry`.
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    dt = ((t - t0) / _DAYS_PER_YEAR).reshape(-1, 1)
    xS = xS0.reshape(1, 2) + dt * muS.reshape(1, 2) * 1e-3
    xL = xL0.reshape(1, 2) + dt * muL.reshape(1, 2) * 1e-3
    if parallax_vectors is not None:
        pvec = jnp.asarray(parallax_vectors, dtype=jnp.float64)
        xS = xS + piS * pvec * 1e-3
        xL = xL + piL * pvec * 1e-3
    thetaS = xS - xL
    u_vec = thetaS / (thetaE_amp * 1e-3)
    u_amp = jnp.linalg.norm(u_vec, axis=1)
    g = (1.0 - b_sff) / b_sff
    sqrt_term = jnp.sqrt(u_amp**2 + 4.0)
    numer_u = u_amp**2 - u_amp * sqrt_term + 3.0
    denom_u = u_amp**2 + 2.0 + g * u_amp * sqrt_term
    numer = thetaS * (1.0 + g * numer_u)[:, jnp.newaxis]
    denom = (1.0 + g) * denom_u
    shift = numer / denom[:, jnp.newaxis]
    return b_sff * xS + (1.0 - b_sff) * xL + shift


def pspl_photometry_param1(
    t,
    mL,
    t0,
    beta,
    dL,
    dL_dS,
    xS0_E,
    xS0_N,
    muL_E,
    muL_N,
    muS_E,
    muS_N,
    mag_src,
    b_sff,
    parallax_vectors=None,
):
    """PSPL photometry from PSPL_PhotAstromParam1 physical fitter parameters."""
    (
        u0,
        thetaE_hat,
        tE,
        piE_E,
        piE_N,
        _xS0,
        _xL0,
        _muS,
        _muL,
        _thetaE_amp,
        _piS,
        _piL,
    ) = derive_pspl_photastrom_param1_geometry(
        mL, t0, beta, dL, dL_dS, xS0_E, xS0_N, muL_E, muL_N, muS_E, muS_N
    )
    return pspl_photometry(
        t,
        t0,
        tE,
        u0,
        thetaE_hat,
        mag_src,
        b_sff=b_sff,
        parallax_vectors=parallax_vectors,
        piE_E=piE_E,
        piE_N=piE_N,
    )


def gaussian_log_likelihood_sum(mag_model, mag_obs, mag_err):
    """
    Sum of per-point Gaussian log-likelihoods (includes normalization).

    Matches :meth:`bagle.model.PSPL.log_likely_photometry`.
    """
    mag_model = jnp.asarray(mag_model, dtype=jnp.float64)
    mag_obs = jnp.asarray(mag_obs, dtype=jnp.float64)
    mag_err = jnp.asarray(mag_err, dtype=jnp.float64)
    chi2 = ((mag_obs - mag_model) / mag_err) ** 2
    lnL_const = -0.5 * jnp.log(2.0 * jnp.pi * mag_err**2)
    return jnp.sum((-0.5 * chi2) + lnL_const)


@dataclass(frozen=True)
class PhotFilterLikelihoodData:
    """Precomputed photometry arrays for one filter."""

    t: np.ndarray
    mag_obs: np.ndarray
    mag_err: np.ndarray
    weight: float
    parallax_vectors: np.ndarray | None
    idx_b_sff: int
    idx_mag_src: int


@dataclass(frozen=True)
class JaxPhotLikelihoodContext:
    """Host-side context for jitted photometry log-likelihood."""

    model_kind: str
    fitter_param_names: tuple[str, ...]
    base_fitter_names: tuple[str, ...]
    base_param_indices: tuple[int, ...]
    filters: tuple[PhotFilterLikelihoodData, ...]
    root_tol: float


def supports_jax_phot_loglik(fitter) -> str | None:
    """
    Return layout id when ``fitter`` can use the JAX photometry likelihood.

    Delegates to :mod:`bagle.jax.layout_registry` when the model resolves to a
    registered layout; falls back to legacy class-name allowlist.
    """
    if "GP_" in fitter.model_class.__name__:
        return None
    try:
        from bagle.jax.likelihood import supports_jax_loglik_for_fitter

        layout_id = supports_jax_loglik_for_fitter(fitter)
        if layout_id is not None:
            from bagle.jax.layout_registry import resolve_layout

            layout = resolve_layout(fitter.model_class)
            if layout and layout.has_gp:
                return None
            if layout and layout.likelihood_mode in ("phot", "phot_gp"):
                return layout.family
            if layout and layout.likelihood_mode == "joint":
                return None
            if layout_id:
                return layout.family if layout else layout_id
    except ImportError:
        pass

    model_class = fitter.model_class
    kind = _JAX_PHOT_MODEL_KIND.get(model_class.__name__)
    if kind is None:
        return None

    if getattr(model_class, "astrometryFlag", False) and fitter.n_ast_sets > 0:
        return None

    if not getattr(model_class, "photometryFlag", False) or fitter.n_phot_sets == 0:
        return None

    if _fitter_has_blocked_params(fitter):
        return None

    return kind


def gaussian_astrometry_log_likelihood_sum(pos_model, x_obs, y_obs, x_err, y_err):
    """Joint x/y Gaussian astrometry log-likelihood (matches PSPL astrometry term)."""
    pos_model = jnp.asarray(pos_model, dtype=jnp.float64)
    x_obs = jnp.asarray(x_obs, dtype=jnp.float64)
    y_obs = jnp.asarray(y_obs, dtype=jnp.float64)
    x_err = jnp.asarray(x_err, dtype=jnp.float64)
    y_err = jnp.asarray(y_err, dtype=jnp.float64)
    chi2_x = ((x_obs - pos_model[:, 0]) / x_err) ** 2
    chi2_y = ((y_obs - pos_model[:, 1]) / y_err) ** 2
    lnL_const_x = -0.5 * jnp.log(2.0 * jnp.pi * x_err**2)
    lnL_const_y = -0.5 * jnp.log(2.0 * jnp.pi * y_err**2)
    return jnp.sum((-0.5 * (chi2_x + chi2_y)) + lnL_const_x + lnL_const_y)


def supports_jax_joint_loglik(fitter) -> str | None:
    """
    Return layout id when joint phot+astrometry JAX likelihood is supported.
    """
    try:
        from bagle.jax.likelihood import supports_jax_loglik_for_fitter
        from bagle.jax.layout_registry import resolve_layout

        from bagle.jax.layout_registry import legacy_layout_id

        layout_id = supports_jax_loglik_for_fitter(fitter)
        layout = resolve_layout(fitter.model_class)
        if layout_id and layout and layout.likelihood_mode in ("joint", "joint_gp", "ast"):
            return legacy_layout_id(layout)
    except ImportError:
        pass

    model_class = fitter.model_class
    kind = _JAX_JOINT_MODEL_KIND.get(model_class.__name__)
    if kind is None:
        return None
    if not getattr(model_class, "photometryFlag", False) or fitter.n_phot_sets == 0:
        return None
    if not getattr(model_class, "astrometryFlag", False) or fitter.n_ast_sets == 0:
        return None
    if _fitter_has_blocked_params(fitter):
        return None
    names = tuple(fitter.fitter_param_names)
    if names[: len(PSPL_PHOTASTROM_PARAM1_FITTER_NAMES)] != PSPL_PHOTASTROM_PARAM1_FITTER_NAMES:
        return None
    if kind == "pspl_photastrom_param1" and model_class.__name__.endswith("_Par_Param1"):
        if "raL" not in fitter.data or "decL" not in fitter.data:
            return None
    return kind


def _fitter_weight(fitter, idx: int, default: float = 1.0) -> float:
    w = getattr(fitter, "weights", None)
    if w is None or idx >= len(w):
        return default
    return float(w[idx])


def _param_index(names: Sequence[str], param_base: str, filt_idx: int | None) -> int:
    if filt_idx is None:
        key = param_base
    else:
        key = f"{param_base}{filt_idx}"
    try:
        return names.index(key)
    except ValueError as exc:
        raise KeyError(f"Parameter {key!r} not in fitter_param_names") from exc


def build_jax_phot_likelihood_context(fitter) -> JaxPhotLikelihoodContext | None:
    """
    Build a host-side context for :func:`log_likelihood_phot_from_vec`.

    Returns ``None`` when the solver configuration is not supported.
    """
    kind = supports_jax_phot_loglik(fitter)
    if kind is None:
        return None

    names = tuple(fitter.fitter_param_names)
    if kind == "pspl":
        base_names = PSPL_PHOT_PARAM1_FITTER_NAMES
    else:
        base_names = PSBL_PHOT_PARAM1_FITTER_NAMES

    if names[: len(base_names)] != base_names:
        return None

    base_param_indices = tuple(range(len(base_names)))
    use_parallax = "raL" in fitter.data and "decL" in fitter.data
    ra_l = float(fitter.data["raL"]) if use_parallax else None
    dec_l = float(fitter.data["decL"]) if use_parallax else None

    filters: list[PhotFilterLikelihoodData] = []
    for i in range(fitter.n_phot_sets):
        filt_1 = i + 1
        t = np.asarray(fitter.data[f"t_phot{filt_1}"], dtype=np.float64)
        mag_obs = np.asarray(fitter.data[f"mag{filt_1}"], dtype=np.float64)
        mag_err = np.asarray(fitter.data[f"mag_err{filt_1}"], dtype=np.float64)
        weight = float(getattr(fitter, "weights", [1.0] * fitter.n_phot_sets)[i])

        pvec = None
        if use_parallax:
            pvec = precompute_parallax_vectors(ra_l, dec_l, t)

        idx_b = _param_index(names, "b_sff", filt_1 if f"b_sff{filt_1}" in names else None)
        if f"mag_src{filt_1}" in names:
            idx_m = _param_index(names, "mag_src", filt_1)
        elif "mag_src" in names:
            idx_m = _param_index(names, "mag_src", None)
        else:
            return None

        filters.append(
            PhotFilterLikelihoodData(
                t=t,
                mag_obs=mag_obs,
                mag_err=mag_err,
                weight=weight,
                parallax_vectors=pvec,
                idx_b_sff=idx_b,
                idx_mag_src=idx_m,
            )
        )

    root_tol = 1e-8
    return JaxPhotLikelihoodContext(
        model_kind=kind,
        fitter_param_names=names,
        base_fitter_names=base_names,
        base_param_indices=base_param_indices,
        filters=tuple(filters),
        root_tol=root_tol,
    )


def log_likelihood_phot_from_vec(param_vec, ctx: JaxPhotLikelihoodContext):
    """Differentiable photometry log-likelihood for supported model layouts."""
    param_vec = jnp.asarray(param_vec, dtype=jnp.float64).reshape(-1)
    lnL = 0.0

    for phot in ctx.filters:
        fitter_vec = param_vec[jnp.array(ctx.base_param_indices)]
        b_sff = param_vec[phot.idx_b_sff]
        mag_src = param_vec[phot.idx_mag_src]
        t = jnp.asarray(phot.t, dtype=jnp.float64)
        mag_obs = jnp.asarray(phot.mag_obs, dtype=jnp.float64)
        mag_err = jnp.asarray(phot.mag_err, dtype=jnp.float64)
        pvec = (
            None
            if phot.parallax_vectors is None
            else jnp.asarray(phot.parallax_vectors, dtype=jnp.float64)
        )

        if ctx.model_kind == "pspl":
            mag_model = pspl_photometry_from_fitter_vec(
                t,
                fitter_vec,
                mag_src,
                b_sff=b_sff,
                parallax_vectors=pvec,
            )
        else:
            mag_model = psbl_photometry_from_fitter_vec(
                t,
                fitter_vec,
                mag_src,
                b_sff=b_sff,
                root_tol=ctx.root_tol,
                parallax_vectors=pvec,
            )

        lnL = lnL + phot.weight * gaussian_log_likelihood_sum(
            mag_model, mag_obs, mag_err
        )

    return lnL


def build_jax_phot_loglik_fn(fitter):
    """
    Return a ``jax.jit``-compiled ``log_likelihood(param_vec)`` or ``None``.
    """
    ctx = build_jax_phot_likelihood_context(fitter)
    if ctx is None:
        return None, None

    base_idx = jnp.array(ctx.base_param_indices, dtype=jnp.int32)

    def _loglik(param_vec):
        param_vec = jnp.asarray(param_vec, dtype=jnp.float64).reshape(-1)
        lnL = 0.0
        fitter_vec = param_vec[base_idx]
        for phot in ctx.filters:
            b_sff = param_vec[phot.idx_b_sff]
            mag_src = param_vec[phot.idx_mag_src]
            t = jnp.asarray(phot.t, dtype=jnp.float64)
            mag_obs = jnp.asarray(phot.mag_obs, dtype=jnp.float64)
            mag_err = jnp.asarray(phot.mag_err, dtype=jnp.float64)
            pvec = (
                None
                if phot.parallax_vectors is None
                else jnp.asarray(phot.parallax_vectors, dtype=jnp.float64)
            )
            if ctx.model_kind == "pspl":
                mag_model = pspl_photometry_from_fitter_vec(
                    t,
                    fitter_vec,
                    mag_src,
                    b_sff=b_sff,
                    parallax_vectors=pvec,
                )
            else:
                mag_model = psbl_photometry_from_fitter_vec(
                    t,
                    fitter_vec,
                    mag_src,
                    b_sff=b_sff,
                    root_tol=ctx.root_tol,
                    parallax_vectors=pvec,
                )
            lnL = lnL + phot.weight * gaussian_log_likelihood_sum(
                mag_model, mag_obs, mag_err
            )
        return lnL

    return jax.jit(_loglik), ctx


@dataclass(frozen=True)
class AstFilterLikelihoodData:
    """Precomputed astrometry arrays for one data set."""

    t: np.ndarray
    x_obs: np.ndarray
    y_obs: np.ndarray
    x_err: np.ndarray
    y_err: np.ndarray
    weight: float
    parallax_vectors: np.ndarray | None
    idx_b_sff: int
    phot_filt_idx: int


@dataclass(frozen=True)
class PhotAstromFilterLikelihoodData:
    """Joint phot+ast arrays for one mapped filter pair."""

    phot: PhotFilterLikelihoodData | None
    ast: AstFilterLikelihoodData


@dataclass(frozen=True)
class JaxJointLikelihoodContext:
    """Host context for PSPL PhotAstrom Param1 joint likelihood."""

    layout: str
    use_parallax: bool
    fitter_param_names: tuple[str, ...]
    base_indices: tuple[int, ...]
    filters: tuple[PhotAstromFilterLikelihoodData, ...]


def build_jax_joint_likelihood_context(fitter) -> JaxJointLikelihoodContext | None:
    """Build host context for joint photometry + astrometry JAX likelihood."""
    layout = supports_jax_joint_loglik(fitter)
    if layout is None:
        return None

    names = tuple(fitter.fitter_param_names)
    base_indices = tuple(range(len(PSPL_PHOTASTROM_PARAM1_FITTER_NAMES)))
    use_parallax = layout == "pspl_photastrom_param1" and "raL" in fitter.data
    ra_l = float(fitter.data["raL"]) if use_parallax else None
    dec_l = float(fitter.data["decL"]) if use_parallax else None
    map_phot = getattr(fitter, "map_phot_idx_to_ast_idx", [])
    joint_filters: list[PhotAstromFilterLikelihoodData] = []
    for i in range(fitter.n_ast_sets):
        ast_filt = i + 1
        phot_idx = map_phot[i] if len(map_phot) > i else i
        phot_filt = phot_idx + 1

        t_ast = np.asarray(fitter.data[f"t_ast{ast_filt}"], dtype=np.float64)
        x_obs = np.asarray(fitter.data[f"xpos{ast_filt}"], dtype=np.float64)
        y_obs = np.asarray(fitter.data[f"ypos{ast_filt}"], dtype=np.float64)
        x_err = np.asarray(fitter.data[f"xpos_err{ast_filt}"], dtype=np.float64)
        y_err = np.asarray(fitter.data[f"ypos_err{ast_filt}"], dtype=np.float64)
        ast_weight = _fitter_weight(fitter, fitter.n_phot_sets + i)
        pvec_ast = None
        if use_parallax:
            pvec_ast = precompute_parallax_vectors(ra_l, dec_l, t_ast)

        idx_b_ast = _param_index(
            names, "b_sff", phot_filt if f"b_sff{phot_filt}" in names else None
        )

        phot_block = None
        if phot_idx < fitter.n_phot_sets:
            t_phot = np.asarray(fitter.data[f"t_phot{phot_filt}"], dtype=np.float64)
            mag_obs = np.asarray(fitter.data[f"mag{phot_filt}"], dtype=np.float64)
            mag_err = np.asarray(fitter.data[f"mag_err{phot_filt}"], dtype=np.float64)
            phot_weight = _fitter_weight(fitter, phot_idx)
            pvec_phot = None
            if use_parallax:
                pvec_phot = precompute_parallax_vectors(ra_l, dec_l, t_phot)
            idx_m = _param_index(
                names, "mag_src", phot_filt if f"mag_src{phot_filt}" in names else None
            )
            phot_block = PhotFilterLikelihoodData(
                t=t_phot,
                mag_obs=mag_obs,
                mag_err=mag_err,
                weight=phot_weight,
                parallax_vectors=pvec_phot,
                idx_b_sff=idx_b_ast,
                idx_mag_src=idx_m,
            )

        ast_block = AstFilterLikelihoodData(
            t=t_ast,
            x_obs=x_obs,
            y_obs=y_obs,
            x_err=x_err,
            y_err=y_err,
            weight=ast_weight,
            parallax_vectors=pvec_ast,
            idx_b_sff=idx_b_ast,
            phot_filt_idx=phot_idx,
        )
        joint_filters.append(PhotAstromFilterLikelihoodData(phot=phot_block, ast=ast_block))

    # Photometry-only sets (no astrometry counterpart)
    mapped_phot = set(map_phot) if map_phot else set(range(fitter.n_ast_sets))
    for phot_idx in range(fitter.n_phot_sets):
        if phot_idx in mapped_phot:
            continue
        phot_filt = phot_idx + 1
        t_phot = np.asarray(fitter.data[f"t_phot{phot_filt}"], dtype=np.float64)
        mag_obs = np.asarray(fitter.data[f"mag{phot_filt}"], dtype=np.float64)
        mag_err = np.asarray(fitter.data[f"mag_err{phot_filt}"], dtype=np.float64)
        phot_weight = _fitter_weight(fitter, phot_idx)
        pvec_phot = None
        if use_parallax:
            pvec_phot = precompute_parallax_vectors(ra_l, dec_l, t_phot)
        idx_b = _param_index(
            names, "b_sff", phot_filt if f"b_sff{phot_filt}" in names else None
        )
        idx_m = _param_index(
            names, "mag_src", phot_filt if f"mag_src{phot_filt}" in names else None
        )
        phot_block = PhotFilterLikelihoodData(
            t=t_phot,
            mag_obs=mag_obs,
            mag_err=mag_err,
            weight=phot_weight,
            parallax_vectors=pvec_phot,
            idx_b_sff=idx_b,
            idx_mag_src=idx_m,
        )
        joint_filters.append(PhotAstromFilterLikelihoodData(phot=phot_block, ast=None))

    return JaxJointLikelihoodContext(
        layout=layout,
        use_parallax=use_parallax,
        fitter_param_names=names,
        base_indices=base_indices,
        filters=tuple(joint_filters),
    )


def _joint_loglik_pspl_param1(param_vec, ctx: JaxJointLikelihoodContext):
    """Joint log-likelihood for PSPL_PhotAstrom Param1 layouts."""
    param_vec = jnp.asarray(param_vec, dtype=jnp.float64).reshape(-1)
    base = param_vec[jnp.array(ctx.base_indices, dtype=jnp.int32)]
    mL, t0, beta, dL, dL_dS = base[0], base[1], base[2], base[3], base[4]
    xS0_E, xS0_N = base[5], base[6]
    muL_E, muL_N, muS_E, muS_N = base[7], base[8], base[9], base[10]

    (
        u0,
        thetaE_hat,
        tE,
        piE_E,
        piE_N,
        xS0,
        xL0,
        muS,
        muL,
        thetaE_amp,
        piS,
        piL,
    ) = derive_pspl_photastrom_param1_geometry(
        mL, t0, beta, dL, dL_dS, xS0_E, xS0_N, muL_E, muL_N, muS_E, muS_N
    )

    lnL = 0.0
    for block in ctx.filters:
        b_sff = param_vec[block.ast.idx_b_sff if block.ast else block.phot.idx_b_sff]
        pvec_ast = None
        if block.ast and block.ast.parallax_vectors is not None:
            pvec_ast = jnp.asarray(block.ast.parallax_vectors, dtype=jnp.float64)

        if block.ast is not None:
            t_ast = jnp.asarray(block.ast.t, dtype=jnp.float64)
            pos = pspl_astrometry_param1(
                t_ast,
                t0,
                xS0,
                xL0,
                muS,
                muL,
                thetaE_amp,
                b_sff,
                parallax_vectors=pvec_ast,
                piS=piS,
                piL=piL,
            )
            lnL = lnL + block.ast.weight * gaussian_astrometry_log_likelihood_sum(
                pos,
                block.ast.x_obs,
                block.ast.y_obs,
                block.ast.x_err,
                block.ast.y_err,
            )

        if block.phot is not None:
            mag_src = param_vec[block.phot.idx_mag_src]
            pvec_phot = None
            if block.phot.parallax_vectors is not None:
                pvec_phot = jnp.asarray(block.phot.parallax_vectors, dtype=jnp.float64)
            mag_model = pspl_photometry(
                jnp.asarray(block.phot.t, dtype=jnp.float64),
                t0,
                tE,
                u0,
                thetaE_hat,
                mag_src,
                b_sff=b_sff,
                parallax_vectors=pvec_phot,
                piE_E=piE_E,
                piE_N=piE_N,
            )
            lnL = lnL + block.phot.weight * gaussian_log_likelihood_sum(
                mag_model, block.phot.mag_obs, block.phot.mag_err
            )

    return lnL


def build_jax_joint_loglik_fn(fitter):
    """Return ``(jit_loglik_fn, context)`` for joint fits, or ``(None, None)``."""
    ctx = build_jax_joint_likelihood_context(fitter)
    if ctx is None:
        return None, None

    def _loglik(param_vec):
        if ctx.layout == "pspl_photastrom_param1":
            return _joint_loglik_pspl_param1(param_vec, ctx)
        raise NotImplementedError(f"JAX joint layout {ctx.layout!r} not implemented")

    return jax.jit(_loglik), ctx


def supports_jax_loglik(fitter) -> str | None:
    """Return layout id when JAX autodiff is available (registry-backed)."""
    try:
        from bagle.jax.likelihood import supports_jax_loglik_for_fitter

        return supports_jax_loglik_for_fitter(fitter)
    except ImportError:
        return None


def build_jax_loglik_fn(fitter):
    """
    Return the best available ``jax.jit`` log-likelihood and context.

    Prefers joint phot+astrometry when supported, else photometry-only.
    """
    try:
        from bagle.jax.likelihood import build_jax_loglik_fn as registry_build

        fn, ctx = registry_build(fitter)
        if fn is not None:
            return fn, ctx
    except ImportError:
        pass
    joint_fn, joint_ctx = build_jax_joint_loglik_fn(fitter)
    if joint_fn is not None:
        return joint_fn, joint_ctx
    return build_jax_phot_loglik_fn(fitter)


# ---------------------------------------------------------------------------
# Jitted entry points
# ---------------------------------------------------------------------------

pspl_amplification_jit = jax.jit(pspl_amplification)
log_likelihood_phot_from_vec_jit = jax.jit(log_likelihood_phot_from_vec, static_argnames=("ctx",))
pspl_photometry_jit = jax.jit(pspl_photometry)
pspl_photometry_from_fitter_vec_jit = jax.jit(pspl_photometry_from_fitter_vec)
psbl_photometry_jit = jax.jit(
    psbl_photometry, static_argnames=("check_sols", "rescale")
)
psbl_photometry_from_fitter_vec_jit = jax.jit(
    psbl_photometry_from_fitter_vec, static_argnames=("check_sols", "rescale")
)
