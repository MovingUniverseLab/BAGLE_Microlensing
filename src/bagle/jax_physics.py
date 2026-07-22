"""
Stateless JAX numerical kernels for microlensing models.

Pure functions suitable for ``jax.jit`` and ``jax.grad``.  Host methods in
``bagle.model_jax`` pack instance attributes and call these kernels directly.
Fitter log-likelihoods call explicit methods on the model Param mixins.

This module is the kernel library only — layout registries and string
``eval_kind`` dispatch do not belong here.  GP-enabled photometry may still
use celerite on the host.

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


def pack_fitter_params(names: Sequence[str],
    params: Mapping[str, float]) -> jnp.ndarray:
    """
    Pack named fitter parameters into a 1-D float64 vector.

    Parameters
    ----------
    names : sequence of str
        Ordered fitter parameter names.
    params : mapping
        Name -> value dictionary of fitter parameters.

    Returns
    -------
    param_vec
        See summary above.
    """
    # One float64 entry per fitter name, in the declared order.
    param_vec = jnp.array([float(params[name]) for name in names], dtype=jnp.float64)
    return param_vec


def unpack_fitter_params(names: Sequence[str], vec) -> dict[str, jnp.ndarray]:
    """
    Unpack a fitter parameter vector into a name -> scalar mapping.

    Parameters
    ----------
    names : sequence of str
        Ordered fitter parameter names.
    vec : array_like
        Packed parameter vector.

    Returns
    -------
    params
        See summary above.
    """
    vec = jnp.asarray(vec, dtype=jnp.float64).reshape(-1)

    # Guard against packing / unpacking mismatches.
    if vec.shape[0] != len(names):
        raise ValueError(
            f"Expected vector of length {len(names)}, got {vec.shape[0]}"
        )

    # Map each name to its corresponding scalar entry.
    params = {name: vec[i] for i, name in enumerate(names)}
    return params


def pack_psbl_phot_param1(params: Mapping[str, float]) -> jnp.ndarray:
    """
    Pack PSBL_PhotParam1 fitter parameters.

    Parameters
    ----------
    params : mapping
        Name -> value dictionary of fitter parameters.

    Returns
    -------
    param_vec
        See summary above.
    """
    param_vec = pack_fitter_params(PSBL_PHOT_PARAM1_FITTER_NAMES, params)
    return param_vec


def unpack_psbl_phot_param1(vec) -> dict[str, jnp.ndarray]:
    """
    Unpack PSBL_PhotParam1 fitter parameters.

    Parameters
    ----------
    vec : array_like
        Packed parameter vector.

    Returns
    -------
    params
        See summary above.
    """
    params = unpack_fitter_params(PSBL_PHOT_PARAM1_FITTER_NAMES, vec)
    return params


def pack_psbl_phot_param1_phot(params: Mapping[str, float]) -> jnp.ndarray:
    """
    Pack per-filter photometry parameters ``(b_sff, mag_src)``.

    Parameters
    ----------
    params : mapping
        Name -> value dictionary of fitter parameters.

    Returns
    -------
    param_vec
        See summary above.
    """
    param_vec = pack_fitter_params(PSBL_PHOT_PARAM1_PHOT_NAMES, params)
    return param_vec


def unpack_psbl_phot_param1_phot(vec) -> dict[str, jnp.ndarray]:
    """
    Unpack per-filter photometry parameters.

    Parameters
    ----------
    vec : array_like
        Packed parameter vector.

    Returns
    -------
    params
        See summary above.
    """
    params = unpack_fitter_params(PSBL_PHOT_PARAM1_PHOT_NAMES, vec)
    return params


def pack_pspl_phot_param1(params: Mapping[str, float]) -> jnp.ndarray:
    """
    Pack PSPL_PhotParam1 fitter parameters.

    Parameters
    ----------
    params : mapping
        Name -> value dictionary of fitter parameters.

    Returns
    -------
    param_vec
        See summary above.
    """
    param_vec = pack_fitter_params(PSPL_PHOT_PARAM1_FITTER_NAMES, params)
    return param_vec


def unpack_pspl_phot_param1(vec) -> dict[str, jnp.ndarray]:
    """
    Unpack PSPL_PhotParam1 fitter parameters.

    Parameters
    ----------
    vec : array_like
        Packed parameter vector.

    Returns
    -------
    params
        See summary above.
    """
    params = unpack_fitter_params(PSPL_PHOT_PARAM1_FITTER_NAMES, vec)
    return params


def pack_pspl_phot_param1_phot(params: Mapping[str, float]) -> jnp.ndarray:
    """
    Pack per-filter photometry parameters ``(b_sff, mag_src)``.

    Parameters
    ----------
    params : mapping
        Name -> value dictionary of fitter parameters.

    Returns
    -------
    param_vec
        See summary above.
    """
    param_vec = pack_fitter_params(PSPL_PHOT_PARAM1_PHOT_NAMES, params)
    return param_vec


def unpack_pspl_phot_param1_phot(vec) -> dict[str, jnp.ndarray]:
    """
    Unpack per-filter photometry parameters.

    Parameters
    ----------
    vec : array_like
        Packed parameter vector.

    Returns
    -------
    params
        See summary above.
    """
    params = unpack_fitter_params(PSPL_PHOT_PARAM1_PHOT_NAMES, vec)
    return params


# ---------------------------------------------------------------------------
# Parallax tables (host-side; Astropy ephemerides)
# ---------------------------------------------------------------------------


def precompute_parallax_vectors(raL: float, decL: float, t,
                                obs_location: str = "earth") -> jnp.ndarray:
    """
    Precompute parallax direction vectors on the host.

    Uses :func:`bagle.parallax.parallax_in_direction` (Astropy + JPL
    ephemerides).  The returned array has shape ``(N_times, 2)`` with
    columns ``[East, North]`` in AU, suitable as ``parallax_vectors`` in
    the jitted trajectory kernels.

    Parameters
    ----------
    raL, decL : float
        Lens right ascension and declination in degrees (J2000).
    t : array_like
        Observation times in MJD.
    obs_location : str
        Observer location passed to the parallax module (e.g. ``'earth'``,
        ``'spitzer'``, ``'jwst'``).
    """
    from bagle import parallax

    # Calculate parallax vectors for each epoch: shape (N_times, 2) = [East, North] in AU.
    parallax_vectors = np.asarray(
        parallax.parallax_in_direction(raL, decL, t, obsLocation=obs_location),
        dtype=np.float64,
    )
    return parallax_vectors


def compute_parallax_offset(parallax_vectors, piE_E, piE_N):
    """
    Microlensing parallax offset in Einstein-radius units.

    Parameters
    ----------
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).

    Returns
    -------
    parallax_offset
        See summary above.
    """
    parallax_vectors = jnp.asarray(parallax_vectors, dtype=jnp.float64)

    # Amplitude of the microlensing parallax vector.
    piE_amp = jnp.sqrt(piE_E**2 + piE_N**2)

    # Offset of the source trajectory in Einstein-radius units.
    parallax_offset = piE_amp * parallax_vectors
    return parallax_offset


# ---------------------------------------------------------------------------
# Flux helpers (JIT-safe; no host-side prints)
# ---------------------------------------------------------------------------


def mag2flux_jax(mag):
    """
    mag2flux_jax.

    Parameters
    ----------
    mag : array_like
        Magnitudes.

    Returns
    -------
    flux
        See summary above.
    """
    flux = _FLUX_ZP * 10.0 ** ((mag - _MAG_ZP) / -2.5)
    flux = jnp.nan_to_num(flux, nan=0.0)

    # Negative fluxes are unphysical; mark as NaN for downstream masking.
    flux = jnp.where(flux < 0, jnp.nan, flux)
    return flux


def flux2mag_jax(flux):
    """
    flux2mag_jax.

    Parameters
    ----------
    flux : array_like
        Fluxes.

    Returns
    -------
    mag
        See summary above.
    """
    flux = jnp.asarray(flux, dtype=jnp.float64)

    # Inverse of mag2flux_jax with the same zero-point convention.
    mag = -2.5 * jnp.log10(flux / _FLUX_ZP) + _MAG_ZP
    return mag


# ---------------------------------------------------------------------------
# PSBL geometry derived from fitter parameters
# ---------------------------------------------------------------------------


def u0_hat_from_thetaE_hat_jax(thetaE_hat, beta):
    """
    JAX version of :func:`bagle.model.u0_hat_from_thetaE_hat`.

    Parameters
    ----------
    thetaE_hat : array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    beta : float
        Signed source-lens impact parameter (mas or Einstein radii).

    Returns
    -------
    u0_hat
        See summary above.
    """
    thetaE_hat = jnp.asarray(thetaE_hat, dtype=jnp.float64).reshape(2)
    beta = jnp.asarray(beta, dtype=jnp.float64)

    # Same-sign East/North components of thetaE_hat change the u0 orientation.
    sign_prod_pos = jnp.sign(thetaE_hat[0]) * jnp.sign(thetaE_hat[1]) > 0

    # Candidate u0_hat for beta > 0 (and the opposite for beta < 0).
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

    # Sign of beta selects which hemisphere for the closest-approach vector.
    u0_hat = jnp.where(beta > 0, u0_pos, u0_neg)
    return u0_hat


def derive_pspl_static_geometry(u0_amp, piE_E, piE_N):
    """
    Derive static PSPL geometry from fitter parameters.

    Parameters
    ----------
    u0_amp : float
        Signed impact parameter amplitude in Einstein radii.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).

    Returns
    -------
    geometry
        See summary above.
    """
    # Microlensing parallax vector and its amplitude.
    piE = jnp.stack([piE_E, piE_N])
    piE_amp = jnp.linalg.norm(piE)

    # Relative proper-motion direction (same as piE direction for PSPL).
    thetaE_hat = piE / piE_amp

    # Closest-approach vector in the Einstein ring.
    u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, u0_amp)
    u0 = jnp.abs(u0_amp) * u0_hat

    return u0, thetaE_hat, piE_amp


def derive_psbl_static_geometry(u0_amp, piE_E, piE_N, q, sep, phi):
    """
    Derive static PSBL geometry from fitter parameters.

    Parameters
    ----------
    u0_amp : float
        Signed impact parameter amplitude in Einstein radii.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).
    q : float
        Binary mass ratio ``m2/m1``.
    sep : float
        Binary projected separation in Einstein radii.
    phi : float
        Binary orientation angle (degrees).

    Returns
    -------
    geometry
        See summary above.
    """
    # Same PSPL-style trajectory geometry for the source.
    piE = jnp.stack([piE_E, piE_N])
    piE_amp = jnp.linalg.norm(piE)
    thetaE_hat = piE / piE_amp
    u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, u0_amp)
    u0 = jnp.abs(u0_amp) * u0_hat

    # Binary axis orientation relative to the piE direction.
    phi_rad = phi * _DEG2RAD
    phi_piE_rad = jnp.arctan2(piE_E, piE_N)
    phi_rho1_rad = phi_piE_rad + phi_rad

    # Primary / secondary lens positions about the binary midpoint.
    xL1_over_theta = jnp.stack(
        [
            0.5 * sep * jnp.sin(phi_rho1_rad),
            0.5 * sep * jnp.cos(phi_rho1_rad),
        ]
    )
    xL2_over_theta = -xL1_over_theta

    # Mass fractions for the binary lens equation.
    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)

    return m1, m2, u0, thetaE_hat, xL1_over_theta, xL2_over_theta, piE_amp


def einstein_source_position(t, t0, tE, u0, thetaE_hat, parallax_vectors=None,
                             piE_E=None, piE_N=None, parallax_correction=None):
    """
    Unlensed source–lens separation in Einstein-radius units.

    Parameters
    ----------
    t : jnp.array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : jnp.array_like
        Source-lens separation at ``t0`` (Einstein radii), shape ``(2,)``.
    thetaE_hat : jnp.array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    parallax_vectors : jnp.array_like, optional
        Precomputed parallax table from :func:`precompute_parallax_vectors`,
        shape ``(N_times, 2)``.  Combined with ``piE_E`` and ``piE_N`` inside
        the jitted kernel so gradients w.r.t. parallax parameters are available.
    piE_E, piE_N : float, optional
        Microlensing parallax components in Einstein-radius units.  Required
        when ``parallax_vectors`` is provided.
    parallax_correction : jnp.array_like, optional
        Legacy alias for a pre-multiplied table ``piE_amp * parallax_vectors``.
        When provided, ``parallax_vectors`` / ``piE_*`` are ignored.
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    u0 = jnp.asarray(u0, dtype=jnp.float64).reshape(2)
    thetaE_hat = jnp.asarray(thetaE_hat, dtype=jnp.float64).reshape(2)

    # tau: time in Einstein-crossing units; shape [N_times, 1].
    tau = ((t - t0) / tE).reshape(-1, 1)

    # Rectilinear source-lens separation; shape [N_times, 2].
    u = u0.reshape(1, 2) + tau * thetaE_hat.reshape(1, 2)

    # Optional parallax deflection of the trajectory.
    if parallax_correction is not None:
        u = u - jnp.asarray(parallax_correction, dtype=jnp.float64)
    elif parallax_vectors is not None:
        u = u - compute_parallax_offset(parallax_vectors, piE_E, piE_N)

    return u


def pspl_u(t, t0, tE, u0, thetaE_hat, 
           parallax_vectors=None, 
           piE_E=None, piE_N=None, 
           parallax_correction=None):
    """
    PSPL separation vector ``u(t)`` with shape ``(N_times, 2)``.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Source-lens separation at ``t0`` (Einstein radii), shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).
    parallax_correction : array_like or None
        Legacy pre-multiplied ``piE_amp * parallax_vectors``.

    Returns
    -------
    u
        See summary above.
    """
    # Thin wrapper: PSPL uses the same Einstein-frame source trajectory.
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
    return u


def psbl_source_position(t, t0, tE, u0, thetaE_hat, parallax_vectors=None,
    piE_E=None, piE_N=None, parallax_correction=None):
    """
    Unlensed source position as a complex array (East + i North).

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Source-lens separation at ``t0`` (Einstein radii), shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).
    parallax_correction : array_like or None
        Legacy pre-multiplied ``piE_amp * parallax_vectors``.

    Returns
    -------
    w
        See summary above.
    """
    # Real-valued source track in Einstein radii (East, North).
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

    # Pack into complex w = East + i North for the Witt quintic.
    w = u[:, 0] + 1j * u[:, 1]
    return w


def psbl_static_lens_positions(xL1_over_theta, xL2_over_theta, n_times):
    """
    Broadcast static lens positions to ``n_times`` complex arrays.

    Parameters
    ----------
    xL1_over_theta : array_like
        Primary lens position in Einstein radii, shape ``(2,)``.
    xL2_over_theta : array_like
        Secondary lens position in Einstein radii, shape ``(2,)``.
    n_times : int
        Number of epochs to broadcast to.

    Returns
    -------
    lens_pos
        See summary above.
    """
    xL1 = jnp.asarray(xL1_over_theta, dtype=jnp.float64).reshape(2)
    xL2 = jnp.asarray(xL2_over_theta, dtype=jnp.float64).reshape(2)

    # Complex lens positions (East + i North), constant in time for static PSBL.
    z1 = jnp.full(n_times, xL1[0] + 1j * xL1[1], dtype=jnp.complex128)
    z2 = jnp.full(n_times, xL2[0] + 1j * xL2[1], dtype=jnp.complex128)

    return z1, z2


def psbl_complex_pos_static(t, t0, tE, u0, thetaE_hat, xL1_over_theta,
    xL2_over_theta, parallax_vectors=None, piE_E=None, piE_N=None,
    parallax_correction=None):
    """
    Source and static binary-lens positions as complex arrays.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Source-lens separation at ``t0`` (Einstein radii), shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    xL1_over_theta : array_like
        Primary lens position in Einstein radii, shape ``(2,)``.
    xL2_over_theta : array_like
        Secondary lens position in Einstein radii, shape ``(2,)``.
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).
    parallax_correction : array_like or None
        Legacy pre-multiplied ``piE_amp * parallax_vectors``.

    Returns
    -------
    complex_pos
        See summary above.
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)

    # Complex source trajectory (with optional microlensing parallax).
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

    # Static binary lenses, broadcast to the same number of epochs.
    z1, z2 = psbl_static_lens_positions(xL1_over_theta, xL2_over_theta, w.shape[0])
    return w, z1, z2


def psbl_keplerian_lens_positions(t, w, o, i, e, p, tp, aleph, aleph_sec):
    """
    Time-varying PSBL phot orbit lens positions (Einstein-radius units).

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    w : array_like
        Complex source position(s) or orbit argument of periapsis.
    o : float
        Orbit ascending-node angle.
    i : float
        Orbit inclination.
    e : float
        Orbit eccentricity.
    p : float
        Orbit period.
    tp : float
        Time of periapsis.
    aleph : float
        Primary semi-major axis (Einstein radii).
    aleph_sec : float
        Secondary semi-major axis (Einstein radii).

    Returns
    -------
    lens_pos
        See summary above.
    """
    from bagle.jax.orbits import oal2xy

    # Cartesian orbit positions for primary and secondary.
    x, y, x2, y2 = oal2xy(t, w, o, i, e, p, tp, aleph, aleph_sec)

    # Complex lens positions for the Witt quintic.
    z1 = x + 1j * y
    z2 = x2 + 1j * y2

    return z1, z2


def psbl_complex_pos_keplerian(t, t0, tE, u0, thetaE_hat, w, o, i, e, p, tp,
    aleph, aleph_sec, parallax_vectors=None, piE_E=None, piE_N=None,
    parallax_correction=None):
    """
    Source and Keplerian binary-lens positions as complex arrays.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Source-lens separation at ``t0`` (Einstein radii), shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    w : array_like
        Complex source position(s) or orbit argument of periapsis.
    o : float
        Orbit ascending-node angle.
    i : float
        Orbit inclination.
    e : float
        Orbit eccentricity.
    p : float
        Orbit period.
    tp : float
        Time of periapsis.
    aleph : float
        Primary semi-major axis (Einstein radii).
    aleph_sec : float
        Secondary semi-major axis (Einstein radii).
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).
    parallax_correction : array_like or None
        Legacy pre-multiplied ``piE_amp * parallax_vectors``.

    Returns
    -------
    complex_pos
        See summary above.
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)

    # Complex source trajectory (with optional microlensing parallax).
    src_w = psbl_source_position(
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

    # Time-varying lens positions from the Keplerian phot orbit.
    z1, z2 = psbl_keplerian_lens_positions(t, w, o, i, e, p, tp, aleph, aleph_sec)
    return src_w, z1, z2


# ---------------------------------------------------------------------------
# PSPL amplification and photometry
# ---------------------------------------------------------------------------


def pspl_amplification_from_u(u):
    """
    Total PSPL amplification from separation vectors.

    Parameters
    ----------
    u : array_like
        Source-lens separation vectors, shape ``(N_times, 2)``.

    Returns
    -------
    amp
        See summary above.
    """
    u = jnp.asarray(u, dtype=jnp.float64)

    # Separation amplitude; shape [N_times].
    u_amp = jnp.linalg.norm(u, axis=1)

    # Point-source point-lens magnification formula.
    amp = (u_amp**2 + 2) / (u_amp * jnp.sqrt(u_amp**2 + 4))
    return amp


def pspl_amplification(t, t0, tE, u0, thetaE_hat, parallax_vectors=None,
    piE_E=None, piE_N=None, parallax_correction=None):
    """
    Total PSPL amplification at times ``t``.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Source-lens separation at ``t0`` (Einstein radii), shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).
    parallax_correction : array_like or None
        Legacy pre-multiplied ``piE_amp * parallax_vectors``.

    Returns
    -------
    amp
        See summary above.
    """
    # Source-lens separation, then PSPL magnification formula.
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
    amp = pspl_amplification_from_u(u)
    return amp


def pspl_photometry_from_amp(amp, mag_src, b_sff=None):
    """
    Unresolved PSPL magnitude from total amplification.

    Parameters
    ----------
    amp : array_like
        Total magnification.
    mag_src : float
        Unlensed source magnitude.
    b_sff : float or None
        Source flux fraction (blend parameter).

    Returns
    -------
    mag
        See summary above.
    """
    flux_src = mag2flux_jax(mag_src)

    # Magnified source flux.
    flux_model = flux_src * amp

    # Optional blend / neighbor contribution via source flux fraction.
    if b_sff is not None:
        flux_model = flux_model + flux_src * (1.0 - b_sff) / b_sff

    mag = flux2mag_jax(flux_model)
    return mag


def pspl_photometry(t, t0, tE, u0, thetaE_hat, mag_src, b_sff=None,
    parallax_vectors=None, piE_E=None, piE_N=None, parallax_correction=None):
    """
    PSPL unresolved photometry at times ``t``.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Source-lens separation at ``t0`` (Einstein radii), shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    mag_src : float
        Unlensed source magnitude.
    b_sff : float or None
        Source flux fraction (blend parameter).
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).
    parallax_correction : array_like or None
        Legacy pre-multiplied ``piE_amp * parallax_vectors``.

    Returns
    -------
    mag
        See summary above.
    """
    # Amplification first, then convert to unresolved magnitudes.
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
    mag = pspl_photometry_from_amp(amp, mag_src, b_sff=b_sff)
    return mag


def pspl_resolved_amplification_from_u(u):
    """
    Plus/minus PSPL image amplifications from separation ``u`` (N, 2).

    Parameters
    ----------
    u : array_like
        Source-lens separation vectors, shape ``(N_times, 2)``.

    Returns
    -------
    amp_pm
        See summary above.
    """
    u = jnp.asarray(u, dtype=jnp.float64)

    # Separation amplitude and common sqrt term for image magnifications.
    u_amp = jnp.linalg.norm(u, axis=1)
    sqrt_term = jnp.sqrt(u_amp**2 + 4.0)

    # Plus / minus image amplifications.
    a_plus = 0.5 * ((u_amp**2 + 2.0) / (u_amp * sqrt_term) + 1.0)
    a_minus = 0.5 * ((u_amp**2 + 2.0) / (u_amp * sqrt_term) - 1.0)

    return a_plus, a_minus


def pspl_resolved_astrometry_from_u(u):
    """
    Plus/minus PSPL image positions in Einstein radii.

    Parameters
    ----------
    u : array_like
        Source-lens separation vectors, shape ``(N_times, 2)``.

    Returns
    -------
    u_pm
        See summary above.
    """
    u = jnp.asarray(u, dtype=jnp.float64)

    # Unit vector along the source-lens separation.
    u_amp = jnp.linalg.norm(u, axis=1, keepdims=True)
    u_hat = u / u_amp
    sqrt_term = jnp.sqrt(u_amp**2 + 4.0)

    # Image positions in Einstein radii (relative to the lens).
    u_plus = ((u_amp + sqrt_term) / 2.0) * u_hat
    u_minus = ((u_amp - sqrt_term) / 2.0) * u_hat

    return u_plus, u_minus


def pspl_phot_astrometry(t, t0, tE, u0, thetaE_hat, mag_src, b_sff,
    parallax_vectors=None, piE_E=None, piE_N=None, parallax_correction=None):
    """
    PSPL_Phot flux-weighted unresolved centroid in Einstein radii.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Source-lens separation at ``t0`` (Einstein radii), shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    mag_src : float
        Unlensed source magnitude.
    b_sff : float or None
        Source flux fraction (blend parameter).
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).
    parallax_correction : array_like or None
        Legacy pre-multiplied ``piE_amp * parallax_vectors``.

    Returns
    -------
    u_cent
        See summary above.
    """
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

    # Resolved image positions and amplifications in Einstein radii.
    u_plus, u_minus = pspl_resolved_astrometry_from_u(u)
    a_plus, a_minus = pspl_resolved_amplification_from_u(u)
    a_total = a_plus + a_minus

    # Amplification-weighted image centroid (still relative to the lens).
    u_cent = (
        u_plus * a_plus[:, jnp.newaxis] + u_minus * a_minus[:, jnp.newaxis]
    ) / a_total[:, jnp.newaxis]

    # Source and blend fluxes; blend light is centered on the lens (u=0).
    f_src = mag2flux_jax(mag_src)
    f_l = f_src * (1.0 - b_sff) / b_sff

    # Flux-weighted unresolved centroid including blend.
    u_cent = (u_cent * f_src * a_total[:, jnp.newaxis]) / (
        f_src * a_total[:, jnp.newaxis] + f_l
    )
    return u_cent


def pspl_phot_astrometry_unlensed(t, t0, tE, u0, thetaE_hat, b_sff,
    parallax_vectors=None, piE_E=None, piE_N=None, parallax_correction=None):
    """
    Unlensed PSPL_Phot flux-weighted centroid in Einstein radii.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Source-lens separation at ``t0`` (Einstein radii), shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    b_sff : float or None
        Source flux fraction (blend parameter).
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).
    parallax_correction : array_like or None
        Legacy pre-multiplied ``piE_amp * parallax_vectors``.

    Returns
    -------
    u_cent
        See summary above.
    """
    # Source-lens separation, then flux-weighted centroid with blend on lens.
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

    # Without lensing, the centroid is just the blend-weighted source track.
    u_cent = jnp.asarray(b_sff, dtype=jnp.float64) * u
    return u_cent


def pspl_linear_astrometry(t, t0, x0, mu, parallax_vectors=None, pi=None):
    """
    Linear sky motion in arcsec (PSPL source or lens).

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    x0 : array_like
        Sky position at ``t0`` (arcsec), shape ``(2,)``.
    mu : array_like
        Proper motion (mas/yr), shape ``(2,)``.
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    pi : float or None
        Parallax amplitude applied to ``parallax_vectors`` (mas).

    Returns
    -------
    pos
        See summary above.
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)

    # Time since t0 in years; shape [N_times, 1].
    dt = ((t - t0) / _DAYS_PER_YEAR).reshape(-1, 1)

    # Linear proper motion; convert mas/yr -> arcsec/yr with 1e-3.
    pos = x0.reshape(1, 2) + dt * mu.reshape(1, 2) * 1e-3

    # Optional annual parallax shift (mas -> arcsec).
    if parallax_vectors is not None and pi is not None:
        pos = pos + jnp.asarray(pi, dtype=jnp.float64) * jnp.asarray(
            parallax_vectors, dtype=jnp.float64
        ) * 1e-3

    return pos


def pspl_source_astrometry_unlensed(t, t0, xS0, muS, 
                                    parallax_vectors=None, piS=None):
    """
    Unlensed source astrometry in arcsec.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    xS0 : array_like
        Source sky position at ``t0`` (arcsec), shape ``(2,)``.
    muS : array_like
        Source proper motion (mas/yr), shape ``(2,)``.
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piS : float or None
        Source parallax (mas).

    Returns
    -------
    pos
        See summary above.
    """
    # Thin wrapper around the shared linear sky-motion kernel.
    pos = pspl_linear_astrometry(t, t0, xS0, muS, parallax_vectors, piS)
    return pos


def pspl_resolved_amplification(t, t0, tE, u0, thetaE_hat, 
                                parallax_vectors=None, piE_E=None, piE_N=None, 
                                parallax_correction=None):
    """
    Plus/minus PSPL amplifications; shape ``(2, N_times)``.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Source-lens separation at ``t0`` (Einstein radii), shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).
    parallax_correction : array_like or None
        Legacy pre-multiplied ``piE_amp * parallax_vectors``.

    Returns
    -------
    amp_pm
        See summary above.
    """
    # Separation vector, then plus/minus image magnifications.
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
    a_plus, a_minus = pspl_resolved_amplification_from_u(u)

    # Stack as (2, N_times) to match the NumPy API.
    amp_pm = jnp.stack((a_plus, a_minus))
    return amp_pm


def pspl_resolved_astrometry(t, t0, tE, u0, thetaE_hat, xL0, muL, thetaE_amp,
    parallax_vectors=None, piE_E=None, piE_N=None, piL=None,
    parallax_correction=None):
    """
    Plus/minus PSPL image astrometry in arcsec; shape ``(2, N_times, 2)``.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Source-lens separation at ``t0`` (Einstein radii), shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    xL0 : array_like
        Lens sky position at ``t0`` (arcsec), shape ``(2,)``.
    muL : array_like
        Lens proper motion (mas/yr), shape ``(2,)``.
    thetaE_amp : float
        Einstein radius amplitude (mas).
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).
    piL : float or None
        Lens parallax (mas).
    parallax_correction : array_like or None
        Legacy pre-multiplied ``piE_amp * parallax_vectors``.

    Returns
    -------
    pos_images
        See summary above.
    """
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

    # Image positions relative to the lens (Einstein radii).
    u_plus, u_minus = pspl_resolved_astrometry_from_u(u)

    # Lens sky track and Einstein radius in arcsec.
    xL = pspl_linear_astrometry(t, t0, xL0, muL, parallax_vectors, piL)
    scale = jnp.asarray(thetaE_amp, dtype=jnp.float64) * 1e-3

    # Absolute image positions on the sky; shape (2, N_times, 2).
    pos_images = jnp.stack((xL + u_plus * scale, xL + u_minus * scale))
    return pos_images


def gaussian_chi2_photometry(mag_model, mag_obs, mag_err):
    """
    Per-point photometric chi^2.

    Parameters
    ----------
    mag_model : array_like
        Model magnitudes.
    mag_obs : array_like
        Observed magnitudes.
    mag_err : array_like
        Magnitude uncertainties.

    Returns
    -------
    chi2
        See summary above.
    """
    mag_model = jnp.asarray(mag_model, dtype=jnp.float64)
    mag_obs = jnp.asarray(mag_obs, dtype=jnp.float64)
    mag_err = jnp.asarray(mag_err, dtype=jnp.float64)

    # Per-point photometric chi^2 (no sum).
    chi2 = ((mag_obs - mag_model) / mag_err) ** 2
    return chi2


def gaussian_log_likelihood_photometry_each(mag_model, mag_obs, mag_err):
    """
    Per-point photometric ln(likelihood) including normalization.

    Parameters
    ----------
    mag_model : array_like
        Model magnitudes.
    mag_obs : array_like
        Observed magnitudes.
    mag_err : array_like
        Magnitude uncertainties.

    Returns
    -------
    lnL
        See summary above.
    """
    chi2 = gaussian_chi2_photometry(mag_model, mag_obs, mag_err)

    # Gaussian normalization term for each datum.
    lnL_const = -0.5 * jnp.log(2.0 * jnp.pi * mag_err**2)
    lnL = (-0.5 * chi2) + lnL_const
    return lnL


def gaussian_chi2_astrometry(pos_model, x_obs, y_obs, x_err, y_err):
    """
    Per-point joint x/y astrometric chi^2.

    Parameters
    ----------
    pos_model : array_like
        Model sky positions, shape ``(N_times, 2)``.
    x_obs : array_like
        Observed RA positions (arcsec).
    y_obs : array_like
        Observed Dec positions (arcsec).
    x_err : array_like
        RA uncertainties (arcsec).
    y_err : array_like
        Dec uncertainties (arcsec).

    Returns
    -------
    chi2
        See summary above.
    """
    pos_model = jnp.asarray(pos_model, dtype=jnp.float64)
    x_obs = jnp.asarray(x_obs, dtype=jnp.float64)
    y_obs = jnp.asarray(y_obs, dtype=jnp.float64)
    x_err = jnp.asarray(x_err, dtype=jnp.float64)
    y_err = jnp.asarray(y_err, dtype=jnp.float64)

    # Separate East / North chi^2 contributions.
    chi2_x = ((x_obs - pos_model[:, 0]) / x_err) ** 2
    chi2_y = ((y_obs - pos_model[:, 1]) / y_err) ** 2
    chi2 = chi2_x + chi2_y
    return chi2


def gaussian_log_likelihood_astrometry_each(pos_model, x_obs, y_obs, x_err, y_err):
    """
    Per-point astrometric ln(likelihood) including normalization.

    Parameters
    ----------
    pos_model : array_like
        Model sky positions, shape ``(N_times, 2)``.
    x_obs : array_like
        Observed RA positions (arcsec).
    y_obs : array_like
        Observed Dec positions (arcsec).
    x_err : array_like
        RA uncertainties (arcsec).
    y_err : array_like
        Dec uncertainties (arcsec).

    Returns
    -------
    lnL
        See summary above.
    """
    chi2 = gaussian_chi2_astrometry(pos_model, x_obs, y_obs, x_err, y_err)

    # Separate East / North Gaussian normalization terms.
    lnL_const_x = -0.5 * jnp.log(2.0 * jnp.pi * x_err**2)
    lnL_const_y = -0.5 * jnp.log(2.0 * jnp.pi * y_err**2)
    lnL = (-0.5 * chi2) + lnL_const_x + lnL_const_y
    return lnL


def pspl_photometry_from_fitter_vec(t, fitter_vec, mag_src, b_sff=None,
    parallax_vectors=None):
    """
    PSPL photometry from a packed PSPL_PhotParam1 fitter vector.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    fitter_vec : array_like
        Packed fitter parameter vector.
    mag_src : float
        Unlensed source magnitude.
    b_sff : float or None
        Source flux fraction (blend parameter).
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.

    Returns
    -------
    mag
        See summary above.
    """
    # Unpack PSPL_PhotParam1 and derive Einstein-frame geometry.
    params = unpack_pspl_phot_param1(fitter_vec)
    u0, thetaE_hat, _ = derive_pspl_static_geometry(
        params["u0_amp"],
        params["piE_E"],
        params["piE_N"],
    )

    # Forward unresolved photometry with optional parallax.
    mag = pspl_photometry(
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
    return mag


# ---------------------------------------------------------------------------
# Quintic root solver (Witt / BAGLE coefficients)
# ---------------------------------------------------------------------------


def quintic_coefficients(w, z1, z2, m1, m2):
    """
    Return quintic coefficients ``(a5, a4, a3, a2, a1, a0)`` high-to-low.

    Parameters
    ----------
    w : array_like
        Complex source position(s) or orbit argument of periapsis.
    z1 : array_like
        Complex primary lens position(s).
    z2 : array_like
        Complex secondary lens position(s).
    m1 : float
        Primary mass fraction (or physical mass where noted).
    m2 : float
        Secondary mass fraction (or physical mass where noted).

    Returns
    -------
    coeffs
        See summary above.
    """
    w = jnp.asarray(w, dtype=jnp.complex128).reshape(-1)
    z1 = jnp.asarray(z1, dtype=jnp.complex128).reshape(-1)
    z2 = jnp.asarray(z2, dtype=jnp.complex128).reshape(-1)
    m1 = jnp.asarray(m1, dtype=jnp.float64)
    m2 = jnp.asarray(m2, dtype=jnp.float64)

    # Conjugates enter the Witt binary-lens quintic.
    wbar = jnp.conj(w)
    z1bar = jnp.conj(z1)
    z2bar = jnp.conj(z2)

    # Coefficients a5..a0 of the complex quintic (high degree to constant).
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
    """
    Solve a single quintic via the companion matrix (scalar coefficients).

    Parameters
    ----------
    a5 : complex
        Quintic coefficient of ``z^5``.
    a4 : complex
        Quintic coefficient of ``z^4``.
    a3 : complex
        Quintic coefficient of ``z^3``.
    a2 : complex
        Quintic coefficient of ``z^2``.
    a1 : complex
        Quintic coefficient of ``z^1``.
    a0 : complex
        Quintic coefficient of ``z^0``.

    Returns
    -------
    roots
        See summary above.
    """
    # Companion matrix of the monic quintic; eigenvalues are the roots.
    C = jnp.complex128(
        [
            [-a4 / a5, -a3 / a5, -a2 / a5, -a1 / a5, -a0 / a5],
            [1.0 + 0j, 0, 0, 0, 0],
            [0, 1.0 + 0j, 0, 0, 0],
            [0, 0, 1.0 + 0j, 0, 0],
            [0, 0, 0, 1.0 + 0j, 0],
        ]
    )
    roots = jnp.linalg.eigvals(C)
    return roots


_vmap_quintic_roots = jax.vmap(
    quintic_roots_companion, in_axes=(0, 0, 0, 0, 0, 0)
)


def _psbl_invalid_root_mask(z_arr, w, z1, z2, m1, m2, root_tol):
    """
    Boolean mask of Witt roots that fail the complex lens equation.

    Parameters
    ----------
    z_arr : array_like
        Complex image positions, shape ``(N_times, N_images)``.
    w : array_like
        Complex source position(s).
    z1 : array_like
        Complex primary lens position(s).
    z2 : array_like
        Complex secondary lens position(s).
    m1 : float
        Primary mass fraction (or physical mass where noted).
    m2 : float
        Secondary mass fraction (or physical mass where noted).
    root_tol : float
        Lens-equation root tolerance.

    Returns
    -------
    bad : jnp.ndarray, dtype=bool, shape (N_times, N_images)
        True where the lens-equation residual exceeds ``root_tol``.
    """
    n = w.shape[0]

    # Broadcast scalar masses / tolerance to per-epoch arrays when needed.
    m1_arr = m1 if jnp.ndim(m1) else jnp.full((n,), m1)
    m2_arr = m2 if jnp.ndim(m2) else jnp.full((n,), m2)
    tol = root_tol if jnp.ndim(root_tol) else jnp.full((n,), root_tol)

    # Residual of the complex binary lens equation at each candidate root.
    diff = w[:, jnp.newaxis] - (
        z_arr
        - m1_arr[:, jnp.newaxis] / jnp.conj(z_arr - z1[:, jnp.newaxis])
        - m2_arr[:, jnp.newaxis] / jnp.conj(z_arr - z2[:, jnp.newaxis])
    )

    return jnp.abs(diff) > tol[:, jnp.newaxis]


def _mask_psbl_roots(z_arr, w, z1, z2, m1, m2, root_tol):
    """
    Replace invalid Witt roots with NaN (NumPy-host parity helper).

    Notes
    -----
    Prefer :func:`_psbl_invalid_root_mask` + zeroing amplifications inside
    :func:`psbl_all_arrays` for autodiff-safe likelihood evaluations.
    """
    bad = _psbl_invalid_root_mask(z_arr, w, z1, z2, m1, m2, root_tol)
    return jnp.where(bad, jnp.nan + 0j, z_arr)


@jax.custom_jvp
def _zero_nonfinite(x):
    """Replace non-finite values with zero without poisoning gradients."""
    return jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x))


@_zero_nonfinite.defjvp
def _zero_nonfinite_jvp(primals, tangents):
    """JVP for :func:`_zero_nonfinite`."""
    (x,) = primals
    (t,) = tangents

    # Forward: drop NaN/Inf; backward: treat those entries as locally constant.
    y = jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x))
    dy = jnp.where(jnp.isfinite(x), t, jnp.zeros_like(t))
    return y, dy


def psbl_image_positions(w, z1, z2, m1, m2, root_tol, check_sols: bool):
    """
    Binary-lens image positions from the Witt quintic (companion-matrix roots).

    Parameters
    ----------
    root_tol : float or array
        Lens-equation tolerance; may be per-epoch when rescaling is used.
    check_sols : bool
        When ``True``, mask roots that fail the lens equation with NaN.
        Autodiff-safe callers should prefer ``check_sols=False`` here and
        zero invalid amplifications in :func:`psbl_all_arrays`.
    """
    a5, a4, a3, a2, a1, a0 = quintic_coefficients(w, z1, z2, m1, m2)

    # Solve one quintic per epoch (vmap over companion-matrix eigvals).
    z_arr = jnp.asarray(_vmap_quintic_roots(a5, a4, a3, a2, a1, a0))

    def _mask(z_arr):
        images = _mask_psbl_roots(z_arr, w, z1, z2, m1, m2, root_tol)
        return images

    # Optionally discard roots that fail the lens equation.
    images = jax.lax.cond(check_sols, _mask, lambda x: x, z_arr)
    return images


psbl_image_positions_jit = jax.jit(psbl_image_positions, static_argnames=("check_sols",))


# ---------------------------------------------------------------------------
# Rescaling, amplification, photometry
# ---------------------------------------------------------------------------

def rescale_complex_pos(w, z1, z2, m1, m2):
    """
    Center and scale complex positions into roughly a 1 x 1 box.

    Parameters
    ----------
    w : array_like
        Complex source position(s) or orbit argument of periapsis.
    z1 : array_like
        Complex primary lens position(s).
    z2 : array_like
        Complex secondary lens position(s).
    m1 : float
        Primary mass fraction (or physical mass where noted).
    m2 : float
        Secondary mass fraction (or physical mass where noted).

    Returns
    -------
    scaled
        See summary above.
    """
    w = jnp.asarray(w, dtype=jnp.complex128)
    z1 = jnp.asarray(z1, dtype=jnp.complex128)
    z2 = jnp.asarray(z2, dtype=jnp.complex128)
    m1 = jnp.asarray(m1, dtype=jnp.float64)
    m2 = jnp.asarray(m2, dtype=jnp.float64)

    # Stack source and lens positions; shift to the centroid of each epoch.
    pos = jnp.vstack([w, z1, z2]).T
    shift = jnp.average(pos, axis=1)
    s = shift[:, jnp.newaxis] if w.ndim > 1 else shift
    w = w - s
    z1 = z1 - s
    z2 = z2 - s

    # Scale so the bounding box is roughly unit size (improves root finding).
    pr, pi = jnp.real(pos), jnp.imag(pos)
    xscale = jnp.max(pr, axis=1) - jnp.min(pr, axis=1)
    yscale = jnp.max(pi, axis=1) - jnp.min(pi, axis=1)
    xyscale = jnp.stack([xscale, yscale], axis=1)
    scale = 1.0 / jnp.max(xyscale, axis=1)
    sc = scale[:, jnp.newaxis] if w.ndim > 1 else scale
    w = w * sc
    z1 = z1 * sc
    z2 = z2 * sc

    # Masses scale as length^2 under this coordinate transform.
    m1 = m1 * (scale**2)
    m2 = m2 * (scale**2)

    return w, z1, z2, m1, m2, scale, shift


def psbl_amp_arr(z_arr, z1, z2, m1, m2):
    """
    Magnification of each image from the binary-lens Jacobian.

    Parameters
    ----------
    z_arr : array_like
        Complex image positions, shape ``(N_times, N_images)``.
    z1 : array_like
        Complex primary lens position(s).
    z2 : array_like
        Complex secondary lens position(s).
    m1 : float
        Primary mass fraction (or physical mass where noted).
    m2 : float
        Secondary mass fraction (or physical mass where noted).

    Returns
    -------
    amp_arr
        See summary above.
    """
    n_times = z1.shape[0]
    m1 = jnp.asarray(m1, dtype=jnp.float64)
    m2 = jnp.asarray(m2, dtype=jnp.float64)

    # Complex derivative of the lens mapping (Jacobian determinant pieces).
    dwbardz = m1 / (z_arr - z1.reshape((n_times, 1))) ** 2
    dwbardz += m2 / (z_arr - z2.reshape((n_times, 1))) ** 2
    jacobian = 1.0 - jnp.abs(dwbardz) ** 2

    # Image magnification is 1 / |det J|.
    amp_arr = 1.0 / jnp.abs(jacobian)
    return amp_arr


def psbl_all_arrays(w, z1, z2, m1, m2, root_tol, check_sols: bool = True, rescale: bool = True):
    """
    Image positions and per-image amplifications.

    Parameters
    ----------
    w : array_like
        Complex source position(s) or orbit argument of periapsis.
    z1 : array_like
        Complex primary lens position(s).
    z2 : array_like
        Complex secondary lens position(s).
    m1 : float
        Primary mass fraction (or physical mass where noted).
    m2 : float
        Secondary mass fraction (or physical mass where noted).
    root_tol : float
        Lens-equation root tolerance.
    check_sols : bool
        If True, mask roots that fail the lens equation.
    rescale : bool
        If True, rescale complex positions before root finding.

    Returns
    -------
    images_and_amps
        See summary above.
    """
    w = jnp.asarray(w, dtype=jnp.complex128)
    z1 = jnp.asarray(z1, dtype=jnp.complex128)
    z2 = jnp.asarray(z2, dtype=jnp.complex128)
    m1_phys = jnp.asarray(m1, dtype=jnp.float64)
    m2_phys = jnp.asarray(m2, dtype=jnp.float64)

    # Keep raw companion-matrix roots (no NaN masking) so amplifications stay
    # autodiff-safe. Invalid roots are zeroed via the lens-equation residual.
    if rescale:
        # Solve in a scaled frame, then map images back to physical units.
        rw, rz1, rz2, rm1, rm2, scale, shift = rescale_complex_pos(
            w, z1, z2, m1_phys, m2_phys
        )
        rt = root_tol * scale if jnp.ndim(root_tol) else root_tol * scale
        rimages = psbl_image_positions_jit(
            rw, rz1, rz2, rm1, rm2, rt, False
        )
        images = (rimages / scale.reshape(-1, 1)) + shift.reshape(-1, 1)
        amps = psbl_amp_arr(images, z1, z2, m1_phys, m2_phys)
        if check_sols:
            bad = _psbl_invalid_root_mask(
                rimages, rw, rz1, rz2, rm1, rm2, rt
            )
            amps = jnp.where(bad, 0.0, amps)
    else:
        images = psbl_image_positions_jit(
            w, z1, z2, m1_phys, m2_phys, root_tol, False
        )
        amps = psbl_amp_arr(images, z1, z2, m1_phys, m2_phys)
        if check_sols:
            bad = _psbl_invalid_root_mask(
                images, w, z1, z2, m1_phys, m2_phys, root_tol
            )
            amps = jnp.where(bad, 0.0, amps)

    # Drop non-finite Jacobian amps without introducing NaN cotangents.
    amps = _zero_nonfinite(amps)
    return images, amps


def psbl_total_amplification(amp_arr):
    """
    Sum finite per-image amplifications.

    Parameters
    ----------
    amp_arr : array_like
        Per-image magnifications.

    Returns
    -------
    amp
        See summary above.
    """
    amp_arr = jnp.asarray(amp_arr)

    # Masked / failed roots are NaN; treat them as zero magnification.
    amp = jnp.sum(jnp.where(jnp.isfinite(amp_arr), amp_arr, 0.0), axis=1)
    return amp


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

    # Magnified source flux.
    flux_model = flux_src * amp

    # Optional blend / neighbor contribution via source flux fraction.
    if b_sff is not None:
        flux_model = flux_model + flux_src * (1.0 - b_sff) / b_sff

    mag = flux2mag_jax(flux_model)
    return mag


def psbl_photometry(t, t0, tE, u0, thetaE_hat, xL1_over_theta, xL2_over_theta, m1, m2, 
                    mag_src, b_sff=None, 
                    piE_E=None, piE_N=None, 
                    root_tol=1e-8, 
                    parallax_vectors=None, parallax_correction=None, 
                    check_sols: bool = True, rescale: bool = True):
    """
    PSBL unresolved photometry at times ``t``.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    tE : float
        Einstein crossing time (days).
    u0 : array_like
        Source-lens separation at ``t0`` (Einstein radii), shape ``(2,)``.
    thetaE_hat : array_like
        Unit vector along the relative proper motion, shape ``(2,)``.
    xL1_over_theta : array_like
        Primary lens position in Einstein radii, shape ``(2,)``.
    xL2_over_theta : array_like
        Secondary lens position in Einstein radii, shape ``(2,)``.
    m1 : float
        Primary mass fraction (or physical mass where noted).
    m2 : float
        Secondary mass fraction (or physical mass where noted).
    mag_src : float
        Unlensed source magnitude.
    b_sff : float or None
        Source flux fraction (blend parameter).
    piE_E : float or None
        Microlensing parallax East component (Einstein radii).
    piE_N : float or None
        Microlensing parallax North component (Einstein radii).
    root_tol : float
        Lens-equation root tolerance.
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    parallax_correction : array_like or None
        Legacy pre-multiplied ``piE_amp * parallax_vectors``.
    check_sols : bool
        If True, mask roots that fail the lens equation.
    rescale : bool
        If True, rescale complex positions before root finding.

    Returns
    -------
    mag
        See summary above.
    """
    # Complex source and static binary-lens positions.
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

    # Witt quintic images -> per-image amps -> total magnification.
    _, amp_arr = psbl_all_arrays(
        w, z1, z2, m1, m2, root_tol, check_sols=check_sols, rescale=rescale
    )
    amp = psbl_total_amplification(amp_arr)

    # Convert total amp to unresolved magnitudes (with optional blend).
    mag = psbl_photometry_from_amp(amp, mag_src, b_sff=b_sff)
    return mag


def psbl_photometry_from_fitter_vec(t, fitter_vec, mag_src, b_sff=None,
    root_tol=1e-8, parallax_vectors=None, check_sols: bool = True,
    rescale: bool = True):
    """
    PSBL photometry from a packed PSBL_PhotParam1 fitter vector.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    fitter_vec : array_like
        Packed fitter parameter vector.
    mag_src : float
        Unlensed source magnitude.
    b_sff : float or None
        Source flux fraction (blend parameter).
    root_tol : float
        Lens-equation root tolerance.
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    check_sols : bool
        If True, mask roots that fail the lens equation.
    rescale : bool
        If True, rescale complex positions before root finding.

    Returns
    -------
    mag
        See summary above.
    """
    # Unpack the packed fitter cube into named scalars.
    params = unpack_psbl_phot_param1(fitter_vec)

    # Derive masses and Einstein-frame geometry from (u0, piE, q, sep, phi).
    m1, m2, u0, thetaE_hat, xL1, xL2, _ = derive_psbl_static_geometry(
        params["u0_amp"],
        params["piE_E"],
        params["piE_N"],
        params["q"],
        params["sep"],
        params["phi"],
    )

    # Forward photometry with the derived static PSBL geometry.
    mag = psbl_photometry(
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
    return mag


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
    """
    _fitter_has_blocked_params.

    Parameters
    ----------
    fitter : MicrolensSolver
        Fitter instance providing data and model class.

    Returns
    -------
    blocked
        See summary above.
    """
    # GP kernels and extra error/weight params are not supported here yet.
    for name in fitter.additional_param_names:
        if any(name.startswith(prefix) for prefix in _GP_PARAM_PREFIXES):
            return True
    for name in fitter.fitter_param_names:
        if any(x in name for x in ("add_err", "mult_err", "weights")):
            return True
    return False


def derive_pspl_photastrom_param1_geometry(mL, t0, beta, dL, dL_dS, xS0_E, xS0_N,
    muL_E, muL_N, muS_E, muS_N):
    """
    Physical-parameter geometry for :class:`~bagle.model.PSPL_PhotAstromParam1`.

    Parameters
    ----------
    mL : float
        Lens mass (Solar masses).
    t0 : float
        Time of closest approach (MJD).
    beta : float
        Signed source-lens impact parameter (mas or Einstein radii).
    dL : float
        Lens distance (pc).
    dL_dS : float
        Distance ratio ``dL/dS``.
    xS0_E : float
        Source RA position at ``t0`` (arcsec).
    xS0_N : float
        Source Dec position at ``t0`` (arcsec).
    muL_E : float
        Lens proper motion East (mas/yr).
    muL_N : float
        Lens proper motion North (mas/yr).
    muS_E : float
        Source proper motion East (mas/yr).
    muS_N : float
        Source proper motion North (mas/yr).

    Returns
    -------
    geometry
        See summary above.
    """
    # Source distance and sky / proper-motion vectors.
    dS = dL / dL_dS
    xS0 = jnp.stack([xS0_E, xS0_N])
    muL = jnp.stack([muL_E, muL_N])
    muS = jnp.stack([muS_E, muS_N])

    # Relative and absolute parallaxes (mas).
    inv_dist_diff = 1.0 / dL - 1.0 / dS
    piRel = _PI_MAS_PER_PC * inv_dist_diff
    piS = _PI_MAS_PER_PC / dS
    piL = _PI_MAS_PER_PC / dL

    # Relative proper motion sets the Einstein-ring direction.
    muRel = muS - muL
    muRel_amp = jnp.linalg.norm(muRel)

    # Einstein radius from lens mass and distance geometry (mas).
    thetaE_amp = jnp.sqrt(
        _EINSTEIN_M_PER_MSUN * mL * inv_dist_diff / _PC_M
    ) * _RAD_TO_MAS
    thetaE_hat = muRel / muRel_amp

    # Impact parameter in Einstein radii from physical beta (mas).
    u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, beta)
    u0_amp = beta / thetaE_amp
    u0 = jnp.abs(u0_amp) * u0_hat

    # Microlensing parallax and Einstein crossing time.
    piE_amp = piRel / thetaE_amp
    piE = piE_amp * thetaE_hat
    tE = (thetaE_amp / muRel_amp) * _DAYS_PER_YEAR

    # Lens position at t0 from source position and angular separation.
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


def pspl_astrometry_param1(t, t0, xS0, xL0, muS, muL, thetaE_amp, b_sff,
    parallax_vectors=None, piS=None, piL=None):
    """
    PSPL flux-weighted centroid astrometry (arcsec), matching

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Time of closest approach (MJD).
    xS0 : array_like
        Source sky position at ``t0`` (arcsec), shape ``(2,)``.
    xL0 : array_like
        Lens sky position at ``t0`` (arcsec), shape ``(2,)``.
    muS : array_like
        Source proper motion (mas/yr), shape ``(2,)``.
    muL : array_like
        Lens proper motion (mas/yr), shape ``(2,)``.
    thetaE_amp : float
        Einstein radius amplitude (mas).
    b_sff : float or None
        Source flux fraction (blend parameter).
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.
    piS : float or None
        Source parallax (mas).
    piL : float or None
        Lens parallax (mas).

    Returns
    -------
    pos
        See summary above.
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    dt = ((t - t0) / _DAYS_PER_YEAR).reshape(-1, 1)

    # Unlensed source and lens tracks (arcsec); mu in mas/yr.
    xS = xS0.reshape(1, 2) + dt * muS.reshape(1, 2) * 1e-3
    xL = xL0.reshape(1, 2) + dt * muL.reshape(1, 2) * 1e-3

    # Optional parallax for source and lens separately.
    if parallax_vectors is not None:
        pvec = jnp.asarray(parallax_vectors, dtype=jnp.float64)
        xS = xS + piS * pvec * 1e-3
        xL = xL + piL * pvec * 1e-3

    # Angular separation and Einstein-normalized u vector.
    thetaS = xS - xL
    u_vec = thetaS / (thetaE_amp * 1e-3)
    u_amp = jnp.linalg.norm(u_vec, axis=1)

    # Blend ratio and flux-weighted centroid shift (matches PSPL.get_astrometry).
    g = (1.0 - b_sff) / b_sff
    sqrt_term = jnp.sqrt(u_amp**2 + 4.0)
    numer_u = u_amp**2 - u_amp * sqrt_term + 3.0
    denom_u = u_amp**2 + 2.0 + g * u_amp * sqrt_term
    numer = thetaS * (1.0 + g * numer_u)[:, jnp.newaxis]
    denom = (1.0 + g) * denom_u
    shift = numer / denom[:, jnp.newaxis]

    # Blend of source / lens positions plus the microlensing centroid shift.
    pos = b_sff * xS + (1.0 - b_sff) * xL + shift
    return pos


def pspl_photometry_param1(t, mL, t0, beta, dL, dL_dS, xS0_E, xS0_N, muL_E,
    muL_N, muS_E, muS_N, mag_src, b_sff, parallax_vectors=None):
    """
    PSPL photometry from PSPL_PhotAstromParam1 physical fitter parameters.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    mL : float
        Lens mass (Solar masses).
    t0 : float
        Time of closest approach (MJD).
    beta : float
        Signed source-lens impact parameter (mas or Einstein radii).
    dL : float
        Lens distance (pc).
    dL_dS : float
        Distance ratio ``dL/dS``.
    xS0_E : float
        Source RA position at ``t0`` (arcsec).
    xS0_N : float
        Source Dec position at ``t0`` (arcsec).
    muL_E : float
        Lens proper motion East (mas/yr).
    muL_N : float
        Lens proper motion North (mas/yr).
    muS_E : float
        Source proper motion East (mas/yr).
    muS_N : float
        Source proper motion North (mas/yr).
    mag_src : float
        Unlensed source magnitude.
    b_sff : float or None
        Source flux fraction (blend parameter).
    parallax_vectors : array_like or None
        Precomputed parallax table, shape ``(N_times, 2)``.

    Returns
    -------
    mag
        See summary above.
    """
    # Physical -> geometric parameters (u0, tE, piE, ...).
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

    # Standard PSPL photometry in the Einstein frame.
    mag = pspl_photometry(
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
    return mag


def gaussian_log_likelihood_sum(mag_model, mag_obs, mag_err):
    """
    Sum of per-point Gaussian log-likelihoods (includes normalization).

    Parameters
    ----------
    mag_model : array_like
        Model magnitudes.
    mag_obs : array_like
        Observed magnitudes.
    mag_err : array_like
        Magnitude uncertainties.

    Returns
    -------
    lnL
        See summary above.
    """
    mag_model = jnp.asarray(mag_model, dtype=jnp.float64)
    mag_obs = jnp.asarray(mag_obs, dtype=jnp.float64)
    mag_err = jnp.asarray(mag_err, dtype=jnp.float64)

    # Per-point chi^2 plus Gaussian normalization, then sum over times.
    chi2 = ((mag_obs - mag_model) / mag_err) ** 2
    lnL_const = -0.5 * jnp.log(2.0 * jnp.pi * mag_err**2)
    lnL = jnp.sum((-0.5 * chi2) + lnL_const)
    return lnL


def pspl_log_likely_photometry(t, t0, tE, u0, thetaE_hat, mag_src,
                               b_sff, mag_obs, mag_err,
                               parallax_vectors=None, piE_E=None,
                               piE_N=None, gp_params=None,
                               fixed_jitter=True):
    """Evaluate a PSPL photometric Gaussian log-likelihood.

    This is intentionally a small composition of the PSPL forward model and
    the normalized Gaussian likelihood. Dataset weights and parameter-vector
    indexing belong to the caller.
    """
    mag_model = pspl_photometry(
        t, t0, tE, u0, thetaE_hat, mag_src, b_sff=b_sff,
        parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N
    )
    if gp_params is None:
        lnL = gaussian_log_likelihood_sum(mag_model, mag_obs, mag_err)
    else:
        from bagle.jax.gp import gp_log_likely_photometry

        lnL = gp_log_likely_photometry(
            t, mag_obs, mag_err, mag_model, gp_params,
            fixed_jitter=fixed_jitter
        )
    return lnL


def psbl_log_likely_photometry(t, t0, tE, u0, thetaE_hat, xL1, xL2,
                               m1, m2, mag_src, b_sff, mag_obs, mag_err,
                               parallax_vectors=None, piE_E=None,
                               piE_N=None, root_tol=1e-8, gp_params=None,
                               fixed_jitter=True):
    """Evaluate a static PSBL photometric Gaussian log-likelihood."""
    mag_model = psbl_photometry(
        t, t0, tE, u0, thetaE_hat, xL1, xL2, m1, m2, mag_src,
        b_sff=b_sff, root_tol=root_tol,
        parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N
    )
    if gp_params is None:
        lnL = gaussian_log_likelihood_sum(mag_model, mag_obs, mag_err)
    else:
        from bagle.jax.gp import gp_log_likely_photometry

        lnL = gp_log_likely_photometry(
            t, mag_obs, mag_err, mag_model, gp_params,
            fixed_jitter=fixed_jitter
        )
    return lnL


def bspl_log_likely_photometry(t, t0_pri, t0_sec, tE, u0_pri,
                               u0_sec, thetaE_hat, mag_src_pri,
                               mag_src_sec, b_sff, mag_obs, mag_err,
                               parallax_vectors=None, piE_E=None,
                               piE_N=None, gp_params=None,
                               fixed_jitter=True):
    """Evaluate a static BSPL photometric Gaussian log-likelihood."""
    u_pri = einstein_source_position(
        t, t0_pri, tE, u0_pri, thetaE_hat,
        parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N
    )
    u_sec = einstein_source_position(
        t, t0_sec, tE, u0_sec, thetaE_hat,
        parallax_vectors=parallax_vectors, piE_E=piE_E, piE_N=piE_N
    )
    amp_pri = pspl_amplification_from_u(u_pri)
    amp_sec = pspl_amplification_from_u(u_sec)
    flux_pri = mag2flux_jax(mag_src_pri)
    flux_sec = mag2flux_jax(mag_src_sec)
    flux_base = (flux_pri + flux_sec) * (1.0 - b_sff) / b_sff
    mag_model = flux2mag_jax(
        flux_pri * amp_pri + flux_sec * amp_sec + flux_base
    )
    if gp_params is None:
        lnL = gaussian_log_likelihood_sum(mag_model, mag_obs, mag_err)
    else:
        from bagle.jax.gp import gp_log_likely_photometry

        lnL = gp_log_likely_photometry(
            t, mag_obs, mag_err, mag_model, gp_params,
            fixed_jitter=fixed_jitter
        )
    return lnL


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
    Return a family id when ``fitter`` can use the JAX photometry likelihood.

    Parameters
    ----------
    fitter : MicrolensSolver
        Fitter instance providing data and model class.

    Returns
    -------
    kind
        See summary above.
    """
    model_class = fitter.model_class
    if not hasattr(model_class, "jax_log_likely_photometry"):
        return None
    if getattr(model_class, "astrometryFlag", False) and fitter.n_ast_sets > 0:
        return None
    if not getattr(model_class, "photometryFlag", False) or fitter.n_phot_sets == 0:
        return None
    if any(
        "add_err" in name or "mult_err" in name
        for name in fitter.fitter_param_names
    ):
        return None
    family = model_class.__name__.split("_", maxsplit=1)[0].lower()
    return family


def gaussian_astrometry_log_likelihood_sum(pos_model, x_obs, y_obs, x_err, y_err):
    """
    Joint x/y Gaussian astrometry log-likelihood (matches PSPL astrometry term).

    Parameters
    ----------
    pos_model : array_like
        Model sky positions, shape ``(N_times, 2)``.
    x_obs : array_like
        Observed RA positions (arcsec).
    y_obs : array_like
        Observed Dec positions (arcsec).
    x_err : array_like
        RA uncertainties (arcsec).
    y_err : array_like
        Dec uncertainties (arcsec).

    Returns
    -------
    lnL
        See summary above.
    """
    pos_model = jnp.asarray(pos_model, dtype=jnp.float64)
    x_obs = jnp.asarray(x_obs, dtype=jnp.float64)
    y_obs = jnp.asarray(y_obs, dtype=jnp.float64)
    x_err = jnp.asarray(x_err, dtype=jnp.float64)
    y_err = jnp.asarray(y_err, dtype=jnp.float64)

    # Separate East / North chi^2 contributions.
    chi2_x = ((x_obs - pos_model[:, 0]) / x_err) ** 2
    chi2_y = ((y_obs - pos_model[:, 1]) / y_err) ** 2

    # Gaussian normalization for each coordinate, then sum over epochs.
    lnL_const_x = -0.5 * jnp.log(2.0 * jnp.pi * x_err**2)
    lnL_const_y = -0.5 * jnp.log(2.0 * jnp.pi * y_err**2)
    lnL = jnp.sum((-0.5 * (chi2_x + chi2_y)) + lnL_const_x + lnL_const_y)
    return lnL


def pspl_log_likely_astrometry(t, t0, xS0, xL0, muS, muL, thetaE_amp,
                               b_sff, x_obs, y_obs, x_err, y_err,
                               parallax_vectors=None, piS=None, piL=None):
    """Evaluate a PSPL absolute-astrometry Gaussian log-likelihood."""
    pos_model = pspl_astrometry_param1(
        t, t0, xS0, xL0, muS, muL, thetaE_amp, b_sff,
        parallax_vectors=parallax_vectors, piS=piS, piL=piL
    )
    lnL = gaussian_astrometry_log_likelihood_sum(
        pos_model, x_obs, y_obs, x_err, y_err
    )
    return lnL


def psbl_astrometry_param1(t, t0, xS0, xL0, muS, muL, thetaE_amp,
                           xL1_over_theta, xL2_over_theta, m1, m2,
                           mag_src, b_sff, dmag_Lp_Ls=20.0,
                           parallax_vectors=None, piS=None, piL=None,
                           root_tol=1e-8, check_sols: bool = True,
                           rescale: bool = True):
    """
    PSBL flux-weighted unresolved centroid astrometry (arcsec).

    Matches :meth:`bagle.model_jax.PSBL.get_astrometry` for static lenses.

    Parameters
    ----------
    t : array_like
        Observation times in MJD.
    t0 : float
        Reference time (MJD).
    xS0, xL0 : array_like
        Source / geometric-center lens sky position at ``t0`` (arcsec).
    muS, muL : array_like
        Proper motions (mas/yr).
    thetaE_amp : float
        Einstein radius (mas).
    xL1_over_theta, xL2_over_theta : array_like
        Companion offsets from geometric center in Einstein radii.
    m1, m2 : float
        Normalized lens masses.
    mag_src : float
        Unlensed source magnitude.
    b_sff : float
        Source flux fraction.
    dmag_Lp_Ls : float, optional
        Primary-minus-secondary lens magnitude difference.
    parallax_vectors : array_like or None
        Shape ``(N_times, 2)`` parallax table.
    piS, piL : float or None
        Source / lens parallax (mas).
    root_tol : float
        Witt quintic root tolerance.
    check_sols, rescale : bool
        Lens-equation solver options.

    Returns
    -------
    pos : jnp.ndarray, shape (N_times, 2)
        East / North centroid positions in arcsec.
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    dt = ((t - t0) / _DAYS_PER_YEAR).reshape(-1, 1)

    # Unlensed source and geometric-center lens tracks (arcsec).
    xS = xS0.reshape(1, 2) + dt * muS.reshape(1, 2) * 1e-3
    xL = xL0.reshape(1, 2) + dt * muL.reshape(1, 2) * 1e-3
    if parallax_vectors is not None:
        pvec = jnp.asarray(parallax_vectors, dtype=jnp.float64)
        xS = xS + piS * pvec * 1e-3
        xL = xL + piL * pvec * 1e-3

    # Companion positions relative to geometric center (arcsec).
    thetaE_as = thetaE_amp * 1e-3
    xL1 = xL + xL1_over_theta.reshape(1, 2) * thetaE_as
    xL2 = xL + xL2_over_theta.reshape(1, 2) * thetaE_as

    # Host PhotAstrom uses m1, m2 in arcsec^2 (= mass fraction * thetaE^2).
    m1 = jnp.asarray(m1, dtype=jnp.float64) * thetaE_as ** 2
    m2 = jnp.asarray(m2, dtype=jnp.float64) * thetaE_as ** 2

    # Complex arcsec positions for the Witt quintic.
    w = xS[:, 0] + 1j * xS[:, 1]
    z1 = xL1[:, 0] + 1j * xL1[:, 1]
    z2 = xL2[:, 0] + 1j * xL2[:, 1]

    image_arr, amp_arr = psbl_all_arrays(
        w, z1, z2, m1, m2, root_tol,
        check_sols=check_sols, rescale=rescale
    )

    # Image positions as (N_times, N_images, 2).
    xS_img = jnp.stack(
        [jnp.real(image_arr), jnp.imag(image_arr)], axis=-1
    )
    amp_f = jnp.where(jnp.isfinite(amp_arr), amp_arr, 0.0)
    amp_3 = amp_f.reshape((amp_f.shape[0], amp_f.shape[1], 1))
    xS_f = jnp.where(jnp.isfinite(xS_img), xS_img, 0.0)

    # Source and luminous-lens fluxes (neighbor light assumed zero).
    fS = mag2flux_jax(mag_src)
    flux_non = fS * (1.0 - b_sff) / jnp.maximum(b_sff, 1e-12)
    fr = jnp.nan_to_num(10.0 ** (dmag_Lp_Ls / -2.5), nan=0.0)
    fL1 = flux_non * fr / (1.0 + fr)
    fL2 = flux_non / (1.0 + fr)

    # Flux-weighted unresolved centroid.
    numer = (
        jnp.sum(xS_f * amp_3 * fS, axis=1)
        + xL1 * fL1
        + xL2 * fL2
    )
    denom = jnp.sum(amp_3 * fS, axis=1) + fL1 + fL2
    return numer / denom


def psbl_log_likely_astrometry(t, t0, xS0, xL0, muS, muL, thetaE_amp,
                               xL1_over_theta, xL2_over_theta, m1, m2,
                               mag_src, b_sff, x_obs, y_obs, x_err, y_err,
                               dmag_Lp_Ls=20.0, parallax_vectors=None,
                               piS=None, piL=None, root_tol=1e-8):
    """Evaluate a static PSBL absolute-astrometry Gaussian log-likelihood."""
    pos_model = psbl_astrometry_param1(
        t, t0, xS0, xL0, muS, muL, thetaE_amp,
        xL1_over_theta, xL2_over_theta, m1, m2,
        mag_src, b_sff, dmag_Lp_Ls=dmag_Lp_Ls,
        parallax_vectors=parallax_vectors, piS=piS, piL=piL,
        root_tol=root_tol
    )
    lnL = gaussian_astrometry_log_likelihood_sum(
        pos_model, x_obs, y_obs, x_err, y_err
    )
    return lnL


def supports_jax_joint_loglik(fitter) -> str | None:
    """
    Return a Param mixin name when joint phot+astrometry JAX likelihood is supported.

    Parameters
    ----------
    fitter : MicrolensSolver
        Fitter instance providing data and model class.

    Returns
    -------
    layout_id
        See summary above.
    """
    model_class = fitter.model_class
    need_phot = bool(
        getattr(model_class, "photometryFlag", False)
        and fitter.n_phot_sets
    )
    need_ast = bool(
        getattr(model_class, "astrometryFlag", False)
        and fitter.n_ast_sets
    )
    if not need_ast:
        return None
    if need_phot and not hasattr(
        model_class, "jax_log_likely_photometry"
    ):
        return None
    if not hasattr(model_class, "jax_log_likely_astrometry"):
        return None
    if any(
        "add_err" in name or "mult_err" in name
        for name in fitter.fitter_param_names
    ):
        return None
    support = model_class.__name__
    return support


def _fitter_weight(fitter, idx: int, default: float = 1.0) -> float:
    """
    _fitter_weight.

    Parameters
    ----------
    fitter : MicrolensSolver
        Fitter instance providing data and model class.
    idx : int
        Filter or data-set index.
    default : float
        Default weight when unset.

    Returns
    -------
    weight
        See summary above.
    """
    w = getattr(fitter, "weights", None)
    if w is None or idx >= len(w):
        return default
    weight = float(w[idx])
    return weight


def _param_index(names: Sequence[str], param_base: str, filt_idx: int | None) -> int:
    """
    _param_index.

    Parameters
    ----------
    names : sequence of str
        Ordered fitter parameter names.
    param_base : str
        Base parameter name without filter suffix.
    filt_idx : int or None
        1-based filter index, or None for unindexed names.

    Returns
    -------
    index
        See summary above.
    """
    if filt_idx is None:
        key = param_base
    else:
        key = f"{param_base}{filt_idx}"
    try:
        index = names.index(key)
        return index
    except ValueError as exc:
        raise KeyError(f"Parameter {key!r} not in fitter_param_names") from exc


def build_jax_phot_likelihood_context(fitter) -> JaxPhotLikelihoodContext | None:
    """
    Build a host-side context for :func:`log_likelihood_phot_from_vec`.

    Parameters
    ----------
    fitter : MicrolensSolver
        Fitter instance providing data and model class.

    Returns
    -------
    ctx
        See summary above.
    """
    kind = supports_jax_phot_loglik(fitter)
    if kind is None:
        return None

    # Select the base geometric cube for this phot family.
    names = tuple(fitter.fitter_param_names)
    if kind == "pspl":
        base_names = PSPL_PHOT_PARAM1_FITTER_NAMES
    else:
        base_names = PSBL_PHOT_PARAM1_FITTER_NAMES

    # Reject fitters whose leading parameter order does not match.
    if names[: len(base_names)] != base_names:
        return None

    base_param_indices = tuple(range(len(base_names)))

    # Parallax tables need lens sky coordinates on the host.
    use_parallax = "raL" in fitter.data and "decL" in fitter.data
    ra_l = float(fitter.data["raL"]) if use_parallax else None
    dec_l = float(fitter.data["decL"]) if use_parallax else None

    filters: list[PhotFilterLikelihoodData] = []
    for i in range(fitter.n_phot_sets):
        filt_1 = i + 1

        # Pull photometry arrays for this filter.
        t = np.asarray(fitter.data[f"t_phot{filt_1}"], dtype=np.float64)
        mag_obs = np.asarray(fitter.data[f"mag{filt_1}"], dtype=np.float64)
        mag_err = np.asarray(fitter.data[f"mag_err{filt_1}"], dtype=np.float64)
        weight = float(getattr(fitter, "weights", [1.0] * fitter.n_phot_sets)[i])

        # Optional host-side parallax direction table.
        pvec = None
        if use_parallax:
            pvec = precompute_parallax_vectors(ra_l, dec_l, t)

        # Indices into the full fitter vector for blend and mag_src.
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
    ctx = JaxPhotLikelihoodContext(
        model_kind=kind,
        fitter_param_names=names,
        base_fitter_names=base_names,
        base_param_indices=base_param_indices,
        filters=tuple(filters),
        root_tol=root_tol,
    )
    return ctx


def log_likelihood_phot_from_vec(param_vec, ctx: JaxPhotLikelihoodContext):
    """
    Differentiable photometry log-likelihood for supported model layouts.

    Parameters
    ----------
    param_vec : array_like
        Full fitter parameter vector.
    ctx : object
        Frozen likelihood context.

    Returns
    -------
    lnL
        See summary above.
    """
    param_vec = jnp.asarray(param_vec, dtype=jnp.float64).reshape(-1)
    lnL = 0.0

    for phot in ctx.filters:
        # Shared geometric cube plus per-filter blend / source magnitude.
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

        # Forward model: PSPL or PSBL photometry from the packed cube.
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

        # Weighted Gaussian photometry term for this filter.
        lnL = lnL + phot.weight * gaussian_log_likelihood_sum(
            mag_model, mag_obs, mag_err
        )

    return lnL


def build_jax_phot_loglik_fn(fitter):
    """
    Return a ``jax.jit``-compiled ``log_likelihood(param_vec)`` or ``None``.

    Parameters
    ----------
    fitter : MicrolensSolver
        Fitter instance providing data and model class.

    Returns
    -------
    loglik_and_ctx
        See summary above.
    """
    ctx = build_jax_phot_likelihood_context(fitter)
    if ctx is None:
        return None, None

    # Capture base indices once so the jitted closure stays pure.
    base_idx = jnp.array(ctx.base_param_indices, dtype=jnp.int32)

    def _loglik(param_vec):
        param_vec = jnp.asarray(param_vec, dtype=jnp.float64).reshape(-1)
        lnL = 0.0
        fitter_vec = param_vec[base_idx]

        for phot in ctx.filters:
            # Per-filter blend and source magnitude from the full vector.
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

            # Forward photometry for this filter's model kind.
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

    loglik_and_ctx = jax.jit(_loglik), ctx
    return loglik_and_ctx


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
    """
    Build host context for joint photometry + astrometry JAX likelihood.

    Parameters
    ----------
    fitter : MicrolensSolver
        Fitter instance providing data and model class.

    Returns
    -------
    ctx
        See summary above.
    """
    layout = supports_jax_joint_loglik(fitter)
    if layout is None:
        return None

    names = tuple(fitter.fitter_param_names)
    base_names = tuple(fitter.model_class.fitter_param_names)
    try:
        base_indices = tuple(names.index(name) for name in base_names)
    except ValueError:
        return None

    # Parallax tables need lens sky coordinates when raL/decL are present.
    use_parallax = "raL" in fitter.data and "decL" in fitter.data
    ra_l = float(fitter.data["raL"]) if use_parallax else None
    dec_l = float(fitter.data["decL"]) if use_parallax else None
    map_phot = getattr(fitter, "map_phot_idx_to_ast_idx", [])
    joint_filters: list[PhotAstromFilterLikelihoodData] = []

    for i in range(fitter.n_ast_sets):
        # Map each astrometry set to its paired photometry filter.
        ast_filt = i + 1
        phot_idx = map_phot[i] if len(map_phot) > i else i
        phot_filt = phot_idx + 1

        # Astrometry observations and optional parallax table.
        t_ast = np.asarray(fitter.data[f"t_ast{ast_filt}"], dtype=np.float64)
        x_obs = np.asarray(fitter.data[f"xpos{ast_filt}"], dtype=np.float64)
        y_obs = np.asarray(fitter.data[f"ypos{ast_filt}"], dtype=np.float64)
        x_err = np.asarray(fitter.data[f"xpos_err{ast_filt}"], dtype=np.float64)
        y_err = np.asarray(fitter.data[f"ypos_err{ast_filt}"], dtype=np.float64)
        ast_weight = _fitter_weight(fitter, fitter.n_phot_sets + i)
        pvec_ast = None
        if use_parallax:
            pvec_ast = precompute_parallax_vectors(ra_l, dec_l, t_ast)

        # Blend index comes from the paired photometry filter.
        idx_b_ast = _param_index(
            names, "b_sff", phot_filt if f"b_sff{phot_filt}" in names else None
        )

        phot_block = None
        if phot_idx < fitter.n_phot_sets:
            # Photometry observations for the matched filter.
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

    # Photometry-only sets (no astrometry counterpart).
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

    ctx = JaxJointLikelihoodContext(
        layout=layout,
        use_parallax=use_parallax,
        fitter_param_names=names,
        base_indices=base_indices,
        filters=tuple(joint_filters),
    )
    return ctx


def build_jax_joint_loglik_fn(fitter):
    """
    Compatibility wrapper for the explicit Param-mixin likelihood.

    Parameters
    ----------
    fitter : MicrolensSolver
        Fitter instance providing data and model class.

    Returns
    -------
    loglik_and_ctx : tuple
        ``(jit_loglik, ctx)`` from
        :func:`bagle.model_fitter_jax.build_explicit_jax_loglik_fn`,
        or ``(None, None)`` when unsupported.
    """
    from bagle.model_fitter_jax import build_explicit_jax_loglik_fn

    result = build_explicit_jax_loglik_fn(fitter)
    
    return result


def supports_jax_loglik(fitter) -> str | None:
    """
    Return model class name when JAX autodiff likelihood is available.

    Parameters
    ----------
    fitter : MicrolensSolver
        Fitter instance providing data and model class.

    Returns
    -------
    support : str or None
        ``fitter.model_class.__name__`` when an explicit JAX log-likelihood
        can be built, otherwise ``None``.
    """
    # Probe the same builder used by MultiNest / PyMC / NumPyro paths.
    fn, _ = build_jax_loglik_fn(fitter)
    if fn is None:
        return None

    return fitter.model_class.__name__


def build_jax_loglik_fn(fitter):
    """
    Return the best available ``jax.jit`` log-likelihood and context.

    Parameters
    ----------
    fitter : MicrolensSolver
        Fitter instance providing data and model class.

    Returns
    -------
    loglik_and_ctx : tuple
        ``(jit_loglik, ctx)`` from
        :func:`bagle.model_fitter_jax.build_explicit_jax_loglik_fn`,
        or ``(None, None)`` when unsupported.
    """
    # Lazy import avoids a circular import with model_fitter_jax.
    from bagle.model_fitter_jax import build_explicit_jax_loglik_fn

    result = build_explicit_jax_loglik_fn(fitter)
    return result


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
