"""Derive microlensing geometry from fitter parameter vectors (JAX)."""
from __future__ import annotations

import jax.numpy as jnp

from bagle.jax_physics import (
    _DAYS_PER_YEAR,
    _EINSTEIN_M_PER_MSUN,
    _PC_M,
    _PI_MAS_PER_PC,
    _RAD_TO_MAS,
    derive_pspl_photastrom_param1_geometry,
    derive_pspl_static_geometry,
    derive_psbl_static_geometry,
    u0_hat_from_thetaE_hat_jax,
)

_KAPPA_MAS_MSUN = (
    4.0
    * _EINSTEIN_M_PER_MSUN
    / _PC_M
    * _RAD_TO_MAS
)


def mag_src_from_base(mag_base, b_sff):
    return mag_base - 2.5 * jnp.log10(b_sff)


def mag_src_from_fitter(mag_value, mag_fitter: str, b_sff):
    if mag_fitter == "mag_base":
        return mag_src_from_base(mag_value, b_sff)
    return mag_value


def derive_pspl_photastrom_reduced(
    t0,
    u0_amp,
    tE,
    thetaE_amp,
    piS,
    piE_E,
    piE_N,
    xS0_E,
    xS0_N,
    muS_E,
    muS_N,
):
    """Geometry for PSPL PhotAstrom Param2 / Astrom Param4 style parameters."""
    piE = jnp.stack([piE_E, piE_N])
    piE_amp = jnp.linalg.norm(piE)
    piRel = piE_amp * thetaE_amp
    muRel_amp = thetaE_amp / (tE / _DAYS_PER_YEAR)
    mL = thetaE_amp**2 / (piRel * _KAPPA_MAS_MSUN)
    piL = piRel + piS
    dL = _PI_MAS_PER_PC / piL
    dS = _PI_MAS_PER_PC / piS
    thetaE_hat = piE / piE_amp
    muRel = muRel_amp * thetaE_hat
    muS = jnp.stack([muS_E, muS_N])
    muL = muS - muRel
    u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, u0_amp)
    u0 = jnp.abs(u0_amp) * u0_hat
    xS0 = jnp.stack([xS0_E, xS0_N])
    thetaS0 = u0 * thetaE_amp
    xL0 = xS0 - thetaS0 * 1e-3
    return (
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
    )


def derive_pspl_photastrom_log10_thetaE(
    t0,
    u0_amp,
    tE,
    log10_thetaE,
    piS,
    piE_E,
    piE_N,
    xS0_E,
    xS0_N,
    muS_E,
    muS_N,
):
    thetaE_amp = 10.0**log10_thetaE
    return derive_pspl_photastrom_reduced(
        t0,
        u0_amp,
        tE,
        thetaE_amp,
        piS,
        piE_E,
        piE_N,
        xS0_E,
        xS0_N,
        muS_E,
        muS_N,
    )


def derive_pspl_phot_log(
    t0,
    u0_amp,
    log_tE,
    log_piE,
    phi_muRel_deg,
):
    """PSPL_PhotParam3: log tE, log piE, phi_muRel."""
    tE = 10.0**log_tE
    piE_amp = 10.0**log_piE
    phi = phi_muRel_deg * jnp.pi / 180.0
    piE_E = piE_amp * jnp.sin(phi)
    piE_N = piE_amp * jnp.cos(phi)
    u0, thetaE_hat, _ = derive_pspl_static_geometry(u0_amp, piE_E, piE_N)
    return u0, thetaE_hat, tE, piE_E, piE_N


def derive_psbl_photastrom_param1(
    mLp,
    mLs,
    t0,
    xS0_E,
    xS0_N,
    beta,
    muL_E,
    muL_N,
    muS_E,
    muS_N,
    dL,
    dS,
    sep,
    alpha_deg,
):
    """Static PSBL PhotAstrom Param1 geometry (mas, arcsec)."""
    mL = mLp + mLs
    xS0 = jnp.stack([xS0_E, xS0_N])
    muL = jnp.stack([muL_E, muL_N])
    muS = jnp.stack([muS_E, muS_N])
    inv_dist_diff = 1.0 / dL - 1.0 / dS
    piRel = _PI_MAS_PER_PC * inv_dist_diff
    piS = _PI_MAS_PER_PC / dS
    piL = _PI_MAS_PER_PC / dL
    muRel = muS - muL
    muRel_amp = jnp.linalg.norm(muRel)
    thetaE_amp = jnp.sqrt(_EINSTEIN_M_PER_MSUN * mL * inv_dist_diff / _PC_M) * _RAD_TO_MAS
    thetaE_hat = muRel / muRel_amp
    u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, beta)
    u0_amp = beta / thetaE_amp
    u0 = jnp.abs(u0_amp) * u0_hat
    piE_amp = piRel / thetaE_amp
    piE_E = piE_amp * thetaE_hat[0]
    piE_N = piE_amp * thetaE_hat[1]
    tE = (thetaE_amp / muRel_amp) * _DAYS_PER_YEAR
    thetaS0 = u0 * thetaE_amp
    xL0 = xS0 - thetaS0 * 1e-3
    alpha_rad = alpha_deg * jnp.pi / 180.0
    phi_piE = jnp.arctan2(piE_E, piE_N)
    phi_rho1 = phi_piE + alpha_rad
    xL1_over_theta = jnp.stack(
        [0.5 * sep * jnp.sin(phi_rho1), 0.5 * sep * jnp.cos(phi_rho1)]
    )
    xL2_over_theta = -xL1_over_theta
    q = mLs / mLp
    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)
    return (
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
        m1,
        m2,
        xL1_over_theta,
        xL2_over_theta,
        mLp,
        mLs,
    )


def unpack_base_params(names: tuple[str, ...], vec):
    vec = jnp.asarray(vec, dtype=jnp.float64).reshape(-1)
    return {n: vec[i] for i, n in enumerate(names)}


def derive_geometry_from_layout(layout_id: str, eval_kind: str, base_vec, names: tuple[str, ...]):
    """Dispatch geometry derivation from a base fitter parameter vector."""
    p = unpack_base_params(names, base_vec)
    if eval_kind == "pspl_photastrom_physical":
        return derive_pspl_photastrom_param1_geometry(
            p["mL"],
            p["t0"],
            p["beta"],
            p["dL"],
            p["dL_dS"],
            p["xS0_E"],
            p["xS0_N"],
            p["muL_E"],
            p["muL_N"],
            p["muS_E"],
            p["muS_N"],
        )
    if eval_kind in ("pspl_photastrom_reduced", "pspl_astrom_reduced"):
        te_key = "log10_thetaE" if "log10_thetaE" in p else "thetaE"
        theta = 10.0 ** p[te_key] if te_key == "log10_thetaE" else p[te_key]
        if "log_piE" in p and "phi_muRel" in p:
            piE_amp = 10.0 ** p["log_piE"]
            phi = p["phi_muRel"] * jnp.pi / 180.0
            piE_E = piE_amp * jnp.sin(phi)
            piE_N = piE_amp * jnp.cos(phi)
        elif "piEN_piEE" in p:
            piE_E = p["piE_E"]
            piE_N = p["piE_E"] * p["piEN_piEE"]
        else:
            piE_E = p["piE_E"]
            piE_N = p["piE_N"]
        return derive_pspl_photastrom_reduced(
            p["t0"],
            p["u0_amp"],
            p["tE"],
            theta,
            p["piS"],
            piE_E,
            piE_N,
            p["xS0_E"],
            p["xS0_N"],
            p["muS_E"],
            p["muS_N"],
        )
    if eval_kind in ("pspl_phot_static", "pspl_phot_log"):
        if "log_tE" in p:
            u0, thetaE_hat, tE, piE_E, piE_N = derive_pspl_phot_log(
                p["t0"],
                p["u0_amp"],
                p["log_tE"],
                p["log_piE"],
                p["phi_muRel"],
            )
        else:
            u0, thetaE_hat, _ = derive_pspl_static_geometry(
                p["u0_amp"], p["piE_E"], p["piE_N"]
            )
            tE = p["tE"]
            piE_E = p["piE_E"]
            piE_N = p["piE_N"]
        return ("pspl_phot", u0, thetaE_hat, tE, piE_E, piE_N)
    if eval_kind.startswith("psbl_phot"):
        m1, m2, u0, thetaE_hat, xL1, xL2, _ = derive_psbl_static_geometry(
            p["u0_amp"],
            p["piE_E"],
            p["piE_N"],
            p["q"],
            p["sep"],
            p["phi"],
        )
        return (
            "psbl_phot",
            u0,
            thetaE_hat,
            p["t0"],
            p["tE"],
            m1,
            m2,
            xL1,
            xL2,
            p["piE_E"],
            p["piE_N"],
        )
    if eval_kind == "bspl_phot":
        piE = jnp.stack([p["piE_E"], p["piE_N"]])
        piE_amp = jnp.linalg.norm(piE)
        thetaE_hat = piE / piE_amp
        u0_hat = u0_hat_from_thetaE_hat_jax(thetaE_hat, p["u0_amp"])
        u0_pri = jnp.abs(p["u0_amp"]) * u0_hat
        phi_rad = p["phi"] * jnp.pi / 180.0
        phi_piE = jnp.arctan2(p["piE_E"], p["piE_N"])
        phi_rho1 = phi_piE + phi_rad
        sep_vec = p["sep"] * jnp.stack([jnp.sin(phi_rho1), jnp.cos(phi_rho1)])
        u0_amp_sec = p["u0_amp"] + jnp.dot(sep_vec, u0_hat)
        u0_sec = u0_amp_sec * u0_hat
        return (
            "bspl_phot",
            u0_pri,
            u0_sec,
            thetaE_hat,
            p["t0"],
            p["t0"],
            p["tE"],
            p["piE_E"],
            p["piE_N"],
        )
    if eval_kind.startswith("psbl_photastrom"):
        return derive_psbl_photastrom_param1(
            p["mLp"],
            p["mLs"],
            p["t0"],
            p["xS0_E"],
            p["xS0_N"],
            p["beta"],
            p["muL_E"],
            p["muL_N"],
            p["muS_E"],
            p["muS_N"],
            p["dL"],
            p["dS"],
            p["sep"],
            p["alpha"],
        )
    raise NotImplementedError(f"derive_geometry for eval_kind={eval_kind!r}")
