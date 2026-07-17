"""Pack fitter cubes into shared jnp microlensing parameters for kernels."""
from __future__ import annotations

import jax.numpy as jnp

from bagle.jax.geometry import (
    derive_pspl_phot_log,
    derive_pspl_photastrom_log10_thetaE,
    derive_pspl_photastrom_reduced,
    derive_pspl_static_geometry,
)
from bagle.jax_physics import (
    derive_psbl_static_geometry,
    derive_pspl_photastrom_param1_geometry,
)


def _named(vec, names):
    vec = jnp.asarray(vec, dtype=jnp.float64).reshape(-1)
    return {n: vec[i] for i, n in enumerate(names)}


def pack_pspl_phot_param1(vec, names):
    p = _named(vec, names)
    u0, thetaE_hat, _ = derive_pspl_static_geometry(
        p["u0_amp"], p["piE_E"], p["piE_N"]
    )
    return dict(
        t0=p["t0"],
        tE=p["tE"],
        u0=u0,
        thetaE_hat=thetaE_hat,
        piE_E=p["piE_E"],
        piE_N=p["piE_N"],
    )


def pack_pspl_phot_param2(vec, names):
    """Same static geometry as Param1 (mag_base is phot-only extra)."""
    return pack_pspl_phot_param1(vec, names)


def pack_pspl_phot_param3(vec, names):
    p = _named(vec, names)
    u0, thetaE_hat, tE, piE_E, piE_N = derive_pspl_phot_log(
        p["t0"], p["u0_amp"], p["log_tE"], p["log_piE"], p["phi_muRel"]
    )
    return dict(
        t0=p["t0"],
        tE=tE,
        u0=u0,
        thetaE_hat=thetaE_hat,
        piE_E=piE_E,
        piE_N=piE_N,
    )


def pack_pspl_photastrom_param1(vec, names):
    p = _named(vec, names)
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
    return dict(
        t0=p["t0"],
        tE=tE,
        u0=u0,
        thetaE_hat=thetaE_hat,
        piE_E=piE_E,
        piE_N=piE_N,
        xS0=xS0,
        xL0=xL0,
        muS=muS,
        muL=muL,
        thetaE_amp=thetaE_amp,
        piS=piS,
        piL=piL,
    )


def pack_pspl_photastrom_param2(vec, names):
    p = _named(vec, names)
    theta_key = "thetaE" if "thetaE" in p else "thetaE_amp"
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
    ) = derive_pspl_photastrom_reduced(
        p["t0"],
        p["u0_amp"],
        p["tE"],
        p[theta_key],
        p["piS"],
        p["piE_E"],
        p["piE_N"],
        p["xS0_E"],
        p["xS0_N"],
        p["muS_E"],
        p["muS_N"],
    )
    return dict(
        t0=p["t0"],
        tE=tE,
        u0=u0,
        thetaE_hat=thetaE_hat,
        piE_E=piE_E,
        piE_N=piE_N,
        xS0=xS0,
        xL0=xL0,
        muS=muS,
        muL=muL,
        thetaE_amp=thetaE_amp,
        piS=piS,
        piL=piL,
    )


def pack_pspl_photastrom_param3(vec, names):
    p = _named(vec, names)
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
    ) = derive_pspl_photastrom_log10_thetaE(
        p["t0"],
        p["u0_amp"],
        p["tE"],
        p["log10_thetaE"],
        p["piS"],
        p["piE_E"],
        p["piE_N"],
        p["xS0_E"],
        p["xS0_N"],
        p["muS_E"],
        p["muS_N"],
    )
    return dict(
        t0=p["t0"],
        tE=tE,
        u0=u0,
        thetaE_hat=thetaE_hat,
        piE_E=piE_E,
        piE_N=piE_N,
        xS0=xS0,
        xL0=xL0,
        muS=muS,
        muL=muL,
        thetaE_amp=thetaE_amp,
        piS=piS,
        piL=piL,
    )


def pack_bspl_phot_param1(vec, names):
    """BSPL Phot Param1 dual-source static geometry."""
    p = _named(vec, names)
    u0_pri, thetaE_hat, _ = derive_pspl_static_geometry(
        p["u0_amp_pri"], p["piE_E"], p["piE_N"]
    )
    u0_sec, _, _ = derive_pspl_static_geometry(
        p["u0_amp_sec"], p["piE_E"], p["piE_N"]
    )
    return dict(
        t0_pri=p["t0_pri"],
        t0_sec=p["t0_sec"],
        tE=p["tE"],
        u0_pri=u0_pri,
        u0_sec=u0_sec,
        thetaE_hat=thetaE_hat,
        piE_E=p["piE_E"],
        piE_N=p["piE_N"],
    )


def pack_psbl_phot_param1(vec, names):
    """PSBL phot-only Param1 static binary geometry."""
    p = _named(vec, names)
    phi = p["phi"] if "phi" in p else p.get("alpha", 0.0)
    m1, m2, u0, thetaE_hat, xL1, xL2, _piE_amp = derive_psbl_static_geometry(
        p["u0_amp"],
        p["piE_E"],
        p["piE_N"],
        p["q"],
        p["sep"],
        phi,
    )
    return dict(
        t0=p["t0"],
        tE=p["tE"],
        u0=u0,
        thetaE_hat=thetaE_hat,
        m1=m1,
        m2=m2,
        xL1=xL1,
        xL2=xL2,
        piE_E=p["piE_E"],
        piE_N=p["piE_N"],
    )


def pack_generic(vec, names):
    """Named scalar pack plus PSPL-like static geometry when possible."""
    p = _named(vec, names)
    out = dict(p)
    if "u0_amp" in p and "piE_E" in p and "piE_N" in p:
        u0, thetaE_hat, _ = derive_pspl_static_geometry(
            p["u0_amp"], p["piE_E"], p["piE_N"]
        )
        out["u0"] = u0
        out["thetaE_hat"] = thetaE_hat
    if "log10_thetaE" in p and "t0" in p and "u0_amp" in p and "piS" in p:
        try:
            return pack_pspl_photastrom_param3(vec, names)
        except Exception:
            pass
    if "thetaE" in p and "piS" in p and "xS0_E" in p:
        try:
            return pack_pspl_photastrom_param2(vec, names)
        except Exception:
            pass
    if "mL" in p and "beta" in p and "dL" in p and "dL_dS" in p:
        try:
            return pack_pspl_photastrom_param1(vec, names)
        except Exception:
            pass
    if "log_tE" in p and "log_piE" in p and "phi_muRel" in p:
        try:
            return pack_pspl_phot_param3(vec, names)
        except Exception:
            pass
    if "u0_amp_pri" in p and "u0_amp_sec" in p:
        try:
            return pack_bspl_phot_param1(vec, names)
        except Exception:
            pass
    if "q" in p and "sep" in p and ("phi" in p or "alpha" in p):
        try:
            return pack_psbl_phot_param1(vec, names)
        except Exception:
            pass
    return out


def pack_params_for_class(cls, vec):
    """Dispatch packing for any Param mixin class."""
    names = tuple(cls.fitter_param_names)
    name = cls.__name__

    # Explicit PSPL packers
    if name in ("PSPL_PhotParam1", "PSPL_PhotParam1_geoproj", "PSPL_GP_PhotParam1",
                "PSPL_GP_PhotParam1_2"):
        return pack_pspl_phot_param1(vec, names)
    if name in ("PSPL_PhotParam2", "PSPL_GP_PhotParam2", "PSPL_GP_PhotParam2_2",
                "PSPL_GP_PhotParam2_3", "PSPL_GP_PhotParam2_4", "PSPL_GP_PhotParam2_5"):
        return pack_pspl_phot_param2(vec, names)
    if name in ("PSPL_PhotParam3", "PSPL_GP_PhotParam3"):
        return pack_pspl_phot_param3(vec, names)
    if name in ("PSPL_PhotAstromParam1", "PSPL_GP_PhotAstromParam1"):
        return pack_pspl_photastrom_param1(vec, names)
    if name in ("PSPL_PhotAstromParam2", "PSPL_GP_PhotAstromParam2",
                "PSPL_PhotAstromParam4", "PSPL_PhotAstromParam4_geoproj",
                "PSPL_GP_PhotAstromParam4", "PSPL_GP_PhotAstromParam4_1",
                "PSPL_GP_PhotAstromParam4_2", "PSPL_AstromParam4"):
        return pack_pspl_photastrom_param2(vec, names)
    if name in ("PSPL_PhotAstromParam3", "PSPL_AstromParam3",
                "PSPL_GP_PhotAstromParam3", "PSPL_GP_PhotAstromParam3_1",
                "PSPL_GP_PhotAstromParam3_2"):
        return pack_pspl_photastrom_param3(vec, names)

    # BSPL phot
    if "BSPL" in name and "u0_amp_pri" in names:
        return pack_bspl_phot_param1(vec, names)

    # PSBL phot with q/sep
    if "PSBL" in name and "q" in names and "sep" in names:
        try:
            return pack_psbl_phot_param1(vec, names)
        except Exception:
            return pack_generic(vec, names)

    return pack_generic(vec, names)


def params_from_self_phot(model):
    return dict(
        t0=jnp.asarray(model.t0, dtype=jnp.float64),
        tE=jnp.asarray(model.tE, dtype=jnp.float64),
        u0=jnp.asarray(model.u0, dtype=jnp.float64),
        thetaE_hat=jnp.asarray(model.thetaE_hat, dtype=jnp.float64),
        piE_E=jnp.asarray(model.piE[0], dtype=jnp.float64),
        piE_N=jnp.asarray(model.piE[1], dtype=jnp.float64),
    )


def params_from_self_photastrom(model):
    d = params_from_self_phot(model)
    d.update(
        xS0=jnp.asarray(model.xS0, dtype=jnp.float64),
        xL0=jnp.asarray(model.xL0, dtype=jnp.float64),
        muS=jnp.asarray(model.muS, dtype=jnp.float64),
        muL=jnp.asarray(model.muL, dtype=jnp.float64),
        thetaE_amp=jnp.asarray(model.thetaE_amp, dtype=jnp.float64),
        piS=jnp.asarray(model.piS, dtype=jnp.float64),
        piL=jnp.asarray(model.piL, dtype=jnp.float64),
    )
    return d
