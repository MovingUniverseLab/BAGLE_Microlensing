"""BSPL (binary source, point lens) JAX kernels."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from astropy import constants as _const
from astropy import units as _u

import bagle.jax_physics as jax_physics

from bagle.jax_physics import (
    _fitter_weight,
    _param_index,
    gaussian_astrometry_log_likelihood_sum,
    gaussian_log_likelihood_sum,
    mag2flux_jax,
    flux2mag_jax,
    precompute_parallax_vectors,
    pspl_amplification_from_u,
    pspl_resolved_amplification_from_u,
    einstein_source_position,
    derive_pspl_static_geometry,
)


def _filt_scalar(model, name: str, filt_idx: int) -> float:
    val = getattr(model, name)
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"empty parameter {name!r}")
    return float(arr[filt_idx]) if arr.size > filt_idx else float(arr[0])


def _bspl_u_dual(t, t0_pri, t0_sec, tE, u0_pri, u0_sec, thetaE_hat, pvec, piE_E, piE_N):
    u1 = einstein_source_position(
        t, t0_pri, tE, u0_pri, thetaE_hat, parallax_vectors=pvec, piE_E=piE_E, piE_N=piE_N
    )
    u2 = einstein_source_position(
        t, t0_sec, tE, u0_sec, thetaE_hat, parallax_vectors=pvec, piE_E=piE_E, piE_N=piE_N
    )
    return u1, u2


def bspl_photometry_jax(
    t,
    t0_pri,
    t0_sec,
    tE,
    u0_pri,
    u0_sec,
    thetaE_hat,
    mag_pri,
    mag_sec,
    b_sff,
    pvec=None,
    piE_E=None,
    piE_N=None,
):
    u1, u2 = _bspl_u_dual(t, t0_pri, t0_sec, tE, u0_pri, u0_sec, thetaE_hat, pvec, piE_E, piE_N)
    A1 = pspl_amplification_from_u(u1)
    A2 = pspl_amplification_from_u(u2)
    f1 = mag2flux_jax(mag_pri)
    f2 = mag2flux_jax(mag_sec)
    flux = f1 * A1 + f2 * A2 + (f1 + f2) * (1.0 - b_sff) / b_sff
    return flux2mag_jax(flux)


def _bspl_times(model):
    """
    Primary and secondary reference epochs for photometry.

    Option A uses a single epoch (primary CA); ``get_t0_sec`` remains a
    derived diagnostic for the secondary's own closest approach.

    Parameters
    ----------
    model : object
        BSPL-like model with ``t0_pri``.

    Returns
    -------
    t0_pri, t0_sec : float
        Both equal to ``t0_pri`` under Option A.
    """
    t0_pri = float(model.t0_pri)
    return t0_pri, t0_pri


def _bspl_u0_pair(model, filt_idx: int):
    """
    Primary and secondary ``u0`` vectors at ``t0_pri``.

    Parameters
    ----------
    model : object
        BSPL-like model.
    filt_idx : int
        Unused; kept for call-site compatibility.

    Returns
    -------
    u0_pri, u0_sec : jnp.ndarray, shape = (2,)
        Lens-relative positions at ``t0_pri`` (Einstein radii).
    thetaE_hat : jnp.ndarray, shape = (2,)
        Unit vector along relative proper motion.
    """
    thetaE_hat = jnp.asarray(model.thetaE_hat, dtype=jnp.float64).reshape(2)

    # Prefer snapshot vectors already stored on the model (Option A).
    if hasattr(model, "u0_pri") and hasattr(model, "u0_sec"):
        u0_pri = jnp.asarray(model.u0_pri, dtype=jnp.float64).reshape(2)
        u0_sec = jnp.asarray(model.u0_sec, dtype=jnp.float64).reshape(2)
        return u0_pri, u0_sec, thetaE_hat

    # Reconstruct the full static offset from sep / angle.
    u0_pri_j, thetaE_hat, _ = derive_pspl_static_geometry(
        float(getattr(model, "u0_amp_pri", model.u0_amp)),
        float(model.piE[0]),
        float(model.piE[1]),
    )
    # Phot-only: sep is already in Einstein radii. PhotAstrom: mas.
    if hasattr(model, "thetaE_amp"):
        sep = float(model.sep) / float(model.thetaE_amp)
    else:
        sep = float(model.sep)
    alpha = float(getattr(model, "phi_rho1_rad", getattr(model, "alpha_rad", 0.0)))
    sep_vec = sep * np.array([np.sin(alpha), np.cos(alpha)])
    u0_sec_j = jnp.asarray(u0_pri_j, dtype=jnp.float64) + jnp.asarray(
        sep_vec, dtype=jnp.float64)

    return u0_pri_j, u0_sec_j, thetaE_hat


def bspl_photometry_from_model(model, t, filt_idx, pvec):
    u0_pri, u0_sec, thetaE_hat = _bspl_u0_pair(model, filt_idx)
    t0_pri, t0_sec = _bspl_times(model)
    mag = bspl_photometry_jax(
        jnp.asarray(t, dtype=jnp.float64),
        t0_pri,
        t0_sec,
        float(model.tE),
        u0_pri,
        u0_sec,
        thetaE_hat,
        _filt_scalar(model, "mag_src_pri", filt_idx),
        _filt_scalar(model, "mag_src_sec", filt_idx),
        _filt_scalar(model, "b_sff", filt_idx),
        pvec=pvec,
        piE_E=float(model.piE[0]),
        piE_N=float(model.piE[1]),
    )
    return np.asarray(mag, dtype=np.float64)


def bspl_amplification_from_model(model, t, filt_idx, pvec):
    u0_pri, u0_sec, thetaE_hat = _bspl_u0_pair(model, filt_idx)
    t0_pri, t0_sec = _bspl_times(model)
    t_j = jnp.asarray(t, dtype=jnp.float64)
    u1, u2 = _bspl_u_dual(
        t_j,
        t0_pri,
        t0_sec,
        float(model.tE),
        u0_pri,
        u0_sec,
        thetaE_hat,
        pvec,
        float(model.piE[0]),
        float(model.piE[1]),
    )
    a1 = pspl_amplification_from_u(u1)
    a2 = pspl_amplification_from_u(u2)
    f1 = mag2flux_jax(_filt_scalar(model, "mag_src_pri", filt_idx))
    f2 = mag2flux_jax(_filt_scalar(model, "mag_src_sec", filt_idx))
    amp = (f1 * a1 + f2 * a2) / (f1 + f2)
    return np.asarray(amp, dtype=np.float64)


def bspl_resolved_amplification_from_model(model, t, filt_idx, pvec):
    """Dual-source ± image amplifications, shape ``(N_t, 2 sources, 2 images)``.

    Uses host ``get_u`` so static and orbital BSPL layouts match NumPy reference.
    """
    u_vec = np.asarray(model.get_u(t, filt_idx=filt_idx), dtype=np.float64)
    u1 = jnp.asarray(u_vec[:, 0, :], dtype=jnp.float64)
    u2 = jnp.asarray(u_vec[:, 1, :], dtype=jnp.float64)
    a1_plus, a1_minus = pspl_resolved_amplification_from_u(u1)
    a2_plus, a2_minus = pspl_resolved_amplification_from_u(u2)
    a_src1 = jnp.stack([a1_plus, a1_minus], axis=1)
    a_src2 = jnp.stack([a2_plus, a2_minus], axis=1)
    amp = jnp.stack([a_src1, a_src2], axis=1)
    return np.asarray(amp, dtype=np.float64)


def bspl_source_astrometry_unlensed_from_model(model, t, filt_idx, pvec):
    """Flux-weighted unlensed source centroid in Einstein radii (BSPL phot-only).

    Uses host ``get_u`` so static and orbital layouts match NumPy reference.
    """
    u_vec = np.asarray(model.get_u(t, filt_idx=filt_idx), dtype=np.float64)
    u1 = jnp.asarray(u_vec[:, 0, :], dtype=jnp.float64)
    u2 = jnp.asarray(u_vec[:, 1, :], dtype=jnp.float64)
    f1 = mag2flux_jax(_filt_scalar(model, "mag_src_pri", filt_idx))
    f2 = mag2flux_jax(_filt_scalar(model, "mag_src_sec", filt_idx))
    u = (u1 * f1 + u2 * f2) / (f1 + f2)
    return np.asarray(u, dtype=np.float64)


def bspl_photastrom_source_astrometry_unlensed_from_model(
    model, t, filt_idx, pvec
):
    """Flux-weighted unlensed source centroid in arcsec (BSPL PhotAstrom).

    Uses host ``get_resolved_source_astrometry_unlensed`` for orbital motion.
    """
    xS_both = np.asarray(
        model.get_resolved_source_astrometry_unlensed(t, filt_idx=filt_idx),
        dtype=np.float64,
    )
    x1 = jnp.asarray(xS_both[:, 0, :], dtype=jnp.float64)
    x2 = jnp.asarray(xS_both[:, 1, :], dtype=jnp.float64)
    f1 = mag2flux_jax(_filt_scalar(model, "mag_src_pri", filt_idx))
    f2 = mag2flux_jax(_filt_scalar(model, "mag_src_sec", filt_idx))
    xS = (x1 * f1 + x2 * f2) / (f1 + f2)
    return np.asarray(xS, dtype=np.float64)


# Days in a Julian year, matching model_jax.days_per_year.
_DAYS_PER_YEAR = 365.25

# Kepler's law: P (days) = _KEPLER_DAY * sqrt(a_AU**3 / M_Msun).
# a_AU is the host's ``dS_pc * a_mas * 1e-3``.
_KEPLER_DAY = float(
    (2.0 * np.pi * np.sqrt((1.0 * _u.AU) ** 3 / (_const.G * _u.Msun)))
    .to(_u.day)
    .value
)


def _param_dict(names, vec):
    """Map a class-order vector onto parameter names.

    Parameters
    ----------
    names : sequence of str
        ``fitter_param_names`` for the class that owns ``vec``.
    vec : array_like
        Filter-local values in that order.

    Returns
    -------
    params : dict
    """
    values = jnp.asarray(vec, dtype=jnp.float64)
    params = dict(zip(tuple(names), values))
    return params


def _primary_geometry(p):
    """Primary-source PSPL geometry shared by every BSPL PhotAstrom param.

    Parameters
    ----------
    p : dict
        Filter-local fitter parameters.

    Returns
    -------
    t0, xS0, xL0, muS, muL, thetaE_amp, piS, piL : float or array
        ``xS0`` is the primary at ``t0`` (arcsec). Proper motions are
        mas/yr. ``thetaE_amp``, ``piS``, and ``piL`` are mas.
    """
    from bagle.jax.geometry import (
        derive_pspl_photastrom_log10_thetaE,
        derive_pspl_photastrom_reduced,
    )

    t0 = p["t0"]
    xS0_E = p["xS0_E"]
    xS0_N = p["xS0_N"]
    muS_E = p["muS_E"]
    muS_N = p["muS_N"]

    # Physical lens mass and distances (Param1 and its orbit children).
    if "mL" in p:
        geom = jax_physics.derive_pspl_photastrom_param1_geometry(
            p["mL"], t0, p["beta"], p["dL"], p["dL_dS"],
            xS0_E, xS0_N, p["muL_E"], p["muL_N"], muS_E, muS_N,
        )
    elif "log10_thetaE" in p:
        geom = derive_pspl_photastrom_log10_thetaE(
            t0, p["u0_amp"], p["tE"], p["log10_thetaE"], p["piS"],
            p["piE_E"], p["piE_N"], xS0_E, xS0_N, muS_E, muS_N,
        )
    else:
        geom = derive_pspl_photastrom_reduced(
            t0, p["u0_amp"], p["tE"], p["thetaE"], p["piS"],
            p["piE_E"], p["piE_N"], xS0_E, xS0_N, muS_E, muS_N,
        )

    (_u0, _hat, _tE, _piE_E, _piE_N, xS0, xL0, muS, muL,
     thetaE_amp, piS, piL) = geom
    return t0, xS0, xL0, muS, muL, thetaE_amp, piS, piL


def _source_magnitudes(p):
    """Primary and secondary baseline magnitudes.

    Param1 stores the two magnitudes. Param2 and Param3 store
    ``mag_base``, ``b_sff``, and ``fratio_bin``.

    Parameters
    ----------
    p : dict
        Filter-local fitter parameters.

    Returns
    -------
    mag_pri, mag_sec : float
    """
    if "mag_src_pri" in p:
        return p["mag_src_pri"], p["mag_src_sec"]

    # Same split as BSPL_PhotAstromParam2.__init__.
    mag_pri = (
        p["mag_base"]
        - 2.5 * jnp.log10(p["b_sff"])
        + 2.5 * jnp.log10(1.0 + p["fratio_bin"])
    )
    mag_sec = (
        p["mag_base"]
        - 2.5 * jnp.log10(p["b_sff"])
        + 2.5 * jnp.log10(1.0 + 1.0 / p["fratio_bin"])
    )
    return mag_pri, mag_sec


def _kepler_elements(p, dS):
    """Source-orbit elements in the units ``oal2xy`` expects.

    Parameters
    ----------
    p : dict
        Filter-local fitter parameters. ``log_a`` is Param4; otherwise
        ``aleph``, ``aleph_sec``, and ``p`` are fitted. A missing ``e``
        is a circular orbit.
    dS : float
        Source distance (pc), used only to turn Param4's semi-major
        axis into a period.

    Returns
    -------
    aleph, aleph_sec : float
        Primary and secondary semi-major axes (arcsec).
    period : float
        Period (days).
    ecc : float
        Eccentricity.
    """
    if "log_a" in p:
        a_mas = 10.0 ** p["log_a"]
        m_tot = p["mSp"] + p["mSs"]
        aleph_sec = p["mSp"] / m_tot * a_mas
        aleph = a_mas - aleph_sec
        a_AU = dS * a_mas * 1e-3
        period = _KEPLER_DAY * jnp.sqrt(a_AU ** 3 / m_tot)
        ecc = p["e"]
    else:
        aleph = p["aleph"]
        aleph_sec = p["aleph_sec"]
        period = p["p"]
        ecc = p["e"] if "e" in p else jnp.asarray(0.0)

    # Host Orbit stores aleph in arcsec.
    return aleph * 1e-3, aleph_sec * 1e-3, period, ecc


def _sky_tracks(p, t, parallax_vectors):
    """Unlensed primary, secondary, and lens positions (arcsec).

    Static sources share ``muS``. A linear orbit adds
    ``delta_muS_sec``. An accelerated orbit also adds ``accSsec``.
    A Keplerian orbit moves both sources about their center of mass
    with ``oal2xy``, and places that center so the primary sits on
    ``xS0`` at ``t0``.

    Parameters
    ----------
    p : dict
        Filter-local fitter parameters.
    t : array_like, shape (N_times,)
        Observation times (MJD).
    parallax_vectors : array_like or None
        Parallax table, shape ``(N_times, 2)``. ``None`` for noPar.

    Returns
    -------
    xS1, xS2, xL : jnp.ndarray, shape (N_times, 2)
    thetaE_amp : float
        Einstein radius (mas).
    mag_pri, mag_sec : float
    """
    from bagle.jax.orbits import oal2xy

    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    t0, xS0, xL0, muS, muL, thetaE_amp, piS, piL = _primary_geometry(p)
    dt = ((t - t0) / _DAYS_PER_YEAR).reshape(-1, 1)

    # Lens track. mu is mas/yr; positions are arcsec.
    xL = xL0.reshape(1, 2) + dt * muL.reshape(1, 2) * 1e-3

    if "omega_pri" in p:
        # dS (pc) only enters Param4's period. Param1 stores dL/dL_dS.
        if "dL" in p:
            dS = p["dL"] / p["dL_dS"]
        else:
            dS = jax_physics._PI_MAS_PER_PC / piS
        aleph, aleph_sec, period, ecc = _kepler_elements(p, dS)

        # Offsets from the source CoM. Shift the CoM so the primary
        # is at the input xS0 when t = t0.
        x0, y0, x20, y20 = oal2xy(
            jnp.asarray([t0]), p["omega_pri"], p["big_omega_sec"],
            p["i"], ecc, period, p["tp"], aleph, aleph_sec,
        )
        d_pri0 = jnp.stack([x0[0], y0[0]])
        xS0_com = xS0 - d_pri0
        x_com = xS0_com.reshape(1, 2) + dt * muS.reshape(1, 2) * 1e-3
        x, y, x2, y2 = oal2xy(
            t, p["omega_pri"], p["big_omega_sec"], p["i"], ecc,
            period, p["tp"], aleph, aleph_sec,
        )
        xS1 = x_com + jnp.stack([x, y], axis=1)
        xS2 = x_com + jnp.stack([x2, y2], axis=1)
    else:
        alpha = p["alpha"] * jnp.pi / 180.0
        sep_vec = p["sep"] * 1e-3 * jnp.stack(
            [jnp.sin(alpha), jnp.cos(alpha)])
        xS0_sec = xS0 + sep_vec

        # Secondary proper motion. Absent delta means a static binary.
        if "delta_muS_sec_E" in p:
            muS_sec = muS + jnp.stack(
                [p["delta_muS_sec_E"], p["delta_muS_sec_N"]])
        else:
            muS_sec = muS

        xS1 = xS0.reshape(1, 2) + dt * muS.reshape(1, 2) * 1e-3
        xS2 = xS0_sec.reshape(1, 2) + dt * muS_sec.reshape(1, 2) * 1e-3
        if "accSsec_E" in p:
            acc = jnp.stack([p["accSsec_E"], p["accSsec_N"]])
            xS2 = xS2 + 0.5 * (dt ** 2) * acc.reshape(1, 2) * 1e-3

    # Same parallax on both sources; the lens uses piL.
    if parallax_vectors is not None:
        pvec = jnp.asarray(parallax_vectors, dtype=jnp.float64)
        shift_s = piS * pvec * 1e-3
        xS1 = xS1 + shift_s
        xS2 = xS2 + shift_s
        xL = xL + piL * pvec * 1e-3

    mag_pri, mag_sec = _source_magnitudes(p)
    return xS1, xS2, xL, thetaE_amp, mag_pri, mag_sec


def _image_state(u_vec, xL, thetaE_amp):
    """Plus and minus image positions (arcsec) and amplifications.

    Parameters
    ----------
    u_vec : array_like, shape (N_times, 2)
        Einstein-radius separation, source minus lens.
    xL : array_like, shape (N_times, 2)
        Lens position (arcsec).
    thetaE_amp : float
        Einstein radius (mas).

    Returns
    -------
    x_plus, x_minus : jnp.ndarray, shape (N_times, 2)
    a_plus, a_minus : jnp.ndarray, shape (N_times,)
    amp : jnp.ndarray, shape (N_times,)
        Sum of the two image amplifications.
    """
    u_amp = jnp.linalg.norm(u_vec, axis=1)
    u_hat = u_vec / u_amp[:, jnp.newaxis]
    sqrt_term = jnp.sqrt(u_amp ** 2 + 4.0)
    u_plus = ((u_amp + sqrt_term) / 2.0)[:, jnp.newaxis] * u_hat
    u_minus = ((u_amp - sqrt_term) / 2.0)[:, jnp.newaxis] * u_hat

    # thetaE_amp is mas; image offsets enter the sky in arcsec.
    scale = thetaE_amp * 1e-3
    x_plus = xL + u_plus * scale
    x_minus = xL + u_minus * scale
    amp = (u_amp ** 2 + 2.0) / (u_amp * sqrt_term)
    a_plus = 0.5 * (amp + 1.0)
    a_minus = 0.5 * (amp - 1.0)
    return x_plus, x_minus, a_plus, a_minus, amp


def bspl_photastrom_centroid(xS1, xS2, xL, thetaE_amp, mag_pri, mag_sec,
                             b_sff):
    """Flux-weighted lensed centroid, matching ``BSPL_PhotAstrom.get_astrometry``.

    Parameters
    ----------
    xS1, xS2, xL : array_like, shape (N_times, 2)
        Unlensed primary, secondary, and lens (arcsec).
    thetaE_amp : float
        Einstein radius (mas).
    mag_pri, mag_sec : float
        Unlensed source magnitudes.
    b_sff : float
        Source flux fraction. A negative lens flux is clipped to zero.

    Returns
    -------
    pos : jnp.ndarray, shape (N_times, 2)
        Unresolved centroid (arcsec).
    """
    thetaE = thetaE_amp * 1e-3
    u1 = (xS1 - xL) / thetaE
    u2 = (xS2 - xL) / thetaE
    x1p, x1m, a1p, a1m, _a1 = _image_state(u1, xL, thetaE_amp)
    x2p, x2m, a2p, a2m, _a2 = _image_state(u2, xL, thetaE_amp)

    f1 = mag2flux_jax(mag_pri)
    f2 = mag2flux_jax(mag_sec)
    f_lensed = a1p * f1 + a1m * f1 + a2p * f2 + a2m * f2
    x_lensed = (
        x1p * (a1p * f1)[:, jnp.newaxis]
        + x1m * (a1m * f1)[:, jnp.newaxis]
        + x2p * (a2p * f2)[:, jnp.newaxis]
        + x2m * (a2m * f2)[:, jnp.newaxis]
    ) / f_lensed[:, jnp.newaxis]

    # Blend light sits on the lens. b_sff > 1 is a dark lens.
    fL = jax_physics.lens_flux_for_astrometry_jax(f1 + f2, b_sff)
    pos = (
        x_lensed * f_lensed[:, jnp.newaxis] + xL * fL
    ) / (f_lensed + fL)[:, jnp.newaxis]
    return pos


def bspl_photastrom_magnitudes(xS1, xS2, xL, thetaE_amp, mag_pri, mag_sec,
                               b_sff):
    """Unresolved magnitude, matching ``BSPL.get_photometry``.

    The blend term is not clipped. Astrometry clips it separately.

    Parameters
    ----------
    xS1, xS2, xL : array_like, shape (N_times, 2)
        Unlensed primary, secondary, and lens (arcsec).
    thetaE_amp : float
        Einstein radius (mas).
    mag_pri, mag_sec, b_sff : float
        Source magnitudes and the source flux fraction.

    Returns
    -------
    mag : jnp.ndarray, shape (N_times,)
    """
    thetaE = thetaE_amp * 1e-3
    u1 = (xS1 - xL) / thetaE
    u2 = (xS2 - xL) / thetaE
    _x1p, _x1m, _a1p, _a1m, amp1 = _image_state(u1, xL, thetaE_amp)
    _x2p, _x2m, _a2p, _a2m, amp2 = _image_state(u2, xL, thetaE_amp)

    f1 = mag2flux_jax(mag_pri)
    f2 = mag2flux_jax(mag_sec)
    flux = f1 * amp1 + f2 * amp2
    flux = flux + (f1 + f2) * (1.0 - b_sff) / b_sff
    mag = flux2mag_jax(flux)
    return mag


def bspl_photastrom_log_likely_photometry(names, vec, t, mag_obs, mag_err,
                                          b_sff, parallax_vectors=None,
                                          gp_params=None,
                                          fixed_jitter=True):
    """Photometric log-likelihood for one BSPL PhotAstrom filter.

    Parameters
    ----------
    names : sequence of str
        ``fitter_param_names``.
    vec : array_like
        Filter-local values in that order.
    t, mag_obs, mag_err : array_like, shape (N_times,)
        Times (MJD), magnitudes, and uncertainties.
    b_sff : float
        Source flux fraction for this filter.
    parallax_vectors : array_like or None
        Parallax table, shape ``(N_times, 2)``.
    gp_params : dict or None
        Gaussian-process hyperparameters.
    fixed_jitter : bool
        Use a fixed jitter term when a GP is present.

    Returns
    -------
    lnL : float
    """
    p = _param_dict(names, vec)
    xS1, xS2, xL, thetaE_amp, mag_pri, mag_sec = _sky_tracks(
        p, t, parallax_vectors)
    mag_model = bspl_photastrom_magnitudes(
        xS1, xS2, xL, thetaE_amp, mag_pri, mag_sec, b_sff)

    if gp_params is None:
        lnL = gaussian_log_likelihood_sum(mag_model, mag_obs, mag_err)
    else:
        from bagle.jax.gp import gp_log_likely_photometry

        lnL = gp_log_likely_photometry(
            t, mag_obs, mag_err, mag_model, gp_params,
            fixed_jitter=fixed_jitter,
        )
    return lnL


def bspl_photastrom_log_likely_astrometry(names, vec, t, x_obs, y_obs,
                                          x_err, y_err, b_sff=1.0,
                                          parallax_vectors=None):
    """Astrometric log-likelihood for one BSPL PhotAstrom filter.

    Parameters
    ----------
    names : sequence of str
        ``fitter_param_names``.
    vec : array_like
        Filter-local values in that order.
    t, x_obs, y_obs, x_err, y_err : array_like, shape (N_times,)
        Times (MJD) and East/North positions and uncertainties (arcsec).
    b_sff : float, optional
        Source flux fraction.
    parallax_vectors : array_like or None
        Parallax table, shape ``(N_times, 2)``.

    Returns
    -------
    lnL : float
    """
    p = _param_dict(names, vec)
    xS1, xS2, xL, thetaE_amp, mag_pri, mag_sec = _sky_tracks(
        p, t, parallax_vectors)
    pos = bspl_photastrom_centroid(
        xS1, xS2, xL, thetaE_amp, mag_pri, mag_sec, b_sff)
    lnL = gaussian_astrometry_log_likelihood_sum(
        pos, x_obs, y_obs, x_err, y_err)
    return lnL
