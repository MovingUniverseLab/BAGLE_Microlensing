"""
Phot-only binary-source along-track (Option A).

Phot-only BSPL / BSBL / BFSPL use one epoch (primary CA) and a static
``(sep, phi)`` offset for the secondary. ``u_sec(t) = u_pri(t) + sep_vec``
keeps the along-track piece, so the secondary peaks at its own time when
``phi`` is not +/-90 deg.
"""
from __future__ import annotations

import numpy as np
import pytest

from bagle import model
from bagle import model_jax as mj


T0 = 57000.0
TE = 80.0
SEP = 0.5
U0_AMP = 0.3


def _bspl_kw(phi):
    """
    Constructor kwargs for phot-only BSPL / BFSPL Param1.

    Parameters
    ----------
    phi : float
        Binary-axis angle relative to muRel (deg).

    Returns
    -------
    dict
    """
    return dict(
        t0=T0,
        u0_amp=U0_AMP,
        tE=TE,
        piE_E=0.05,
        piE_N=0.0,
        sep=SEP,
        phi=phi,
        mag_src_pri=[19.0],
        mag_src_sec=[20.0],
        b_sff=[1.0],
    )


def _bsbl_kw(phi_S):
    """
    Constructor kwargs for phot-only BSBL Param1.

    Parameters
    ----------
    phi_S : float
        Source-binary angle relative to muRel (deg).

    Returns
    -------
    dict
    """
    return dict(
        t0=T0,
        u0_amp=U0_AMP,
        tE=TE,
        piE_E=0.05,
        piE_N=0.0,
        q=0.4,
        sep_SL=0.4,
        sep_S=SEP,
        phi_S=phi_S,
        sep_L=0.8,
        phi_L=50.0,
        mag_src_pri=[19.0],
        mag_src_sec=[20.0],
        b_sff=[1.0],
        dmag_Lp_Ls=[20.0],
    )


@pytest.mark.parametrize('mod', [model, mj])
@pytest.mark.parametrize('phi', [0.0, 45.0, 90.0])
def test_bspl_phot_u_sec_is_static_offset(mod, phi):
    """
    Phot-only BSPL: u_sec - u_pri equals sep_vec at every time.
    """
    inst = mod.BSPL_Phot_noPar_Param1(**_bspl_kw(phi))
    t = np.linspace(T0 - 2.0 * TE, T0 + 2.0 * TE, 41)
    u = np.asarray(inst.get_u(t), dtype=float)
    sep_vec = SEP * np.array(
        (np.sin(inst.phi_rho1_rad), np.cos(inst.phi_rho1_rad)))

    # Constant sky offset between the two sources.
    du = u[:, 1, :] - u[:, 0, :]
    np.testing.assert_allclose(
        du, np.broadcast_to(sep_vec, du.shape), atol=1e-12)

    # Snapshot attribute matches the same offset at t0.
    np.testing.assert_allclose(inst.u0_sec, inst.u0_pri + sep_vec, atol=1e-12)
    return None


@pytest.mark.parametrize('mod', [model, mj])
@pytest.mark.parametrize('phi, expect_shift', [
    (0.0, True),
    (45.0, True),
    (90.0, False),
])
def test_bspl_phot_secondary_peak_vs_phi(mod, phi, expect_shift):
    """
    Secondary |u| minimum shifts off t0 when phi has an along-track piece.
    """
    inst = mod.BSPL_Phot_noPar_Param1(**_bspl_kw(phi))
    t = np.linspace(T0 - 2.5 * TE, T0 + 2.5 * TE, 5001)
    u = np.asarray(inst.get_u(t), dtype=float)
    u_amp_sec = np.linalg.norm(u[:, 1, :], axis=1)
    t_peak = t[np.argmin(u_amp_sec)]

    if expect_shift:
        # Derived CA epoch from the along-track projection.
        t0_sec = float(inst.get_t0_sec())
        assert abs(t_peak - T0) > 0.2 * TE
        assert abs(t_peak - t0_sec) < 0.05 * TE
    else:
        # Pure cross-track: secondary CA coincides with the primary.
        assert abs(t_peak - T0) < 0.05 * TE
    return None


@pytest.mark.parametrize('mod', [model, mj])
def test_bsbl_phot_u_sec_is_static_offset(mod):
    """
    Phot-only BSBL: source-pair offset is sep_S along phi_S at every time.
    """
    inst = mod.BSBL_Phot_noPar_Param1(**_bsbl_kw(0.0))
    t = np.linspace(T0 - TE, T0 + TE, 21)
    u = np.asarray(inst.get_u(t), dtype=float)
    sep_vec = SEP * np.array(
        (np.sin(inst.phi_rho1_rad), np.cos(inst.phi_rho1_rad)))
    du = u[:, 1, :] - u[:, 0, :]
    np.testing.assert_allclose(
        du, np.broadcast_to(sep_vec, du.shape), atol=1e-12)
    return None


@pytest.mark.parametrize('mod', [model, mj])
def test_bfspl_phot_matches_bspl_alongtrack(mod):
    """
    Tiny-rho BFSPL phot-only inherits the same Option A get_u as BSPL.
    """
    kw = _bspl_kw(0.0)
    bspl = mod.BSPL_Phot_noPar_Param1(**kw)
    bfspl = mod.BFSPL_Phot_noPar_Param1(
        **kw, radiusS_pri=1e-4, radiusS_sec=1e-4)
    t = np.linspace(T0 - TE, T0 + TE, 31)
    np.testing.assert_allclose(
        np.asarray(bfspl.get_u(t), dtype=float),
        np.asarray(bspl.get_u(t), dtype=float),
        atol=1e-12)
    return None


def test_model_vs_model_jax_bspl_phot_u():
    """
    NumPy and JAX hosts agree on phot-only BSPL get_u for phi=0.
    """
    kw = _bspl_kw(0.0)
    m_np = model.BSPL_Phot_noPar_Param1(**kw)
    m_jx = mj.BSPL_Phot_noPar_Param1(**kw)
    t = np.linspace(T0 - 1.5 * TE, T0 + 1.5 * TE, 51)
    np.testing.assert_allclose(
        np.asarray(m_np.get_u(t), dtype=float),
        np.asarray(m_jx.get_u(t), dtype=float),
        atol=1e-12)
    return None
