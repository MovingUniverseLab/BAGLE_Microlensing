"""Construct and evaluate photometry-only leaves for every family."""
import numpy as np
import pytest

from bagle import model
from bagle import model_jax as mj


T = np.array([56900.0, 57000.0, 57100.0])

# Shared phot-only geometry (Einstein radii). piE != 0 so thetaE_hat exists.
_PHOT_BASE = dict(
    t0=57000.0,
    u0_amp=0.4,
    tE=80.0,
    piE_E=0.05,
    piE_N=-0.02,
)


def _bspl_phot_kw():
    """Constructor kwargs for BSPL / BFSPL phot-only Param1."""
    return dict(
        **_PHOT_BASE,
        sep=0.3,
        phi=40.0,
        mag_src_pri=[19.0],
        mag_src_sec=[20.0],
        b_sff=[1.0],
    )


def _bsbl_phot_kw():
    """Constructor kwargs for BSBL / BFSBL phot-only Param1."""
    return dict(
        **_PHOT_BASE,
        q=0.4,
        sep_SL=0.4,
        sep_S=0.25,
        phi_S=30.0,
        sep_L=0.8,
        phi_L=50.0,
        mag_src_pri=[19.0],
        mag_src_sec=[20.0],
        b_sff=[1.0],
        dmag_Lp_Ls=[20.0],
    )


def _pstl_phot_kw(q_13=0.2):
    """Constructor kwargs for PSTL phot-only Param1."""
    return dict(
        **_PHOT_BASE,
        q_12=0.5,
        q_13=q_13,
        sep12=0.8,
        sep13=0.6,
        phi12=40.0,
        phi13=80.0,
        b_sff=[1.0],
        mag_src=[18.0],
        root_tol=1e-4,
    )


def _psbl_phot_kw():
    """Constructor kwargs for PSBL phot-only Param1."""
    return dict(
        **_PHOT_BASE,
        q=0.5,
        sep=0.8,
        phi=40.0,
        b_sff=[1.0],
        mag_src=[18.0],
        root_tol=1e-4,
    )


@pytest.mark.parametrize('mod', [model, mj])
def test_bsbl_phot_noPar_constructs(mod):
    """BSBL_Phot_noPar_Param1 constructs and returns finite photometry."""
    inst = mod.BSBL_Phot_noPar_Param1(**_bsbl_phot_kw())
    phot = np.asarray(inst.get_photometry(T), dtype=float)
    assert phot.shape == (len(T),)
    assert np.all(np.isfinite(phot))
    xL = np.asarray(inst.get_resolved_lens_astrometry(T), dtype=float)
    assert xL.shape == (len(T), 2, 2)
    return None


@pytest.mark.parametrize('mod', [model, mj])
def test_bfspl_phot_noPar_constructs(mod):
    """BFSPL_Phot_noPar_Param1 constructs and returns finite photometry."""
    kw = _bspl_phot_kw()
    inst = mod.BFSPL_Phot_noPar_Param1(
        **kw, radiusS_pri=0.01, radiusS_sec=0.008)
    phot = np.asarray(inst.get_photometry(T), dtype=float)
    assert phot.shape == (len(T),)
    assert np.all(np.isfinite(phot))
    return None


@pytest.mark.parametrize('mod', [model, mj])
def test_pstl_phot_noPar_constructs(mod):
    """PSTL_Phot_noPar_Param1 constructs and returns finite photometry."""
    inst = mod.PSTL_Phot_noPar_Param1(**_pstl_phot_kw())
    phot = np.asarray(inst.get_photometry(T), dtype=float)
    assert phot.shape == (len(T),)
    assert np.all(np.isfinite(phot))
    xL = np.asarray(inst.get_resolved_lens_astrometry(T), dtype=float)
    assert xL.shape == (len(T), 3, 2)
    images, amps = inst.get_all_arrays(T)
    assert np.asarray(images).shape[0] == len(T)
    assert np.asarray(amps).shape == np.asarray(images).shape
    return None


@pytest.mark.parametrize('mod', [model, mj])
def test_bfspl_phot_tiny_rho_matches_bspl(mod):
    """A-19: tiny-rho BFSPL phot-only matches BSPL_Phot."""
    kw = _bspl_phot_kw()
    bspl = mod.BSPL_Phot_noPar_Param1(**kw)
    bfspl = mod.BFSPL_Phot_noPar_Param1(
        **kw, radiusS_pri=1e-4, radiusS_sec=1e-4)
    phot_b = np.asarray(bspl.get_photometry(T), dtype=float)
    phot_f = np.asarray(bfspl.get_photometry(T), dtype=float)
    np.testing.assert_allclose(phot_f, phot_b, rtol=1e-3, atol=1e-3)
    return None


@pytest.mark.parametrize('mod', [model, mj])
def test_pstl_phot_q13_zero_matches_psbl(mod):
    """PSTL phot-only with q_13=0 recovers PSBL_Phot photometry."""
    pstl = mod.PSTL_Phot_noPar_Param1(**_pstl_phot_kw(q_13=0.0))
    psbl = mod.PSBL_Phot_noPar_Param1(**_psbl_phot_kw())
    phot_t = np.asarray(pstl.get_photometry(T), dtype=float)
    phot_b = np.asarray(psbl.get_photometry(T), dtype=float)
    np.testing.assert_allclose(phot_t, phot_b, rtol=1e-6, atol=1e-6)
    return None


def test_bfsbl_phot_noPar_constructs():
    """BFSBL_Phot_noPar_Param1 lives only on model_jax."""
    inst = mj.BFSBL_Phot_noPar_Param1(
        **_bsbl_phot_kw(), radiusS_pri=0.01, radiusS_sec=0.008)
    phot = np.asarray(inst.get_photometry(T), dtype=float)
    assert phot.shape == (len(T),)
    assert np.all(np.isfinite(phot))
    return None


def test_pstl_phot_jax_log_likely_matches_host():
    """Traced PSTL_PhotParam1 log-likelihood matches host photometry."""
    pytest.importorskip('jax')
    import jax.numpy as jnp

    kw = _pstl_phot_kw()
    inst = mj.PSTL_Phot_noPar_Param1(**kw)
    mag = np.asarray(inst.get_photometry(T), dtype=float)
    err = np.full_like(mag, 0.02)

    names = inst.fitter_param_names
    vec = []
    for name in names:
        if name == 'b_sff':
            vec.append(kw['b_sff'][0])
        elif name == 'mag_src':
            vec.append(kw['mag_src'][0])
        else:
            vec.append(kw[name])

    lnL_jax = inst.jax_log_likely_photometry(
        jnp.asarray(vec), T, mag, err, kw['b_sff'][0], kw['mag_src'][0])
    resid = (mag - mag) / err
    lnL_host = -0.5 * np.sum(resid ** 2 + np.log(2 * np.pi * err ** 2))
    np.testing.assert_allclose(np.asarray(lnL_jax), lnL_host, rtol=1e-8, atol=1e-8)
    return None


@pytest.mark.parametrize('mod, names', [
    (model, [
        'PSPL_Phot_noPar_Param1',
        'PSBL_Phot_noPar_Param1',
        'BSPL_Phot_noPar_Param1',
        'BSBL_Phot_noPar_Param1',
        'FSPL_Phot_noPar_Param1',
        'BFSPL_Phot_noPar_Param1',
        'PSTL_Phot_noPar_Param1',
    ]),
    (mj, [
        'PSPL_Phot_noPar_Param1',
        'PSBL_Phot_noPar_Param1',
        'BSPL_Phot_noPar_Param1',
        'BSBL_Phot_noPar_Param1',
        'FSPL_Phot_noPar_Param1',
        'FSBL_Phot_noPar_Param1',
        'BFSPL_Phot_noPar_Param1',
        'BFSBL_Phot_noPar_Param1',
        'PSTL_Phot_noPar_Param1',
    ]),
])
def test_every_family_has_phot_leaf(mod, names):
    """Every model family exposes a photometry-only composed leaf."""
    for name in names:
        assert hasattr(mod, name), f'{mod.__name__} missing {name}'
        cls = getattr(mod, name)
        assert cls.photometryFlag is True
        assert cls.astrometryFlag is False
    return None
