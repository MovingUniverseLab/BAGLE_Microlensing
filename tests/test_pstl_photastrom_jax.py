"""Host vs traced PSTL PhotAstrom photometry and astrometry."""
import numpy as np
import pytest

pytest.importorskip('jax')
import jax.numpy as jnp

from bagle import model_jax as mj


def _pstl_photastrom_kw(mLt=2.0):
    """Constructor kwargs for PSTL_PhotAstrom_noPar_Param1."""
    return dict(
        mLp=10.0,
        mLs=5.0,
        mLt=mLt,
        t0=57000.0,
        xS0_E=np.array([0.0]),
        xS0_N=np.array([0.01]),
        beta=0.4,
        muL_E=0.0,
        muL_N=0.0,
        sep12=3.0,
        sep13=2.0,
        muS_E=4.0,
        muS_N=-3.0,
        dL=2000.0,
        dS=4000.0,
        alpha12=90.0,
        alpha13=45.0,
        b_sff=np.array([1.0]),
        mag_src=np.array([18.0]),
        dmag_Lp_Ls=np.array([20.0]),
        dmag_Lp_Lt=np.array([-20.0]),
        root_tol=1e-4,
    )


def _fitter_vec(inst, kw):
    """Pack ``fitter_param_names`` from constructor kwargs.

    Parameters
    ----------
    inst : PSTL_PhotAstromParam1
        Instantiated model (provides ``fitter_param_names``).
    kw : dict
        Constructor kwargs.

    Returns
    -------
    vec : list of float
    """
    vec = []
    for name in inst.fitter_param_names:
        val = kw[name]
        if np.ndim(val) > 0:
            vec.append(float(np.asarray(val).reshape(-1)[0]))
        else:
            vec.append(float(val))
    return vec


def test_pstl_photastrom_jax_photometry_matches_host():
    """Traced PhotAstrom photometry log-likelihood matches host photometry."""
    kw = _pstl_photastrom_kw()
    inst = mj.PSTL_PhotAstrom_noPar_Param1(**kw)
    t = np.array([56900.0, 57000.0, 57100.0, 57200.0])
    mag = np.asarray(inst.get_photometry(t), dtype=float)
    err = np.full_like(mag, 0.02)
    vec = _fitter_vec(inst, kw)

    lnL_jax = inst.jax_log_likely_photometry(
        jnp.asarray(vec), t, mag, err, kw['b_sff'][0], kw['mag_src'][0])
    resid = (mag - mag) / err
    lnL_host = -0.5 * np.sum(resid ** 2 + np.log(2 * np.pi * err ** 2))
    np.testing.assert_allclose(
        np.asarray(lnL_jax), lnL_host, rtol=1e-8, atol=1e-8)
    return None


def test_pstl_photastrom_jax_astrometry_matches_host():
    """Traced PhotAstrom astrometry log-likelihood matches host centroid."""
    kw = _pstl_photastrom_kw()
    inst = mj.PSTL_PhotAstrom_noPar_Param1(**kw)
    t = np.array([56900.0, 57000.0, 57100.0, 57200.0])
    pos = np.asarray(inst.get_astrometry(t), dtype=float)
    x_err = np.full(len(t), 1e-4)
    y_err = np.full(len(t), 1e-4)
    vec = _fitter_vec(inst, kw)

    lnL_jax = inst.jax_log_likely_astrometry(
        jnp.asarray(vec), t, pos[:, 0], pos[:, 1], x_err, y_err,
        b_sff=kw['b_sff'][0], mag_src=kw['mag_src'][0],
        dmag_Lp_Ls=kw['dmag_Lp_Ls'][0],
        dmag_Lp_Lt=kw['dmag_Lp_Lt'][0])

    chi2_x = ((pos[:, 0] - pos[:, 0]) / x_err) ** 2
    chi2_y = ((pos[:, 1] - pos[:, 1]) / y_err) ** 2
    lnL_host = np.sum(
        -0.5 * (chi2_x + chi2_y)
        - 0.5 * np.log(2 * np.pi * x_err ** 2)
        - 0.5 * np.log(2 * np.pi * y_err ** 2))
    np.testing.assert_allclose(
        np.asarray(lnL_jax), lnL_host, rtol=1e-8, atol=1e-8)
    return None


def test_pstl_photastrom_luminous_tertiary_shifts_lens_centroid():
    """A luminous tertiary pulls the unlensed lens centroid toward xL3."""
    kw = _pstl_photastrom_kw()
    kw['b_sff'] = np.array([0.5])
    kw['dmag_Lp_Ls'] = np.array([-20.0])
    kw['dmag_Lp_Lt'] = np.array([0.0])
    inst = mj.PSTL_PhotAstrom_noPar_Param1(**kw)
    t = np.array([57000.0])

    xL = np.asarray(inst.get_resolved_lens_astrometry(t), dtype=float)
    xL_cent = np.asarray(inst.get_lens_astrometry(t), dtype=float)
    expected = 0.5 * (xL[:, 0] + xL[:, 2])
    np.testing.assert_allclose(xL_cent, expected, rtol=0, atol=1e-10)

    # Dark tertiary recovers the two-lens (primary-only) centroid.
    kw_dark = dict(kw)
    kw_dark['dmag_Lp_Lt'] = np.array([-20.0])
    inst_dark = mj.PSTL_PhotAstrom_noPar_Param1(**kw_dark)
    xL_dark = np.asarray(inst_dark.get_lens_astrometry(t), dtype=float)
    np.testing.assert_allclose(xL_dark, xL[:, 0], rtol=0, atol=1e-8)
    return None


def test_pstl_photastrom_jax_astrometry_luminous_tertiary():
    """Traced astrometry matches host when the tertiary carries light."""
    kw = _pstl_photastrom_kw()
    kw['b_sff'] = np.array([0.5])
    kw['dmag_Lp_Ls'] = np.array([-20.0])
    kw['dmag_Lp_Lt'] = np.array([0.0])
    inst = mj.PSTL_PhotAstrom_noPar_Param1(**kw)
    t = np.array([56900.0, 57000.0, 57100.0, 57200.0])
    pos = np.asarray(inst.get_astrometry(t), dtype=float)
    x_err = np.full(len(t), 1e-4)
    y_err = np.full(len(t), 1e-4)
    vec = _fitter_vec(inst, kw)

    lnL_jax = inst.jax_log_likely_astrometry(
        jnp.asarray(vec), t, pos[:, 0], pos[:, 1], x_err, y_err,
        b_sff=kw['b_sff'][0], mag_src=kw['mag_src'][0],
        dmag_Lp_Ls=kw['dmag_Lp_Ls'][0],
        dmag_Lp_Lt=kw['dmag_Lp_Lt'][0])
    chi2_x = ((pos[:, 0] - pos[:, 0]) / x_err) ** 2
    chi2_y = ((pos[:, 1] - pos[:, 1]) / y_err) ** 2
    lnL_host = np.sum(
        -0.5 * (chi2_x + chi2_y)
        - 0.5 * np.log(2 * np.pi * x_err ** 2)
        - 0.5 * np.log(2 * np.pi * y_err ** 2))
    np.testing.assert_allclose(
        np.asarray(lnL_jax), lnL_host, rtol=1e-8, atol=1e-8)
    return None
