"""
Traced binary-source finite-source likelihoods (:mod:`bagle.jax.bfs`).

Checks the BFSPL and BFSBL Param1 ``jax_log_likely_*`` methods against the
host photometry and astrometry, and that their gradients are finite.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp

from bagle import jax_physics
from bagle import model_jax as mj

# Per-filter parameters (lists on the host model).
FILT = ('xS0_E', 'xS0_N', 'b_sff', 'mag_src_pri', 'mag_src_sec', 'dmag_Lp_Ls',
        'gamma_pri', 'gamma_sec')

# Bright, luminous-lens events with finite-source peaks.
BFSPL_KW = dict(mL=0.6, t0=57000.0, beta=0.02, dL=4000.0, dL_dS=0.5,
                xS0_E=0.001, xS0_N=-0.002, muL_E=1.0, muL_N=3.0, alpha=40.0,
                muS_E=-2.0, muS_N=-1.0, radiusS_pri=8e-6, radiusS_sec=4e-6,
                sep=0.3, b_sff=0.7, mag_src_pri=19.0, mag_src_sec=19.6,
                raL=268.0, decL=-29.0)
BFSBL_KW = dict(mLp=0.5, mLs=0.05, t0=57000.0, xS0_E=0.001, xS0_N=-0.002,
                beta=0.04, muL_E=1.0, muL_N=3.0, muS_E=-2.0, muS_N=-1.0,
                dL=4000.0, dS=8000.0, sepL=0.75, alphaL=70.0, sepS=0.2,
                alphaS=20.0, mag_src_pri=19.0, mag_src_sec=19.6, b_sff=0.7,
                dmag_Lp_Ls=1.5, radiusS_pri=8e-6, radiusS_sec=4e-6,
                raL=268.0, decL=-29.0)

# Tiny data errors: the chi^2 of the traced model at the host values
# measures the model difference.
ERR = 1e-8


def _traced_vs_host(name, kw, t):
    """
    RMS traced-minus-host photometry and astrometry for one class.

    Parameters
    ----------
    name : str
        Composed class name in :mod:`bagle.model_jax`.
    kw : dict
        Scalar parameters (per-filter values are wrapped for the host).
    t : ndarray, shape (N_times,)
        Times (MJD).

    Returns
    -------
    rms_mag : float
        RMS magnitude difference.
    rms_mas : float
        RMS per-axis position difference (mas).
    grad_ok : bool
        Whether the joint log-likelihood gradient is finite.
    """
    cls = getattr(mj, name)
    kw = dict(kw)
    if 'Limb' in name:
        kw.update(gamma_pri=0.6, gamma_sec=0.3)

    # Host model and its photometry / astrometry.
    mod = cls(**{k: ([v] if k in FILT else v) for k, v in kw.items()})
    mag_h = np.asarray(mod.get_photometry(t))
    pos_h = np.asarray(mod.get_astrometry(t))

    # Packed vector, parallax table and fitter-style magnitude tuple.
    vec = jnp.asarray([kw[n] for n in cls.fitter_param_names])
    pvec = None
    if '_Par_' in name:
        pvec = jax_physics.precompute_parallax_vectors(
            kw['raL'], kw['decL'], t, obs_location='earth')
    mags = (kw['mag_src_pri'], kw['mag_src_sec'])
    tj = jnp.asarray(t)

    def lnl(v, dmag=0.0, dpos=0.0, scale=1.0):
        """Joint traced lnL with optional data offsets and error scale."""
        err = jnp.full(t.size, ERR * scale)
        lp = cls.jax_log_likely_photometry(
            v, tj, jnp.asarray(mag_h) + dmag, err, kw['b_sff'], mags,
            parallax_vectors=pvec)
        la = cls.jax_log_likely_astrometry(
            v, tj, jnp.asarray(pos_h[:, 0]) + dpos, jnp.asarray(pos_h[:, 1]),
            err, err, b_sff=kw['b_sff'], parallax_vectors=pvec)
        return lp, la

    # chi^2 at the host values -> RMS differences.
    err = np.full(t.size, ERR)
    lp, la = (float(x) for x in lnl(vec))
    lp0 = float(jax_physics.gaussian_log_likelihood_sum(mag_h, mag_h, err))
    la0 = float(jax_physics.gaussian_astrometry_log_likelihood_sum(
        jnp.asarray(pos_h), pos_h[:, 0], pos_h[:, 1], err, err))
    rms_mag = ERR * np.sqrt(max(-2.0 * (lp - lp0), 0.0) / t.size)
    rms_mas = 1e3 * ERR * np.sqrt(max(-2.0 * (la - la0), 0.0) / (2 * t.size))

    # Gradient at realistic errors (1 mmag-ish, 1 mas) and offset data.
    grad = jax.grad(lambda v: sum(lnl(v, dmag=0.01, dpos=3e-5,
                                      scale=1e5)))(vec)

    return rms_mag, rms_mas, bool(np.all(np.isfinite(np.asarray(grad))))


@pytest.mark.parametrize("name", [
    'BFSPL_PhotAstrom_noPar_Param1',
    'BFSPL_Limb_PhotAstrom_noPar_Param1',
    'BFSPL_Limb_PhotAstrom_Par_Param1',
])
def test_bfspl_traced_matches_host(name):
    """Traced BFSPL photometry and astrometry match the host images."""
    t = np.linspace(56985.0, 57015.0, 30)
    rms_mag, rms_mas, grad_ok = _traced_vs_host(name, BFSPL_KW, t)

    # Host far-field epochs use a 50-point limb (traced: 400).
    assert rms_mag < 2e-5
    assert rms_mas < 2e-4
    assert grad_ok

    return None


@pytest.mark.parametrize("name,n_t", [
    ('BFSBL_PhotAstrom_noPar_Param1', 24),
    ('BFSBL_PhotAstrom_Par_Param1', 24),
    ('BFSBL_Limb_PhotAstrom_noPar_Param1', 8),
])
def test_bfsbl_traced_matches_host(name, n_t):
    """Traced BFSBL photometry and astrometry match the host contour model."""
    t = np.linspace(56985.0, 57015.0, n_t)
    rms_mag, rms_mas, grad_ok = _traced_vs_host(name, BFSBL_KW, t)

    assert rms_mag < 1e-5
    assert rms_mas < 1e-5
    assert grad_ok

    return None


@pytest.mark.parametrize("module", ["model", "model_jax"])
def test_bfspl_host_astrometry_matches_bspl(module):
    """Host BFSPL with tiny sources equals point-source BSPL astrometry."""
    import importlib

    mod = importlib.import_module(f'bagle.{module}')
    kw = {k: ([v] if k in FILT else v) for k, v in BFSPL_KW.items()
          if not k.startswith('radiusS') and k not in ('raL', 'decL')}
    t = np.linspace(56985.0, 57015.0, 31)

    # Blended (b_sff < 1) unequal sources: lens light and per-source weights
    # both matter.
    ps = mod.BSPL_PhotAstrom_noPar_Param1(**kw)
    fs = mod.BFSPL_PhotAstrom_noPar_Param1(radiusS_pri=1e-9, radiusS_sec=1e-9,
                                           **kw)
    dmag = np.asarray(fs.get_photometry(t)) - np.asarray(ps.get_photometry(t))
    dpos = np.asarray(fs.get_astrometry(t)) - np.asarray(ps.get_astrometry(t))
    assert np.max(np.abs(dmag)) < 1e-5
    assert np.max(np.abs(dpos)) * 1e3 < 1e-5

    return None


@pytest.mark.parametrize("module", ["model", "model_jax"])
def test_bfspl_host_astrometry_translation_invariant(module):
    """Shifting the whole BFSPL event on the sky shifts the centroid rigidly."""
    import importlib

    mod = importlib.import_module(f'bagle.{module}')
    kw = {k: ([v] if k in FILT else v) for k, v in BFSPL_KW.items()
          if k not in ('raL', 'decL')}
    t = np.linspace(56985.0, 57015.0, 15)
    pos0 = np.asarray(mod.BFSPL_PhotAstrom_noPar_Param1(**kw).get_astrometry(t))

    # Move the source (and hence the lens) by (1, -2) arcsec.
    shift = np.array([1.0, -2.0])
    kw['xS0_E'] = [kw['xS0_E'][0] + shift[0]]
    kw['xS0_N'] = [kw['xS0_N'][0] + shift[1]]
    pos1 = np.asarray(mod.BFSPL_PhotAstrom_noPar_Param1(**kw).get_astrometry(t))
    assert np.max(np.abs(pos1 - pos0 - shift)) * 1e3 < 1e-6

    return None


@pytest.mark.parametrize("module", ["model", "model_jax"])
def test_bfspl_amplification_photometry_consistent(module):
    """BFSPL amplification / resolved photometry share one axis convention.

    ``get_all_arrays`` amps are ``[t, source, image]``. ``get_amplification``
    is the flux-weighted total (BSPL convention), and the resolved image
    fluxes add back up to ``get_photometry``.

    Parameters
    ----------
    module : str
        ``'model'`` (NumPy reference) or ``'model_jax'``.

    Returns
    -------
    None
    """
    import importlib

    mod = importlib.import_module(f'bagle.{module}')
    kw = {k: ([v] if k in FILT else v) for k, v in BFSPL_KW.items()
          if k not in ('raL', 'decL')}
    t = np.linspace(56985.0, 57015.0, 21)
    fs = mod.BFSPL_PhotAstrom_noPar_Param1(**kw)

    # Image amplifications, shape [t, source, image].
    img_arr, amp_arr = fs.get_all_arrays(t)
    assert amp_arr.shape == (t.size, 2, 2)
    amp_copy = amp_arr.copy()

    # Source fluxes and the blend flux added by b_sff.
    f_src = np.array([mod.mag2flux(kw['mag_src_pri'][0]),
                      mod.mag2flux(kw['mag_src_sec'][0])])
    b_sff = kw['b_sff'][0]
    f_blend = f_src.sum() * (1.0 - b_sff) / b_sff

    # Flux-weighted total amplification reproduces get_photometry.
    A = np.asarray(fs.get_amplification(t, amp_arr=amp_arr))
    assert A.shape == (t.size,)
    f_tot = mod.mag2flux(np.asarray(fs.get_photometry(t, amp_arr=amp_arr)))
    np.testing.assert_allclose(f_src.sum() * A + f_blend, f_tot, rtol=1e-10)

    # Resolved images: same layout as amps; fluxes sum back to the total.
    mag_res = np.asarray(fs.get_resolved_photometry(t, amp_arr=amp_arr))
    assert mag_res.shape == (t.size, 2, 2)
    # Image fluxes (no blend) plus the blend give get_photometry.
    f_res = np.nan_to_num(mod.mag2flux(mag_res))
    np.testing.assert_allclose(f_res.sum(axis=(1, 2)) + f_blend, f_tot,
                               rtol=1e-10)
    np.testing.assert_array_equal(amp_arr, amp_copy)

    # Point-source limit matches BSPL's flux-weighted amplification.
    kw_ps = {k: v for k, v in kw.items() if not k.startswith('radiusS')}
    ps = mod.BSPL_PhotAstrom_noPar_Param1(**kw_ps)
    tiny = mod.BFSPL_PhotAstrom_noPar_Param1(radiusS_pri=1e-9,
                                             radiusS_sec=1e-9, **kw_ps)
    np.testing.assert_allclose(tiny.get_amplification(t),
                               ps.get_amplification(t), rtol=1e-5)

    return None


def test_bfsbl_host_frame_matches_bsbl():
    """BFSBL with tiny sources equals point-source BSBL (luminous lenses)."""
    kw = {k: ([v] if k in FILT else v) for k, v in BFSBL_KW.items()
          if not k.startswith('radiusS')}
    kw['beta'] = 0.3
    t = np.linspace(56900.0, 57100.0, 9)

    # Unequal lens fluxes put the lens light centroid off the binary centre.
    ps = mj.BSBL_PhotAstrom_noPar_Param1(**kw)
    fs = mj.BFSBL_PhotAstrom_noPar_Param1(radiusS_pri=1e-7, radiusS_sec=1e-7,
                                          **kw)
    dmag = np.abs(np.asarray(fs.get_photometry(t))
                  - np.asarray(ps.get_photometry(t)))
    dpos = np.abs(np.asarray(fs.get_astrometry(t))
                  - np.asarray(ps.get_astrometry(t)))
    assert dmag.max() < 1e-4
    assert dpos.max() * 1e3 < 1e-4

    return None
