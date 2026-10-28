"""
Uniform time-first layouts for image, amplification, photometry and
astrometry arrays across every model family (model.py and model_jax.py).

Conventions (N times, S sources, I images per source, L lenses):

* ``get_all_arrays`` -> ``images`` complex ``(N, S, I)`` (East + i North,
  in each family's native frame) and ``amps`` ``(N, S, I)``.
* ``get_resolved_amplification`` -> ``(N, S, I)``, the same as ``amps``.
* ``get_amplification`` -> ``(N,)``, flux-weighted over sources of the sum
  over images.
* ``get_resolved_photometry`` -> ``(N, S, I)`` magnitudes of each image,
  without blend flux.
* ``get_resolved_astrometry`` -> ``(N, S, I, 2)`` real ``[E, N]``.
* ``get_resolved_lens_astrometry`` -> ``(N, L, 2)``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("jax")

import bagle.model as m_np
import bagle.model_jax as m_jx

import model_old_vs_jax_fixtures as fx

# Family representative -> (S sources, I images, L lenses or None).
FAMILIES = {
    'PSPL_Phot_noPar_Param1': (1, 2, None),
    'PSPL_PhotAstrom_noPar_Param1': (1, 2, None),
    'PSBL_Phot_noPar_Param1': (1, 5, 2),
    'PSBL_PhotAstrom_noPar_Param1': (1, 5, 2),
    'BSPL_Phot_noPar_Param1': (2, 2, None),
    'BSPL_PhotAstrom_noPar_Param1': (2, 2, None),
    'BSBL_PhotAstrom_noPar_Param1': (2, 5, 2),
    'FSPL_Phot_noPar_Param1': (1, 2, None),
    'FSPL_PhotAstrom_noPar_Param1': (1, 2, None),
    'FSBL_Phot_noPar_Param1': (1, 1, 2),
    'FSBL_PhotAstrom_noPar_Param1': (1, 1, 2),
    'BFSPL_PhotAstrom_noPar_Param1': (2, 2, None),
    'BFSBL_PhotAstrom_noPar_Param1': (2, 1, 2),
}

# Family prefixes not converted yet (xfail until their layout lands).
PENDING = {'BSBL', 'FSPL', 'FSBL', 'BFSPL', 'BFSBL'}


def _cases():
    """
    Parametrize (module, class name) pairs, xfail for pending families.

    Returns
    -------
    cases : list of pytest.param
    """
    cases = []
    for name in FAMILIES:
        for mod, tag in ((m_np, 'model'), (m_jx, 'model_jax')):
            if not hasattr(mod, name):
                continue
            marks = ()
            if name.split('_')[0] in PENDING:
                marks = (pytest.mark.xfail(reason='layout pending', strict=False),)
            cases.append(pytest.param(mod, name, id=f'{tag}-{name}', marks=marks))

    return cases


def _build(mod, name):
    """
    Build a fixture instance and an epoch grid around t0.

    Parameters
    ----------
    mod : module
        ``bagle.model`` or ``bagle.model_jax``.
    name : str
        Concrete class name.

    Returns
    -------
    inst : model instance
    t : ndarray, shape (7,)
        Times in MJD.
    """
    cls = getattr(mod, name)
    args, kw = fx.build_init_args(cls, mod)
    inst = cls(*args, **kw)
    fx._post_init(inst)

    t0 = float(np.atleast_1d(inst.t0)[0])
    tE = float(np.atleast_1d(inst.tE)[0])
    t = np.linspace(t0 - tE, t0 + tE, 7)

    return inst, t


def _source_fluxes(inst):
    """
    Unlensed flux of each source in filter 0.

    Parameters
    ----------
    inst : model instance

    Returns
    -------
    f_src : ndarray, shape (S,)
    """
    if hasattr(inst, 'mag_src_pri'):
        mags = [inst.mag_src_pri[0], inst.mag_src_sec[0]]
    else:
        mags = [inst.mag_src[0]]

    return np.array([m_np.mag2flux(m) for m in mags])


@pytest.mark.parametrize('mod, name', _cases())
def test_get_all_arrays_layout(mod, name):
    """
    get_all_arrays returns complex images and amps, both (N, S, I).
    """
    S, I, _L = FAMILIES[name]
    inst, t = _build(mod, name)

    out = inst.get_all_arrays(t)
    assert isinstance(out, tuple) and len(out) == 2
    images, amps = (np.asarray(a) for a in out)

    assert np.iscomplexobj(images)
    assert images.shape == (t.size, S, I)
    assert amps.shape == (t.size, S, I)

    # A missing image is missing in both arrays.
    np.testing.assert_array_equal(np.isfinite(images), np.isfinite(amps))

    return None


@pytest.mark.parametrize('mod, name', _cases())
def test_amplification_layout(mod, name):
    """
    Resolved amps equal get_all_arrays amps and sum to get_amplification.
    """
    S, I, _L = FAMILIES[name]
    inst, t = _build(mod, name)
    _img, amps = inst.get_all_arrays(t)
    amps = np.asarray(amps, dtype=float)

    A_res = np.asarray(inst.get_resolved_amplification(t), dtype=float)
    assert A_res.shape == (t.size, S, I)
    np.testing.assert_allclose(A_res, amps, rtol=1e-10, equal_nan=True)

    # Total: sum over images, flux-weighted over sources.
    f_src = _source_fluxes(inst)
    A_src = np.nansum(amps, axis=2)
    A_exp = A_src @ f_src / f_src.sum()

    A = np.asarray(inst.get_amplification(t), dtype=float)
    assert A.shape == (t.size,)
    np.testing.assert_allclose(A, A_exp, rtol=1e-10)

    return None


@pytest.mark.parametrize('mod, name', _cases())
def test_resolved_photometry_layout(mod, name):
    """
    Resolved photometry is per image without blend; fluxes add up.
    """
    S, I, _L = FAMILIES[name]
    inst, t = _build(mod, name)
    _img, amps = inst.get_all_arrays(t)
    amps = np.asarray(amps, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mag_res = np.asarray(inst.get_resolved_photometry(t), dtype=float)
    assert mag_res.shape == (t.size, S, I)

    # Each image: source flux times its own amplification (no blend).
    f_src = _source_fluxes(inst)
    flux_img = amps * f_src[np.newaxis, :, np.newaxis]
    ok = np.isfinite(flux_img) & (flux_img > 0)
    np.testing.assert_allclose(m_np.mag2flux(mag_res[ok]), flux_img[ok],
                               rtol=1e-8)

    # Image fluxes plus the blend give the unresolved photometry.
    f_blend = f_src.sum() * (1.0 - inst.b_sff[0]) / inst.b_sff[0]
    flux_tot = np.nansum(flux_img, axis=(1, 2)) + f_blend
    mag = np.asarray(inst.get_photometry(t), dtype=float)
    np.testing.assert_allclose(m_np.mag2flux(mag), flux_tot, rtol=1e-8)

    return None


@pytest.mark.parametrize('mod, name', _cases())
def test_resolved_astrometry_layout(mod, name):
    """
    Resolved astrometry is real (N, S, I, 2), finite where images exist.
    """
    S, I, _L = FAMILIES[name]
    inst, t = _build(mod, name)
    images, _amps = inst.get_all_arrays(t)

    x_res = np.asarray(inst.get_resolved_astrometry(t), dtype=float)
    assert x_res.shape == (t.size, S, I, 2)
    np.testing.assert_array_equal(np.isfinite(x_res[..., 0]),
                                  np.isfinite(np.asarray(images)))

    return None


@pytest.mark.xfail(reason='lens astrometry layout pending', strict=False)
@pytest.mark.parametrize('mod, name', [
    c for c in _cases() if FAMILIES[c.values[1]][2] is not None])
def test_resolved_lens_astrometry_layout(mod, name):
    """
    Resolved lens astrometry is time first, (N, L, 2).
    """
    _S, _I, L = FAMILIES[name]
    inst, t = _build(mod, name)

    xL = np.asarray(inst.get_resolved_lens_astrometry(t), dtype=float)
    assert xL.shape == (t.size, L, 2)

    return None
