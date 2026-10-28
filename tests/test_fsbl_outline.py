"""
FSBL lensed-image outlines (``get_resolved_astrometry_outline``).

The outlines are binary-lens images of the source limb. They are checked
against the lens equation, against the contour integrator (same frame and,
with ``adaptive=True``, the same limb angles), and against the shift-outline
method. FSPL ``get_resolved_shift_outline`` gets a regression check too.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from bagle import fsbl_contour
from bagle import model as m_np
from bagle import model_jax as m_jx

import model_old_vs_jax_fixtures as fx

# Caustic-crossing equal-mass binary (docs/notebooks/example_FSBL.ipynb).
CAUSTIC_ARGS = (0.5, 0.5, 57000.0, 1e-4, [0.0], [0.0], 0.02, 0.0, 0.0, 5.0,
                0.0, 4000.0, 8000.0, 0.8, 0.0, [1.0], [18.0], [20.0])

# Epochs (days from t0): before, across, and after the caustic.
DT = np.array([-18.43, 8.84, 12.91])


@pytest.fixture(scope="module")
def fsbl():
    """
    Caustic-crossing FSBL_PhotAstrom model.

    Returns
    -------
    fsbl : bagle.model_jax.FSBL_PhotAstrom_noPar_Param1
    """
    mod = m_jx.FSBL_PhotAstrom_noPar_Param1(*CAUSTIC_ARGS, n_outline=120)

    return mod


def _sky_to_frame(mod, t, xS):
    """
    Convert outline sky positions back to the contour-integral frame.

    Parameters
    ----------
    mod : FSBL_PhotAstrom model
    t : ndarray, shape (N_times,)
        Times in MJD.
    xS : ndarray, shape (N_times, N_pts, 5, 2)
        Outline sky positions in arcsec.

    Returns
    -------
    z : ndarray, complex, shape (N_times, N_pts, 5)
        Image positions relative to the lens midpoint (Einstein radii).
    frame : tuple
        Output of ``mod._contour_frame(t)``.
    """
    frame = mod._contour_frame(t)
    shift = frame[-1]

    # Arcsec offsets from the lens origin -> Einstein radii.
    xL0 = np.asarray(mod.get_lens_origin_astrometry(t))
    dx = (xS - xL0[:, None, None, :]) * 1e3 / mod.thetaE_amp

    # Lens-origin frame -> lens-midpoint frame.
    z = dx[..., 0] + 1j * dx[..., 1] - shift[:, None, None]

    return z, frame


def test_outline_images_lie_on_source_limb(fsbl):
    """
    Every outline point maps back onto the source limb.
    """
    t = fsbl.t0 + DT
    n_out = 256
    xS, parity, theta, counts = fsbl.get_resolved_astrometry_outline(
        t, n_outline=n_out, return_all=True)

    # Shapes: [N_times, N_pts, 5, 2] and friends; uniform grid, no padding.
    assert xS.shape == (t.size, n_out, 5, 2)
    assert parity.shape == (t.size, n_out, 5)
    assert theta.shape == (t.size, n_out)
    np.testing.assert_array_equal(counts, n_out)

    z, (zeta_c, z1, z2, m1, m2, rho, _shift) = _sky_to_frame(fsbl, t, xS)
    valid = np.isfinite(z)

    # 3 or 5 images at every limb point, with +/-1 parity where valid.
    n_img = valid.sum(axis=2)
    assert np.all(np.isin(n_img, (3, 5)))
    np.testing.assert_array_equal(np.abs(parity[valid]), 1.0)
    np.testing.assert_array_equal(parity[~valid], 0.0)

    # Lens map of each image = its limb point.
    for i in range(t.size):
        w_limb = zeta_c[i] + rho * np.exp(1j * theta[i])
        w_img = fsbl_contour.lens_map(z[i], z1[i], z2[i], m1, m2)
        resid = np.abs(w_img - w_limb[:, None])[valid[i]]
        assert resid.max() < 1e-8

    # The caustic epochs have 5 images somewhere on the limb.
    assert np.any(n_img == 5)

    return None


def test_adaptive_outline_reproduces_integrator(fsbl):
    """
    Integrating the adaptive outline gives get_all_arrays exactly.
    """
    t = fsbl.t0 + DT
    xS, parity, theta, counts = fsbl.get_resolved_astrometry_outline(
        t, adaptive=True, return_all=True)
    img_ref, _p, amp_ref, _a = fsbl.get_all_arrays(t)

    # Refinement added points where the images are stretched.
    assert np.all(counts >= fsbl._contour_n_init())
    assert np.any(counts > fsbl._contour_n_init())
    assert xS.shape[1] == counts.max()
    assert np.all(np.isnan(theta[np.arange(theta.shape[1]) >= counts[:, None]]))

    zeta_c, z1, z2, m1, m2, rho, shift = fsbl._contour_frame(t)
    for i in range(t.size):
        # Re-integrate on exactly the returned angles.
        th = theta[i, :counts[i]]
        z_img, par, dz, wedge, jac, _n = fsbl_contour._limb_sample_images(
            zeta_c[i], rho, th, z1[i], z2[i], m1, m2, fsbl_contour._ROOT_TOL)
        area, mx, my, _err = fsbl_contour._integrate_epoch(
            th, z_img, par, dz, wedge, jac, rho)

        # Same magnification and centroid as the integrator.
        np.testing.assert_allclose(area / (np.pi * rho ** 2), amp_ref[i],
                                   rtol=1e-12)
        cent = np.array([mx / area + shift[i].real, my / area + shift[i].imag])
        np.testing.assert_allclose(cent, img_ref[i], rtol=0, atol=1e-12)

        # Outline images are the integrator's images (slot order may differ).
        z_out, _frame = _sky_to_frame(fsbl, t[i:i + 1], xS[i:i + 1])
        z_out = z_out[0, :counts[i]]
        got = np.sort_complex(np.where(np.isfinite(z_out), z_out, 0))
        exp = np.sort_complex(np.where(np.isfinite(z_img), z_img, 0))
        np.testing.assert_allclose(got, exp, rtol=0, atol=1e-9)

    return None


def test_shift_outline_matches_astrometry_outline(fsbl):
    """
    Shift outline (mas, lens-origin relative) agrees with the sky outline.
    """
    t = fsbl.t0 + DT
    xS = fsbl.get_resolved_astrometry_outline(t, n_outline=64)
    xSL = fsbl.get_resolved_shift_outline(t, n_outline=64)
    xL0 = np.asarray(fsbl.get_lens_origin_astrometry(t))

    np.testing.assert_allclose(xL0[:, None, None, :] + xSL * 1e-3, xS,
                               rtol=0, atol=1e-14, equal_nan=True)

    return None


def test_limb_darkened_adaptive_outline_uses_outer_disk():
    """
    Limb-darkened FSBL outline uses the outer-disk adaptive angles.
    """
    cls = m_jx.FSBL_Limb_PhotAstrom_noPar_Param1
    args, kw = fx.build_init_args(cls, m_jx)
    mod = cls(*args, **kw)
    fx._post_init(mod)
    assert mod.limb_gamma(0) != 0

    t = mod.t0 + np.array([-0.2, 0.0, 0.2]) * mod.tE
    xS, _par, theta, counts = mod.get_resolved_astrometry_outline(
        t, adaptive=True, return_all=True)

    # Uniform-disk adaptive integrator at r = rho (first disk of the LD sum).
    zeta_c, z1, z2, m1, m2, rho, _shift = mod._contour_frame(t)
    _img, _amp, th_ref, n_ref = fsbl_contour.adaptive_limb_images_amps(
        zeta_c, rho, s=1.0, q=1.0, n_init=mod._contour_n_init(),
        tol=mod.amg_tol, max_pts=mod.amg_max_pts, z1=z1, z2=z2, m1=m1, m2=m2,
        return_theta=True)

    np.testing.assert_array_equal(counts, n_ref)
    np.testing.assert_array_equal(theta, th_ref)
    assert np.all(np.isfinite(xS[:, :, :, 0]).sum(axis=2)[
        np.arange(xS.shape[1]) < counts[:, None]] >= 3)

    return None


@pytest.mark.parametrize("module", [m_np, m_jx], ids=["model", "model_jax"])
def test_fspl_shift_outline_runs(module):
    """
    FSPL get_resolved_shift_outline (was a NameError) matches the outline.
    """
    cls = module.FSPL_PhotAstrom_noPar_Param1
    args, kw = fx.build_init_args(cls, module)
    mod = cls(*args, **kw)
    fx._post_init(mod)

    t = mod.t0 + np.array([-0.5, 0.0, 0.5]) * mod.tE
    xSL = np.asarray(mod.get_resolved_shift_outline(t))
    xS = np.asarray(mod.get_resolved_astrometry_outline(t))
    xL = np.asarray(mod.get_lens_astrometry(t))

    # Shape: [N_times, n_outline, [+, -], [E, N]].
    assert xSL.shape == (t.size, mod.n_outline, 2, 2)
    np.testing.assert_allclose(xL[:, None, None, :] + xSL * 1e-3, xS,
                               rtol=0, atol=1e-12)

    return None
