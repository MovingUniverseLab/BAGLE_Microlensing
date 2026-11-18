"""
Reproduce paper 1 Figures ``magmap.png`` and ``csmap.png``.

``magmap`` -- PAPER: BSBL, q = 1, lens separation 6 mas at tp,
thetaE = 4.9 mas, source separation 13 mas at tp, mags 16 / 20, b_sff = 1.
The map only depends on the lens geometry (caption), so a static PSBL with
the same lenses is used. ASSUMED (read off the published figure): primary
lens at (-5.6, 1.8) mas and secondary at (-1.0, 5.6) mas, i.e. sep = 5.97
mas and alpha = 230.4 deg; source markers at (4.4, 3.6) and (-0.8, 3.6) mas
(5.2 mas apart, not 13 mas, so the figure is not drawn at the source tp).

``csmap`` -- PAPER: PSBL, q = 1, lens separation 5 mas at tp, thetaE =
5.2 mas, mag_src = 20, b_sff = 1. ASSUMED (read off the published figure):
the lenses are drawn only ~1.8 mas apart, at (0.15, -1.15) and (-0.15,
-2.9) mas, which also matches the small central caustic (s ~ 0.34); we use
that drawn geometry. Dark lens (no lens light).
"""
import numpy as np
import matplotlib.pyplot as plt

from bagle import model, plot_models

from common import dist_for_thetaE, save, sky_axes

T0 = 60000.0
DS = 8000.0
M_TOT = 10.0


def static_psbl(mid_mas, sep_mas, alpha_deg, thetaE, mag_src=20.0):
    """
    Static PSBL with its geometric lens midpoint at ``mid_mas``.

    Parameters
    ----------
    mid_mas : tuple of float
        (East, North) lens midpoint (mas).
    sep_mas : float
        Lens separation (mas).
    alpha_deg : float
        Binary axis angle (deg East of North).
    thetaE : float
        Einstein radius of the total mass (mas).
    mag_src : float, optional
        Source magnitude.

    Returns
    -------
    psbl : PSBL_PhotAstrom_noPar_Param1
        Model instance (q = 1, dark lenses).
    """

    # Equal masses; dL picked to give the requested thetaE.
    dL = dist_for_thetaE(M_TOT, thetaE, DS)

    # Source sits on the lens midpoint at t0 (beta = 0); a tiny source
    # proper motion only defines the trajectory direction.
    psbl = model.PSBL_PhotAstrom_noPar_Param1(
        M_TOT / 2, M_TOT / 2, T0, mid_mas[0] * 1e-3, mid_mas[1] * 1e-3,
        0.0, 0.0, 0.0, 1.0, 0.0, dL, DS, sep_mas, alpha_deg,
        [1.0], [mag_src], 0.0)
    return psbl


def images_for_sources(psbl, w):
    """
    Image positions and amplifications for arbitrary source positions.

    Mirrors :meth:`PSBL.get_all_arrays` (rescaled root finding) for a
    fixed lens configuration at ``t0``.

    Parameters
    ----------
    psbl : PSBL model
        Model supplying the lens positions and masses.
    w : numpy array, dtype=complex, shape = [N]
        Source positions (arcsec, East + i North).

    Returns
    -------
    images : numpy array, dtype=complex, shape = [N, 5]
        Image positions (arcsec); NaN for missing images.
    amps : numpy array, shape = [N, 5]
        Image amplifications.
    """

    # Lens positions at t0, repeated for every source position.
    zs, _m = plot_models._resolved_lens_complex(psbl, np.array([psbl.t0]))
    zs = np.asarray(zs).ravel()
    z1 = np.full(w.shape, zs[0], dtype=complex)
    z2 = np.full(w.shape, zs[1], dtype=complex)

    # Rescale into a unit box, solve, and undo the rescaling.
    rw, rz1, rz2, m1, m2, scale, shift = psbl.rescale_complex_pos(
        w.copy(), z1.copy(), z2.copy())
    tol = psbl.root_tol
    psbl.root_tol = tol * scale
    rimg = psbl.get_image_pos_arr(rw, rz1, rz2, m1, m2)
    psbl.root_tol = tol
    images = rimg / scale[:, None] + shift[:, None]

    amps = psbl.get_amp_arr(images, z1, z2)
    return images, amps


def magmap():
    """
    Inverse-ray-shooting magnification map for the BSBL lens geometry.

    Returns
    -------
    None
    """

    xLp, xLs = np.array([-5.6, 1.8]), np.array([-1.0, 5.6])
    d = xLp - xLs
    sep = np.hypot(*d)
    alpha = np.degrees(np.arctan2(d[0], d[1])) % 360
    psbl = static_psbl((xLp + xLs) / 2, sep, alpha, 4.9)
    zs, masses = plot_models._resolved_lens_complex(psbl, np.array([T0]))
    zs = np.asarray(zs).ravel()
    print(f"magmap: sep={sep:.2f} mas alpha={alpha:.1f} deg; lenses (mas) "
          f"{np.round(zs * 1e3, 2)}")

    # Image-plane grid (arcsec), wider than the plotted source window.
    half, n = 0.030, 3000
    x = np.linspace(-half, half, n)
    zg = (x[None, :] + 1j * x[:, None]).ravel()
    w = plot_models.get_source_pos(zg, masses[0], masses[1], zs[0], zs[1])

    # Bin source positions; normalise by the unlensed ray density.
    edges = np.linspace(-0.010, 0.010, 401)
    H, _, _ = np.histogram2d(w.real, w.imag, bins=[edges, edges])
    pix_img = (x[1] - x[0]) ** 2
    pix_src = (edges[1] - edges[0]) ** 2
    mag = H.T * pix_img / pix_src

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(mag, origin="lower", cmap="inferno", vmin=1,
                   vmax=np.percentile(mag, 99.95),
                   extent=[edges[0] * 1e3, edges[-1] * 1e3] * 2)
    fig.colorbar(im, ax=ax, label="Magnification")
    ax.plot(*xLp, "o", color="gray", ms=5, label="Primary Lens")
    ax.plot(*xLs, "^", color="green", ms=9, label="Secondary Lens")
    ax.plot(4.4, 3.6, "h", color="red", ms=10, label="Primary Source")
    ax.plot(-0.8, 3.6, "D", color="blue", ms=9, label="Secondary Source")
    ax.set_xlim(10, -10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel(r"$\Delta\alpha^*$ (mas)")
    ax.set_ylabel(r"$\Delta\delta$ (mas)")
    ax.legend(loc="lower left")
    save(fig, "magmap")
    return None


def csmap():
    """
    Centroid-shift map: |flux-weighted image centroid - source| per source.

    Returns
    -------
    None
    """

    xLp, xLs = np.array([0.15, -1.15]), np.array([-0.15, -2.9])
    d = xLp - xLs
    sep = np.hypot(*d)
    alpha = np.degrees(np.arctan2(d[0], d[1])) % 360
    psbl = static_psbl((xLp + xLs) / 2, sep, alpha, 5.2)

    # Source-plane grid (mas -> arcsec).
    n = 241
    x = np.linspace(-6, 6, n)
    w = ((x[None, :] + 1j * x[:, None]).ravel()) * 1e-3

    # Flux-weighted centroid of the images (dark lenses).
    images, amps = images_for_sources(psbl, w)
    ok = np.isfinite(images) & np.isfinite(amps)
    a = np.where(ok, amps, 0.0)
    cen = np.sum(np.where(ok, images, 0) * a, axis=1) / np.sum(a, axis=1)
    shift = np.abs(cen - w).reshape(n, n) * 1e3

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(shift, origin="lower", cmap="inferno", vmin=0,
                   vmax=np.nanpercentile(shift, 99.9),
                   extent=[x[0], x[-1], x[0], x[-1]])
    fig.colorbar(im, ax=ax, label=r"$|\vec{\delta}_{c,obs}|$ (mas)")
    ax.plot(*xLp, "o", color="gray", ms=6, label="Primary Lens")
    ax.plot(*xLs, "^", color="green", ms=10, label="Secondary Lens")
    ax.set_xlim(6, -6)
    ax.set_ylim(-6, 6)
    ax.set_xlabel(r"$\Delta\alpha^*$ (mas)")
    ax.set_ylabel(r"$\Delta\delta$ (mas)")
    ax.legend(loc="upper right")
    print(f"csmap: sep={sep:.2f} mas alpha={alpha:.1f} deg, max shift "
          f"{np.nanmax(shift):.2f} mas, shift at 6 mas {shift[n // 2, 0]:.2f}")
    save(fig, "csmap")
    return None


if __name__ == "__main__":
    magmap()
    csmap()
