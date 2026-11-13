"""
Trajectory-plotting helpers shared by the PSBL / BSPL / BSBL figure scripts.

All BAGLE positions are in arcsec; everything here is plotted in mas.
"""
import numpy as np

from common import sky_axes

LENS_COLORS = ("k", "purple")
SRC_COLORS = ("orange", "tab:pink")


def binary_lens_images(mod, t):
    """
    Lensed image positions for a binary-lens model.

    Parameters
    ----------
    mod : PSBL or BSBL model
        Model instance.
    t : numpy array, shape = [N]
        Times (MJD).

    Returns
    -------
    img : numpy array, dtype=complex, shape = [N, N_src, 5]
        Image positions (mas); NaN where an image does not exist.
    """

    # Shape of images: [N_times, N_sources, 5] (arcsec).
    images, _amps = mod.get_all_arrays(t)
    return np.asarray(images) * 1e3


def draw_binary_lens(ax, mod, t, img_color="tab:olive", img_size=1.5):
    """
    Plot the lenses, unlensed sources, images and observed centroid.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    mod : PSBL or BSBL model
        Model instance.
    t : numpy array, shape = [N]
        Times (MJD).
    img_color : str, optional
        Colour of the lensed images.
    img_size : float, optional
        Marker size of the image points.

    Returns
    -------
    None
    """

    # Lenses: [N, 2, 2]; centroid: [N, 2].
    xL = np.asarray(mod.get_resolved_lens_astrometry(t)) * 1e3
    cen = np.asarray(mod.get_astrometry(t)) * 1e3
    img = binary_lens_images(mod, t).ravel()

    # Unlensed sources: [N, 2] for one source or [N, 2, 2] for two.
    xS = np.asarray(mod.get_source_astrometry_unlensed(t)) * 1e3
    if hasattr(mod, "get_resolved_source_astrometry_unlensed"):
        xS = np.asarray(mod.get_resolved_source_astrometry_unlensed(t)) * 1e3
    if xS.ndim == 2:
        xS = xS[:, None, :]

    ax.scatter(img.real, img.imag, s=img_size, color=img_color, zorder=1,
               label="Images")
    for k, name in enumerate(["Primary Lens", "Secondary Lens"]):
        ax.plot(xL[:, k, 0], xL[:, k, 1], color=LENS_COLORS[k], lw=2.5,
                label=name, zorder=2)
    for k in range(xS.shape[1]):
        name = "Source" if xS.shape[1] == 1 else \
            ["Primary Source", "Secondary Source"][k]
        if np.ptp(xS[:, k, 0]) + np.ptp(xS[:, k, 1]) < 1e-6:
            ax.plot(xS[0, k, 0], xS[0, k, 1], "o", color=SRC_COLORS[k],
                    ms=9, label=name, zorder=4)
        else:
            ax.plot(xS[:, k, 0], xS[:, k, 1], color=SRC_COLORS[k], lw=2.5,
                    label=name, zorder=2)
    ax.plot(cen[:, 0], cen[:, 1], color="palegreen", lw=3,
            label="Observed Source Position", zorder=3)
    sky_axes(ax)
    return None


def draw_bspl(ax, bspl, t):
    """
    Plot unlensed sources, lens, +/- images and the observed centroid.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    bspl : BSPL model
        Model instance.
    t : numpy array, shape = [N]
        Times (MJD).

    Returns
    -------
    None
    """

    # Shapes: xS [N, 2 src, 2], xL [N, 2], img [N, 2 src, 2 img, 2].
    xS = np.asarray(bspl.get_resolved_source_astrometry_unlensed(t)) * 1e3
    xL = np.asarray(bspl.get_lens_astrometry(t)) * 1e3
    img = np.asarray(bspl.get_resolved_astrometry(t)) * 1e3
    cen = np.asarray(bspl.get_astrometry(t)) * 1e3

    # The major (+) image is the one farther from the lens.
    d = np.linalg.norm(img - xL[:, None, None, :], axis=-1)
    major = np.argmax(d, axis=-1)
    idx = np.arange(len(t))

    ax.plot(xS[:, 0, 0], xS[:, 0, 1], color="orange", lw=2,
            label="Unlensed Primary Source")
    ax.plot(xS[:, 1, 0], xS[:, 1, 1], color="tab:pink", lw=2,
            label="Unlensed Secondary Source")
    ax.plot(xL[:, 0], xL[:, 1], "k.", ms=6, label="Lens")
    styles = [("tab:olive", "Primary"), ("tab:cyan", "Secondary")]
    for s, (col, name) in enumerate(styles):
        ip = img[idx, s, major[:, s]]
        im = img[idx, s, 1 - major[:, s]]
        ax.plot(ip[:, 0], ip[:, 1], ":", color=col, lw=3,
                label=f"Lensed {name} +Image")
        ax.plot(im[:, 0], im[:, 1], "--", color=col, lw=3,
                label=f"Lensed {name} -Image")
    ax.plot(cen[:, 0], cen[:, 1], color="palegreen", lw=3,
            label="Observed Source Position")
    sky_axes(ax)
    return None
