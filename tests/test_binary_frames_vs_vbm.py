"""
Binary-lens t0 / u0 input frames vs VBMicrolensing.

VBMicrolensing ``BinaryLightCurve`` takes ``t0`` / ``u0`` about the lens
center of mass, with the primary (mass ``1/(1+q)``) at ``x = -s q/(1+q)``.
Each test builds a static binary-lens PhotAstrom model, converts its *input*
closest approach (about the geometric center, the CoM, or the primary) into
VBM's CoM frame, and compares light curves. Only the orientation (the VBM
``alpha`` angle) is read from the model's lens and source positions.

Several motion directions are used because the sign of the CoM / primary
``u0_amp`` shift depends on which side of the motion the positive-``u0_amp``
direction lies (see ``docs/dev/audits/binary_frames/binary_frame_audit.md``).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

VBM = pytest.importorskip("VBMicrolensing")

from bagle import model, model_jax

# Inputs; each class takes the subset its constructor needs.
VALS = dict(
    mLp=10.0, mLs=5.0, t0=57000.0, t0_com=57000.0, t0_p=57000.0,
    t0_prim=57000.0, u0_amp=0.1, u0_amp_com=0.1, u0_amp_prim=0.1,
    beta=0.4, beta_p=0.4, tE=100.0, thetaE=5.0, log10_thetaE=np.log10(5.0),
    piS=0.25, piE_E=0.05, piE_N=0.03, xS0_E=[0.0], xS0_N=[0.01], muL_E=0.0,
    muL_N=0.0, muS_E=4.0, muS_N=-3.0, dL=2000.0, dS=4000.0, q=0.5, sep=3.0,
    alpha=90.0, b_sff=[1.0], mag_src=[18.0], mag_base=[18.0],
    dmag_Lp_Ls=[20.0], radiusS=5e-6, raL=268.0, decL=-29.0)

# Motion directions (via piE or muS, depending on the class) and binary
# angles: northward, southward, and south-westward relative motion. Angles
# keep |sin(phi)| large (a big u0 shift) and the tracks off the caustics.
GEOMETRIES = {
    "north": dict(piE_E=0.05, piE_N=0.03, muS_E=4.0, muS_N=3.0, alpha=200.0),
    "south": dict(piE_E=0.05, piE_N=-0.03, muS_E=4.0, muS_N=-3.0, alpha=160.0),
    "southwest": dict(piE_E=-0.04, piE_N=-0.02, muS_E=-4.0, muS_N=-2.0,
                      alpha=330.0),
}
for _g in GEOMETRIES.values():
    _g["piEN_piEE"] = _g["piE_N"] / _g["piE_E"]

# (class, input frame, input time name, input impact-parameter name).
CASES = [
    ("PSBL_PhotAstrom_noPar_Param1", "geometric", "t0", "beta"),
    ("PSBL_PhotAstrom_noPar_Param2", "geometric", "t0", "u0_amp"),
    ("PSBL_PhotAstrom_noPar_Param4", "CoM", "t0_com", "u0_amp_com"),
    ("PSBL_PhotAstrom_Par_Param5", "primary", "t0_prim", "u0_amp_prim"),
    ("PSBL_PhotAstrom_Par_Param7", "primary", "t0_p", "beta_p"),
    ("FSBL_PhotAstrom_noPar_Param4", "CoM", "t0_com", "u0_amp_com"),
    ("FSBL_PhotAstrom_Par_Param5", "primary", "t0_prim", "u0_amp_prim"),
    ("FSBL_PhotAstrom_Par_Param7", "primary", "t0_p", "beta_p"),
]


def build(mod, name, geometry):
    """
    Static binary-lens model from ``VALS``, with parallax switched off.

    Parameters
    ----------
    mod : module
        :mod:`bagle.model` or :mod:`bagle.model_jax`.
    name : str
        Composed class name.
    geometry : dict
        Overrides of ``VALS`` setting the motion direction and binary angle.

    Returns
    -------
    m : model instance
        Model with rectilinear source-lens motion.
    """
    if not hasattr(mod, name):
        pytest.skip(f"{name} is not in {mod.__name__}")
    cls = getattr(mod, name)
    vals = {**VALS, **geometry}

    # Constructor arguments come from the Param mixin (the class defining
    # fitter_param_names); composed leaves take *args, **kwargs.
    param_cls = next(b for b in cls.__mro__[1:]
                     if "fitter_param_names" in b.__dict__)
    args = inspect.signature(param_cls.__init__).parameters
    m = cls(**{k: vals[k] for k in args if k in vals})

    # Rectilinear tracks, so the closest approach is well defined.
    m.parallaxFlag = False
    return m


def vbm_orientation(m):
    """
    VBM binary-frame quantities read from the model's geometry.

    Parameters
    ----------
    m : model instance
        Static binary-lens PhotAstrom model with parallax off.

    Returns
    -------
    s : float
        Lens separation (Einstein radii of the total mass).
    q : float
        Mass ratio ``mLs / mLp``.
    alpha : float
        VBM trajectory angle (rad).
    """
    t = np.array([m.t0 - m.tE, m.t0 + m.tE])
    scale = m.thetaE_amp * 1e-3
    lens = m.get_resolved_lens_astrometry(t) / scale
    src = m.get_source_astrometry_unlensed(t) / scale

    # VBM x axis points from the primary (at -x) toward the secondary.
    axis = lens[0, 1] - lens[0, 0]
    s = np.hypot(*axis)
    ex = axis / s
    ey = np.array([-ex[1], ex[0]])

    # Source motion relative to the (static) lenses, per tE, in VBM axes.
    rel = (src[1] - lens[1, 0]) - (src[0] - lens[0, 0])
    v = np.array([rel @ ex, rel @ ey]) / 2.0

    # VBM track: y1 = -tau cos(alpha) + u0 sin(alpha),
    #            y2 = -tau sin(alpha) - u0 cos(alpha).
    alpha = np.arctan2(-v[1], -v[0])
    return s, m.mLs / m.mLp, alpha


def vbm_lightcurve(m, t, frame, t_ref, u_ref):
    """
    VBM light curve for a closest approach given in ``frame``.

    Parameters
    ----------
    m : model instance
        Model supplying ``tE``, the source size, and the binary orientation.
    t : ndarray, shape (N_times,)
        Times (MJD).
    frame : {'geometric', 'CoM', 'primary'}
        Lens point the input closest approach refers to.
    t_ref : float
        Input closest-approach time (MJD).
    u_ref : float
        Input impact parameter (Einstein radii, signed in VBM's frame).

    Returns
    -------
    amp : ndarray, shape (N_times,)
        VBM magnification.
    """
    s, q, alpha = vbm_orientation(m)

    # Position of the input reference point on VBM's binary axis.
    x_p = {"CoM": 0.0, "geometric": s * (1 - q) / (2 * (1 + q)),
           "primary": -s * q / (1 + q)}[frame]

    # Shift the closest approach from that point to the CoM origin.
    t0_vbm = t_ref + x_p * np.cos(alpha) * m.tE
    u0_vbm = u_ref + x_p * np.sin(alpha)

    # Finite-source classes carry radiusS (arcsec); point sources get a tiny rho.
    rho = getattr(m, "radiusS", 0.0) * 1e3 / m.thetaE_amp or 1e-5

    vbm = VBM.VBMicrolensing()
    vbm.Tol = 1e-6
    vbm.RelTol = 1e-7
    par = [np.log(s), np.log(q), u0_vbm, alpha, np.log(rho), np.log(m.tE),
           t0_vbm]
    return np.array(vbm.BinaryLightCurve(par, t)[0])


@pytest.mark.parametrize("geom", GEOMETRIES)
@pytest.mark.parametrize("mod", [model, model_jax], ids=["model", "model_jax"])
@pytest.mark.parametrize("name,frame,t_name,u_name", CASES)
def test_binary_input_frame_matches_vbm(mod, name, frame, t_name, u_name, geom):
    """BAGLE light curve equals VBM's for the class's own t0 / u0 inputs."""
    m = build(mod, name, GEOMETRIES[geom])

    # Frame errors are O(1); the FSBL contour integrator is good to a few 1e-3.
    finite = name.startswith("FSBL")
    n_t, tol = (121, 5e-3) if finite else (601, 1e-3)
    t = np.linspace(m.t0 - 1.5 * m.tE, m.t0 + 1.5 * m.tE, n_t)
    a_bagle = np.asarray(m.get_amplification(t))

    # Input closest approach; beta-style impact parameters are in mas.
    t_ref = getattr(m, t_name)
    u_ref = abs(getattr(m, u_name))
    if u_name.startswith("beta"):
        u_ref = u_ref / m.thetaE_amp

    # BAGLE's u0 sign convention is not VBM's: one side must match.
    err = min(np.max(np.abs(vbm_lightcurve(m, t, frame, t_ref, sign * u_ref)
                            / a_bagle - 1)) for sign in (+1, -1))
    assert err < tol
    return None
