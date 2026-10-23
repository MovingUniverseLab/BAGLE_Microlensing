"""Hyper-Catalan root tracker: series step, caustic tracks, JAX."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from bagle.hyper_catalan import step_roots
from bagle.model import PSBL


def _nn_err(got, truth):
    """Max nearest-neighbour distance from ``got`` onto ``truth``."""
    used = np.zeros(len(truth), dtype=bool)
    err = 0.0

    for zi in np.asarray(got):
        dist = np.abs(np.asarray(truth) - zi)
        dist[used] = np.inf
        j = int(np.argmin(dist))
        used[j] = True
        err = max(err, float(dist[j]) if np.isfinite(dist[j]) else 1e9)

    return err


def _geometry(s, q, w):
    """Witt inputs for a static binary. ``w`` is complex, shape (N,)."""
    w = np.asarray(w, dtype=np.complex128)
    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)
    z1 = np.full(w.shape, -s * m2 + 0.0j)
    z2 = np.full(w.shape, s * m1 + 0.0j)

    return w, z1, z2, m1, m2


def _psbl():
    inst = PSBL()
    inst.root_tol = 1e-8

    return inst


def _n_physical(z_row):
    return int(np.sum(np.isfinite(np.real(z_row))))


def test_hc_step_matches_numpy_roots():
    """Eq. 9 plus two Newton steps recovers a mildly perturbed root."""
    rng = np.random.default_rng(0)
    worst = 0.0

    for _ in range(25):
        coeffs = rng.normal(size=6) + 1j * rng.normal(size=6)
        coeffs = np.asarray(coeffs, dtype=np.complex128)
        coeffs[-1] = 1.0 + 0.0j
        truth = np.roots(coeffs[::-1])
        seed = truth + 1e-3 * (
            rng.normal(size=5) + 1j * rng.normal(size=5)
        )
        got = step_roots(coeffs, seed, hc_order=2, n_newton=2)
        worst = max(worst, _nn_err(got, truth))

    assert worst < 1e-10


@pytest.mark.parametrize("s", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("q", [1e-3, 0.1, 1.0])
def test_psbl_track_matches_companion(s, q):
    """Offset track, including folds / near-cusps / grazes, to 1e-10."""
    # Imaginary offset misses the lenses (a root through infinity) and
    # still crosses or grazes the caustic for these (s, q).
    imag = 0.05 if q < 0.05 else 0.02
    w = np.linspace(-1.4, 1.4, 31) + 1j * imag
    w, z1, z2, m1, m2 = _geometry(s, q, w)
    inst = _psbl()
    hc = inst.get_image_pos_arr_HC(w, z1, z2, m1, m2, check_sols=False)
    ref = inst.get_image_pos_arr(w, z1, z2, m1, m2, check_sols=False)
    errs = [_nn_err(hc[i], ref[i]) for i in range(len(w))]

    assert max(errs) < 1e-10


def test_axis_fold_segment():
    """Real-axis fold for s=1, q=1, stopping short of the lenses."""
    w = np.linspace(-0.46, -0.22, 25) + 0.0j
    w, z1, z2, m1, m2 = _geometry(1.0, 1.0, w)
    inst = _psbl()
    hc = inst.get_image_pos_arr_HC(w, z1, z2, m1, m2, check_sols=False)
    ref = inst.get_image_pos_arr(w, z1, z2, m1, m2, check_sols=False)
    errs = [_nn_err(hc[i], ref[i]) for i in range(len(w))]

    assert max(errs) < 1e-10


def _near_lens(z, z1, z2, radius=1e-2):
    """True when ``z`` lies within ``radius`` of either lens."""
    return min(abs(z - z1), abs(z - z2)) < radius


@pytest.mark.parametrize("s", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("q", [1e-3, 0.1, 1.0])
def test_image_count_matches(s, q):
    """3 or 5 physical images, same count as the companion solve.

    A planetary image within ~1e-3 of its lens makes ``w = z - m/conj(z)``
    ill-conditioned. ``numpy.roots`` can sit just outside ``root_tol``
    while the polished tracker sits just inside, so that one epoch
    may differ by one image. It is accepted only when all five
    polynomial roots still match and the disputed image is the one
    next to the lens.
    """
    imag = 0.05 if q < 0.05 else 0.02
    w = np.linspace(-1.2, 1.2, 21) + 1j * imag
    w, z1, z2, m1, m2 = _geometry(s, q, w)
    inst = _psbl()
    hc = inst.get_image_pos_arr_HC(w, z1, z2, m1, m2, check_sols=True)
    ref = inst.get_image_pos_arr(w, z1, z2, m1, m2, check_sols=True)
    hc_all = inst.get_image_pos_arr_HC(
        w, z1, z2, m1, m2, check_sols=False
    )
    ref_all = inst.get_image_pos_arr(
        w, z1, z2, m1, m2, check_sols=False
    )

    for i in range(len(w)):
        n_hc = _n_physical(hc[i])
        n_ref = _n_physical(ref[i])

        if n_hc == n_ref:
            assert n_hc in (3, 5)
            continue

        assert _nn_err(hc_all[i], ref_all[i]) < 1e-8
        assert n_hc in (2, 3, 5)
        assert n_ref in (2, 3, 5)
        extra = hc[i] if n_hc > n_ref else ref[i]
        other = ref[i] if n_hc > n_ref else hc[i]

        for z in extra:
            if not np.isfinite(z):
                continue

            dist = np.abs(other - z)
            matched = np.isfinite(other) & (dist < 1e-6)

            if np.any(matched):
                continue

            assert _near_lens(z, z1[i], z2[i])


def test_numpy_jax_hc_agree():
    """NumPy and JAX trackers agree to 1e-12 after the same seed order."""
    from bagle.jax.jax_physics import psbl_image_positions_hc

    w = np.linspace(-0.7, 0.2, 6) + 0.05j
    w, z1, z2, m1, m2 = _geometry(1.0, 1.0, w)
    inst = _psbl()
    numpy_z = inst.get_image_pos_arr_HC(
        w, z1, z2, m1, m2, check_sols=False
    )
    jax_z = np.asarray(
        psbl_image_positions_hc(w, z1, z2, m1, m2, 1e-8, False, 2, 2)
    )

    for i in range(len(w)):
        assert _nn_err(jax_z[i], numpy_z[i]) < 1e-12


def test_hc_jit_and_vmap():
    """The JAX tracker compiles under jit and vmap."""
    from bagle.jax.jax_physics import psbl_image_positions_hc

    m1 = 0.5
    m2 = 0.5
    z1 = -0.25 + 0.0j
    z2 = 0.25 + 0.0j
    batch = jnp.array([0.4 + 0.3j, 0.55 + 0.28j, 0.7 + 0.22j])

    def _one(w):
        z = psbl_image_positions_hc(
            jnp.reshape(w, (1,)), z1, z2, m1, m2, 1e-8, False, 2, 2
        )

        return z[0]

    compiled = jax.jit(jax.vmap(_one))
    out = compiled(batch)
    out.block_until_ready()

    assert out.shape == (3, 5)
    assert np.all(np.isfinite(np.asarray(out)))


@pytest.mark.parametrize("rho", [1e-3, 1e-2])
def test_fsbl_hc_matches_companion(rho):
    """Static-limb magnification, off the caustic, to 1e-8 relative."""
    from bagle.jax.fsbl import fsbl_magnification

    s = 1.4
    q = 0.3
    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)
    w0 = np.linspace(0.55, 0.75, 3) + 0.4j
    z1 = np.full(w0.shape, -s * m2 + 0.0j)
    z2 = np.full(w0.shape, s * m1 + 0.0j)
    mag_hc = np.asarray(
        fsbl_magnification(
            w0, z1, z2, m1, m2, rho, 12, root_tol=1e-8, root_solver="hc"
        )
    )
    mag_ref = np.asarray(
        fsbl_magnification(
            w0, z1, z2, m1, m2, rho, 12, root_tol=1e-8, root_solver="companion"
        )
    )
    scale = np.maximum(np.abs(mag_ref), 1e-6)
    rel = np.max(np.abs(mag_hc - mag_ref) / scale)

    assert rel < 1e-8


def test_lnl_grad_finite_and_matches_fd():
    """jax.grad of an off-caustic limb lnL is finite and matches FD.

    The scalar is a photometry-style residual of the tracked limb
    positions. Differentiating the polygonal magnification as well
    fuses that small area kernel into the tracker program; the root
    tangent is the part that has to come from the implicit VJP.
    """
    from bagle.jax.jax_physics import fsbl_limb_positions_hc

    t_obs = jnp.array([-4.0, 0.0, 4.0])

    def _lnl(t0, u0, tE, s, q, alpha, rho):
        tau = (t_obs - t0) / tE
        c = jnp.cos(alpha)
        sa = jnp.sin(alpha)
        u_e = -u0 * sa + tau * c
        u_n = u0 * c + tau * sa
        w0 = u_e + 1j * u_n
        m1 = 1.0 / (1.0 + q)
        m2 = q / (1.0 + q)
        z1 = jnp.full((3,), -s * m2)
        z2 = jnp.full((3,), s * m1)
        z = fsbl_limb_positions_hc(
            w0, z1, z2, m1, m2, rho, 4, 2, 2
        )
        # Smooth stand-in for a magnification: size of each image track.
        mag = jnp.sum(jnp.abs(z), axis=(1, 2))
        resid = mag - jnp.array([8.0, 9.0, 8.5])

        return -0.5 * jnp.sum((resid / 0.5) ** 2)

    params = (0.0, 0.7, 20.0, 1.5, 0.4, 0.3, 0.01)
    # jit so the seven finite-difference pairs reuse one primal compile.
    _lnl_jit = jax.jit(_lnl)
    grads = jax.grad(_lnl_jit, argnums=tuple(range(7)))(*params)
    grads = [float(g) for g in grads]

    assert all(np.isfinite(g) for g in grads)

    eps = 1e-5
    for i, g in enumerate(grads):
        up = list(params)
        lo = list(params)
        up[i] = params[i] + eps
        lo[i] = params[i] - eps
        fd = (float(_lnl_jit(*up)) - float(_lnl_jit(*lo))) / (2.0 * eps)
        scale = max(abs(fd), abs(g), 1e-4)

        assert abs(g - fd) / scale < 1e-3
