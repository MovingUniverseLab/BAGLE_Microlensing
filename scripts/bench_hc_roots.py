"""Time companion roots against the hyper-Catalan tracker.

Not a test. Prints one table for NumPy (per epoch), JAX on CPU, and
one FSBL limb. Sizes are 1e3 and 1e4 source positions.
"""

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from bagle.model import PSBL


def _geometry(n):
    """Straight off-axis track. Lenses stay put."""
    s = 1.2
    q = 0.5
    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)
    w = np.linspace(-1.0, 1.0, n) + 0.35j
    z1 = np.full(n, -s * m2 + 0.0j)
    z2 = np.full(n, s * m1 + 0.0j)

    return w, z1, z2, m1, m2


def _time(fn, repeat=3):
    """Best of ``repeat`` calls, in seconds. ``fn`` is warmed up once."""
    fn()
    best = None

    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)

    return best


def main():
    """Print the timing table and return None."""
    inst = PSBL()
    inst.root_tol = 1e-8
    rows = []

    for n in (1000, 10000):
        w, z1, z2, m1, m2 = _geometry(n)
        dt_np = _time(
            lambda: inst.get_image_pos_arr(
                w, z1, z2, m1, m2, check_sols=False
            )
        )
        dt_hc = _time(
            lambda: inst.get_image_pos_arr_HC(
                w, z1, z2, m1, m2, check_sols=False
            )
        )
        rows.append(
            (
                "numpy roots",
                n,
                dt_np,
                1e6 * dt_np / n,
            )
        )
        rows.append(
            (
                "numpy HC",
                n,
                dt_hc,
                1e6 * dt_hc / n,
            )
        )

    import jax

    from bagle.jax.jax_physics import psbl_image_positions_hc
    from bagle.jax_physics import psbl_image_positions_jit

    for n in (1000, 10000):
        w, z1, z2, m1, m2 = _geometry(n)

        def _run_companion(w=w, z1=z1, z2=z2, m1=m1, m2=m2):
            z = psbl_image_positions_jit(w, z1, z2, m1, m2, 1e-8, False)
            z.block_until_ready()

            return z

        def _run_hc(w=w, z1=z1, z2=z2, m1=m1, m2=m2):
            z = psbl_image_positions_hc(w, z1, z2, m1, m2, 1e-8, False, 2, 2)
            z.block_until_ready()

            return z

        # Compile outside the timed loop.
        _run_companion()
        _run_hc()
        dt_c = _time(_run_companion, repeat=3)
        dt_h = _time(_run_hc, repeat=3)
        rows.append(("JAX companion", n, dt_c, 1e6 * dt_c / n))
        rows.append(("JAX HC", n, dt_h, 1e6 * dt_h / n))

    from bagle.jax.fsbl import fsbl_magnification

    for n_limb, label in ((32, "FSBL limb 32"), (64, "FSBL limb 64")):
        w0 = np.array([0.6 + 0.4j])
        z1 = np.array([-0.3 + 0.0j])
        z2 = np.array([0.3 + 0.0j])

        def _run(n_limb=n_limb):
            mag = fsbl_magnification(
                w0, z1, z2, 0.5, 0.5, 1e-2, n_limb,
                root_tol=1e-8, root_solver="hc",
            )
            mag.block_until_ready()

            return mag

        _run()
        dt = _time(_run, repeat=3)
        rows.append((label + " HC", 1, dt, 1e6 * dt))

    print(f"{'solver':<16} {'N':>8} {'seconds':>12} {'us/epoch':>12}")

    for name, n, dt, us in rows:
        print(f"{name:<16} {n:8d} {dt:12.4f} {us:12.2f}")

    return None


if __name__ == "__main__":
    main()
