"""
Reproduce paper 1 Figure ``com_motion.png`` (Elliptical Orbit).

PAPER (caption): i = 0, e = 0.8, omega = 30 deg, Omega = 10 deg.
ASSUMED (read off the published figure): aleph_pri = 3.2 mas,
aleph_sec = 4.8 mas (lengths of the drawn semi-major axis bars); the period
and tp only set the time sampling, so P = 1 and tp = 0.
"""
import numpy as np
import matplotlib.pyplot as plt

from bagle import orbits

from common import save, sky_axes


def main():
    """
    Draw both orbits about the centre of mass with the semi-major axes.

    Returns
    -------
    None
    """

    orb = orbits.Orbit()
    orb.w, orb.o, orb.i, orb.e = 30.0, 10.0, 0.0, 0.8
    orb.p, orb.tp = 1.0, 0.0
    orb.aleph, orb.aleph2 = 3.2, 4.8

    # One full period, plus the periastron epoch.
    t = np.linspace(0, 1, 2001)
    xp, yp, xs, ys = orb.oal2xy(t)
    xp_tp, yp_tp, xs_tp, ys_tp = orb.oal2xy(np.array([0.0]))

    # Ellipse centre (X = -e) and apoastron (X = -1 - e) in the orbit plane,
    # mapped to (East, North) with the Thiele-Innes constants.
    e = orb.e
    cen_p = np.array([-e * orb.conB, -e * orb.conA])
    apo_p = np.array([(-1 - e) * orb.conB, (-1 - e) * orb.conA])
    cen_s = np.array([-e * orb.conB2, -e * orb.conA2])
    apo_s = np.array([(-1 - e) * orb.conB2, (-1 - e) * orb.conA2])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(xp, yp, color="purple", lw=4, label="Primary Orbit")
    ax.plot(xs, ys, color="gray", lw=4, label="Secondary Companion Orbit")
    ax.plot(xp_tp, yp_tp, "o", color="gold", ms=14, label="Primary at $t_p$")
    ax.plot(xs_tp, ys_tp, "h", color="#56B4E9", ms=18,
            label="Secondary Companion at $t_p$")
    ax.plot(*np.c_[cen_p, apo_p], color="gold", lw=4, label=r"$\aleph_{pri}$")
    ax.plot(*np.c_[cen_s, apo_s], color="#56B4E9", lw=4,
            label=r"$\aleph_{sec}$")
    ax.plot(0, 0, "o", color="deeppink", ms=8, label="Center of Mass")
    ax.set_aspect("equal")
    sky_axes(ax)
    ax.set_title("Elliptical Orbit (BAGLE reproduction)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    # Report the PA of the primary at tp: expected Omega + omega for i = 0.
    pa = np.degrees(np.arctan2(xp_tp[0], yp_tp[0])) % 360
    print(f"primary PA at tp = {pa:.2f} deg (Omega + omega = 40)")
    print(f"aleph bars: pri {np.hypot(*(apo_p - cen_p)):.2f}, "
          f"sec {np.hypot(*(apo_s - cen_s)):.2f} mas")

    save(fig, "com_motion")
    return None


if __name__ == "__main__":
    main()
