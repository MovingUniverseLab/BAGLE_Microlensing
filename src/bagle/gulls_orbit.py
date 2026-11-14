"""The analytic Roman ORBIT5 observer used for RMDC26 Beginner events.

This module has no magnification-engine dependency.
"""
import numpy as np


def _orbit5_component(bjd, component):
    t = np.asarray(bjd, dtype=float)
    if component == 0:
        years = (t-2451545.)/365.25
        scale = 1+(3.00447e-6/3)**(1/3)
        semimajor = (1.00000261+years*.0000000562)*scale
        eccentricity = .01671123-years*.0000004392
        inc = np.deg2rad(-.00001531-years*.0001294668)
        longitude = np.deg2rad(100.46457166+years*359.9937244981)
        perihelion = np.deg2rad(102.93768193+years*.0032327364)
        node = 0.
    else:
        years = (t-2459581.)/365.25
        semimajor = np.full_like(t, .8e6/1.49597871e8)
        eccentricity = np.zeros_like(t)
        inc = np.deg2rad(np.full_like(t, 73.))
        longitude = np.deg2rad(years*738.7078652)
        perihelion = np.zeros_like(t)
        node = np.pi/2
    anomaly = (longitude-perihelion+np.pi) % (2*np.pi)-np.pi
    eccentric = anomaly + eccentricity*np.sin(anomaly)
    for _ in range(8):
        eccentric -= (eccentric-eccentricity*np.sin(eccentric)-anomaly)/(1-eccentricity*np.cos(eccentric))
    x = semimajor*(np.cos(eccentric)-eccentricity)
    y = semimajor*np.sqrt(1-eccentricity**2)*np.sin(eccentric)
    w = perihelion-node
    cw, sw, cn, sn, ci, si = np.cos(w), np.sin(w), np.cos(node), np.sin(node), np.cos(inc), np.sin(inc)
    return np.stack(((cw*cn-sw*sn*ci)*x+(-sw*cn-cw*sn*ci)*y,
                     (cw*sn+sw*cn*ci)*x+(-sw*sn+cw*cn*ci)*y,
                     sw*si*x+cw*si*y), axis=-1)


def roman_orbit5(observer_bjd, reference_bjd):
    """Return ORBIT 5 observer XYZ and its reference position and velocity.

    Positions are heliocentric ecliptic AU; velocity is AU/day. This copies
    the Earth-L2 plus JWST-like surrogate documented in the related Gulls
    source. The exact Beginner executable revision is not known. Native curve
    comparisons establish the accuracy reported in docs/gulls.rst, not exact
    identity. Input epochs and reference use the native absolute BJD notation.
    """
    t = np.asarray(observer_bjd, dtype=float)
    ref = float(reference_bjd)
    if not np.all(np.isfinite(t)) or not np.isfinite(ref):
        raise ValueError('observer epochs must be finite')
    positions = _orbit5_component(t, 0) + _orbit5_component(t, 1)
    origin = _orbit5_component(ref, 0) + _orbit5_component(ref, 1)
    velocity = np.zeros(3)
    for component, rate in enumerate((359.9937244981, 738.7078652)):
        step = 1e-6*365.25*360/rate
        velocity += (_orbit5_component(ref+step, component)-_orbit5_component(ref-step, component))/(2*step)
    return positions, origin, velocity
