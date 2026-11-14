"""MulensModel observer adapter for BAGLE's historical Beginner convention.

MulensModel calculates trajectories and magnifications. Only the effective
satellite ephemeris differs from its standard observer. This module loads
only when ``frame_convert.get_mulens_model`` converts a Gulls model.
"""
import numpy as np
import MulensModel as mm
from astropy.coordinates import SkyCoord
from astropy import units as u

from bagle.gulls import JD_TO_MJD


class _ObserverTrajectory(mm.Trajectory):
    def _get_delta_satellite(self):
        # MulensModel 3.12's shared satellite cache omits the ephemeris from
        # its key. Different tcroin values at the same sky position/times
        # would otherwise reuse the previous model's observer. Calculate this
        # projection locally; do not patch or clear another model's cache.
        xyz = self.satellite_skycoord.cartesian.xyz.to_value(u.au).T
        return {'N': -xyz @ self.coords.north_projected,
                'E': -xyz @ self.coords.east_projected,
                'D': -xyz @ self.coords.cartesian.xyz.value}


class _ObserverCurve(mm.MagnificationCurve):
    def _setup_trajectory(self, selection):
        satellite = (None if self.satellite_skycoord is None
                     else self.satellite_skycoord[selection])
        return _ObserverTrajectory(
            self.times[selection], parameters=self.parameters,
            parallax=self.parallax, coords=self.coords,
            satellite_skycoord=satellite)


class RomanModel(mm.Model):
    """A MulensModel model carrying an analytic effective Roman observer.

    Query times are JD. ``source_fluxes`` and ``blend_flux`` use the input
    BAGLE magnitude zero point (flux = 10**(-0.4 * magnitude)). They can be
    combined with magnification to obtain model flux. MulensModel's magnitude
    utilities use a different zero point. Binary ``get_magnification`` defaults to
    the flux ratio for the filter selected during conversion; ``separate=True``
    gives both source magnifications. All magnification methods are inherited.

    ``get_satellite_coords`` returns an effective, reference-subtracted
    observer with its sign converted for MulensModel. It is not a physical
    geocentric Roman position. The parallax vector alone cannot encode it.
    """
    def __init__(self, parameters, *, coords, observer):
        super().__init__(parameters, coords=coords)
        self._bagle_observer = observer

    def get_satellite_coords(self, times):
        times = np.atleast_1d(np.asarray(times, dtype=float))
        displacement = -self._bagle_observer.get_observer_displacement(
            times-JD_TO_MJD)
        return SkyCoord(x=displacement[:, 0]*u.au,
                        y=displacement[:, 1]*u.au,
                        z=displacement[:, 2]*u.au,
                        representation_type='cartesian', frame='icrs')

    def get_trajectory(self, times, satellite_skycoord=None):
        times = np.atleast_1d(np.asarray(times, dtype=float))
        satellite = (self.get_satellite_coords(times) if satellite_skycoord is None
                     else satellite_skycoord)
        parameters = ([self.parameters] if self.n_sources == 1 else
                      [getattr(self.parameters, f'source_{i+1}_parameters')
                       for i in range(self.n_sources)])
        trajectories = [_ObserverTrajectory(
            times, parameters=p, parallax=self._parallax, coords=self.coords,
            satellite_skycoord=satellite) for p in parameters]
        return trajectories[0] if self.n_sources == 1 else trajectories

    def _observer_curve(self, time, parameters, satellite, gamma, methods):
        curve = _ObserverCurve(
            time, parameters=parameters, parallax=self._parallax,
            coords=self.coords, satellite_skycoord=satellite, gamma=gamma)
        curve.set_magnification_methods(methods, self._default_magnification_method)
        curve.set_magnification_methods_parameters(self._methods_parameters)
        return curve

    def get_magnification_curve(self, time, satellite_skycoord, gamma):
        return self._observer_curve(time, self.parameters, satellite_skycoord,
                                    gamma, self._methods)

    def get_magnification_curves(self, time, satellite_skycoord, gamma):
        curves = []
        for i in range(self.n_sources):
            methods = (self._methods.get(i+1) if isinstance(self._methods, dict)
                       else self._methods)
            coefficient = gamma[i] if isinstance(gamma, list) else gamma
            curves.append(self._observer_curve(
                time, getattr(self.parameters, f'source_{i+1}_parameters'),
                satellite_skycoord, coefficient, methods))
        return curves

    def get_magnification(self, time, satellite_skycoord=None, gamma=None,
                          bandpass=None, source_flux_ratio=None, separate=None):
        if self.n_sources == 2 and source_flux_ratio is None and not separate:
            source_flux_ratio = float(self.source_fluxes[1]/self.source_fluxes[0])
        return super().get_magnification(
            time, satellite_skycoord=satellite_skycoord, gamma=gamma,
            bandpass=bandpass, source_flux_ratio=source_flux_ratio,
            separate=separate)
