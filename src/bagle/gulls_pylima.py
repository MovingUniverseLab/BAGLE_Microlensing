"""pyLIMA telescope with the effective historical Beginner observer."""
import numpy as np
from pyLIMA.telescopes import Telescope

from bagle.gulls import JD_TO_MJD


class RomanTelescope(Telescope):
    """Provide reference-subtracted observer offsets to pyLIMA's parallax.

    This is an effective photometric observer, not a physical geocentric
    ephemeris. ``compute_parallax`` uses the supplied analytic BAGLE observer
    and never downloads Earth or satellite positions. pyLIMA computes the
    trajectories from these offsets and its own model parameters.
    """
    def __init__(self, *, observer, **kwargs):
        super().__init__(**kwargs)
        self._bagle_observer = observer

    def compute_parallax(self, parallax_model, North_vector, East_vector):
        """Populate projected observer displacements at telescope epochs.

        Parameters
        ----------
        parallax_model : list
            pyLIMA parallax mode and reference JD; the supplied observer
            already includes its historical reference subtraction.
        North_vector, East_vector : array_like
            Unit vectors in equatorial Cartesian coordinates.
        """
        for kind, data in [('photometry', self.lightcurve),
                           ('astrometry', self.astrometry)]:
            if data is not None:
                xyz = self._bagle_observer.get_observer_displacement(
                    data['time'].value-JD_TO_MJD)
                self.deltas_positions[kind] = np.array(
                    [xyz @ North_vector, xyz @ East_vector])
