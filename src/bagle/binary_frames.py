"""
Reference-point frames shared by the binary-lens / binary-source models.

Keplerian (orbiting) binary models store their internal ``t0`` / ``u0_amp``
/ ``xL0`` in the CoM-CoM frame, the only pair whose closest approach is
exact. The lens-midpoint pair (``t0_mid``, ``u0_amp_mid``, ``beta_mid``,
``xL0_mid``) and the primary-primary pair (``t0_p``, ``u0_amp_p``,
``beta_p``, ``xS0_p``) are defined at an instant and need a root solve, so
they are computed on first access and cached. Static, linear and
accelerated models expose the ``_mid`` names as aliases of their internal
(midpoint) values.

Suffix rule: ``_com`` is source CoM to lens CoM, ``_p`` is primary source
to primary lens, and ``_mid`` (or no suffix for non-Keplerian models) is
primary source to lens geometric midpoint.
"""
import numpy as np

from bagle import frame_convert as fc

days_per_year = 365.25

# Lazily computed attribute groups.
_MID_NAMES = ('t0_mid', 'u0_amp_mid', 'beta_mid', 'xL0_mid')
_P_NAMES = ('t0_p', 'u0_amp_p', 'beta_p', 'xS0_p')

# Non-Keplerian aliases: _mid name -> internal attribute.
_MID_ALIASES = {'t0_mid': 't0', 'u0_amp_mid': 'u0_amp', 'beta_mid': 'beta',
                'xL0_mid': 'xL0'}


class KeplerianFrame(object):
    """
    CoM-CoM geometry of an orbiting binary model, for lazy frame pairs.

    Parameters
    ----------
    t0_com : float
        Source CoM - lens CoM closest-approach time (MJD).
    u0_amp_com : float
        Signed CoM-CoM impact parameter (Einstein units).
    tE : float
        Einstein crossing time (days).
    thetaE_hat : array_like, shape (2,)
        Unit vector along the relative proper motion (E, N).
    thetaE_amp : float
        Einstein radius (mas).
    orbL : bagle.orbits.Orbit or None
        Lens orbit about the lens CoM, in arcsec; ``None`` for one lens.
    orbS : bagle.orbits.Orbit or None
        Source orbit about the source CoM, in arcsec; ``None`` for one
        source.
    xL0_com, xS0_com : array_like, shape (n_filters, 2)
        Lens and source CoM positions at ``t0_com`` (arcsec).
    muL, muS : array_like, shape (2,)
        Lens and source CoM proper motions (mas/yr).
    """

    def __init__(self, t0_com, u0_amp_com, tE, thetaE_hat, thetaE_amp, orbL,
                 orbS, xL0_com, xS0_com, muL, muS):
        """
        Store the CoM-CoM geometry.

        Parameters are described in the class docstring.

        Returns
        -------
        None
        """
        self.t0_com = float(t0_com)
        self.u0_amp_com = float(u0_amp_com)
        self.tE = float(tE)
        self.thetaE_hat = np.asarray(thetaE_hat, dtype=float)
        self.thetaE_amp = float(thetaE_amp)
        self.orbL = orbL
        self.orbS = orbS
        self.xL0_com = np.atleast_2d(np.asarray(xL0_com, dtype=float))
        self.xS0_com = np.atleast_2d(np.asarray(xS0_com, dtype=float))
        self.muL = np.asarray(muL, dtype=float)
        self.muS = np.asarray(muS, dtype=float)

        # The source point of the _mid / _p pairs is the primary source.
        self.src_point = 'prim_center' if orbS is not None else 'COM'

        return None

    def pair(self, lens):
        """
        Reference epoch and impact parameter of one lens point.

        Parameters
        ----------
        lens : {'geom_mid', 'prim_center', 'COM'}
            Lens reference point; the source point is the primary source.

        Returns
        -------
        t0 : float
            Reference epoch (MJD).
        u0_amp : float
            Signed impact parameter (Einstein units).
        """
        t0, u0_amp = fc.convert_u0_amp_t0_bsbl_orbit(
            self.t0_com, self.u0_amp_com, self.tE, self.thetaE_hat,
            self.orbL, self.orbS, self.thetaE_amp * 1e-3,
            lens_in='COM', lens_out=lens, src_in='COM', src_out=self.src_point)

        return t0, u0_amp

    def lens_point(self, t, lens):
        """
        Sky position of a lens reference point, without parallax.

        Parameters
        ----------
        t : float
            Time (MJD).
        lens : {'geom_mid', 'prim_center', 'COM'}
            Lens reference point.

        Returns
        -------
        xL : numpy array, shape (n_filters, 2)
            Position (arcsec).
        """
        # Lens CoM moves linearly from t0_com.
        dt = (t - self.t0_com) / days_per_year
        xL = self.xL0_com + self.muL * dt * 1e-3

        # Orbital offset of the point (arcsec).
        if self.orbL is not None and lens != 'COM':
            xL = xL + fc.keplerian_point_offset(self.orbL, t, lens)

        return xL

    def source_point(self, t):
        """
        Sky position of the primary source, without parallax.

        Parameters
        ----------
        t : float
            Time (MJD).

        Returns
        -------
        xS : numpy array, shape (n_filters, 2)
            Position (arcsec).
        """
        # Source CoM moves linearly from t0_com.
        dt = (t - self.t0_com) / days_per_year
        xS = self.xS0_com + self.muS * dt * 1e-3

        # Orbital offset of the primary source (arcsec).
        if self.orbS is not None:
            xS = xS + fc.keplerian_point_offset(self.orbS, t, 'prim_center')

        return xS

    def lazy_values(self, group):
        """
        Compute one group of lazily evaluated attributes.

        Parameters
        ----------
        group : {'mid', 'p'}
            Lens midpoint pair or primary-primary pair.

        Returns
        -------
        values : dict
            Attribute name -> value for every name in the group.
        """
        if group == 'mid':
            # Primary source relative to the lens geometric midpoint.
            t0, u0_amp = self.pair('geom_mid')
            values = {'t0_mid': t0, 'u0_amp_mid': u0_amp,
                      'beta_mid': u0_amp * self.thetaE_amp,
                      'xL0_mid': self.lens_point(t0, 'geom_mid')}
        else:
            # Primary source relative to the primary lens.
            t0, u0_amp = self.pair('prim_center')
            values = {'t0_p': t0, 'u0_amp_p': u0_amp,
                      'beta_p': u0_amp * self.thetaE_amp,
                      'xS0_p': self.source_point(t0)}

        return values


class BinaryFramesMixin(object):
    """
    Lazy ``_mid`` / ``_p`` frame attributes for binary models.

    Notes
    -----
    Mixed into the PSBL / BSBL model families. ``__getattr__`` only runs for
    names an instance does not have, so a parameterization that takes
    ``t0_p`` / ``beta_p`` / ``xS0_p`` as inputs keeps them unchanged.

    * Keplerian models set ``self._kep_frame`` (a :class:`KeplerianFrame`);
      the ``_mid`` and ``_p`` groups are root-solved on first access and
      cached on the instance. A failed root solve raises ``ValueError``.
    * Other models expose ``t0_mid``, ``u0_amp_mid``, ``beta_mid`` and
      ``xL0_mid`` as their internal (midpoint) ``t0``, ``u0_amp``, ``beta``
      and ``xL0``.
    """

    def __getattr__(self, name):
        """
        Resolve a lazy frame attribute.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        value : object
            The attribute value.

        Raises
        ------
        AttributeError
            If ``name`` is not a frame attribute or is unavailable.
        """
        if name not in _MID_NAMES and name not in _P_NAMES:
            raise AttributeError(name)

        # Avoid recursion: look only in the instance dictionary.
        attrs = self.__dict__
        frame = attrs.get('_kep_frame')

        if frame is None:
            # Non-Keplerian: _mid names alias the internal midpoint values.
            base = _MID_ALIASES.get(name)
            if base is not None and base in attrs:
                return attrs[base]
            raise AttributeError(name)

        # Keplerian: solve the group once and cache it on the instance. The
        # cached keys are recorded so a re-init can drop them.
        group = 'mid' if name in _MID_NAMES else 'p'
        values = frame.lazy_values(group)
        cached = attrs.setdefault('_kep_lazy_keys', set())
        for key, value in values.items():
            if key not in attrs:
                attrs[key] = value
                cached.add(key)

        return attrs[name]


def clear_lazy_frames(model):
    """
    Drop lazily cached ``_mid`` / ``_p`` values from a model instance.

    Parameters
    ----------
    model : BAGLE model instance
        Instance whose frame is being (re)built.

    Returns
    -------
    None
    """
    attrs = model.__dict__
    for key in attrs.pop('_kep_lazy_keys', set()):
        attrs.pop(key, None)

    return None
