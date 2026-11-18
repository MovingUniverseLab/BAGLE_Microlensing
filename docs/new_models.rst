.. _new_models:

================================
Making Your Own Microlens Models
================================

Each model class is built up from a menu of different features
by inheriting from multiple base classes, each from a different 'family' of
related classes.

Each microlensing model must contain:

    1) A class from the Data Class Family:
    
        * `PSPL` -- base class for all Data classes:
        
          -  `PSPL_Phot`
          -  `PSPL_PhotAstrom`
          -  `PSPL_GP_Phot`
          -  `PSPL_GP_PhotAstrom`
          
    2) A class from the Parallax Class Family:
    
        * `ParallaxClassABC` -- base class for all Parallax classes:

          - `PSPL_noParallax`
          - `PSPL_Parallax`
        
    3) A class from the GP Class Family: (optional)
    
        * `PSPL_GP` -- base class for all GP classes.
        
    4) A class from the Parametrization Class Family:
    
        * `PSPL_Param` -- base class for all Param classes
        
          - `PSPL_PhotParam1`
          - `PSPL_PhotParam2`
          - `PSPL_PhotAstromParam1`
          - `PSPL_PhotAstromParam2`
          - `PSPL_PhotAstromParam3`
          - `PSPL_PhotAstromParam4`
          - `PSPL_PhotAstromParam5`
          - `PSPL_GP_PhotParam1`
          - `PSPL_GP_PhotParam2`
          - `PSPL_GP_PhotAstromParam1`
          - `PSPL_GP_PhotAstromParam2`
          - `PSPL_GP_PhotAstromParam3`
          - `PSPL_GP_PhotAstromParam4`

There is a similar hierarchy for PSBL, etc.

For example, the `PSPL_PhotAstrom_noPar_Param1` model is declared as::

    class PSPL_PhotAstrom_noPar_Param1(ModelClassABC,
                                         PSPL_PhotAstrom,
                                         PSPL_noParallax,
                                         PSPL_PhotAstromParam1)

Class Families
==============

Model Class Family
------------------
These are the classes that can be instantiated by the user.
The base class is ModelClassABC.

Data Class Family
-----------------

These classes inform the model of what type of data will be used by the model.
If the model will be for photometry only, then a model with the `PSPL_Phot`
class must be selected. These models have the words `Phot` in their names.
If the model will be using photometry and astrometry data, then a model with
the `PSPL_PhotAstrom` must be selected. These models have the words
`PhotAstrom` in their names.

Data containing astrometry will generate a warning that astrometry data will
not be used in the model when run through a model using `PSPL_Phot`. Data that
does not contain astrometry run through a model using `PSPL_PhotAstrom` will
generate a RuntimeError.

The base class is PSPL.

Parallax Class Family
----------------------

These classes set whether the model uses parallax when calculating
photometry, calculating astrometry, and fitting data. There are only two
options for this class family, `PSPL_noParallax` and `PSPL_Parallax`. Models
that do not have parallax have the words `noPar` in their names, while models
that do contain parallax have the words `Par` in their names.

The base class is ParallaxClassABC.

Parameterization Class Family
------------------------------

These classes determine which physical parameters define the model.
A common need is to make a new parameterization for an existing model.
First, decide whether your parameterization allows for only photometry (`Phot`),
only astrometry (`Astrom`) or photometry and astrometry (`PhotAstrom`).
The base class of all parameterizations is typically `PSPL_Param`.
A new parameterization class should only implement an ``__init__``
class. Within this initializer method, the new input parameters should
be converted to the barycentric variables expected to calculate the
model throughout (e.g. vector u0, tE, muRel_hat). See other parameterizations to see the complete
set of expected variables in a model class.

Some example parameterizations with brief explanations of their parameters
are described below.

    PhotParam1 :
        Point source point lens model for microlensing photometry only.
        This model includes the relative proper motion between the lens
        and the source. Note, the parameters below are all barycentric
        (i.e. at the center of the Solar System). Despite the parameterization,
        models always render lightcurves and astrometric trajectory curves
        in the observers' reference frame (e.g. Earth or a satellite).

        `Parameters`: 
            | t0, u0_amp, tE, 
            | piE_E, piE_N, 
            | b_sff, mag_src,
            | (ra, dec)

    PhotAstromParam1 :
        Point Source Point Lens model for microlensing. This model includes
        proper motions of both the lens and source.

        `Parameters`:
            | mL, t0, beta, 
            | dL, dL_dS, 
            | xS0_E, xS0_N,
            | muL_E, muL_N, 
            | muS_E, muS_N,
            | b_sff, mag_src,
            | (ra, dec)

    PhotAstromParam2 :
        Point Source Point Lens model for microlensing. This model includes
        proper motions of the source and the source position on the sky.

        `Parameters`: 
            | t0, u0_amp, tE, thetaE, piS,
            | piE_E, piE_N,
            | xS0_E, xS0_N,
            | muS_E, muS_N,
            | b_sff, mag_src,
            | (ra, dec)

    PhotAstromParam3 :
        Point Source Point Lens model for microlensing. This model includes
        proper motions of the source and the source position on the sky.
        Note it fits the baseline magnitude rather than the unmagnified source 
        brightness.

        `Parameters`: 
            | t0, u0_amp, tE, log10_thetaE, piS,
            | piE_E, piE_N,
            | xS0_E, xS0_N,
            | muS_E, muS_N,
            | b_sff, mag_base,
            | (ra, dec)

`(ra, dec)` are only required if the model is created with a parallax class.
``xS0_E``, ``xS0_N``, the source flux fraction ``b_sff``, and the
magnitudes are arrays of length ``n_filters``. ``xS0`` is always
``(n_filters, 2)``, including one filter. ``xS0[filt_idx]`` is
that filter's East/North pair. ``xS0[0, 0]`` is East and
``xS0[0, 1]`` is North for filter 0.
Declare those names on ``filt_param_names`` with a matching
``filt_param_usage`` (``'phot'``, ``'astrom'``, or ``'both'``). See
:doc:`multi_location`.
More details about each parameterization can be found in the Parameterization
Class docstring.

.. _binary-reference-frames:

Reference points and the internal frame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binary models (PSBL, FSBL, BSPL, BSBL, BFSBL) can describe the closest
approach about several reference points. A suffix names the pair:

==========  ====================  ========================  ========================
Suffix      Source point          Lens point                Kind
==========  ====================  ========================  ========================
``_com``    source CoM            lens CoM                  true closest approach
``_p``      primary source        primary lens              instantaneous (below)
``_mid``    primary source        lens geometric midpoint   instantaneous (below)
==========  ====================  ========================  ========================

* The suffix applies to the whole input set: ``t0_*``, ``u0_amp_*`` /
  ``beta_*`` and the source position, ``xS0_com_E`` / ``xS0_com_N`` or
  ``xS0_p_E`` / ``xS0_p_N`` (unsuffixed ``xS0_E`` / ``xS0_N`` for the
  midpoint pair). The input ``xS0`` is always the input source point at the
  input ``t0``. For a single source the source CoM is the source itself.
* Projected binary-axis angles (``alpha``, ``phi``, ``alphaL``, ``alphaS``)
  and the companion projected separations (``sep``, ``sepL_t0``,
  ``sepS_t0``) are defined at that same input ``t0_*``, not at periapse
  (``tp``). Keplerian elements ``Omega``, ``omega_pri`` / ``omega``, and
  ``i`` are orbit invariants and are not tied to ``t0``.
* Every ``u0_amp`` carries the Gould (2004) sign: ``u0_amp > 0`` means the
  source is East of the lens point.
* **Internal frame.** The unsuffixed ``t0`` / ``u0_amp`` / ``u0`` /
  ``beta`` / ``xL0`` are the midpoint pair for static, linear and
  accelerated models, and the CoM pair for Keplerian models (except the
  midpoint-input PSBL Param2 / Param3, which keep the midpoint pair).
  Phot-only Keplerian models (``PSBL_Phot_*Orbs``) are defined by the binary
  state at the input ``t0`` and keep the midpoint frame.
* **Instantaneous pairs.** An orbiting midpoint or primary moves on a curve,
  so its ``(t0, u0)`` is defined at an instant: at ``t0`` the source point
  minus the lens point (at that instant) is perpendicular to the relative
  proper motion, and ``u0`` is that offset. Such a ``t0`` is a reference
  epoch, not a closest-approach time. Only the CoMs move on straight lines.
* **Lazy attributes.** For Keplerian models, ``t0_mid``, ``u0_amp_mid``,
  ``beta_mid``, ``xL0_mid`` and (when they are not inputs) ``t0_p``,
  ``u0_amp_p``, ``beta_p``, ``xS0_p`` are root-solved on first access with
  :func:`bagle.frame_convert.convert_u0_amp_t0_bsbl_orbit` and cached, so
  constructing a model or evaluating a likelihood never needs the root
  solve. For other models the ``_mid`` names alias the internal values.
* Source masses are ``mS``, ``mSp``, ``mSs`` (Msun) with ``qS = mSs / mSp``,
  matching the lens ``mL``, ``mLp``, ``mLs`` and ``q = mLs / mLp``.
* Orbit angles follow :meth:`bagle.orbits.Orbit.oal2xy`: Omega is the
  ascending node (shared by both components) measured from North through
  East, and ``omega_pri`` is measured from it to the primary's periastron
  in the direction of motion; the secondary uses ``omega_pri + 180``.

Making a New Model
------------------

Each model is, as described above, constructed by combining inheriting from
different parent classes that contain the desired features for the model. Each
model must have one class from each class family. In addition to this, there
are several rules that must be followed when creating a new class.

    1)  The data class must match the parameterization class. For example,
        if the chosen data class is `PSPL_Phot`, then the parameter class
        must be `PSPL_PhotParam1` (or a different PhotParam in a future
        version). If the data class is `PSPL_PhotAstrom`, then the parameter
        class must be one of the classes with a PhotAstromParam.

    2)  Models are built using python's multiple inheritance feature. Therefore
        the order in which the parent classes are listed in the model class'
        definition matters. Parent classes to models should always be listed
        in the order:
        
            a) ModelClassABC
            b) Data Class
            c) Parallax Class
            d) Parameterization Class
            
        If using the optional GP class, then the order is
        
            a) ModelClassABC
            b) GP Class
            c) Data Class
            d) Parallax Class
            e) Parameterization Class

    3)  Each class must be given the `@inheritdocstring` decorator, and include
        the following commands in the model's ``__init__``:
        
            * ``a.super().__init__(*args, **kwargs)``
            * ``startbases(self)``
            * ``checkconflicts(self)``
            
        Each of these performs the following function:
        
            * ``super().__init__(*args, **kwargs)``: Inherits the ``__init__`` from the Parameterization Class.
            * ``startbases(self)``: Runs a `start` command on each parent class, giving each parent class a chance to run a set of functions upon instantiation.
            * ``checkconflicts(self)``: Checks to confirm that the combination of parent classes in the model are valid.

    4)  Models should be named to reflect the parents classes used to construct
        it, as outlined in the above sections.


