=========================
Overview of bagle.model
=========================

model.py is a module that contains a set of classes
and functions that allow the user to construct microlensing
models. The available classes for instantiating a microlensing
event are shown in the list below. See the API documentation
for each class for details.

Example
=======

To instantiate a model::

   from bagle import model
   
   mL = 10.0  # msun
   t0 = 57000.00
   xS0 = np.array([0.000, 0.000])
   beta = 1.4  # mas
   muS = np.array([8.0, 0.0])
   muL = np.array([0.00, 0.00])
   dL = 4000.0
   dS = 8000.0
   b_sff = [1.0]    # one for each filter
   mag_src = [19.0] # one for each filter

   event1 = model.PSPL_PhotAstrom_noPar_Param1(mL,
                          t0, beta, dL, dL / dS,
                          xS0[0], xS0[1], muL[0], muL[1],
                          muS[0], muS[1],
                          b_sff, mag_src)
                          
   # Get time range for event
   t = np.arange(event1.t0 - 3000,
                 event1.t0 + 3000, 1) 
   dt = t - event1.t0  

   # Quanties you can print
   A = event1.get_amplification(t)
   shift = event1.get_centroid_shift(t) 
   shift_amp = np.linalg.norm(shift, axis=1)                       


Note, each model class has a name that typically has a structure of 

    <ModelDataType>_<Parallax>_<GP>_<Parameterization>

For example, `PSPL_Phot_noPar_Param2` has a data and model class type of PSPL_Phot,
which contains a point-source, point-lens event with photometry only. The model
has no parallax, no GP and uses parameterization #2.

The complete list of instantiable model classes is: 

Point source, point lens, photometry only:
==========================================

    - PSPL_Phot_noPar_Param1
    - PSPL_Phot_noPar_Param2
    - PSPL_Phot_Par_Param1
    - PSPL_Phot_Par_Param2
    - PSPL_Phot_Par_Param1_geoproj
    - PSPL_Phot_noPar_GP_Param1
    - PSPL_Phot_noPar_GP_Param2
    - PSPL_Phot_Par_GP_Param1
    - PSPL_Phot_Par_GP_Param1_2
    - PSPL_Phot_Par_GP_Param2
    - PSPL_Phot_Par_GP_Param2_2

Point source, point lens, photometry and astrometry:
====================================================

    - PSPL_PhotAstrom_noPar_Param1
    - PSPL_PhotAstrom_noPar_Param2
    - PSPL_PhotAstrom_noPar_Param3
    - PSPL_PhotAstrom_noPar_Param4
    - PSPL_PhotAstrom_Par_Param4_geoproj
    - PSPL_PhotAstrom_Par_Param1
    - PSPL_PhotAstrom_Par_Param2
    - PSPL_PhotAstrom_Par_Param3
    - PSPL_PhotAstrom_Par_Param4
    - PSPL_PhotAstrom_Par_Param5
    - PSPL_PhotAstrom_LumLens_Par_Param1
    - PSPL_PhotAstrom_LumLens_Par_Param2
    - PSPL_PhotAstrom_LumLens_Par_Param4
    - PSPL_PhotAstrom_noPar_GP_Param1
    - PSPL_PhotAstrom_noPar_GP_Param2
    - PSPL_PhotAstrom_Par_GP_Param1
    - PSPL_PhotAstrom_Par_GP_Param2
    - PSPL_PhotAstrom_Par_GP_Param3
    - PSPL_PhotAstrom_Par_GP_Param4
    - PSPL_PhotAstrom_Par_LumLens_GP_Param1
    - PSPL_PhotAstrom_Par_LumLens_GP_Param2
    - PSPL_PhotAstrom_Par_LumLens_GP_Param3
    - PSPL_PhotAstrom_Par_LumLens_GP_Param4

Point source, point lens, astrometry only
=========================================

    - PSPL_Astrom_Par_Param4
    - PSPL_Astrom_Par_Param3

Point soruce, binary lens, photometry only
==========================================

    - PSBL_Phot_noPar_Param1
    - PSBL_Phot_Par_Param1
    - PSBL_Phot_noPar_GP_Param1
    - PSBL_Phot_Par_GP_Param1

Point source, binary lens, photometry and astrometry
====================================================
    - PSBL_PhotAstrom_noPar_Param1
    - PSBL_PhotAstrom_noPar_Param2
    - PSBL_PhotAstrom_noPar_Param3
    - PSBL_PhotAstrom_Par_Param1
    - PSBL_PhotAstrom_Par_Param2
    - PSBL_PhotAstrom_Par_Param3
    - PSBL_PhotAstrom_Par_Param4
    - PSBL_PhotAstrom_Par_Param5
    - PSBL_PhotAstrom_noPar_GP_Param1
    - PSBL_PhotAstrom_noPar_GP_Param2
    - PSBL_PhotAstrom_Par_GP_Param1
    - PSBL_PhotAstrom_Par_GP_Param2

Binary source, point lens, photometry and only
==============================================

    - BSPL_Phot_noPar_Param1
    - BSPL_Phot_Par_Param1
    - BSPL_Phot_noPar_GP_Param1
    - BSPL_Phot_Par_GP_Param1

Binary source, point lens, photometry and astrometry
====================================================

    - BSPL_PhotAstrom_noPar_Param1
    - BSPL_PhotAstrom_noPar_Param2
    - BSPL_PhotAstrom_noPar_Param3
    - BSPL_PhotAstrom_Par_Param1
    - BSPL_PhotAstrom_Par_Param2
    - BSPL_PhotAstrom_Par_Param3
    - BSPL_PhotAstrom_noPar_GP_Param1
    - BSPL_PhotAstrom_noPar_GP_Param2
    - BSPL_PhotAstrom_noPar_GP_Param3
    - BSPL_PhotAstrom_Par_GP_Param1
    - BSPL_PhotAstrom_Par_GP_Param2
    - BSPL_PhotAstrom_Par_GP_Param3

Finite source, point lens, photometry and astrometry (broken)
=============================================================

    - FSPL_PhotAstrom_Par_Param1


=========================
Developers
=========================

Each model class i built up from a menu of different features
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
=================

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

These classes determine which physical parameters define the model. Currently
this file supports one parameterization when using only photometry (`Phot`)
and three parametrizations when using photometry and astrometery
(`PhotAstrom`).

The base class is PSPL_Param.

The parameters for each parameterization are:

    PhotParam1 :
        Point source point lens model for microlensing photometry only.
        This model includes the relative proper motion between the lens
        and the source. Parameters are reduced with the use of piRel

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
More details about each parameterization can be found in the Parameterization
Class docstring.

Making a New Model
--------------------

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


