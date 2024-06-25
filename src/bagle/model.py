#.. module:: model
#    :platform: Unix, Mac, Windows
#    :synopsis: Microlensing model objects.
#
#.. moduleauthor:: Jessica Lu <jlu.astro@berkeley.edu>
#.. moduleauthor:: Michael Medford <MichaelMedford@berkeley.edu>
#.. moduleauthor:: Casey Lam <casey_lam@berkeley.edu>get_resolved_lens_astrometry
#.. moduleauthor:: Edward Broadberry
"""
=========================
Overview
=========================

This set of classes and functions allows the user to construct microlensing
models. The available classes for instantiating a microlensing event are shown
in the list below. See the API documentation for each class for details.

Note, each model class has a name that typically has a structure of 

    <ModelDataType>_<Parallax>_<GP>_<Parameterization>

For example, `PSPL_Phot_noPar_Param2` has a data and model class type of PSPL_Phot,
which contains a point-source, point-lens event with photometry only. The model
has no parallax, no GP and uses parameterization #2.

The complete list of instantiable model classes is: 

Point source, point lens, photometry only:
    - 'PSPL_Phot_noPar_Param1'
    - 'PSPL_Phot_noPar_Param2'
    - 'PSPL_Phot_Par_Param1'
    - 'PSPL_Phot_Par_Param2'
    - 'PSPL_Phot_Par_Param1_geoproj'
    - 'PSPL_Phot_noPar_GP_Param1'
    - 'PSPL_Phot_noPar_GP_Param2'
    - 'PSPL_Phot_Par_GP_Param1'
    - 'PSPL_Phot_Par_GP_Param1_2'
    - 'PSPL_Phot_Par_GP_Param2'
    - 'PSPL_Phot_Par_GP_Param2_2'

Point source, point lens, photometry and astrometry:
    - 'PSPL_PhotAstrom_noPar_Param1'
    - 'PSPL_PhotAstrom_noPar_Param2'
    - 'PSPL_PhotAstrom_noPar_Param3'
    - 'PSPL_PhotAstrom_noPar_Param4'
    - 'PSPL_PhotAstrom_Par_Param4_geoproj'
    - 'PSPL_PhotAstrom_Par_Param1'
    - 'PSPL_PhotAstrom_Par_Param2'
    - 'PSPL_PhotAstrom_Par_Param3'
    - 'PSPL_PhotAstrom_Par_Param4'
    - 'PSPL_PhotAstrom_Par_Param5'
    - 'PSPL_PhotAstrom_LumLens_Par_Param1'
    - 'PSPL_PhotAstrom_LumLens_Par_Param2'
    - 'PSPL_PhotAstrom_LumLens_Par_Param4'
    - 'PSPL_PhotAstrom_noPar_GP_Param1'
    - 'PSPL_PhotAstrom_noPar_GP_Param2'
    - 'PSPL_PhotAstrom_Par_GP_Param1'
    - 'PSPL_PhotAstrom_Par_GP_Param2'
    - 'PSPL_PhotAstrom_Par_GP_Param3'
    - 'PSPL_PhotAstrom_Par_GP_Param4'
    - 'PSPL_PhotAstrom_Par_LumLens_GP_Param1'
    - 'PSPL_PhotAstrom_Par_LumLens_GP_Param2'
    - 'PSPL_PhotAstrom_Par_LumLens_GP_Param3'
    - 'PSPL_PhotAstrom_Par_LumLens_GP_Param4'

Point source, point lens, astrometry only
    - 'PSPL_Astrom_Par_Param4'
    - 'PSPL_Astrom_Par_Param3'

Point soruce, binary lens, photometry only
    - 'PSBL_Phot_noPar_Param1'
    - 'PSBL_Phot_Par_Param1'
    - 'PSBL_Phot_noPar_GP_Param1'
    - 'PSBL_Phot_Par_GP_Param1'

Point source, binary lens, photometry and astrometry
    - 'PSBL_PhotAstrom_CircOrbs_noPar_Param1'
    - 'PSBL_PhotAstrom_EllOrbs_noPar_Param1'
    - 'PSBL_PhotAstrom_CircOrbs_noPar_Param4'
    - 'PSBL_PhotAstrom_EllOrbs_noPar_Param4'
    - 'PSBL_PhotAstrom_CircOrbs_noPar_Param8'
    - 'PSBL_PhotAstrom_EllOrbs_noPar_Param8'
    - 'PSBL_PhotAstrom_CircOrbs_Par_Param1'
    - 'PSBL_PhotAstrom_EllOrbs_Par_Param1'
    - 'PSBL_PhotAstrom_CircOrbs_Par_Param4'
    - 'PSBL_PhotAstrom_EllOrbs_Par_Param4'
    - 'PSBL_PhotAstrom_CircOrbs_Par_Param8'
    - 'PSBL_PhotAstrom_EllOrbs_Par_Param8'
    - 'PSBL_PhotAstrom_AccOrbs_noPar_Param6'
    - 'PSBL_PhotAstrom_LinOrbs_noPar_Param6'
    - 'PSBL_PhotAstrom_AccOrbs_noPar_Param7'
    - 'PSBL_PhotAstrom_LinOrbs_noPar_Param7'
    - 'PSBL_PhotAstrom_AccOrbs_Par_Param6'
    - 'PSBL_PhotAstrom_LinOrbs_Par_Param6'
    - 'PSBL_PhotAstrom_AccOrbs_Par_Param7'
    - 'PSBL_PhotAstrom_LinOrbs_Par_Param7'
    - 'PSBL_PhotAstrom_noPar_Param1'
    - 'PSBL_PhotAstrom_noPar_Param2'
    - 'PSBL_PhotAstrom_noPar_Param3'
    - 'PSBL_PhotAstrom_Par_Param1'
    - 'PSBL_PhotAstrom_Par_Param2'
    - 'PSBL_PhotAstrom_Par_Param3'
    - 'PSBL_PhotAstrom_Par_Param4'
    - 'PSBL_PhotAstrom_Par_Param5'
    - 'PSBL_PhotAstrom_Par_Param7'
    - 'PSBL_PhotAstrom_noPar_GP_Param1'
    - 'PSBL_PhotAstrom_noPar_GP_Param2'
    - 'PSBL_PhotAstrom_Par_GP_Param1'
    - 'PSBL_PhotAstrom_Par_GP_Param2'

Binary source, point lens, photometry only
    - 'BSPL_Phot_noPar_Param1'
    - 'BSPL_Phot_Par_Param1'
    - 'BSPL_Phot_noPar_GP_Param1'
    - 'BSPL_Phot_Par_GP_Param1'

Binary source, point lens, photometry and astrometry
    - 'BSPL_PhotAstrom_noPar_LinOrbs_Param1'
    - 'BSPL_PhotAstrom_noPar_AccOrbs_Param1'
    - 'BSPL_PhotAstrom_noPar_CircOrbs_Param1'
    - 'BSPL_PhotAstrom_noPar_LinOrbs_Param2'
    - 'BSPL_PhotAstrom_noPar_AccOrbs_Param2'
    - 'BSPL_PhotAstrom_noPar_CircOrbs_Param2'
    - 'BSPL_PhotAstrom_noPar_LinOrbs_Param3'
    - 'BSPL_PhotAstrom_noPar_AccOrbs_Param3'
    - 'BSPL_PhotAstrom_noPar_CircOrbs_Param3'
    - 'BSPL_PhotAstrom_noPar_Param1'
    - 'BSPL_PhotAstrom_noPar_Param2'
    - 'BSPL_PhotAstrom_noPar_Param3'
    
    - 'BSPL_PhotAstrom_Par_LinOrbs_Param1'
    - 'BSPL_PhotAstrom_Par_AccOrbs_Param1'
    - 'BSPL_PhotAstrom_Par_CircOrbs_Param1'
    - 'BSPL_PhotAstrom_Par_LinOrbs_Param2'
    - 'BSPL_PhotAstrom_Par_AccOrbs_Param2'
    - 'BSPL_PhotAstrom_Par_CircOrbs_Param2'
    - 'BSPL_PhotAstrom_Par_LinOrbs_Param3'
    - 'BSPL_PhotAstrom_Par_AccOrbs_Param3'
    - 'BSPL_PhotAstrom_Par_CircOrbs_Param3'
    - 'BSPL_PhotAstrom_Par_Param1'
    - 'BSPL_PhotAstrom_Par_Param2'
    - 'BSPL_PhotAstrom_Par_Param3'

    - 'BSPL_PhotAstrom_noPar_GP_LinOrbs_Param1'
    - 'BSPL_PhotAstrom_noPar_GP_LinOrbs_Param2'
    - 'BSPL_PhotAstrom_noPar_GP_LinOrbs_Param3'
    - 'BSPL_PhotAstrom_noPar_GP_Param1'
    - 'BSPL_PhotAstrom_noPar_GP_Param2'
    - 'BSPL_PhotAstrom_noPar_GP_Param3'
    
    - 'BSPL_PhotAstrom_Par_GP_LinOrbs_Param1'
    - 'BSPL_PhotAstrom_Par_GP_LinOrbs_Param2'
    - 'BSPL_PhotAstrom_Par_GP_LinOrbs_Param3'
    - 'BSPL_PhotAstrom_Par_GP_Param1'
    - 'BSPL_PhotAstrom_Par_GP_Param2'
    - 'BSPL_PhotAstrom_Par_GP_Param3'

Binary source, binary lens, photometry and astrometry
    - 'BSBL_PhotAstrom_noPar_Param1'
    - 'BSBL_PhotAstrom_noPar_Param2'
    - 'BSBL_PhotAstrom_Par_Param1'
    - 'BSBL_PhotAstrom_Par_Param2'

Finite source, point lens, photometry and astrometry (broken)
    - 'FSPL_PhotAstrom_Par_Param1'


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

For example, the `PSPL_PhotAstrom_noPar_Param1` model is declared as:

    ``class PSPL_PhotAstrom_noPar_Param1(ModelClassABC, PSPL_PhotAstrom, PSPL_noParallax, PSPL_PhotAstromParam1)``

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

Other notes
-----------
All times must be reported in MJD.

"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import inspect
import numpy as np
import math
from astropy import constants as const
from astropy import units
from astropy.time import Time
import pdb
import celerite
from astropy.coordinates import get_body_barycentric, SkyCoord, solar_system_ephemeris, get_body_barycentric_posvel
from astropy.coordinates.builtin_frames.utils import get_jd12
import erfa
from joblib import Memory
import os
from functools import lru_cache, wraps
import copy
from bagle import frame_convert as fc
from bagle import orbits as orbits
from abc import ABC

au_day_to_km_s = 1731.45683

# Use the JPL ephemerides.
solar_system_ephemeris.set('jpl')

# Setup a parallax cache
cache_dir = os.path.dirname(__file__) + '/parallax_cache/'
cache_memory = Memory(cache_dir, verbose=0, bytes_limit='1G')
# Default cache size is 1 GB
cache_memory.reduce_size()


######################################################
### POINT SOURCE POINT LENS (PSPL) CLASSES ###
######################################################
# --------------------------------------------------
#
# Parameterization Class Family
#
# --------------------------------------------------
#
#    An abstract class that all Param classes should sub-class.
#    This serves as a reminder for the class variables that
#    MUST be set. 
#
class PSPL_Param(ABC):
    # Fit paramters: Shared fit parameters
    fitter_param_names = []

    # Fit paramters: Filter specific fit parameters -- handled as arrays.
    # Every photometric data-set has them.
    # (e.g. b_sff, mag_src, mag_base)
    phot_param_names = []

    # Fit parameters: Optional data-set specific fit parameters -- handled as dictionaries
    # (with keys on the filter index). Not every data-set needs these.
    # User indicates which data-sets use these parameters by including or not
    # in the dictionary. This is most useful for noise properties.
    phot_optional_param_names = []
    ast_optional_param_names = []

    # Non-fit paramters: Custom paramters that will not be fit.
    # These parameters should be derived from the fit parameters and
    # they must exist as a variable on the PSPL model object
    # (e.g. fit thetaE and track mL).
    # Users can tweak this or add more in PSPL_Solver.
    additional_param_names = []

    # Fixed parameters: These are parameters that are required for the model, but are not 
    # fit quantities. For example, RA and Dec in a parallax model, or the reference time
    # if fitting in the geocentric projected frame.
    fixed_param_names = []

    # need to set this.
    paramAstromFlag = False
    paramPhotFlag = False

    def __init__(self, *args, **kwargs):
        # Check that required phot_params are proper arrays.
        # If not, then make them arrays of len(1).
        for param in self.phot_param_names:
            param_var = getattr(self, param)
            if not isinstance(param_var, (list, np.ndarray)):
                setattr(self, param, np.array([param_var]))

        # Loop through again and check the lengths are the same
        # and all have the same length.
        phot_param_len = None
        for param in self.phot_param_names:
            param_var = getattr(self, param)

            if phot_param_len is None:
                phot_param_len = len(param_var)
            else:
                if len(param_var) != phot_param_len:
                    msg = 'Mis-matched length for photometric parameter: {0:s}.'
                    msg += 'Expected length = {1:d} and got {2:d}'
                    raise RuntimeError(msg.format(param, phot_param_len, len(param_var)))

        # Check that the optional paramaters are proper dictionaries.
        # If not and they contain a single value, then make them
        # a dictionary with the value set for the first photometric filter.
        # Otherwise, set an empty dictionary.
        for param in self.phot_optional_param_names:
            param_var = getattr(self, param)
            if not isinstance(param_var, dict):
                new_param_var = {}

                # Case: single number passed in... assume first filter.
                if isinstance(param_var, (int, float)):
                    new_param_var[0] = param_var

                # Case: array with length of all photometric filters passed in.
                if isinstance(param_var, (list, np.ndarray)):
                    if len(param_var) == phot_param_len:
                        for ii in range(len(param_var)):
                            new_param_var[ii] = param_var[ii]  # move entry to dictionary
                    else:
                        # Case: Too-short list or array.
                        msg = 'Mis-matched fomat for optional photometric parameter: {0:s}.'
                        msg += 'Should be a dictionary with keys matched to the filter indices.'

                        raise RuntimeExcpetion(msg.format(param))

                # Save the proerly formatted param dictionary.    
                setattr(self, param, new_param_var)

        # Check that the optional paramaters are proper dictionaries.
        # If not and they contain a single value, then make them
        # a dictionary with the value set for the first photometric filter.
        # Otherwise, set an empty dictionary.
        for param in self.ast_optional_param_names:
            param_var = getattr(self, param)
            if not isinstance(param_var, dict):
                new_param_var = {}

                # Case: single number passed in... assume first filter.
                if isinstance(param_var, (int, float)):
                    new_param_var[0] = param_var

                # Case: array with length of all photometric filters passed in.
                if isinstance(param_var, (list, np.ndarray)):
                    if len(param_var) == phot_param_len:
                        for ii in range(len(param_var)):
                            new_param_var[ii] = param_var[ii]  # move entry to dictionary
                    else:
                        # Case: Too-short list or array.
                        msg = 'Mis-matched fomat for optional photometric parameter: {0:s}.'
                        msg += 'Should be a dictionary with keys matched to the filter indices.'

                        raise RuntimeExcpetion(msg.format(param))

                # Save the proerly formatted param dictionary.    
                setattr(self, param, new_param_var)

        return


class PSPL_AstromParam4(PSPL_Param):
    """
    Point Source Point Lens model for microlensing. This model includes
    proper motions of the source and the source position on the sky.
    It is the same as PSPL_PhotAstromParam2 except it fits for baseline instead
    of source magnitude.

    Attributes
    ----------
    
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float
        Angular distance between the lens and source on the 
        plane of the sky at closest approach in units of thetaE. Can be
    
            * positive (u0_amp > 0 when u0_hat[0] > 0) or 
            * negative (u0_amp < 0 when u0_hat[0] < 0).
        
    tE: float 
        Einstein crossing time (days).
    thetaE: float
        The size of the Einstein radius in (mas).
    piS: float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    muS_E: float
        RA Source proper motion (mas/yr)
    muS_N: float
        Dec Source proper motion (mas/yr)
    
    
    
    Notes
    -----
    
    .. note:: Required parameters if calculating with parallax
    
        * raL: Right ascension of the lens in decimal degrees.
        * decL: Declination of the lens in decimal degrees.
        
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'xS0_E', 'xS0_N',
                          'muS_E', 'muS_N']
    #    phot_param_names = ['b_sff', 'mag_base']
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']
    #                              'mag_src']

    paramAstromFlag = True
    paramPhotFlag = False

    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 raL=None, decL=None):
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.thetaE_amp = thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.raL = raL
        self.decL = decL

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        # Derived quantities
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)

        self.piL = self.piRel + self.piS

        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_hat = self.muRel / self.muRel_amp
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.u0_amp)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)
        self.xL0_E, self.xL0_N = self.xL0

        return


class PSPL_AstromParam3(PSPL_Param):
    """
    Point Source Point Lens model for microlensing. This model includes
    proper motions of the source and the source position on the sky.
    It is the same as PSPL_PhotAstromParam3 except it fits only astrometry, no
    photometry.


    Attributes
    ----------
    
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float
        Angular distance between the lens and source on the 
        plane of the sky at closest approach in units of thetaE. Can be
    
            * positive (u0_amp > 0 when u0_hat[0] > 0) or 
            * negative (u0_amp < 0 when u0_hat[0] < 0).
        
    tE: float
        Einstein crossing time (days).
    log10_thetaE: float
        The log of the Einstein radius log10(thetaE/mas).
    piS: float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    muS_E: float
        RA Source proper motion (mas/yr)
    muS_N: float
        Dec Source proper motion (mas/yr)
    
    
 
    Notes
    -----
    
    .. note:: Required parameters if calculating with parallax
    
        * raL: Right ascension of the lens in decimal degrees.
        * decL: Declination of the lens in decimal degrees.
        
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'log10_thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'xS0_E', 'xS0_N',
                          'muS_E', 'muS_N']
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = False

    def __init__(self, t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 raL=None, decL=None):
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.thetaE_amp = 10 ** log10_thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.raL = raL
        self.decL = decL

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        # Derived quantities
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)

        self.piL = self.piRel + self.piS

        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_hat = self.muRel / self.muRel_amp
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.u0_amp)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        return


class PSPL_PhotParam1(PSPL_Param):
    """PSPL model for photometry only.

    Point source point lens model for microlensing photometry only.
    This model includes the relative proper motion between the lens
    and the source. Parameters are reduced with the use of piRel
    (rather than dL and dS) and muRel (rather than muL and muS).

    Note the attributes, RA (raL) and Dec (decL) are required 
    if you are calculating a model with parallax. 

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float
         Angular distance between the lens and source on the plane of the
         sky at closest approach in units of thetaE. It can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE: float
        Einstein crossing time in days.
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE.
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    b_sff: numpy array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = f_S / (f_S + f_L + f_N)`. This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_src: numpy array or list
        Photometric magnitude of the source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.

    """

    fitter_param_names = ['t0', 'u0_amp', 'tE',
                          'piE_E', 'piE_N']
    phot_param_names = ['b_sff', 'mag_src']

    paramAstromFlag = False
    paramPhotFlag = True

    def __init__(self, t0, u0_amp, tE, piE_E, piE_N, b_sff, mag_src,
                 raL=None, decL=None):
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.b_sff = b_sff
        self.mag_src = mag_src
        self.raL = raL
        self.decL = decL

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        self.mag_base = self.mag_src + 2.5 * np.log10(self.b_sff)

        # Calculate the microlensing parallax amplitude
        self.piE_amp = np.linalg.norm(self.piE)

        # Get thetaE_hat (same direction as piE
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.u0_amp)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        return


class PSPL_PhotParam2(PSPL_Param):
    """
    Point source point lens model for microlensing photometry only.
    This model includes the relative proper motion between the lens
    and the source. Parameters are reduced with the use of piRel
    (rather than dL and dS) and muRel (rather than muL and muS).
    Same as PSPL_PhotParam1, except fits for mag_base instead of 
    mag_src.
    
    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float
         Angular distance between the lens and source on the plane of the
         sky at closest approach in units of thetaE. It can be
         positive (u0_amp > 0 when u0_hat[0] > 0) or 
         negative (u0_amp < 0 when u0_hat[0] < 0).
    tE: float
        Einstein crossing time in days.
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE.
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    b_sff: numpy array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = f_S / (f_S + f_L + f_N)`. This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_base: numpy array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        
        
    Notes
    -----
    
    .. note:: Required parameters if calculating with parallax
    
        * raL: Right ascension of the lens in decimal degrees.
        * decL: Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['t0', 'u0_amp', 'tE',
                          'piE_E', 'piE_N']
    phot_param_names = ['b_sff', 'mag_base']
    additional_param_names = ['mag_src']

    paramAstromFlag = False
    paramPhotFlag = True

    def __init__(self, t0, u0_amp, tE, piE_E, piE_N, b_sff, mag_base,
                 raL=None, decL=None):
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.b_sff = b_sff
        self.mag_base = mag_base
        self.raL = raL
        self.decL = decL

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        # Derived quantities
        self.mag_src = self.mag_base - 2.5 * np.log10(self.b_sff)

        # Calculate the microlensing parallax amplitude
        self.piE_amp = np.linalg.norm(self.piE)

        # Get thetaE_hat (same direction as piE
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.u0_amp)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        return

class PSPL_PhotParam3(PSPL_Param):
    """
    Point source point lens model for microlensing photometry only.
    Utilizes angle of muRel instead of piEE and pEN. Also
    fits in log(piE) and log(tE).
    
    Attributes
    ----------
    t0: float
        Heliocentric time of closest approach (u0) between source and lens in MJD (MJD.DDD)
    u0_amp: float
         Angular distance between the lens and source on the plane of the
         sky at closest approach in units of thetaE. It can be
         positive (u0_amp > 0 when u0_hat[0] > 0) or 
         negative (u0_amp < 0 when u0_hat[0] < 0).
    log_tE: float
        Einstein crossing time in days.
    log_piE : float
        The log of the microlensing parallax amplitude. 
    phi_muRel : float
        The angle of the muRel vector, in degrees. Angle is measured in degrees
        East of North (counter-clockwise on the sky from North). 
    b_sff: numpy array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = f_S / (f_S + f_L + f_N)`. This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_base: numpy array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        
        
    Notes
    -----
    
    .. note:: Required parameters if calculating with parallax
    
        * raL: Right ascension of the lens in decimal degrees.
        * decL: Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['t0', 'u0_amp', 'log_tE',
                          'log_piE', 'phi_muRel']
    phot_param_names = ['b_sff', 'mag_base']
    additional_param_names = ['mag_src']

    paramAstromFlag = False
    paramPhotFlag = True

    def __init__(self, t0, u0_amp, log_tE, log_piE, phi_muRel, b_sff, mag_base,
                 raL=None, decL=None):
        self.t0 = t0
        self.u0_amp = u0_amp
        self.log_tE = log_tE
        self.log_piE = log_piE
        self.phi_muRel = phi_muRel # degrees
        self.b_sff = b_sff
        self.mag_base = mag_base
        self.raL = raL
        self.decL = decL        

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        # Derived quantities
        self.tE = 10**log_tE
        self.piE_amp = 10**log_piE
        self.phi_muRel_rad = np.deg2rad(phi_muRel) # radians
        self.piE = self.piE_amp * np.array([np.sin(self.phi_muRel_rad),
                                            np.cos(self.phi_muRel_rad)])
        
        self.mag_src = self.mag_base - 2.5 * np.log10(self.b_sff)

        # Get thetaE_hat (same direction as piE
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.u0_amp)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        return
    
class PSPL_PhotParam1_geoproj(PSPL_PhotParam1):
    """PSPL model for photometry only.
    Point source point lens model for microlensing photometry only.
    This model includes the relative proper motion between the lens
    and the source. Parameters are reduced with the use of piRel
    (rather than dL and dS) and muRel (rather than muL and muS).
    Note the attributes, RA (raL) and Dec (decL) are required 
    if you are calculating a model with parallax. 

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float
         Angular distance between the lens and source on the plane of the
         sky at closest approach in units of thetaE. It can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE: float
        Einstein crossing time in days.
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE.
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    b_sff: numpy array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = f_S / (f_S + f_L + f_N)`. This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_src: numpy array or list
        Photometric magnitude of the source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.

    """
    def __init__(self, t0, u0_amp, tE, 
                 piE_E, piE_N, b_sff, mag_src,
                 t0par,
                 raL=None, decL=None):
        self.t0par = t0par
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.b_sff = b_sff
        self.mag_src = mag_src
        self.raL = raL
        self.decL = decL

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__(t0, u0_amp, tE, 
                         piE_E, piE_N, b_sff, mag_src,
                         raL, decL)

        return


class PSPL_PhotAstromParam1(PSPL_Param):
    """PSPL model for astrometry and photometry - physical parameterization.

    A Point Source Point Lens model for microlensing. This model uses a 
    parameterization that depends on only physical quantities such as the 
    lens mass and positions and proper motions of both the lens and source. 

    Note the attributes, RA (raL) and Dec (decL) are required 
    if you are calculating a model with parallax. 

    Attributes
    ----------
    mL: float
        Mass of the lens (Msun)
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    beta: float
        Angular distance between the lens and source on the plane of the sky (mas). Can be

          * positive (u0_amp > 0 when u0_hat[0] < 0) or 
          * negative (u0_amp < 0 when u0_hat[0] > 0).

    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    muL_E: float
        RA Lens proper motion (mas/yr)
    muL_N: float
        Dec Lens proper motion (mas/yr)
    muS_E: float
        RA Source proper motion (mas/yr)
    muS_N: float
        Dec Source proper motion (mas/yr)
    b_sff: numpy array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = f_S / (f_S + f_L + f_N)`. This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_src: numpy array or list
        Photometric magnitude of the source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['mL', 't0', 'beta', 'dL', 'dL_dS',
                          'xS0_E', 'xS0_N',
                          'muL_E', 'muL_N',
                          'muS_E', 'muS_N']
    phot_param_names = ['b_sff', 'mag_src']
    additional_param_names = ['dS', 'tE', 'u0_amp',
                              'thetaE_E', 'thetaE_N',
                              'piE_E', 'piE_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True

    def __init__(self, mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 b_sff, mag_src,
                 raL=None, decL=None):
        self.t0 = t0
        self.mL = mL
        self.xS0 = np.array([xS0_E, xS0_N])
        self.beta = beta
        self.muL = np.array([muL_E, muL_N])
        self.muS = np.array([muS_E, muS_N])
        self.dL = dL
        self.dL_dS = dL_dS
        self.dS = self.dL / self.dL_dS
        self.b_sff = b_sff
        self.mag_src = mag_src
        self.raL = raL
        self.decL = decL

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        self.mag_base = self.mag_src + 2.5 * np.log10(self.b_sff)

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (self.dL * units.pc)) - (1.0 / (self.dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_E, self.muRel_N = self.muRel
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr

        self.muS_E, self.muS_N = self.muS
        self.muL_E, self.muL_N = self.muL

        # Calculate the Einstein radius
        thetaE = units.rad * np.sqrt(
            (4.0 * const.G * mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat
        self.thetaE_E, self.thetaE_N = self.thetaE

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0_amp = self.beta / self.thetaE_amp  # in Einstein units
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens
        # (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat
        self.piE_E, self.piE_N = self.piE

        # Calculate the Einstein crossing time. (days)
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year

        return


class PSPL_PhotAstromParam2(PSPL_Param):
    """PSPL model for photometry and astrometry -- photom-like parameterization

    Point Source Point Lens model for microlensing. This model includes
    proper motions of the source and the source position on the sky.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float 
        Angular distance between the lens and source on the plane of the
        sky at closest approach in units of thetaE. Can be
        
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
          
    tE: float 
        Einstein crossing time (days).
    thetaE: float
        The size of the Einstein radius in (mas).
    piS: float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    muS_E: float
        RA Source proper motion (mas/yr)
    muS_N: float
        Source proper motion (mas/yr)
    b_sff: numpy array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = f_S / (f_S + f_L + f_N)`. This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_src: numpy array or list
        Photometric magnitude of the source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['t0', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'xS0_E', 'xS0_N',
                          'muS_E', 'muS_N']
    phot_param_names = ['b_sff', 'mag_src']
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True

    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 b_sff, mag_src,
                 raL=None, decL=None):
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.thetaE_amp = thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.b_sff = b_sff
        self.mag_src = mag_src
        self.raL = raL
        self.decL = decL

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        self.mag_base = self.mag_src + 2.5 * np.log10(self.b_sff)

        # Derived quantities
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)

        self.piL = self.piRel + self.piS

        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.u0_amp)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        return


class PSPL_PhotAstromParam3(PSPL_Param):
    """
    Point Source Point Lens model for microlensing. This model includes
    proper motions of the source and the source position on the sky.
    It is the same as PSPL_PhotAstromParam4 except it fits for log10(thetaE)
    instead of thetaE.


    Attributes	
    ----------

    t0 : float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can be

           * positive (u0_amp > 0 when u0_hat[0] > 0) or 
           * negative (u0_amp < 0 when u0_hat[0] < 0).

    tE : float
        Einstein crossing time (days).
    log10_thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
           :math:`b_sff = f_S / (f_S + f_L + f_N)`. 
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_base : numpy array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.

    Notes
    -----
    
    .. note:: Required parameters if calculating with parallax
    
        * raL: Right ascension of the lens in decimal degrees.
        * decL: Declination of the lens in decimal degrees.
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'log10_thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'xS0_E', 'xS0_N',
                          'muS_E', 'muS_N']
    phot_param_names = ['b_sff', 'mag_base']
    additional_param_names = ['thetaE_amp', 'mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N',
                              'mag_src']

    paramAstromFlag = True
    paramPhotFlag = True

    def __init__(self, t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 b_sff, mag_base,
                 raL=None, decL=None):
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.log10_thetaE = log10_thetaE
        self.thetaE = 10 ** log10_thetaE
        self.thetaE_amp = 10 ** log10_thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.b_sff = b_sff
        self.mag_base = mag_base
        self.raL = raL
        self.decL = decL

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        # Derived quantities
        self.mag_src = self.mag_base - 2.5 * np.log10(self.b_sff)
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)

        self.piL = self.piRel + self.piS

        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.u0_amp)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        return


class PSPL_PhotAstromParam4(PSPL_Param):
    """
    Point Source Point Lens model for microlensing. This model includes
    proper motions of the source and the source position on the sky.
    It is the same as PSPL_PhotAstromParam2 except it fits for baseline instead
    of source magnitude.

    Parameters
    ----------

    t0 : float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can be

           * positive (u0_amp > 0 when u0_hat[0] > 0) or 
           * negative (u0_amp < 0 when u0_hat[0] < 0).

    tE : float
        Einstein crossing time (days).
    thetaE: 
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
           :math:`b_sff = f_S / (f_S + f_L + f_N)`. 
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_base : numpy array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.

    Notes
    -----
    
    .. note:: Required parameters if calculating with parallax
    
        * raL: Right ascension of the lens in decimal degrees.
        * decL: Declination of the lens in decimal degrees.
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'xS0_E', 'xS0_N',
                          'muS_E', 'muS_N']
    phot_param_names = ['b_sff', 'mag_base']
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N',
                              'mag_src']

    paramAstromFlag = True
    paramPhotFlag = True

    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 b_sff, mag_base,
                 raL=None, decL=None):
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.thetaE_amp = thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.b_sff = b_sff
        self.mag_base = mag_base
        self.raL = raL
        self.decL = decL

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        # Derived quantities
        self.mag_src = self.mag_base - 2.5 * np.log10(self.b_sff)
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)

        self.piL = self.piRel + self.piS

        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.u0_amp)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        return


class PSPL_PhotAstromParam4_geoproj(PSPL_PhotAstromParam4):
    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 b_sff, mag_base,
                 t0par,
                 raL=None, decL=None):

        self.t0par = t0par
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.thetaE_amp = thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.b_sff = b_sff
        self.mag_base = mag_base
        self.raL = raL
        self.decL = decL

        super().__init__(t0, u0_amp, tE, thetaE, piS,
                         piE_E, piE_N,
                         xS0_E, xS0_N,
                         muS_E, muS_N,
                         b_sff, mag_base,
                         raL, decL)

        return


# NOTE: NOT SURE IF THIS WORKs OR NOT
class PSPL_PhotAstromParam5(PSPL_Param):
    """
    Point Source Point Lens model for microlensing. This model includes
    proper motions of the source and the source position on the sky.
    It fits for piEN/piEE and piEE, instead of piEE and piEN.

    Attributes
    ----------
    t0 : float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the lens
        on the plane of the sky at closest approach in units of thetaE. Can
          
           * positive (u0_amp > 0 when u0_hat[0] > 0) or 
           * negative (u0_amp < 0 when u0_hat[0] < 0).
           
    tE : float
        Einstein crossing time (days).
    log10_thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piEN_piEE : float
        Ratio of piE_N to piE_E.
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
           :math:`b_sff = f_S / (f_S + f_L + f_N)`. 
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_base : numpy array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.

    Notes
    -----
    
    .. note:: Required parameters if calculating with parallax
    
        * raL: Right ascension of the lens in decimal degrees.
        * decL: Declination of the lens in decimal degrees.
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'log10_thetaE', 'piS',
                          'piEN_piEE', 'piE_E',
                          'xS0_E', 'xS0_N',
                          'muS_E', 'muS_N']
    phot_param_names = ['b_sff', 'mag_base']
    additional_param_names = ['thetaE_amp', 'mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N',
                              'mag_src']

    paramAstromFlag = True
    paramPhotFlag = True

    def __init__(self, t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piEN_piEE,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 b_sff, mag_base,
                 raL=None, decL=None):
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piEN_piEE = piEN_piEE
        piE_N = piE_E * piEN_piEE
        self.piE = np.array([piE_E, piE_N])
        self.log10_thetaE = log10_thetaE
        self.thetaE = 10**log10_thetaE
        self.thetaE_amp = 10**log10_thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.b_sff = b_sff
        self.mag_base = mag_base
        self.raL = raL
        self.decL = decL

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()
                        
        # Derived quantities
        self.mag_src = self.mag_base - 2.5*np.log10(self.b_sff)
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)

        self.piL = self.piRel + self.piS

        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.u0_amp)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        return

class PSPL_GP_PhotParam1(PSPL_PhotParam1):
    # Optional data-set specific parameters -- handled as dictionaries
    # (with keys on the filter index). Not ever data-set needs these.
    # User indicates which data-sets use these parameters by including or not
    # in the dictionary. This is most useful for noise properties.
    phot_optional_param_names = ['gp_log_sigma', 'gp_log_rho', 'gp_log_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, piE_E, piE_N, b_sff, mag_src,
                 gp_log_sigma, gp_log_rho, gp_log_S0, gp_log_omega0,
                 raL=None, decL=None):
        self.gp_log_sigma = gp_log_sigma
        self.gp_log_rho = gp_log_rho
        self.gp_log_S0 = gp_log_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, tE, piE_E, piE_N, b_sff, mag_src,
                         raL=raL, decL=decL)

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return


class PSPL_GP_PhotParam1_2(PSPL_PhotParam1):
    """
    Figuring out the new prior parametrization.
    """
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']
    additional_param_names = ['gp_log_rho', 'gp_log_S0']

    def __init__(self, t0, u0_amp, tE, piE_E, piE_N, b_sff, mag_src,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        self.gp_log_sigma = gp_log_sigma
        self.gp_rho = gp_rho
        self.gp_log_omega04_S0 = gp_log_omega04_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, tE, piE_E, piE_N, b_sff, mag_src,
                         raL=raL, decL=decL)

        self.gp_log_rho = {}
        for key, val in self.gp_rho.items():
            self.gp_log_rho[key] = np.log(val)

        self.gp_log_S0 = {}
        for key, val in self.gp_log_omega04_S0.items():
            self.gp_log_S0[key] = self.gp_log_omega04_S0[key] - 4 * self.gp_log_omega0[key]

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return


class PSPL_GP_PhotParam2(PSPL_PhotParam2):
    # Optional data-set specific parameters -- handled as dictionaries
    # (with keys on the filter index). Not ever data-set needs these.
    # User indicates which data-sets use these parameters by including or not
    # in the dictionary. This is most useful for noise properties.
    phot_optional_param_names = ['gp_log_sigma', 'gp_log_rho', 'gp_log_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, piE_E, piE_N, b_sff, mag_base,
                 gp_log_sigma, gp_log_rho, gp_log_S0, gp_log_omega0,
                 raL=None, decL=None):
        self.gp_log_sigma = gp_log_sigma
        self.gp_log_rho = gp_log_rho
        self.gp_log_S0 = gp_log_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, tE, piE_E, piE_N, b_sff, mag_base,
                         raL=raL, decL=decL)

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return


class PSPL_GP_PhotParam2_2(PSPL_PhotParam2):
    # Optional data-set specific parameters -- handled as dictionaries
    # (with keys on the filter index). Not ever data-set needs these.
    # User indicates which data-sets use these parameters by including or not
    # in the dictionary. This is most useful for noise properties.

    # This is like the PSPL_GP_PhotParam2 class, EXCEPT the gp parametrization
    # is different (uses the one like PSPL_GP_PhotParam1_2)
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']
    additional_param_names = ['gp_log_rho', 'gp_log_S0']

    def __init__(self, t0, u0_amp, tE, piE_E, piE_N, b_sff, mag_base,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        self.gp_log_sigma = gp_log_sigma
        self.gp_rho = gp_rho
        self.gp_log_omega04_S0 = gp_log_omega04_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, tE, piE_E, piE_N, b_sff, mag_base,
                         raL=raL, decL=decL)

        self.gp_log_rho = {}
        for key, val in self.gp_rho.items():
            self.gp_log_rho[key] = np.log(val)

        self.gp_log_S0 = {}
        for key, val in self.gp_log_omega04_S0.items():
            self.gp_log_S0[key] = self.gp_log_omega04_S0[key] - 4 * self.gp_log_omega0[key]

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return

class PSPL_GP_PhotParam3(PSPL_PhotParam3):
    # Optional data-set specific parameters -- handled as dictionaries
    # (with keys on the filter index). Not ever data-set needs these.
    # User indicates which data-sets use these parameters by including or not
    # in the dictionary. This is most useful for noise properties.
    phot_optional_param_names = ['gp_log_sigma', 'gp_log_rho', 'gp_log_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, log_tE, log_piE, phi_muRel, b_sff, mag_base,
                 gp_log_sigma, gp_log_rho, gp_log_S0, gp_log_omega0,
                 raL=None, decL=None):
        self.gp_log_sigma = gp_log_sigma
        self.gp_log_rho = gp_log_rho
        self.gp_log_S0 = gp_log_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, log_tE, log_piE, phi_muRel, b_sff, mag_base,
                         raL=raL, decL=decL)

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return


class PSPL_GP_PhotAstromParam1(PSPL_PhotAstromParam1):
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 b_sff, mag_src,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        self.gp_log_sigma = gp_log_sigma
        self.gp_rho = gp_rho
        self.gp_log_omega04_S0 = gp_log_omega04_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(mL, t0, beta, dL, dL_dS,
                         xS0_E, xS0_N,
                         muL_E, muL_N,
                         muS_E, muS_N,
                         b_sff, mag_src,
                         raL=raL, decL=decL)

        self.gp_log_rho = {}
        for key, val in self.gp_rho.items():
            self.gp_log_rho[key] = np.log(val)

        self.gp_log_S0 = {}
        for key, val in self.gp_log_omega04_S0.items():
            self.gp_log_S0[key] = self.gp_log_omega04_S0[key] - 4 * self.gp_log_omega0[key]

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return


class PSPL_GP_PhotAstromParam2(PSPL_PhotAstromParam2):
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 b_sff, mag_src,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        self.gp_log_sigma = gp_log_sigma
        self.gp_rho = gp_rho
        self.gp_log_omega04_S0 = gp_log_omega04_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, tE, thetaE, piS,
                         piE_E, piE_N,
                         xS0_E, xS0_N,
                         muS_E, muS_N,
                         b_sff, mag_src,
                         raL=raL, decL=decL)

        self.gp_log_rho = {}
        for key, val in self.gp_rho.items():
            self.gp_log_rho[key] = np.log(val)

        self.gp_log_S0 = {}
        for key, val in self.gp_log_omega04_S0.items():
            self.gp_log_S0[key] = self.gp_log_omega04_S0[key] - 4 * self.gp_log_omega0[key]

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return


class PSPL_GP_PhotAstromParam3(PSPL_PhotAstromParam3):
    """
    Point Source Point Lens with GP model for microlensing. This model includes
    proper motions of the source and the source position on the sky.
    It is the same as PSPL_PhotAstromParam4 except it fits for log10(thetaE)
    instead of thetaE.

    Attributes
    ----------

    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float
        Angular distance between the lens and source on the plane of the
        sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE: float
        Einstein crossing time (days).
    log10_thetaE: float
        log10 of the size of the Einstein radius in (mas).
    piS: float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    muS_E: float
        RA Source proper motion (mas/yr)
    muS_N: float
        Dec Source proper motion (mas/yr)
    b_sff: numpy array or list of floats
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = f_S / (f_S + f_L + f_N)`. This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_base: numpy array or list of floats
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
    gp_log_sigma: float
        Guassian process :math:`log(\sigma)` for the Matern 3/2 kernel. 
    gp_rho: float
        Guassian process :math:`{\\rho}` for the Matern 3/2 kernel.
    gp_log_omega04_S0: float
        Guassian process :math:`log(\omega_0^4 * S_0)` from the SHO kernel.
    gp_log_omega0: float
        Guassian process :math:`log(\omega_0)` from the SHO kernel. 
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.

    Notes
    -----
    .. note::
       `raL` and `decL` are required parameters if calculating with parallax
    """
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 b_sff, mag_base,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        self.gp_log_sigma = gp_log_sigma
        self.gp_rho = gp_rho
        self.gp_log_omega04_S0 = gp_log_omega04_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, tE, log10_thetaE, piS,
                         piE_E, piE_N,
                         xS0_E, xS0_N,
                         muS_E, muS_N,
                         b_sff, mag_base,
                         raL=raL, decL=decL)

        self.gp_log_rho = {}
        for key, val in self.gp_rho.items():
            self.gp_log_rho[key] = np.log(val)

        self.gp_log_S0 = {}
        for key, val in self.gp_log_omega04_S0.items():
            self.gp_log_S0[key] = self.gp_log_omega04_S0[key] - 4 * self.gp_log_omega0[key]

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return


class PSPL_GP_PhotAstromParam4(PSPL_PhotAstromParam4):
    """
    Point Source Point Lens with GP model for microlensing. This model includes
    proper motions of the source and the source position on the sky.
    It is the same as PSPL_PhotAstromParam2 except it fits for baseline instead
    of source magnitude.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)

    u0_amp: float
        Angular distance between the lens and source on the plane of the
        sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE: float
        Einstein crossing time (days).
    thetaE: float
        The size of the Einstein radius in (mas).
    piS: float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    muS_E: float
        RA Source proper motion (mas/yr)
    muS_N: float
        Dec Source proper motion (mas/yr)
    b_sff: float
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = f_S / (f_S + f_L + f_N)`. This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_base: float
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
    gp_log_sigma: float
        Guassian process :math:`log(\sigma)` for the Matern 3/2 kernel. 
    gp_rho: float
        Guassian process :math:`{\\rho}` for the Matern 3/2 kernel.
    gp_log_omega04_S0: float
        Guassian process :math:`log(\omega_0^4 * S_0)` from the SHO kernel.
    gp_log_omega0: float
        Guassian process :math:`log(\omega_0)` from the SHO kernel.

    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.


    Notes
    -----
    .. note::
       | `raL` and `decL` are required parameters if calculating with parallax
       | For an explanation of the Guassian process parameters, see Golovich et al. 2019()

    """
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 b_sff, mag_base,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        self.gp_log_sigma = gp_log_sigma
        self.gp_rho = gp_rho
        self.gp_log_omega04_S0 = gp_log_omega04_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, tE, thetaE, piS,
                         piE_E, piE_N,
                         xS0_E, xS0_N,
                         muS_E, muS_N,
                         b_sff, mag_base,
                         raL=raL, decL=decL)

        self.gp_log_rho = {}
        for key, val in self.gp_rho.items():
            self.gp_log_rho[key] = np.log(val)

        self.gp_log_S0 = {}
        for key, val in self.gp_log_omega04_S0.items():
            self.gp_log_S0[key] = self.gp_log_omega04_S0[key] - 4 * self.gp_log_omega0[key]

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return


# --------------------------------------------------
#
# Data Class Family
#
# --------------------------------------------------
class PSPL(ABC):
    def animate(self, tE, time_steps, frame_time, name, size, zoom,
                astrometry):
        """ Produces animation of microlensing event. 
        This function takes the PSPL and makes an animation, the input variables are as follows

        Parameters
        ----------

        tE: 
            number of einstein crossings times before/after the peak you want the animation to plot
                e.g tE = 2 => graph will go from -2 tE to 2 tE
        time_steps:
            number of time steps before/after peak, so total number of time steps will 
            be 2 times this value
        frame_time:
            times in ms of each frame in the animation
        name: string
            the animation will be saved as name.html
        size: list
            [horizontal, vertical] cm's
        zoom:
            # of einstein radii plotted in vertical direction
        """
        times = np.array(range(-time_steps, time_steps + 1, 1))
        tau = tE * times / (-times[0])
        t = self.t0 + (tau * self.tE)

        A = self.get_amplification(t)
        rA = self.get_resolved_amplification(t)
        aplus = rA[0]
        aminus = rA[1]
        rs = self.get_astrometry_unlensed(t)
        ri = self.get_resolved_astrometry(t)
        plus = ri[0]
        minus = ri[1]
        l = self.get_lens_astrometry(t)

        # Setup the alpha for the lensed source images.
        aplus_alpha = (0.5 * (aplus - 1) / (aplus.max() - 1)) + 0.5
        aminus_alpha = (0.5 * (aminus - 1) / (aplus.max() - 1)) + 0.5

        fig = plt.figure(figsize=[size[0], size[1] + 0.5])  # sets up the figure
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        fig.subplots_adjust(hspace=.5)

        s_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, label="Source",
                            color='gold', linewidth=2)
        s_line2, = ax1.plot([], '-', markersize=size[0] * 0.3,
                            color='gold', linewidth=2)
        l_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, label="Lens",
                            color='black', linewidth=2)
        l_line2, = ax1.plot([], '-', markersize=size[0] * 0.3,
                            color='black', linewidth=2)
        p_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, label="Lensed Source Images",
                            color='coral', linewidth=2)
        p_line2, = ax1.plot([], '-', markersize=size[0] * 0.3,
                            color='coral', linewidth=2)
        m_line1, = ax1.plot([], '.', markersize=size[0] * 1.3,
                            color='coral', linewidth=2)
        m_line2, = ax1.plot([], '-', markersize=size[0] * 0.3,
                            color='coral', linewidth=2)

        dec_lim = 1.1 * np.max(np.abs(np.append(plus[:, 1], minus[:, 1])))
        ax1.set_xlabel('RA (")')
        ax1.set_ylabel('Dec (")')
        ax1.set_xlim(
            (l[0][0] + l[-1][0]) / 2 - 2 * (size[0]) / (2 * size[1]) * (
                    l[-1][1] - l[0][
                1] + 2 * zoom * self.thetaE_amp * 1e-3),
            (l[0][0] + l[-1][0]) / 2 + 2 * (size[0]) / (2 * size[1]) * (
                    l[-1][1] - l[0][
                1] + 2 * zoom * self.thetaE_amp * 1e-3))
        # ax1.set_ylim(l[0][1] - zoom*self.thetaE_amp*0.001, l[-1][1] + zoom*self.thetaE_amp*0.001)
        ax1.set_ylim(-dec_lim, dec_lim)

        title_fmt = r'm$_L$={0:.1f} M$_\odot$, d$_L$={1:.0f} pc, d$_S$={2:.0f} pc '
        title_fmt += r'$\theta_E$={3:.1f} mas, t$_E$={4:.0f} days'
        ax1.set_title(title_fmt.format(self.mL, self.dL, self.dS,
                                       self.thetaE_amp, self.tE), fontsize=12)

        if astrometry == "yes":
            a = self.get_astrometry(t)
            u_line1, = ax1.plot([], '.', markersize=size[0] * 1.3,
                                color='red', linewidth=2,
                                label="Unresolved Astrometry")
            u_line2, = ax1.plot([], '-', markersize=size[0] * 0.3,
                                color='red', linewidth=2)
            mag_line, = ax2.plot(tau, A, color='red', linewidth=2)
            ax1.legend(fontsize=12, loc='upper right')
            ax2.set_xlabel("Time (tE)")
            ax2.set_ylabel("Magnification")

            line = [s_line1, s_line2, l_line1, l_line2,
                    p_line1, p_line2, m_line1, m_line2,
                    u_line1, u_line2, mag_line]

            def update(i, source, lens, plus, minus, astrometry, tau,
                       magnification, line):
                # print(str(i) + ", ", end='', flush=True)
                line[0].set_data(source[i, 0], source[i, 1])
                line[1].set_data(source[:i + 1, 0], source[:i + 1, 1])
                line[2].set_data(lens[i, 0], lens[i, 1])
                line[3].set_data(lens[:i + 1, 0], lens[:i + 1, 1])
                line[4].set_data(plus[i, 0], plus[i, 1])
                line[5].set_data(plus[:i + 1, 0], plus[:i + 1, 1])
                line[6].set_data(minus[i, 0], minus[i, 1])
                line[7].set_data(minus[:i + 1, 0], minus[:i + 1, 1])
                line[8].set_data(astrometry[i, 0], astrometry[i, 1])
                line[9].set_data(astrometry[:i + 1, 0], astrometry[:i + 1, 1])
                line[10].set_data(tau[:i + 1], magnification[:i + 1])
                return line

            """
            FuncAnimation takes in the following arguments

            fig = background figure

            update = function that is called every frame

            len(tau) = the number of frames, so now the first argument passed 
                into update (i) will be (0,1,2...len(tau))

            fargs specifies the other arguments to pass into update

            blit being true means that each frame, if there are elements of it 
                that don't change from the last frame,
                it won't replot them, so this makes it faster

            interval = number of milliseconds between each frame
            alternatively you can specify fps in save after after the file name

            """
            ani = animation.FuncAnimation(fig, update, len(tau),
                                          fargs=[rs, l, plus, minus, a, tau, A,
                                                 line],
                                          blit=True, interval=frame_time)
            ani.save("%s.mp4" % name, writer="ffmpeg")
        elif astrometry == "no":
            line5, = ax2.plot(tau, A, color='red', linewidth=2)
            ax1.legend()
            ax2.set_xlabel("time(tE)")
            ax2.set_ylabel("Magnification")

            line = [line1, line2, line3, line4, line5]

            def update(i, source, lens, plus, minus, tau, magnification, line):
                print(i)
                line[0].set_data(rs[i][0], rs[i][1])
                line[1].set_data(l[i][0], l[i][1])
                line[2].set_data(plus[i][0], plus[i][1])
                line[3].set_data(minus[i][0], minus[i][1])
                line[4].set_data(tau[:i], magnification[:i])

                return line

            ani = animation.FuncAnimation(fig, update, len(tau),
                                          fargs=[rs, l, plus, minus, tau, A,
                                                 line],
                                          blit=True, interval=frame_time)
            ani.save("%s.mp4" % name, writer="ffmpeg")
        else:
            print("Please enter yes or no, for the final argument")

        return ani

    def get_photometry(self, t, filt_idx=0, print_warning=True):
        """
        Get the predicted photomety at the specified times for the specified 
        photometric filter or data set. 

        Parameters
        ----------
        t : array_like
            List of times in MJD for the observations. 
        filt_idx : int, optional
            Index of the photometric filter or data set.

        Other Parameters
        ----------------
        print_warning : bool, optional
            Print a warning in the rare case that the magnitude exceeds a 
            zeropoint of 30 and conversions result in NaN returned.

        Returns
        -------
        mag_model : array_like
            Magnitude of the unresolved microlensing event at t.

        """
        mag_zp = 30.0  # arbitrary but allows for negative blend fractions.
        flux_zp = 1.0

        if hasattr(self, 'fdfdt'):
            flux_src = flux_zp * 10 ** (
                    (self.mag_src[filt_idx] - mag_zp) / -2.5) * (
                               1 + (self.fdfdt / 100.0) * (t - self.t0))
        else:
            flux_src = flux_zp * 10 ** ((self.mag_src[filt_idx] - mag_zp) / -2.5)

        flux_model = flux_src * self.get_amplification(t)

        # Account for blending, if necessary.
        try:
            # Adding flux of neighbors and lens
            # b_sff = fS / (fS + fN + fL)
            flux_model += flux_src * (1.0 - self.b_sff[filt_idx]) / \
                          self.b_sff[filt_idx]
        except AttributeError:
            pass

        # Catch the edge case where we exceed the zeropoint.
        bad = np.where(flux_model <= 0)[0]
        if len(bad) > 0:
            if print_warning:
                print('!!! Warning: get_photometry: bad flux encountered.')
            flux_model[bad] = np.nan

        mag_model = -2.5 * np.log10(flux_model / flux_zp) + mag_zp

        return mag_model

    def get_chi2_photometry(self, t_obs, mag_obs, mag_err_obs, filt_index=0):
        """
        Get chi^2 values for the model and input photometric data in the 
        specified photometric filter or data set. 

        Parameters
        ----------
        t_obs : array_like
            List of times in MJD for the observations. 
        mag_obs : array_like
            List of observed photometric measurements of the microlensing event in magnitudes. 
            Length must be the same as t_obs.
        mag_obs_err : array_like
            List of observed photometric uncertainties of the microlensing event in magnitudes. 
            Length must be the same as t_obs.
        filt_idx : int, optional
            Index of the photometric filter or data set.

        Returns
        -------
        chi2 : array_like
            List of chi^2 values from the model and photometric data. 

        """
        mag_model = self.get_photometry(t_obs, filt_idx=filt_index)

        chi2 = ((mag_obs - mag_model) / mag_err_obs) ** 2

        return chi2

    def get_lnL_constant(self, err_obs):
        """
        Get the natural log of the constant normalization terms of the likelihood.

        .. math:: -0.5 * \ln{2 \pi \sigma_{obs}^2}

        Parameters
        ----------
        err_obs : array_like
            List of the uncertainties.

        Returns
        -------
        List of ln(likelihood constants).

        """
        lnL_const = -0.5 * np.log(2.0 * math.pi * err_obs ** 2)

        return lnL_const

    def log_likely_photometry_each(self, t_obs, mag_obs, mag_err_obs, filt_index=0):
        """
        Get the natural log of the likelihood for the input photometric data in the 
        specified filter or data sets. Note, this function returns a list and it 
        is the full ln(likelihood), including the normalization constant. 

        Parameters
        ----------
        t_obs : array_like
            List of times in MJD for the observations. 
        mag_obs : array_like
            List of observed photometric measurements of the microlensing event in magnitudes. 
            Length must be the same as t_obs.
        mag_obs_err : array_like
            List of observed photometric uncertainties of the microlensing event in magnitudes. 
            Length must be the same as t_obs.
        filt_idx : int, optional
            Index of the photometric filter or data set.

        Returns
        -------
        ln_L : array_like
            List of ln(likelihood) for each photometric measurement. 

        """

        chi2_m = self.get_chi2_photometry(t_obs, mag_obs, mag_err_obs, filt_index=filt_index)

        lnL_const_m = self.get_lnL_constant(mag_err_obs)

        lnL = (-0.5 * chi2_m) + lnL_const_m

        return lnL

    def log_likely_photometry(self, t_obs, mag_obs, mag_err_obs, filt_index=0):
        """
        Get the summed natural log of the likelihood for the input photometric data for the 
        specified filter or data set. Note, this function returns the full ln(likelihood), 
        including the normalization constant. 

        Parameters
        ----------
        t_obs : array_like
            List of times in MJD for the observations. 
        mag_obs : array_like
            List of observed photometric measurements of the microlensing event in magnitudes. 
            Length must be the same as t_obs.
        mag_obs_err : array_like
            List of observed photometric uncertainties of the microlensing event in magnitudes. 
            Length must be the same as t_obs.
        filt_idx : int, optional
            Index of the photometric filter or data set.

        Returns
        -------
        ln_L : float
            ln(likelihood) summed over the photometric measurement

        """
        lnL = self.log_likely_photometry_each(t_obs, mag_obs, mag_err_obs, filt_index=filt_index)

        return lnL.sum()


class PSPL_Phot(PSPL):
    """
    Contains methods for model a PSPL photometry only.
    This is a Data-type class in our hierarchy. It is abstract and should not
    be instantiated. 

    Attributes
    ----------
    Available class variables that should be defined.

    t0
    tE
    u0_amp
    u0_E
    u0_N
    piE_E - valid only if parallax model
    piE_N - valid only if parallax model
    piE_amp
    b_sff[#]
    mag_src[#] -- add in
    mag_base[#] -- add in 
    raL - if parallax model
    decL - if parallax model
    """
    photometryFlag = True
    astrometryFlag = False

    def get_astrometry(self, t, ast_filt_idx=0):
        raise RuntimeError(
            "Astrometry is not supported on this object: " +
            str(self.__class__))

    def get_centroid_shift(self, t, ast_filt_idx=0):
        raise RuntimeError(
            "Astrometry is not supported on this object: " +
            str(self.__class__))

    def get_astrometry_unlensed(self, t):
        raise RuntimeError(
            "Astrometry is not supported on this object: " +
            str(self.__class__))

    def get_lens_astrometry(self, t):
        raise RuntimeError(
            "Astrometry is not supported on this object: " +
            str(self.__class__))

    def get_resolved_astrometry(self, t):
        raise RuntimeError(
            "Astrometry is not supported on this object: " +
            str(self.__class__))

    def get_resolved_amplification(self, t):
        raise RuntimeError(
            "Astrometry is not supported on this object: " +
            str(self.__class__))

    def log_likely_astrometry(self, t, x, y, xerr, yerr, ast_filt_idx=0):
        raise RuntimeError(
            "Astrometry is not supported on this object: " +
            str(self.__class__))


class PSPL_PhotAstrom(PSPL):
    """
    Contains methods for model a PSPL photometry + astrometry.
    This is a Data-type class in our hierarchy. It is abstract and should not
    be instantiated. 

    Attributes
    ----------
    Available class variables that should be defined.

    t0
    tE
    u0_amp
    u0_E
    u0_N
    beta
    piE_E - valid only if parallax model
    piE_N - valid only if parallax model
    piE_amp
    mL
    thetaE_amp
    thetaE_E
    thetaE_N
    xS0_E
    xS0_N
    xL0_E
    xL0_N
    muS_E
    muS_N
    muL_E
    muL_N
    muRel_E
    muRel_N
    muRel_amp
    piS
    piL
    dL
    dS
    dL_dS (dL over dS)
    b_sff[#]
    mag_src[#] -- add in
    mag_base[#] -- add in 
    raL - if parallax model
    decL - if parallax model

    """
    photometryFlag = True
    astrometryFlag = True

    def get_chi2_astrometry(self, t_obs, x_obs, y_obs, x_err_obs, y_err_obs, ast_filt_idx=0):
        """
        Get the chi^2 value for this model given input astrometry data and 
        uncertainties for the specified astrometric data set. 

        Parameters
        ----------
        t_obs : array_like
            List of times in MJD for the observations. 
        x_obs : array_like
            List of relative R.A. astrometric positions on the sky in arcsec. 
            Length must match t_obs.
        y_obs : array_like
            List of relative Dec. astrometric positions on the sky in arcsec. 
            Length must match t_obs.
        x_err_obs : array_like
            List of relative R.A. astrometric positional errors on the sky in arcsec. 
            Length must match t_obs.
        y_err_obs : array_like
            List of relative Dec. astrometric positional errors on the sky in arcsec. 
            Length must match t_obs.
        ast_filt_idx : int, optional
            The index of the astrometric filter or data set.

        Returns
        -------
        chi2 : array_like
            List of chi^2 values from the model and astrometric data. 

        """
        pos_model = self.get_astrometry(t_obs, ast_filt_idx=ast_filt_idx)
        chi2_x = ((x_obs - pos_model[:, 0]) / x_err_obs) ** 2
        chi2_y = ((y_obs - pos_model[:, 1]) / y_err_obs) ** 2

        chi2 = chi2_x + chi2_y

        return chi2

    def get_lnL_constant(self, err_obs):
        """
        Get the natural log of the constant normalization terms of the likelihood.

        .. math:: -0.5 * \ln{2 \pi \sigma_{obs}^2}

        Parameters
        ----------
        err_obs : array_like
            List of the uncertainties.

        Returns
        -------
        List of ln(likelihood constants).

        """
        lnL_const = -0.5 * np.log(2.0 * math.pi * err_obs ** 2)

        return lnL_const

    def log_likely_astrometry_each(self, t_obs, x_obs, y_obs, x_err_obs, y_err_obs, ast_filt_idx=0):
        """
        Get the natural log of the likelihood for the input astrometric data in the 
        specified filter or data sets. Note, this function eturns a list and it 
        is the full ln(likelihood), including the normalization constant. 

        Parameters
        ----------
        t_obs : array_like
            List of times in MJD for the observations. 
        x_obs : array_like
            List of relative R.A. astrometric positions on the sky in arcsec. 
            Length must match t_obs.
        y_obs : array_like
            List of relative Dec. astrometric positions on the sky in arcsec. 
            Length must match t_obs.
        x_err_obs : array_like
            List of relative R.A. astrometric positional errors on the sky in arcsec. 
            Length must match t_obs.
        y_err_obs : array_like
            List of relative Dec. astrometric positional errors on the sky in arcsec. 
            Length must match t_obs.
        ast_filt_idx : int, optional
            Index of the astrometric filter or data set.

        Returns
        -------
        ln_L : array_like
            List of ln(likelihood) for each astrometric measurement. 

        """
        chi2_xy = self.get_chi2_astrometry(t_obs, x_obs, y_obs, x_err_obs, y_err_obs, ast_filt_idx=ast_filt_idx)

        lnL_const_x = self.get_lnL_constant(x_err_obs)
        lnL_const_y = self.get_lnL_constant(y_err_obs)

        lnL = (-0.5 * chi2_xy) + lnL_const_x + lnL_const_y

        return lnL

    def log_likely_astrometry(self, t_obs, x_obs, y_obs, x_err_obs, y_err_obs, ast_filt_idx=0):
        """
        Get the summed natural log of the likelihood for the input astrometric data in the 
        specified filter or data sets. Note, this function returns the full ln(likelihood), 
        including the normalization constant. 

        Parameters
        ----------
        t_obs : array_like
            List of times in MJD for the observations. 
        x_obs : array_like
            List of relative R.A. astrometric positions on the sky in arcsec. 
            Length must match t_obs.
        y_obs : array_like
            List of relative Dec. astrometric positions on the sky in arcsec. 
            Length must match t_obs.
        x_err_obs : array_like
            List of relative R.A. astrometric positional errors on the sky in arcsec. 
            Length must match t_obs.
        y_err_obs : array_like
            List of relative Dec. astrometric positional errors on the sky in arcsec. 
            Length must match t_obs.
        ast_filt_idx : int, optional
            Index of the astrometric filter or data set.

        Returns
        -------
        ln_L : float
            The ln(likelihood) summed over all astrometric measurements. 

        """
        lnL = self.log_likely_astrometry_each(t_obs, x_obs, y_obs, x_err_obs, y_err_obs,
                                              ast_filt_idx=ast_filt_idx)

        return lnL.sum()


class PSPL_Astrom(PSPL):
    """
    Contains methods for model a PSPL photometry + astrometry.
    This is a Data-type class in our hierarchy. It is abstract and should not
    be instantiated. 

    Attributes
    ----------
    Available class variables that should be defined.

    t0
    tE
    u0_amp
    u0_E
    u0_N
    beta
    piE_E - valid only if parallax model
    piE_N - valid only if parallax model
    piE_amp
    mL
    thetaE_amp
    thetaE_E
    thetaE_N
    xS0_E
    xS0_N
    xL0_E
    xL0_N
    muS_E
    muS_N
    muL_E
    muL_N
    muRel_E
    muRel_N
    muRel_amp
    piS
    piL
    dL
    dS
    dL_dS (dL over dS)
    b_sff[#]
    raL - if parallax model
    decL - if parallax model

    """

    photometryFlag = False
    astrometryFlag = True

    def get_chi2_astrometry(self, t_obs, x_obs, y_obs, x_err_obs, y_err_obs, ast_filt_idx=0):
        """
        Get the chi^2 value for this model given input astrometry data and 
        uncertainties for the specified astrometric data set. 

        Parameters
        ----------
        t_obs : array_like
            List of times in MJD for the observations. 
        x_obs : array_like
            List of relative R.A. astrometric positions on the sky in arcsec. 
            Length must match t_obs.
        y_obs : array_like
            List of relative Dec. astrometric positions on the sky in arcsec. 
            Length must match t_obs.
        x_err_obs : array_like
            List of relative R.A. astrometric positional errors on the sky in arcsec. 
            Length must match t_obs.
        y_err_obs : array_like
            List of relative Dec. astrometric positional errors on the sky in arcsec. 
            Length must match t_obs.
        ast_filt_idx : int, optional
            The index of the astrometric filter or data set.

        Returns
        -------
        chi2 : array_like
            List of chi^2 values from the model and astrometric data. 

        """
        pos_model = self.get_astrometry(t_obs, ast_filt_idx=ast_filt_idx)
        chi2_x = ((x_obs - pos_model[:, 0]) / x_err_obs) ** 2
        chi2_y = ((y_obs - pos_model[:, 1]) / y_err_obs) ** 2

        chi2 = chi2_x + chi2_y

        return chi2

    def get_lnL_constant(self, err_obs):
        """
        Get the natural log of the constant normalization terms of the likelihood.

        .. math:: -0.5 * \ln{2 \pi \sigma_{obs}^2}

        Parameters
        ----------
        err_obs : array_like
            List of the uncertainties.

        Returns
        -------
        List of ln(likelihood constants).

        """
        lnL_const = -0.5 * np.log(2.0 * math.pi * err_obs ** 2)

        return lnL_const

    def log_likely_astrometry_each(self, t_obs, x_obs, y_obs, x_err_obs, y_err_obs, ast_filt_idx=0):
        """
        Get the natural log of the likelihood for the input astrometric data in the 
        specified filter or data sets. Note, this function eturns a list and it 
        is the full ln(likelihood), including the normalization constant. 

        Parameters
        ----------
        t_obs : array_like
            List of times in MJD for the observations. 
        x_obs : array_like
            List of relative R.A. astrometric positions on the sky in arcsec. 
            Length must match t_obs.
        y_obs : array_like
            List of relative Dec. astrometric positions on the sky in arcsec. 
            Length must match t_obs.
        x_err_obs : array_like
            List of relative R.A. astrometric positional errors on the sky in arcsec. 
            Length must match t_obs.
        y_err_obs : array_like
            List of relative Dec. astrometric positional errors on the sky in arcsec. 
            Length must match t_obs.
        ast_filt_idx : int, optional
            Index of the astrometric filter or data set.

        Returns
        -------
        ln_L : array_like
            List of ln(likelihood) for each astrometric measurement. 

        """
        chi2_xy = self.get_chi2_astrometry(t_obs, x_obs, y_obs, x_err_obs, y_err_obs, ast_filt_idx=ast_filt_idx)

        lnL_const_x = self.get_lnL_constant(x_err_obs)
        lnL_const_y = self.get_lnL_constant(y_err_obs)

        lnL = (-0.5 * chi2_xy) + lnL_const_x + lnL_const_y

        return lnL

    def log_likely_astrometry(self, t_obs, x_obs, y_obs, x_err_obs, y_err_obs, ast_filt_idx=0):
        """
        Get the natural log of the likelihood for the input astrometric data in the 
        specified filter or data sets. Note, this function eturns a list and it 
        is the full ln(likelihood), including the normalization constant. 

        Parameters
        ----------
        t_obs : array_like
            List of times in MJD for the observations. 
        x_obs : array_like
            List of relative R.A. astrometric positions on the sky in arcsec. 
            Length must match t_obs.
        y_obs : array_like
            List of relative Dec. astrometric positions on the sky in arcsec. 
            Length must match t_obs.
        x_err_obs : array_like
            List of relative R.A. astrometric positional errors on the sky in arcsec. 
            Length must match t_obs.
        y_err_obs : array_like
            List of relative Dec. astrometric positional errors on the sky in arcsec. 
            Length must match t_obs.
        ast_filt_idx : int, optional
            Index of the astrometric filter or data set.

        Returns
        -------
        ln_L : array_like
            List of ln(likelihood) for each astrometric measurement. 

        """
        lnL = self.log_likely_astrometry_each(t_obs, x_obs, y_obs, x_err_obs, y_err_obs,
                                              ast_filt_idx=ast_filt_idx)

        return lnL.sum()

    def animate(self, tE, time_steps, frame_time, name, size, zoom,
                astrometry):
        raise RuntimeError(
            "Animation is not supported on this object: " +
            str(self.__class__))

    def get_photometry(self, t_obs, filt_idx=0, print_warning=True):
        raise RuntimeError(
            "Photometry is not supported on this object: " +
            str(self.__class__))

    def log_likely_photometry(self, t_obs, mag_obs, mag_err_obs, filt_index=0):
        raise RuntimeError(
            "Photometry is not supported on this object: " +
            str(self.__class__))


# --------------------------------------------------
#
# GP Class Family
#
# --------------------------------------------------
class Celerite_GP_Model(celerite.modeling.Model):
    """
    This is nedeed for the GP.
    Just a wrapper over our model so it is a 
    celerite model.
    """

    def __init__(self, pspl_model, filter_index):
        # An instance of a PSPL object (or a PSBL object maybe too?)
        self.pspl_model = pspl_model
        self.filter_index = filter_index

        return

    def get_value(self, t_obs):
        pspl_vals = self.pspl_model.get_photometry(t_obs, filt_idx=self.filter_index)

        return pspl_vals


class PSPL_GP(ABC):
    """
    PSPL object that has optional support for gaussian process on each photometric filter.
    """

    # We don't want to override get_photometry, do we?
    # Otherwise the mean model will be wrong.

    def get_photometry_with_gp(self, t_obs, mag_obs, mag_err_obs, filt_index=0, t_pred=None):
        """Returns photometry with GP noise added in. 

        .. note:: 
            This will throw an error if this is a filter with `use_gp_phot[filt_index] = False`.

        Parameters
        ----------
        t_obs : array_like
            List of times in MJD for the observations. These times are used as input to the GP. 
            If t_pred is not specified, then t_pred = t_obs.
        mag_obs : array_like
            List of observed photometric measurements of the microlensing event in magnitudes. 
            These values are used as input to the GP. Length must be the same as t_obs.
        mag_obs_err : array_like
            List of observed photometric uncertainties of the microlensing event in magnitudes. 
            These values are used as input to the GP. Length must be the same as t_obs.
        filt_idx : int, optional
            Index of the photometric filter or data set.
        t_pred : array_like, optional
            List of times in MJD on which to evalute the model. If t_pred is not specified, then
            t_pred = t_obs. 

        Returns
        -------
        mag_model : array_like
            Magnitude of the unresolved microlensing event at t_obs.


        """
        if self.use_gp_phot[filt_index]:
            if t_pred is None:
                t_pred = t_obs

            # FIXME: is there a better way to write this? Since it totally
            # duplicates everything in log_likely_photometry

            # Fix logQ following Golovich+20
            gp_log_Q = np.log(2 ** -0.5)

            matern = celerite.terms.Matern32Term(self.gp_log_sigma[filt_index], self.gp_log_rho[filt_index])
            sho = celerite.terms.SHOTerm(self.gp_log_S0[filt_index], gp_log_Q, self.gp_log_omega0[filt_index])
            mean_mag_err_obs = np.average(mag_err_obs)
            jitter = celerite.terms.JitterTerm(np.log(mean_mag_err_obs))
            kernel = matern + sho + jitter

            my_model = Celerite_GP_Model(self, filt_index)  # self is any instance of PSPL

            gp = celerite.GP(kernel, mean=my_model, fit_mean=True)
            try:
                gp.compute(t_obs, mag_err_obs)
                mag_model, mag_model_var = gp.predict(mag_obs, t_pred, return_var=True)
                mag_model_std = np.sqrt(mag_model_var)
                return mag_model, mag_model_std
            except celerite.solver.LinAlgError:
                print('celerite LinAlgError')
                return None, None
        else:
            raise RuntimeError(
                'PSPL_GP: Cannot call for filter with use_gp_phot = False (filt_index={0:d})'.format(filt_index))

    def get_log_det_covariance(self, t_obs, mag_obs, mag_err_obs, filt_index=0, t_pred=None):
        """Returns photometry with GP noise added in. 

        .. note::
            This will throw an error if this is a filter with `use_gp_phot[filt_index] = False`.
        """
        if self.use_gp_phot[filt_index]:
            if t_pred is None:
                t_pred = t_obs

            # FIXME: is there a better way to write this? Since it totally
            # duplicates everything in log_likely_photometry

            # Fix logQ following Golovich+20
            gp_log_Q = np.log(2 ** -0.5)

            matern = celerite.terms.Matern32Term(self.gp_log_sigma[filt_index], self.gp_log_rho[filt_index])
            sho = celerite.terms.SHOTerm(self.gp_log_S0[filt_index], gp_log_Q, self.gp_log_omega0[filt_index])
            mean_mag_err_obs = np.average(mag_err_obs)
            jitter = celerite.terms.JitterTerm(np.log(mean_mag_err_obs))

            kernel = matern + sho + jitter

            my_model = Celerite_GP_Model(self, filt_index)  # self is any instance of PSPL

            gp = celerite.GP(kernel, mean=my_model, fit_mean=True)
            try:
                gp.compute(t_obs, mag_err_obs)
                return gp.solver.log_determinant()
            except celerite.solver.LinAlgError:
                print('celerite LinAlgError')
                return None, None
        else:
            raise RuntimeError(
                'PSPL_GP: Cannot call for filter with use_gp_phot = False (filt_index={0:d})'.format(filt_index))

    # Will over-ride from PSPL or PSBL.
    def log_likely_photometry(self, t_obs, mag_obs, mag_err_obs, filt_index=0):
        """
        For models that include a Gaussian Process, get the summed natural log 
        of the likelihood for the input photometric data for the specified filter 
        or data set. Note, this function returns the full ln(likelihood), 
        including the normalization constant. 

        Parameters
        ----------
        t_obs : array_like
            List of times in MJD for the observations. 
        mag_obs : array_like
            List of observed photometric measurements of the microlensing event in magnitudes. 
            Length must be the same as t_obs.
        mag_obs_err : array_like
            List of observed photometric uncertainties of the microlensing event in magnitudes. 
            Length must be the same as t_obs.
        filt_idx : int, optional
            Index of the photometric filter or data set.

        Returns
        -------
        ln_L : float
            ln(likelihood) summed over the photometric measurement

        .. note:: 
            The GP will only be used for filters where `use_gp_phot[filt_index] = True`.        
        """
        if self.use_gp_phot[filt_index]:
            # Fix logQ following Golovich+20
            gp_log_Q = np.log(2 ** -0.5)

            matern = celerite.terms.Matern32Term(self.gp_log_sigma[filt_index], self.gp_log_rho[filt_index])
            sho = celerite.terms.SHOTerm(self.gp_log_S0[filt_index], gp_log_Q, self.gp_log_omega0[filt_index])
            mean_mag_err_obs = np.average(mag_err_obs)
            jitter = celerite.terms.JitterTerm(np.log(mean_mag_err_obs))
            kernel = matern + sho + jitter

            my_model = Celerite_GP_Model(self, filt_index)  # self is any instance of PSPL

            gp = celerite.GP(kernel, mean=my_model, fit_mean=True)

            # Make sure that kernel isn't giving crazy things...
            # otherwise return -np.inf for log likelihood 
            # Reference: https://github.com/dfm/celerite/issues/142
            try:
                gp.compute(t_obs, mag_err_obs)
                lnL_gp = gp.log_likelihood(mag_obs)
            except celerite.solver.LinAlgError:
                lnL_gp = -np.inf

            return lnL_gp

        else:
            lnL = self.log_likely_photometry_each(t_obs, mag_obs, mag_err_obs, filt_index=filt_index)

            return lnL.sum()


        
# --------------------------------------------------
#
# Parallax Class Family
#
# --------------------------------------------------
class ParallaxClassABC(ABC):
    pass

class PSPL_noParallax(ParallaxClassABC):
    parallaxFlag = False

    def get_amplification(self, t):
        """
        Get an array of the photometric amplifications at the input times. 
        No parallax is included.

        Parameters
        ----------
        t : array_like
            Array of times in MJD.DDD

        """
        tau = (t - self.t0) / self.tE

        # Convert to matrices for more efficient operations.
        # Matrix shapes below are:
        #  u0, thetaE_hat: [1, 2]
        #  tau:      [N_times, 1]
        u0 = self.u0.reshape(1, len(self.u0))
        thetaE_hat = self.thetaE_hat.reshape(1, len(self.thetaE_hat))
        tau = tau.reshape(len(tau), 1)

        # Shape of u: [N_times, 2]

        u = u0 + tau * thetaE_hat

        # Shape of u_amp: [N_times]
        u_amp = np.linalg.norm(u, axis=1)

        A = (u_amp ** 2 + 2) / (u_amp * np.sqrt(u_amp ** 2 + 4))

        return A

    def get_lens_astrometry(self, t):
        """
        Get the astrometry for the foreground lens at the input times.
        No parallax is included. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).
        """
        dt_in_years = (t - self.t0) / days_per_year
        xL = self.xL0 + np.outer(dt_in_years, self.muL) * 1e-3

        return xL

    def get_astrometry(self, t, ast_filt_idx=0):
        """
        Get the astrometry of the unresolved (observed) position of the
        lensed source at the input times.
        No parallax is included. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).
        ast_filt_idx : int, optional
            Index of the astrometric filter or data set.

        """

        srce_pos_model = self.xS0 + np.outer((t - self.t0) / days_per_year,
                                             self.muS) * 1e-3
        pos_model = srce_pos_model + (self.get_centroid_shift(t) * 1e-3)
        return pos_model

    def get_centroid_shift(self, t):
        """
        Get the centroid shift (in mas) at the input times. The centroid shift
        is the difference between the lensed, unresolved position and the 
        intrinsic position of the source. 
        No parallax is included. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).

        """
        tau = (t - self.t0) / self.tE

        # Lens-induced astrometric shift of the sum of all source images (in mas)
        numer = (np.outer(tau, self.thetaE_hat) + self.u0) * self.thetaE_amp
        denom = (tau ** 2.0 + self.u0_amp ** 2.0 + 2.0).reshape(numer.shape[0], 1)
        shift = numer / denom

        return shift

    def get_astrometry_unlensed(self, t):
        """
        Get the astrometry of the source if the lens didn't exist.
        No parallax is included. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).

        Returns
        -------
        xS_unlensed : numpy array, dtype=float, shape = [len(t), 2]
            The unlensed positions of the source in arcseconds.
        """
        # Equation of motion for just the background source.
        dt_in_years = (t - self.t0) / days_per_year
        xS_unlensed = self.xS0 + np.outer(dt_in_years, self.muS) * 1e-3

        return xS_unlensed

    def get_resolved_amplification(self, t):
        """
        Get the photometric amplification term at a set of times, t for both the
        plus and minus images.
        No parallax is included. The returned tuple has two entries:
        (A_plus, A_minus), each with len(t) arrays.

        Parameters
        ----------
        t : array_like
            Array of times in MJD.DDD
        """
        # Equation of relative motion (angular on sky) Eq. 16 from Hog+ 1995
        dt_in_years = (t - self.t0) / days_per_year
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel)  # mas
        u = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u, axis=1)

        A_plus = 0.5 * (
                (u_amp ** 2 + 2) / (u_amp * np.sqrt(u_amp ** 2 + 4)) + 1)
        A_minus = 0.5 * (
                (u_amp ** 2 + 2) / (u_amp * np.sqrt(u_amp ** 2 + 4)) - 1)

        return (A_plus, A_minus)

    def get_resolved_astrometry(self, t):
        """
        Get the relative RA and Dec astrometry for each of the two source images,
        which we label plus and minus.
        No parallax is included. The returned tuple has two entries:
        (xS_plus, xS_minus), each with [len(t), 2] arrays where the second dimension
        includes [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).

        Returns
        -------
        (xS_plus, xS_minus) : tuple of numpy arrays
            * xS_plus is the vector position of the plus image in arcsec 
              with shape = [len(t), 2]
            * xS_minus is the vector position of the plus image in arcsec 
              with shape = [len(t), 2]

        """
        dt_in_years = (t - self.t0) / days_per_year

        # Equation of motion for the relative angular separation between the
        # background source and lens.
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel)  # mas

        u_vec = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u_vec, axis=1)
        u_hat = (u_vec.T / u_amp).T

        u_plus = ((u_amp + np.sqrt(u_amp ** 2 + 4)) / 2.0) * u_hat.T
        u_minus = ((u_amp - np.sqrt(u_amp ** 2 + 4)) / 2.0) * u_hat.T
        u_plus = u_plus.T
        u_minus = u_minus.T

        # Lensed Source Images - Lens Image
        xSL_plus = u_plus * self.thetaE_amp  # in mas
        xSL_minus = u_minus * self.thetaE_amp  # in mas

        xL = self.get_lens_astrometry(t)

        xS_plus = xL + (xSL_plus * 1e-3)  # arcsec
        xS_minus = xL + (xSL_minus * 1e-3)  # arcsec

        return (xS_plus, xS_minus)

    def calc_piE_ecliptic(self):
        """Not supported on this object."""
        raise RuntimeError(
            "piE_ecliptic is not supported on this object: "
            + str(self.__class__))

    
class PSPL_Parallax(ParallaxClassABC):
    parallaxFlag = True
    fixed_param_names = ['raL', 'decL']

    def start(self):
        if self.raL is None or self.decL is None:
            raise RuntimeError(
                "raL and decL must be provided when running parallax model.")
        # self.calc_piE_ecliptic()

    def get_amplification(self, t):
        """
        Get an array of the photometric amplifications at the input times. 
        Parallax is included.

        Parameters
        ----------
        t : array_like
            Array of times in MJD.DDD

        """

        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t)

        tau = (t - self.t0) / self.tE

        # Convert to matrices for more efficient operations.
        # Matrix shapes below are:
        #  u0, thetaE_hat: [1, 2]
        #  tau:      [N_times, 1]
        u0 = self.u0.reshape(1, len(self.u0))
        thetaE_hat = self.thetaE_hat.reshape(1, len(self.thetaE_hat))
        tau = tau.reshape(len(tau), 1)

        # Shape of u: [N_times, 2]
        u = u0 + tau * thetaE_hat
        u -= self.piE_amp * parallax_vec

        # Shape of u_amp: [N_times]
        u_amp = np.linalg.norm(u, axis=1)

        A = (u_amp ** 2 + 2) / (u_amp * np.sqrt(u_amp ** 2 + 4))

        return A

    def get_lens_astrometry(self, t_obs):
        """
        Get the astrometry for the foreground lens at the input times.
        Parallax is included. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).
        """
        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)

        # Equation of motion for just the background source.
        dt_in_years = (t_obs - self.t0) / days_per_year
        xL = self.xL0 + np.outer(dt_in_years, self.muL) * 1e-3
        xL += (self.piL * parallax_vec) * 1e-3  # arcsec

        return xL

    def get_astrometry(self, t_obs, ast_filt_idx=0):
        """
        Get the astrometry of the unresolved (observed) position of the
        lensed source at the input times.
        Parallax is included. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).
        ast_filt_idx : int, optional
            Index of the astrometric filter or data set.

        """

        # Things we will need.
        dt_in_years = (t_obs - self.t0) / days_per_year

        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)

        # Equation of motion for just the background source.
        xS_unlensed = self.xS0 + np.outer(dt_in_years, self.muS) * 1e-3
        xS_unlensed += np.squeeze(self.piS * parallax_vec) * 1e-3  # arcsec

        # Equation of motion for the relative angular separation between the background source and lens.
        # Note, we don't just call get_centroid_shift() because parallax_vec calculation is repeated.
        # and it is slow. 
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel)  # mas
        thetaS -= np.squeeze(self.piRel * parallax_vec)  # mas
        u_vec = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u_vec, axis=1)

        denom = u_amp ** 2 + 2.0
        shift = thetaS / denom.reshape((len(u_amp), 1))  # mas

        xS = xS_unlensed + (shift * 1e-3)  # arcsec

        return xS

    def get_centroid_shift(self, t, ast_filt_idx=0):
        """
        Get the centroid shift (in mas) at the input times. The centroid shift
        is the difference between the lensed, unresolved position and the 
        intrinsic position of the source. 
        Parallax is included. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).

        """
        # Things we will need.
        dt_in_years = (t - self.t0) / days_per_year

        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t)

        # Equation of motion for the relative angular separation between the background source and lens.
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel)  # mas
        thetaS -= (self.piRel * parallax_vec)  # mas
        u_vec = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u_vec, axis=1)

        denom = u_amp ** 2 + 2.0

        shift = thetaS / denom.reshape((len(u_amp), 1))  # mas

        return shift

    def get_astrometry_unlensed(self, t_obs):
        """
        Get the astrometry of the source if the lens didn't exist.
        Parallax is included. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).

        Returns
        -------
        xS_unlensed : numpy array, dtype=float, shape = [len(t), 2]
            The unlensed positions of the source in arcseconds.
        """ 
        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)

        # Equation of motion for just the background source.
        dt_in_years = (t_obs - self.t0) / days_per_year
        xS_unlensed = self.xS0 + np.outer(dt_in_years, self.muS) * 1e-3
        xS_unlensed += (self.piS * parallax_vec) * 1e-3  # arcsec

        return xS_unlensed

    def get_resolved_amplification(self, t):
        """
        Get the photometric amplification term at a set of times, t for both the
        plus and minus images.
        Parallax is included. The returned tuple has two entries:
        (A_plus, A_minus), each with len(t) arrays.

        Parameters
        ----------
        t : array_like
            Array of times in MJD.DDD
        """
        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t)

        # Equation of relative motion (angular on sky) Eq. 16 from Hog+ 1995
        dt_in_years = (t - self.t0) / days_per_year
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel) - (
                self.piRel * parallax_vec)  # mas
        u = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u, axis=1)

        A_plus = 0.5 * (
                (u_amp ** 2 + 2) / (u_amp * np.sqrt(u_amp ** 2 + 4)) + 1)
        A_minus = 0.5 * (
                (u_amp ** 2 + 2) / (u_amp * np.sqrt(u_amp ** 2 + 4)) - 1)

        return (A_plus, A_minus)

    def get_resolved_astrometry(self, t_obs):
        """
        Get the relative RA and Dec astrometry for each of the two source images,
        which we label plus and minus.
        Parallax is included. The returned tuple has two entries:
        (xS_plus, xS_minus), each with [len(t), 2] arrays where the second dimension
        includes [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).

        Returns
        -------
        (xS_plus, xS_minus) : tuple of numpy arrays
            * xS_plus is the vector position of the plus image in arcsec 
              with shape = [len(t), 2]
            * xS_minus is the vector position of the plus image in arcsec 
              with shape = [len(t), 2]

        """
        dt_in_years = (t_obs - self.t0) / days_per_year

        # Equation of motion for the relative angular separation between the
        # background source and lens.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel)  # mas
        thetaS -= (self.piRel * parallax_vec)  # mas

        u_vec = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u_vec, axis=1)
        u_hat = (u_vec.T / u_amp).T

        u_plus = ((u_amp + np.sqrt(u_amp ** 2 + 4)) / 2.0) * u_hat.T
        u_minus = ((u_amp - np.sqrt(u_amp ** 2 + 4)) / 2.0) * u_hat.T
        u_plus = u_plus.T
        u_minus = u_minus.T

        # Lensed Source Images - Lens Image
        xSL_plus = u_plus * self.thetaE_amp  # in mas
        xSL_minus = u_minus * self.thetaE_amp  # in mas

        xL = self.get_lens_astrometry(t_obs)

        xS_plus = xL + (xSL_plus * 1e-3)  # arcsec
        xS_minus = xL + (xSL_minus * 1e-3)  # arcsec

        return (xS_plus, xS_minus)

    def calc_piE_ecliptic(self):
        """
        Get piE_ecliptic, the microlensing parallax vector in the 
        ecliptic coorindate system. 
        """
        # Project the microlensing parallax into parallel and perpendicular
        # w.r.t. the ecliptic... useful quantities.
        parallax_vec_at_t0 = \
            parallax_in_direction(self.raL, self.decL, np.array([self.t0]))[0]

        # Unit vector parallel to the ecliptic
        par_hat = parallax_vec_at_t0 / np.linalg.norm(parallax_vec_at_t0)

        # Unit vector perpendicular to the ecliptic
        # Cross product with -z vector (where z increases along the line of sight) in order
        # to define the perpindicular to the ecliptic. Ideally, this would point to the North
        # Galactic pole; but I am not sure if it does.
        perp_hat = np.cross(np.append(par_hat, [0]), np.array([0, 0, -1]))[0:2]

        # Project piE_EN onto piE_parallel_perpendicular
        proj_piE_eclipt = np.dot(self.piE, par_hat)
        proj_piE_eclipt_ortho = np.dot(self.piE, perp_hat)

        # Save into a new vector
        self.piE_eclipt = np.array([proj_piE_eclipt, proj_piE_eclipt_ortho])

        return

    # Label this as phot?
    def get_geoproj_params(self, t0par):
        """
        Get the photometric microlensing model parameters in the geocentric-projected
        coordinate system, which just applies a rectalinear position and 
        velocity offset into the geocentric frame at time t0par.
        Note, this is not a true geocentric frame. It is only geocentric
        at time t0par. However, this is a common convention for photometry-only
        microlens models in the literature. The benefits of the 
        geocentric-projected frame is that the t0_{geoProj} can more closely match
        the observed peak in the light curve.

        Parameters
        ----------
        t0par : float
            Time in MJD at which to convert into the geocentric frame.

        Returns
        -------
        t0_g : float
            The time (in MJD) of closest approach between the lens and source
            in the geocentric-projected frame.
        u0_g : float
            The distance (in thetaE) at closest approach
            in the geocentric-projected frame.
        tE_g : float
            The Einsten crossing time (in MJD)
            in the geocentric-projected frame.
        piEE_g : float
            The East-component of the microlensing parallax vector,
            in the geocentric-projected frame.
            This also indicates the East-component of the
            relative proper motion vector between the source and lens
        piEN_g : float
            The North-component of the microlensing parallax vector,
            in the geocentric-projected frame.
            This also indicates the North-component of the
            relative proper motion vector between the source and lens

        """
        t0_g, u0_g, tE_g, piEE_g, piEN_g = fc.convert_helio_geo_phot(self.raL, self.decL,
                                                                     self.t0, self.u0_amp,
                                                                     self.tE, self.piE[0], self.piE[1],
                                                                     t0par, in_frame='helio',
                                                                     murel_in='SL', murel_out='LS',
                                                                     coord_in='EN', coord_out='tb')

        return t0_g, u0_g, tE_g, piEE_g, piEN_g

    # Make sure this method fails for phot only parametrizations.
    def get_geoproj_ast_params(self, t0par):
        """
        Get the astrometric microlensing model parameters in the geocentric-projected
        coordinate system, which just applies a rectalinear position and 
        velocity offset into the geocentric frame at time t0par.
        Note, this is not a true geocentric frame. It is only geocentric
        at time t0par. However, this is a common convention for photometry-only
        microlens models in the literature. The benefits of the 
        geocentric-projected frame is that the t0_{geoProj} can more closely match
        the observed peak in the light curve.

        Parameters
        ----------
        t0par : float
            Time in MJD at which to convert into the geocentric frame.

        Returns
        -------
        xS0E_g : float
            The East-component of source position vector on the sky,
            in the geocentric-projected frame.
        xS0N_g : float
            The North-component of source position vector on the sky,
            in the geocentric-projected frame.
        muSE_g : float
            The East-component of source proper motion vector,
            in the geocentric-projected frame.
        muSN_g : float
            The North-component of source proper motion vector,
            in the geocentric-projected frame.

        """
        xS0E_g, xS0N_g, muSE_g, muSN_g = convert_helio_geo_ast(self.raL, self.decL,
                                                               self.piS, self.xS0[0], self.xS0[1],
                                                               self.muS[0], self.muS[1],
                                                               self.t0, self.u0_amp,
                                                               self.tE, self.piE[0], self.piE[1],
                                                               t0par,
                                                               in_frame='helio',
                                                               murel_in='SL', murel_out='LS',
                                                               coord_in='EN', coord_out='tb')

        return xS0E_g, xS0N_g, muSE_g, muSN_g


# This is following the geocentric projected formalism
# e.g. P Mroz's code, MulensModel, ...
# Based on P Mroz's code which is based on MulensModel.
# FIGURE OUT: do the functions that are just lens or source
# (not relative to each other) need to be changed?
# I don't think so but I'm not 100% sure...
class PSPL_Parallax_geoproj(PSPL_Parallax):
    fixed_param_names = ['raL', 'decL', 't0par']

    def geta(self, ra, dec, t0par, t):
        """
        idk why it's called "geta"
        times are in MJD.
        """
        if type(ra) == str:
            coord = SkyCoord(ra, dec, unit=(units.hourangle, units.deg))
        if ((type(ra) == float) or (type(ra) == int)):
            coord = SkyCoord(ra, dec, unit=(units.deg, units.deg))

        direction = coord.cartesian.xyz.value    
        north = np.array([0., 0., 1.])
        _east_projected = np.cross(north, direction)/np.linalg.norm(np.cross(north, direction))
        _north_projected = np.cross(direction, _east_projected)/np.linalg.norm(np.cross(direction, _east_projected))

        t = t + 2400000.5
        t0par = t0par + 2400000.5
        _t0par = Time(t0par, format='jd', scale='tdb')
        _t = Time(t, format='jd', scale='tdb')
        (jd1, jd2) = get_jd12(_t0par, 'tdb')
        (earth_pv_helio, earth_pv_bary) = erfa.epv00(jd1, jd2) # this is earth-sun
        velocity = np.asarray(earth_pv_bary[1])

        position = get_body_barycentric(body='earth', time=_t) 
        position_ref = get_body_barycentric(body='earth', time=_t0par)

        delta_s = (position_ref.xyz.T - position.xyz.T).to(units.au).value
        delta_s += np.outer(t - t0par, velocity)

        out_e = np.dot(delta_s, _east_projected)
        out_n = np.dot(delta_s, _north_projected)

        return out_e, out_n

    def get_amplification(self, t):
        """
        Get an array of the photometric amplifications at the input times. 
        Parallax is included.

        Parameters
        ----------
        t : array_like
            Array of times in MJD.DDD

        """
        qe, qn = self.geta(self.raL, self.decL, self.t0par, t)

        tau = (t - self.t0) / self.tE
        dtau = self.piE[1]*qn + self.piE[0]*qe
        dbeta = self.piE[1]*qe - self.piE[0]*qn

        taup = tau + dtau
        betap = self.u0_amp + dbeta

        u_amp = np.hypot(taup, betap)

        A = (u_amp ** 2 + 2) / (u_amp * np.sqrt(u_amp ** 2 + 4))

        return A

    ### FIXME? ###
    def get_lens_astrometry(self, t_obs):
        """
        Get the astrometry for the foreground lens at the input times.
        Parallax is included. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).
        """
        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)

        # Equation of motion for just the background source.
        dt_in_years = (t_obs - self.t0) / days_per_year
        xL = self.xL0 + np.outer(dt_in_years, self.muL) * 1e-3
        xL += (self.piL * parallax_vec) * 1e-3  # arcsec

        return xL

    def get_astrometry(self, t_obs, ast_filt_idx=0):
        """
        Get the astrometry of the unresolved (observed) position of the
        lensed source at the input times.
        Parallax is included. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).
        ast_filt_idx : int, optional
            Index of the astrometric filter or data set.

        """

        # Things we will need.
        dt_in_years = (t_obs - self.t0) / days_per_year

        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)

        # Astrometric shift
        shift = self.get_centroid_shift(t_obs)

        # Equation of motion for just the background source.
        xS_unlensed = self.xS0 + np.outer(dt_in_years, self.muS) * 1e-3
        xS_unlensed += np.squeeze(self.piS * parallax_vec) * 1e-3  # arcsec

        xS = xS_unlensed + (shift * 1e-3)  # arcsec

        return xS

    def get_centroid_shift(self, t, ast_filt_idx=0):
        """
        Get the centroid shift (in mas) at the input times. The centroid shift
        is the difference between the lensed, unresolved position and the 
        intrinsic position of the source. 
        Parallax is included. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).

        """
        qe, qn = self.geta(self.raL, self.decL, self.t0par, t)

        tau = (t - self.t0) / self.tE
        dtau = self.piE[1]*qn + self.piE[0]*qe
        dbeta = self.piE[1]*qe - self.piE[0]*qn

        taup = tau + dtau
        betap = self.u0_amp + dbeta

        u_amp = np.hypot(taup, betap)

        u_N = -betap*self.muRel_hat[0] + taup*self.muRel_hat[1]
        u_E = betap*self.muRel_hat[1] + taup*self.muRel_hat[0]

        delta_N = u_N / (u_amp**2 + 2.0)
        delta_E = u_E / (u_amp**2 + 2.0)

        shift = self.thetaE_amp * np.vstack((delta_E, delta_N)).T

        return shift

    ### FIXME? ###
    def get_astrometry_unlensed(self, t_obs, t0par):
        """
        Get the astrometry of the source if the lens didn't exist.
        Parallax is included. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).

        Returns
        -------
        xS_unlensed : numpy array, dtype=float, shape = [len(t), 2]
            The unlensed positions of the source in arcseconds.
        """ 
        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)

        # Equation of motion for just the background source.
        dt_in_years = (t_obs - self.t0) / days_per_year
        xS_unlensed = self.xS0 + np.outer(dt_in_years, self.muS) * 1e-3
        xS_unlensed += (self.piS * parallax_vec) * 1e-3  # arcsec

        return xS_unlensed

    ### FIXME? ###
    def get_resolved_amplification(self, t):
        """
        Get the photometric amplification term at a set of times, t for both the
        plus and minus images.
        Parallax is included. The returned tuple has two entries:
        (A_plus, A_minus), each with len(t) arrays.

        Parameters
        ----------
        t : array_like
            Array of times in MJD.DDD
        """
        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t)

        # Equation of relative motion (angular on sky) Eq. 16 from Hog+ 1995
        dt_in_years = (t - self.t0) / days_per_year
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel) - (
                self.piRel * parallax_vec)  # mas
        u = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u, axis=1)

        A_plus = 0.5 * (
                (u_amp ** 2 + 2) / (u_amp * np.sqrt(u_amp ** 2 + 4)) + 1)
        A_minus = 0.5 * (
                (u_amp ** 2 + 2) / (u_amp * np.sqrt(u_amp ** 2 + 4)) - 1)

        return (A_plus, A_minus)

    ### FIXME? ###
    def get_resolved_astrometry(self, t_obs, t0par):
        """
        Get the relative RA and Dec astrometry for each of the two source images,
        which we label plus and minus.
        Parallax is included. The returned tuple has two entries:
        (xS_plus, xS_minus), each with [len(t), 2] arrays where the second dimension
        includes [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).

        Returns
        -------
        (xS_plus, xS_minus) : tuple of numpy arrays
            * xS_plus is the vector position of the plus image in arcsec 
              with shape = [len(t), 2]
            * xS_minus is the vector position of the plus image in arcsec 
              with shape = [len(t), 2]

        """
        dt_in_years = (t_obs - self.t0) / days_per_year

        # Equation of motion for the relative angular separation between the
        # background source and lens.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel)  # mas
        thetaS -= (self.piRel * parallax_vec)  # mas

        u_vec = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u_vec, axis=1)
        u_hat = (u_vec.T / u_amp).T

        u_plus = ((u_amp + np.sqrt(u_amp ** 2 + 4)) / 2.0) * u_hat.T
        u_minus = ((u_amp - np.sqrt(u_amp ** 2 + 4)) / 2.0) * u_hat.T
        u_plus = u_plus.T
        u_minus = u_minus.T

        # Lensed Source Images - Lens Image
        xSL_plus = u_plus * self.thetaE_amp  # in mas
        xSL_minus = u_minus * self.thetaE_amp  # in mas

        xL = self.get_lens_astrometry(t_obs)

        xS_plus = xL + (xSL_plus * 1e-3)  # arcsec
        xS_minus = xL + (xSL_minus * 1e-3)  # arcsec

        return (xS_plus, xS_minus)

    def calc_piE_ecliptic(self):
        """
        Get piE_ecliptic, the microlensing parallax vector in the 
        ecliptic coorindate system. 
        """
        # Project the microlensing parallax into parallel and perpendicular
        # w.r.t. the ecliptic... useful quantities.
        parallax_vec_at_t0 = \
            parallax_in_direction(self.raL, self.decL, np.array([self.t0]))[0]

        # Unit vector parallel to the ecliptic
        par_hat = parallax_vec_at_t0 / np.linalg.norm(parallax_vec_at_t0)

        # Unit vector perpendicular to the ecliptic
        # Cross product with -z vector (where z increases along the line of sight) in order
        # to define the perpindicular to the ecliptic. Ideally, this would point to the North
        # Galactic pole; but I am not sure if it does.
        perp_hat = np.cross(np.append(par_hat, [0]), np.array([0, 0, -1]))[0:2]

        # Project piE_EN onto piE_parallel_perpendicular
        proj_piE_eclipt = np.dot(self.piE, par_hat)
        proj_piE_eclipt_ortho = np.dot(self.piE, perp_hat)

        # Save into a new vector
        self.piE_eclipt = np.array([proj_piE_eclipt, proj_piE_eclipt_ortho])

        return

    def get_helio_params(self):
        t0_h, u0_h, tE_h, piEE_h, piEN_h = fc.convert_helio_geo_phot(self.raL, self.decL,
                                                                     self.t0, self.u0_amp,
                                                                     self.tE, self.piE[0], self.piE[1],
                                                                     self.t0par, in_frame='geo',
                                                                     murel_in='LS', murel_out='SL',
                                                                     coord_in='tb', coord_out='EN')

        return t0_h, u0_h, tE_h, piEE_h, piEN_h


class PSPL_noParallax_LumLens(PSPL_noParallax):
    parallaxFlag = False

    def get_centroid_shift(self, t, ast_filt_idx=0):
        """
        Get the centroid shift (in mas) at the input times. The centroid shift
        is the difference between the lensed, unresolved position and the 
        intrinsic position of the source. 
        No parallax is included and the lens is assumed to be luminous. 
        The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).
        ast_filt_idx : int, optional
            Index of the astrometric filter or data set.

        """
        tau = (t - self.t0) / self.tE

        # Assume all neighbor flux is in the lens.
        g = (1.0 - self.b_sff[ast_filt_idx]) / self.b_sff[ast_filt_idx]

        u2 = tau ** 2 + self.u0_amp ** 2
        u = np.sqrt(u2)

        # u^2 - u\sqrt{u^2 + 4} + 3
        numer_u = u2 - u * np.sqrt(u2 + 4) + 3

        # u^2 + 2 + gu\sqrt{u^2+4}
        denom_u = u2 + 2 + g * u * np.sqrt(u2 + 4)

        # \vec{\theta}_S = theta_E \vec{u}
        thetaS = (np.outer(tau, self.thetaE_hat) + self.u0) * self.thetaE_amp

        # Lens-induced astrometric shift of the sum of all source images (in mas)
        numer = thetaS * (1 + g * numer_u)
        denom = (1 + g) * denom_u

        shift = numer / denom.reshape(numer.shape[0], 1)

        return shift

class PSPL_Parallax_LumLens(PSPL_Parallax):
    parallaxFlag = True

    def get_astrometry(self, t_obs, ast_filt_idx=0):
        """
        Get the centroid shift (in mas) at the input times. The centroid shift
        is the difference between the lensed, unresolved position and the 
        intrinsic position of the source. 
        Parallax is included and the lens is assumed to be luminous. 
        The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).
        ast_filt_idx : int, optional
            Index of the astrometric filter or data set.

        """
        # Things we will need.
        dt_in_years = (t_obs - self.t0) / days_per_year

        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
        # Equation of motion for just the background source.
        xS_unlensed = self.xS0 + np.outer(dt_in_years, self.muS) * 1e-3
        xS_unlensed += np.squeeze(self.piS * parallax_vec) * 1e-3  # arcsec

        # Equation of motion for the relative angular separation between the background source and lens.
        # Note, we don't just call get_centroid_shift() because parallax_vec calculation is repeated.
        # and it is slow. 
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel)  # mas
        thetaS -= np.squeeze(self.piRel * parallax_vec)  # mas
        u_vec = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u_vec, axis=1)

        # Assume all neighbor flux is in the lens.
        g = (1.0 - self.b_sff[ast_filt_idx]) / self.b_sff[ast_filt_idx]

        # u^2 - u\sqrt{u^2 + 4} + 3
        numer_u = u_amp ** 2 - u_amp * np.sqrt(u_amp ** 2 + 4) + 3

        # u^2 + 2 + gu\sqrt{u^2+4}
        denom_u = u_amp ** 2 + 2 + g * u_amp * np.sqrt(u_amp ** 2 + 4)

        # Lens-induced astrometric shift of the sum of all source images (in mas)
        numer = thetaS * (1 + g * numer_u).reshape(len(numer_u), 1)
        denom = (1 + g) * denom_u

        shift = numer / denom.reshape((len(u_amp), 1))  # mas

        xS = xS_unlensed + (shift * 1e-3)  # arcsec

        return xS

    def get_centroid_shift(self, t, ast_filt_idx=0):
        """
        Get the centroid shift (in mas) at the input times. The centroid shift
        is the difference between the lensed, unresolved position and the 
        intrinsic position of the source. 
        Parallax is included and the lens is luminous. The returned array is in arcsec and
        has a shape of [len(t), 2] where the second dimension includes 
        [RA, Dec] positions in arcsec.

        Parameters
        ----------
        t : array_like
            Time (in MJD).

        """
        # Things we will need.
        dt_in_years = (t - self.t0) / days_per_year

        # Get the parallax vector for each date.
        parallax_vec = parallax_in_direction(self.raL, self.decL, t)

        # Equation of motion for the relative angular separation between the background source and lens.
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel)  # mas
        thetaS -= (self.piRel * parallax_vec)  # mas
        u_vec = thetaS / self.thetaE_amp
        u_amp = np.linalg.norm(u_vec, axis=1)

        # Assume all neighbor flux is in the lens.
        g = (1.0 - self.b_sff[ast_filt_idx]) / self.b_sff[ast_filt_idx]

        # u^2 - u\sqrt{u^2 + 4} + 3
        numer_u = u_amp ** 2 - u_amp * np.sqrt(u_amp ** 2 + 4) + 3

        # u^2 + 2 + gu\sqrt{u^2+4}
        denom_u = u_amp ** 2 + 2 + g * u_amp * np.sqrt(u_amp ** 2 + 4)

        # Lens-induced astrometric shift of the sum of all source images (in mas)
        numer = thetaS * (1 + g * numer_u)
        denom = (1 + g) * denom_u

        shift = numer / denom.reshape((len(u_amp), 1))  # mas

        return shift


######################################################
### POINT SOURCE BINARY LENS (PSBL) CLASSES ###
######################################################
# --------------------------------------------------
#
# Data Class Family - PSBL
#
# --------------------------------------------------

class PSBL(PSPL):
    """
    Contains methods for model a PSBL photometry + astrometry.
    This is a Data-type class in our hierarchy. It is abstract and should not
    be instantiated.
    """

    def get_amp_arr(self, z_arr, z1, z2):
        """Calculations amplification array

        Calculates the amplification A from the Jacobian J, :math:`A = 1/|J|`

        Parameters
        ----------
        z_arr : array_like
            | Complex position of images. ``Shape = [N_times, N_solutions, 1]``
            | -- note this could be jagged.

        z1 : array_like
            Complex position(s) of lens 1 (primary). ``Shape = [N_times, 1]``

        z2 : array_like
            Complex position(s) of lens 2 (secondary). ``Shape = [N_times, 1]``

        Returns
        -------
        amp_arr : array_like
            BLEH
        """



        N_times = z1.shape[0]
        #print(z_arr)
        #print(z1)
        #print(z2)
        dwbardz = self.m1 / (z_arr - z1.reshape((N_times, 1))) ** 2
        dwbardz += self.m2 / (z_arr - z2.reshape((N_times, 1))) ** 2
        jacobian = 1 - np.absolute(dwbardz) ** 2
        amp_arr = 1.0 / np.absolute(jacobian)  # Absolute value of J
        
        # CASEY: CHECK

        return amp_arr

    def rescale_complex_pos(self, w, z1, z2):
        """
        Make sure everything is roughly centered on the origin
        in a 1 x 1 box.
        """
        m1 = copy.deepcopy(self.m1)
        m2 = copy.deepcopy(self.m2)
        
        # Put the positions of the source and lenses into
        # an array, so we can calculate the average position
        # and "width" of points at each time, in order to center
        # and scale them.
        pos = np.vstack([w, z1, z2]).T

        # Calculate the average position to get the shift.
        shift = np.average(pos, axis=1)
        w -= shift
        z1 -= shift
        z2 -= shift
        
        # Calculate the average spread to get the scale.
        xscale = np.max(pos.real, axis=1) - np.min(pos.real, axis=1)
        yscale = np.max(pos.imag, axis=1) - np.min(pos.imag, axis=1)
        xyscale = np.concatenate([xscale, yscale]).reshape(len(xscale),2)
        scale = 1/np.max(xyscale, axis=1)
        w *= scale
        z1 *= scale
        z2 *= scale
        m1 *= scale**2
        m2 *= scale**2

        return w, z1, z2, m1, m2, scale, shift
    
    def get_image_pos_arr_old(self, w, z1, z2, check_sols=True):
        """Gets image positions.
        | Solve the fifth-order polynomial and get the image positions.
        | See PSBL writeup for full equations.
        | All angular distances are in arcsec.

        Parameters
        ----------
        w : array_like
            Complex position(s) of the source. Shape = [N_times, 1]

        z1 : array_like
            Complex position(s) of lens 1 (primary). Shape = [N_times, 1]

        z2 : array_like
            Complex position(s) of lens 2 (secondary). Shape = [N_times, 1]

        check_sols : bool, optional
            If True, calculated roots are checked against the lens equation,
            and output will only contain those within self.root_tol.
            If False, all calculated roots are returned.

        Returns
        -------
        z_arr : array_like
            Position of the lensed source images.
            Rank-1 array of polynomial roots, possibly complex.
            If check_sols = True, only roots solving the lens
            equation are returned.
        """
        assert (len(w) == len(z1)) & (len(w) == len(z2))

        wbar = np.conj(w)
        z1bar = np.conj(z1)
        z2bar = np.conj(z2)

        #####################################
        # Solve the lens equation!!!!!
        #####################################
        # The lens equation is in the form
        # f(z) = \sum_i a_i z^i = 0 for i = 0 to 5.
        # Here are the coefficients:
        # NIJAID's coeff - matches with Witt 1995 in their limits
        a5 = (wbar - z1bar) * (wbar - z2bar)
        a4 = -((w + 2 * (z1 + z2)) * wbar ** 2) - self.m2 * z2bar - \
             z1bar * (self.m1 + (w + 2 * (z1 + z2)) * z2bar) + \
             wbar * (self.m1 + self.m2 + (w + 2 * (z1 + z2)) * (z1bar + z2bar))
        a3 = (z1 ** 2 + 4 * z1 * z2 + z2 ** 2 + 2 * w * (
                z1 + z2)) * wbar ** 2 + \
             (self.m1 * (w - z1) + self.m2 * (w + 2 * z1 + z2)) * z2bar + \
             z1bar * (self.m2 * (w - z2) + self.m1 * (w + z1 + 2 * z2) + \
                      (z1 ** 2 + 4 * z1 * z2 + z2 ** 2 + 2 * w * (
                              z1 + z2)) * z2bar) - \
             wbar * (2 * (self.m2 * (w + z1) + self.m1 * (w + z2)) + \
                     (z1 ** 2 + 4 * z1 * z2 + z2 ** 2 + 2 * w * (z1 + z2)) * (
                             z1bar + z2bar))
        a2 = -((self.m1 + self.m2) * (
                self.m1 * (w - z1) + self.m2 * (w - z2))) - \
             (2 * z1 * z2 * (z1 + z2) + w * (
                     z1 ** 2 + 4 * z1 * z2 + z2 ** 2)) * wbar ** 2 - \
             (self.m2 * (w - z2) * (2 * z1 + z2) + self.m1 * (
                     w * z1 + 2 * (w + z1) * z2 + z2 ** 2)) * z1bar - \
             (self.m2 * w * (2 * z1 + z2) + self.m1 * (w - z1) * (
                     z1 + 2 * z2) + self.m2 * z1 * (z1 + 2 * z2) + \
              2 * z1 * z2 * (z1 + z2) * z1bar + w * (
                      z1 ** 2 + 4 * z1 * z2 + z2 ** 2) * z1bar) * \
             z2bar + wbar * (z1 * (
                2 * self.m1 * w + 4 * self.m2 * w - self.m1 * z1 + self.m2 * z1) + \
                             2 * (2 * self.m1 + self.m2) * w * z2 + (
                                     self.m1 - self.m2) * z2 ** 2 + \
                             (2 * z1 * z2 * (z1 + z2) + w * (
                                     z1 ** 2 + 4 * z1 * z2 + z2 ** 2)) * (
                                     z1bar + z2bar))
        a1 = 2 * self.m1 ** 2 * w * z2 + 2 * self.m1 * self.m2 * w * z2 - self.m1 * self.m2 * z2 ** 2 - 2 * self.m1 * w * z2 ** 2 * wbar + \
             self.m1 * w * z2 ** 2 * z1bar + self.m1 * w * z2 ** 2 * z2bar + \
             z1 ** 2 * (-(
                self.m1 * self.m2) - 2 * self.m2 * w * wbar + 2 * self.m1 * z2 * wbar + \
                        2 * w * z2 * wbar ** 2 + z2 ** 2 * wbar ** 2 + self.m2 * (
                                w - z2) * z1bar - \
                        2 * w * z2 * wbar * z1bar - z2 ** 2 * wbar * z1bar + \
                        self.m2 * w * z2bar - 2 * self.m1 * z2 * z2bar + self.m2 * z2 * z2bar - \
                        2 * w * z2 * wbar * z2bar - z2 ** 2 * wbar * z2bar + \
                        2 * w * z2 * z1bar * z2bar + z2 ** 2 * z1bar * z2bar) + \
             z1 * (2 * self.m1 * self.m2 * w + 2 * self.m2 ** 2 * (
                w - z2) - 2 * self.m1 ** 2 * z2 - 2 * self.m1 * self.m2 * z2 - \
                   4 * self.m1 * w * z2 * wbar - 4 * self.m2 * w * z2 * wbar + 2 * self.m2 * z2 ** 2 * wbar + \
                   2 * w * z2 ** 2 * wbar ** 2 + 2 * self.m1 * w * z2 * z1bar + \
                   2 * self.m2 * (
                           w - z2) * z2 * z1bar + self.m1 * z2 ** 2 * z1bar - \
                   2 * w * z2 ** 2 * wbar * z1bar + 2 * self.m1 * w * z2 * z2bar + \
                   2 * self.m2 * w * z2 * z2bar - self.m1 * z2 ** 2 * z2bar - \
                   2 * w * z2 ** 2 * wbar * z2bar + 2 * w * z2 ** 2 * z1bar * z2bar)
        a0 = (self.m2 * z1 + self.m1 * z2) * (
                self.m1 * (-w + z1) * z2 + self.m2 * z1 * (-w + z2)) + \
             z1 * z2 * (-(w * z1 * z2 * wbar ** 2) - (
                self.m2 * z1 * (w - z2) + self.m1 * w * z2) * z1bar - \
                        (self.m2 * w * z1 + self.m1 * (
                                w - z1) * z2 + w * z1 * z2 * z1bar) * z2bar + \
                        wbar * (2 * self.m2 * w * z1 + 2 * self.m1 * w * z2 - (
                        self.m1 + self.m2) * z1 * z2 + \
                                w * z1 * z2 * (z1bar + z2bar)))

        # Solve the quintic equation and find all 5 roots.
        # Loop through different time steps and solve each one.
        N_times = len(w)
        z_arr = np.zeros((N_times, 5), dtype=np.complex_)
        for i in range(N_times):
            z_arr[i] = np.roots([a5[i], a4[i], a3[i], a2[i], a1[i], a0[i]])

        # Plug the solutions from the quintic equation back into the lens equation 
        # and see if those roots are actually solutions.
        # There should either be 3 (outside caustic) or 5 (inside caustic).
        # (for our regime, it should be 3)
        if check_sols:
            for i in range(N_times):
                z = z_arr[i, :]
                c1 = self.m1 / np.conj(z - z1[i])
                c2 = self.m2 / np.conj(z - z2[i])
                diff = w[i] - (z - c1 - c2)
                bad_solutions = np.absolute(diff) > self.root_tol
                z_arr[i][bad_solutions] = np.nan + np.nan * 0j

            # nim = (~np.isnan(z_arr)).sum(axis=1)
            # nim_good = (nim == 5).sum() + (nim == 3).sum()
            #
            # if len(nim) != nim_good:
            #     print('Not all solutions have 3 or 5 images-- something is wrong!')
            #     images = []
            #     for ii in np.arange(6):
            #         idx = np.where(nim == ii)[0]
            #         images.append(idx)
            #         print('N images = {0} : {1}'.format(ii, (nim == ii).sum()))
            #
            #     return z_arr, images

        else:
            return z_arr


    def get_image_pos_arr(self, w, z1, z2, m1, m2, check_sols=True):
        """Gets image positions.

        | Solve the fifth-order polynomial and get the image positions.
        | See PSBL writeup for full equations.
        | All angular distances are in arcsec.



        Parameters
        ----------
        w : array_like
            Complex position(s) of the source. Shape = [N_times, 1]

        z1 : array_like
            Complex position(s) of lens 1 (primary). Shape = [N_times, 1]

        z2 : array_like
            Complex position(s) of lens 2 (secondary). Shape = [N_times, 1]

        m1 : float

        m2 : float

        check_sols : bool, optional
            If True, calculated roots are checked against the lens equation,
            and output will only contain those within self.root_tol.
            If False, all calculated roots are returned.

        Returns
        -------
        z_arr : array_like
            Rank-1 array of polynomial roots, possibly complex.
            If check_sols = True, only roots solving the lens
            equation are returned.
        """
        assert (len(w) == len(z1)) & (len(w) == len(z2))

        wbar = np.conj(w)
        z1bar = np.conj(z1)
        z2bar = np.conj(z2)

        #####################################
        # Solve the lens equation!!!!!
        #####################################
        # The lens equation is in the form
        # f(z) = \sum_i a_i z^i = 0 for i = 0 to 5.
        # Here are the coefficients:
        # NIJAID's coeff - matches with Witt 1995 in their limits
        a5 = (wbar - z1bar) * (wbar - z2bar)
        a4 = -((w + 2 * (z1 + z2)) * wbar ** 2) - m2 * z2bar - \
             z1bar * (m1 + (w + 2 * (z1 + z2)) * z2bar) + \
             wbar * (m1 + m2 + (w + 2 * (z1 + z2)) * (z1bar + z2bar))
        a3 = (z1 ** 2 + 4 * z1 * z2 + z2 ** 2 + 2 * w * (
                    z1 + z2)) * wbar ** 2 + \
             (m1 * (w - z1) + m2 * (w + 2 * z1 + z2)) * z2bar + \
             z1bar * (m2 * (w - z2) + m1 * (w + z1 + 2 * z2) + \
                      (z1 ** 2 + 4 * z1 * z2 + z2 ** 2 + 2 * w * (
                                  z1 + z2)) * z2bar) - \
             wbar * (2 * (m2 * (w + z1) + m1 * (w + z2)) + \
                     (z1 ** 2 + 4 * z1 * z2 + z2 ** 2 + 2 * w * (z1 + z2)) * (
                                 z1bar + z2bar))
        a2 = -((m1 + m2) * (
                    m1 * (w - z1) + m2 * (w - z2))) - \
             (2 * z1 * z2 * (z1 + z2) + w * (
                         z1 ** 2 + 4 * z1 * z2 + z2 ** 2)) * wbar ** 2 - \
             (m2 * (w - z2) * (2 * z1 + z2) + m1 * (
                         w * z1 + 2 * (w + z1) * z2 + z2 ** 2)) * z1bar - \
             (m2 * w * (2 * z1 + z2) + m1 * (w - z1) * (
                         z1 + 2 * z2) + m2 * z1 * (z1 + 2 * z2) + \
              2 * z1 * z2 * (z1 + z2) * z1bar + w * (
                          z1 ** 2 + 4 * z1 * z2 + z2 ** 2) * z1bar) * \
             z2bar + wbar * (z1 * (
                    2 * m1 * w + 4 * m2 * w - m1 * z1 + m2 * z1) + \
                             2 * (2 * m1 + m2) * w * z2 + (
                                         m1 - m2) * z2 ** 2 + \
                             (2 * z1 * z2 * (z1 + z2) + w * (
                                         z1 ** 2 + 4 * z1 * z2 + z2 ** 2)) * (
                                         z1bar + z2bar))
        a1 = 2 * m1 ** 2 * w * z2 + 2 * m1 * m2 * w * z2 - m1 * m2 * z2 ** 2 - 2 * m1 * w * z2 ** 2 * wbar + \
             m1 * w * z2 ** 2 * z1bar + m1 * w * z2 ** 2 * z2bar + \
             z1 ** 2 * (-(
                    m1 * m2) - 2 * m2 * w * wbar + 2 * m1 * z2 * wbar + \
                        2 * w * z2 * wbar ** 2 + z2 ** 2 * wbar ** 2 + m2 * (
                                    w - z2) * z1bar - \
                        2 * w * z2 * wbar * z1bar - z2 ** 2 * wbar * z1bar + \
                        m2 * w * z2bar - 2 * m1 * z2 * z2bar + m2 * z2 * z2bar - \
                        2 * w * z2 * wbar * z2bar - z2 ** 2 * wbar * z2bar + \
                        2 * w * z2 * z1bar * z2bar + z2 ** 2 * z1bar * z2bar) + \
             z1 * (2 * m1 * m2 * w + 2 * m2 ** 2 * (
                    w - z2) - 2 * m1 ** 2 * z2 - 2 * m1 * m2 * z2 - \
                   4 * m1 * w * z2 * wbar - 4 * m2 * w * z2 * wbar + 2 * m2 * z2 ** 2 * wbar + \
                   2 * w * z2 ** 2 * wbar ** 2 + 2 * m1 * w * z2 * z1bar + \
                   2 * m2 * (
                               w - z2) * z2 * z1bar + m1 * z2 ** 2 * z1bar - \
                   2 * w * z2 ** 2 * wbar * z1bar + 2 * m1 * w * z2 * z2bar + \
                   2 * m2 * w * z2 * z2bar - m1 * z2 ** 2 * z2bar - \
                   2 * w * z2 ** 2 * wbar * z2bar + 2 * w * z2 ** 2 * z1bar * z2bar)
        a0 = (m2 * z1 + m1 * z2) * (
                    m1 * (-w + z1) * z2 + m2 * z1 * (-w + z2)) + \
             z1 * z2 * (-(w * z1 * z2 * wbar ** 2) - (
                    m2 * z1 * (w - z2) + m1 * w * z2) * z1bar - \
                        (m2 * w * z1 + m1 * (
                                    w - z1) * z2 + w * z1 * z2 * z1bar) * z2bar + \
                        wbar * (2 * m2 * w * z1 + 2 * m1 * w * z2 - (
                            m1 + m2) * z1 * z2 + \
                                w * z1 * z2 * (z1bar + z2bar)))

        # Solve the lens equation and find all 5 roots.
        # Loop through different time steps and solve each one.
        N_times = len(w)
        z_arr = np.zeros((N_times, 5), dtype=np.complex_)
        ai_arr = np.zeros((N_times, 6), dtype=np.complex_)
        for i in range(N_times):
            ai_arr[i] = np.array([a5[i], a4[i], a3[i], a2[i], a1[i], a0[i]])
            z_arr[i] = np.roots([a5[i], a4[i], a3[i], a2[i], a1[i], a0[i]])

        # Plug back into equation and see if those roots are actually solutions.
        # There should either be 3 (outside caustic) or 5 (inside caustic).
        # (for our regime, it should be 3)
        if check_sols:
            for i in range(N_times):
                if type(m1) == np.ndarray:
                    m1_i = m1[i]
                    m2_i = m2[i]
                else:
                    m1_i = m1
                    m2_i = m2

                z = z_arr[i, :]
                c1 = m1_i / np.conj(z - z1[i])
                c2 = m2_i / np.conj(z - z2[i])
                diff = w[i] - (z - c1 - c2)
                bad_solutions = np.absolute(diff) > self.root_tol
                z_arr[i][bad_solutions] = np.nan + np.nan * 0j

            # nim = (~np.isnan(z_arr)).sum(axis=1)
            # nim_good = (nim == 5).sum() + (nim == 3).sum()
            #
            # fig, axes = plt.subplots(2, 3, figsize=(10,8), num=11)
            # plt.clf()
            # fig, axes = plt.subplots(2, 3, figsize=(10,8), num=11)
            # plt.subplots_adjust(left=0.1, hspace=0.4, wspace=0.4)
            # tstr = ['$a_5$', '$a_4$', '$a_3$', '$a_2$', '$a_1$', '$a_0$']
            # cstr = ['blue', 'orange', 'red', 'green', 'purple', 'gray']
            # for ii, ax in enumerate(axes.flat):
            #     for jj in np.arange(6):
            #         idx = np.where(nim == jj)[0]
            #         if len(idx) > 0:
            #             ax.plot(ai_arr[idx,ii].real, ai_arr[idx,ii].imag, '.',
            #                     markersize=2, alpha=0.8, label=str(jj), color=cstr[jj])
            #         ax.set_xlabel('Re')
            #         ax.set_ylabel('Im')
            #     ax.set_title(tstr[ii])
            # leg = axes[0,0].legend(markerscale=10)
            # # for kk in np.arange(len(np.unique(nim))):
            # #     leg.legendHandles[kk]._legmarker.set_markersize(10)
            # plt.suptitle('Coefficients $a_i$ to $f(x) = \sum x^i a_i$')
            #
            # if len(nim) != nim_good:
            #     print('Not all solutions have 3 or 5 images-- something is wrong!')
            #     images = []
            #     for ii in np.arange(6):
            #         idx = np.where(nim == ii)[0]
            #         images.append(idx)
            #         print('N images = {0} : {1}'.format(ii, (nim == ii).sum()))

        return z_arr

    def get_all_arrays(self, t_obs, check_sols=True, rescale=True):
        '''
        Obtain the image and amplitude arrays for each t_obs.

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.

        Returns
        -------
        images : array_like
            Array/tuple of complex positions of each images at each t_obs.
        amp_arr : array_like
            Array/tuple of amplification of each images at each t_obs.
        '''
        kwargs = {'check_sols' : check_sols}

        if rescale:
            # Get complex positions (no rescaling).
            _comp = self.get_complex_pos(t_obs)

            
            # Deepcopy because for some reason in my test it would modify.
            comp = copy.deepcopy(_comp)
            
            # Rescaled complex positions.
            rcomp = self.rescale_complex_pos(*_comp)


            # Image positions derived from rescale complex positions.
            rimages = self.get_image_pos_arr(*rcomp[0:5], **kwargs)

            # Take the image positions derived from the rescaled complex positions
            # and rescale them to get the images back in the original scale.
            images = (rimages/rcomp[5].reshape(len(rcomp[5]), 1)) + rcomp[6].reshape(len(rcomp[6]), 1)
            # Get amplifications.
            amps = self.get_amp_arr(images, *comp[1:])

        else:
            comp = self.get_complex_pos(t_obs)
            images = self.get_image_pos_arr_old(*comp)
            amps = self.get_amp_arr(images, *comp[1:])
        
        return images, amps

    def get_resolved_photometry(self, t_obs, filt_idx=0, amp_arr=None, print_warning=True):
        '''
        Get the photometry for each of the lensed source images.
        Implement with no blending (since we don't support different
        blendings for the different images).

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.

        Other Parameters
        ----------------
        amp_arr : array_like
            Amplifications of each individual image at each time,
            i.e. amp_arr.shape = (len(t_obs), number of images at each t_obs).

            This will over-ride t_obs; but is more efficient when calculating
            both photometry and astrometry. If None, then just use t_obs.
        filt_idx : int
            The filter index (def=0).

        Returns
        -------
        mag_model : array_like
            Magnitude of each lensed image centroid at t_obs.
            Shape = [5, len(t_obs)]
        '''
        mag_zp = 30.0  # arbitrary but allows for negative blend fractions.
        flux_zp = 1.0

        if amp_arr is None:
            img_arr, amp_arr = self.get_all_arrays(t_obs)

        # Mask invalid values from the amplification array.
        amp_arr_mskd = np.masked_invalid(amp_arr)

        flux_src = flux_zp * 10 ** ((self.mag_src[filt_idx] - mag_zp) / -2.5)
        flux_model = flux_src * amp_arr_mskd

        # Account for blending, if necessary.
        try:
            # Adding flux of neighbors and lens
            # b_sff = fS / (fS + fN + fL)
            flux_model += flux_src * (1.0 - self.b_sff[filt_idx]) / \
                          self.b_sff[filt_idx]
        except AttributeError:
            pass

        # Catch the edge case where we exceed the zeropoint.
        bad = np.where(flux_model <= 0)
        if len(bad[0]) > 0:
            if print_warning:
                pdb.set_trace()
                print('!! ! ! !! Warning: get_photometry: bad flux encountered.')
            flux_model[bad] = np.nan

        mag_model = -2.5 * np.log10(flux_model / flux_zp) + mag_zp

        return mag_model

    def get_photometry(self, t_obs, filt_idx=0, amp_arr=None, print_warning=True):
        '''
        Get the photometry for each of the lensed source images.

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.
        filt_idx : int, optional
            Index of the photometric filter or data set.

        Other Parameters
        ----------------
        amp_arr : array_like
            Amplifications of each individual image at each time,
            i.e. ``amp_arr.shape = (len(t_obs)``, number of images at each t_obs).

            This will over-ride t_obs; but is more efficient when calculating
            both photometry and astrometry. If None, then just use t_obs.
        print_warning : bool, optional
            Print a warning in the rare case that the magnitude exceeds a 
            zeropoint of 30 and conversions result in NaN returned.

        Returns
        -------
        mag_model : array_like
            Magnitude of the unresolved microlensing event at t_obs.
        '''
        mag_zp = 30.0  # arbitrary but allows for negative blend fractions.
        flux_zp = 1.0

        if amp_arr is None:
            img_arr, amp_arr = self.get_all_arrays(t_obs)

        # Mask invalid values from the amplification array.
        amp_arr_msk = np.ma.masked_invalid(amp_arr)

        # Sum up all the amplifications b/c surface brightness is conserved.
        amp = np.sum(amp_arr_msk, axis=1)

        flux_src = flux_zp * 10 ** ((self.mag_src[filt_idx] - mag_zp) / -2.5)
        flux_model = flux_src * amp

        # Account for blending, if necessary.
        try:
            # Adding flux of neighbors and lenses
            # b_sff = fS / (fS + fN + fL)
            flux_model += flux_src * (1.0 - self.b_sff[filt_idx]) / self.b_sff[filt_idx]
        except AttributeError:
            pass

        # Catch the edge case where we exceed the zeropoint.
        bad = np.where(flux_model <= 0)[0]
        if len(bad) > 0:
            if print_warning:
                print('!!!!!!!!!! Warning: get_photometry: bad flux encountered.')
                print('')
#                pdb.set_trace()
            flux_model[bad] = np.nan

        mag_model = -2.5 * np.log10(flux_model / flux_zp) + mag_zp

        # Set the masked values (in the data array) to also be nan.
        bad = np.where(flux_model.mask == True)
        if len(bad) > 0:
            mag_model.data[bad] = np.nan

        return mag_model


class PSBL_Phot(PSBL, PSPL_Phot):
    photometryFlag = True
    astrometryFlag = False

    def get_complex_pos(self, t_obs):
        """ Get the positions of the lenses and source as complex numbers. 
        
        This is needed for further calculations.
        Note that all units are still the same as before, this
        is just rewriting vectors :math:`z = (x,y)` as :math:`z = x + iy`.

        Returns
        -------
        w : complex array
            Source position as an array of complex numbers with
            real = east component, imaginary = north component

        z1 : complex array
            Lens primary component position as an array of complex numbers with
            real = east component, imaginary = north component

        z2 : complex array
            Lens secondary component position as an array of complex numbers with
            real = east component, imaginary = north component
        """
        if not isinstance(t_obs, np.ndarray):
            raise RuntimeError("time must be a 1D numpy array")

        # Calculate the position of the source w.r.t. lens (in Einstein radii)
        # Distance along muRel direction
        tau = (t_obs - self.t0) / self.tE
        tau = tau.reshape(len(tau), 1)

        # Distance along u0 direction -- always constant with time.
        u0 = self.u0.reshape(1, len(self.u0))
        thetaE_hat = self.thetaE_hat.reshape(1, len(self.thetaE_hat))

        # Total distance
        u = u0 + tau * thetaE_hat

        # Incorporate parallax
        if self.parallaxFlag:
            parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
            u -= self.piE_amp * parallax_vec

        # Convert positions to complex coordinates
        w = u[:, 0] + u[:, 1] * 1j

        # Get the position of the lenses (in units of Einstein radii)
        z1 = self.xL1_over_theta[0] + self.xL1_over_theta[1] * 1j
        z2 = self.xL2_over_theta[0] + self.xL2_over_theta[1] * 1j

        z1 = np.repeat(z1, w.shape[0])
        z2 = np.repeat(z2, w.shape[0])

        return w, z1, z2

    def get_resolved_lens_astrometry(self, t_obs):
        """Equation of motion for just the foreground lenses, individually.
        
        Parameters
        ----------
        t_obs : array_like
            Time (in MJD).

        Notes
        -----
        .. note::
           Note, this is a photometry only model, so units are in Einstein radii.
        """
        # In phot only fits, lens is at rest. So just duplicate to get
        # the right shape.
        xL1 = np.tile(self.xL1_over_theta, (len(t_obs), 1))
        xL2 = np.tile(self.xL2_over_theta, (len(t_obs), 1))

        return (xL1, xL2)

    def get_astrometry_unlensed(self, t_obs):
        """Get the astrometry of the source if the lens didn't exist.
        Note, this is a photometry only model, so units are in Einstein radii.

        Returns
        -------
        xS_unlensed : numpy array, dtype=float, ``shape = len(t_obs) x 2``
            The unlensed positions of the source in Einstein radii.


        Notes
        -----
        .. note::
           Note, this is a photometry only model, so units are in Einstein radii.
        """
        # Calculate the position of the source w.r.t. lens (in Einstein radii)
        # Distance along muRel direction
        tau = (t_obs - self.t0) / self.tE
        tau = tau.reshape(len(tau), 1)

        # Distance along u0 direction -- always constant with time.
        u0 = self.u0.reshape(1, len(self.u0))
        thetaE_hat = self.thetaE_hat.reshape(1, len(self.thetaE_hat))

        # Total distance
        u = u0 + tau * thetaE_hat

        # Incorporate parallax
        if self.parallaxFlag:
            parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
            u -= self.piE_amp * parallax_vec

        return u

    def get_resolved_astrometry(self, t_obs, image_arr=None, amp_arr=None):
        '''
        Position of the observed source position in Einstein radii.

        Parameters
        ----------
        t_obs : array_like, shape = [N_times]
            Array of times to model.

        Other Parameters
        ----------------
        image_arr : array_like
            Array of complex image positions at each t_obs,
            i.e. image_arr.shape = (len(t_obs), number of images at each t_obs).
            Each value in this array is complex
            (real = north component, imaginary = east component)

        amp_arr : array_like
            Array of magnifications of each images.
            Same shape as image_arr.

        Returns
        -------
        model_pos : array_like. shape = [N_times, N_images, 2]
            Array of vector positions of the centroid at each t_obs.
        '''
        if (image_arr is None) or (amp_arr is None):
            image_arr, amp_arr = self.get_all_arrays(t_obs)

        # In units of Einstein radii.
        xS_lensed_pos = image_arr.view('(2,)float')

        return xS_lensed_pos

    def get_astrometry(self, t_obs, image_arr=None, amp_arr=None, ast_filt_idx=0):
        '''
        Position of the observed (unresolved) source position in Einstein radii.

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.

        Other Parameters
        ----------------
        image_arr : array_like
            Array of complex image positions at each t_obs,
            i.e. image_arr.shape = (len(t_obs), number of images at each t_obs).
            Each value in this array is complex
            (real = north component, imaginary = east component)

        amp_arr : array_like
            Array of magnifications of each images.
            Same shape as image_arr.

        Returns
        -------
        model_pos : array_like
            Array of vector positions of the centroid at each t_obs.
        '''
        if (image_arr is None) or (amp_arr is None):
            image_arr, amp_arr = self.get_all_arrays(t_obs)

        # Split back into x, y such that shape = [N_times, N_images, 2]
        xS_lensed_res = image_arr.view('(2,)float')

        # Mask invalid values from the amplification array.
        amp_arr_mskd = np.ma.masked_invalid(amp_arr)
        amp_arr_mskd2 = amp_arr_mskd.reshape((amp_arr_mskd.shape[0], amp_arr_mskd.shape[1], 1))
        xS_lensed_res_mskd = np.ma.masked_invalid(xS_lensed_res)

        xS_lensed_ures = np.sum(xS_lensed_res_mskd * amp_arr_mskd2, axis=1) / np.sum(amp_arr_mskd2, axis=1)

        return xS_lensed_ures.data


class PSBL_PhotAstrom(PSBL, PSPL_PhotAstrom):
    """
    Contains methods for model a PSPL photometry + astrometry.
    This is a Data-type class in our hierarchy. It is abstract and should not
    be instantiated. 
    """
    photometryFlag = True
    astrometryFlag = True

    def get_complex_pos(self, t_obs):
        """
        Get the positions of the lenses and source as
        complex numbers. This is needed for further calculations.
        Note that all units are still the same as before, this
        is just rewriting vectors :math:`z = (x,y)` as :math:`z = x + iy`.

        Parameters
        ----------
        t : array_like
            Array of times to model.

        Returns
        -------
        w : complex array
            Source position (arcsec) as an array of complex numbers with
            real = east component, imaginary = north component
            shape = [N_times, N_sources].

        z1 : complex array
            Lens primary component position (arcsec) as an array of complex numbers with
            real = east component, imaginary = north component
            shape = [N_times]

        z2 : complex array
            Lens secondary component position (arcsec) as an array of complex numbers with
            real = east component, imaginary = north component
            shape = [N_times]
        """
        if not isinstance(t_obs, np.ndarray):
            raise RuntimeError("time must be a 1D numpy array")

        # Find positions of lens and source over t_obs
        xS_vec = self.get_astrometry_unlensed(t_obs)
        xL1_vec, xL2_vec = self.get_resolved_lens_astrometry(t_obs)

        # Convert positions to complex coordinates
        w = xS_vec[:, 0] + xS_vec[:, 1] * 1j

        z1 = xL1_vec[:, 0] + xL1_vec[:, 1] * 1j
        z2 = xL2_vec[:, 0] + xL2_vec[:, 1] * 1j

        return w, z1, z2

    def get_resolved_astrometry(self, t_obs, image_arr=None, amp_arr=None):
        '''
        Position of the observed source position in arcsec.

        Parameters
        ----------
        t_obs : array_like, shape = [N_times]
            Array of times to model.

        Other Parameters
        ----------------
        image_arr : array_like
            Array of complex image positions at each t_obs,
            i.e. `image_arr.shape = (len(t_obs)`, number of images at each t_obs).
            Each value in this array is complex
            (real = north component, imaginary = east component)

        amp_arr : array_like
            Array of magnifications of each images.
            Same shape as image_arr.

        Returns
        -------
        model_pos : array_like. shape = [N_times, N_images, 2]
            Array of vector positions of the centroid at each t_obs.
        '''
        if (image_arr is None) or (amp_arr is None):
            image_arr, amp_arr = self.get_all_arrays(t_obs)

        xS_lensed_pos = image_arr.view('(2,)float')

        return xS_lensed_pos

    def get_astrometry(self, t_obs, image_arr=None, amp_arr=None, ast_filt_idx=0):
        """
        Position of the observed (unresolved) source position in arcsec.

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.

        Other Parameters
        ----------------
        image_arr : array_like
            Array of complex image positions at each t_obs,
            i.e. `image_arr.shape = (len(t_obs)`, number of images at each t_obs).
            Each value in this array is complex
            (real = north component, imaginary = east component)

        amp_arr : array_like
            Array of magnifications of each images.
            Same shape as image_arr.

        ast_filt_idx : int
            The filter index for the astrometry.

        Returns
        -------
        model_pos : array_like
            Array of vector positions of the centroid at each t_obs.
        """
        if (image_arr is None) or (amp_arr is None):
            image_arr, amp_arr = self.get_all_arrays(t_obs)

        # Split back into x, y such that shape = [N_times, N_images, 2]
        xS_lensed_res = image_arr.view('(2,)float')

        # Mask invalid values from the amplification array.
        amp_arr_mskd = np.ma.masked_invalid(amp_arr)
        # amp_arr_mskd2 = np.tile(amp_arr_mskd, (1, 1, 2))
        amp_arr_mskd2 = amp_arr_mskd.reshape(
            (amp_arr_mskd.shape[0], amp_arr_mskd.shape[1], 1))
        xS_lensed_res_mskd = np.ma.masked_invalid(xS_lensed_res)

        xS_lensed_ures = np.sum(xS_lensed_res_mskd * amp_arr_mskd2,
                                axis=1) / np.sum(amp_arr_mskd2, axis=1)

        return xS_lensed_ures.data

    def get_astrometry_unlensed(self, t_obs):
        """Get the astrometry of the source if the lens didn't exist.

        Returns
        -------
        xS_unlensed : numpy array, dtype=float, shape = len(t_obs) x 2
            The unlensed positions of the source in arcseconds.
        """
        # Equation of motion for just the background source.
        dt_in_years = (t_obs - self.t0) / days_per_year
        xS_unlensed = self.xS0 + np.outer(dt_in_years, self.muS) * 1e-3

        if self.parallaxFlag:
            # Get the parallax vector for each date.
            parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
            xS_unlensed += (self.piS * parallax_vec) * 1e-3  # arcsec

        return xS_unlensed

    def get_lens_astrometry(self, t_obs):
        """Equation of motion for just the foreground lens system.

        Parameters
        ----------
        t_obs : array_like
            Time (in MJD).
        Return
        ------
        xL : array_like, shape = [N_times, 2 directions]
            Position of the lens system (geometric center) over time.
        """
        dt_in_years = (t_obs - self.t0) / days_per_year
        xL = self.xL0 + np.outer(dt_in_years, self.muL) * 1e-3

        if self.parallaxFlag:
            # Get the parallax vector for each date.
            parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
            xL += (self.piL * parallax_vec) * 1e-3  # arcsec

        return xL

    def get_resolved_lens_astrometry(self, t_obs):
        """Equation of motion for just the foreground lenses, individually.

        Parameters
        ----------
        t_obs : array_like
            Time (in MJD).

        Return
        ------
        xL1 : array_like, shape = [N_times, 2 directions]
            Position of the lens primary
        xL2 : array_like, shape = [N_times, 2 directions]
            Position of the lens secondary
        """
        
        xL = self.get_lens_astrometry(t_obs)

        offset = 0.5 * self.sep * np.array([np.sin(self.alpha_rad),
                                            np.cos(self.alpha_rad)])
        offset *= 1e-3  # convert to arcsec

        xL1 = xL + offset  # primary
        xL2 = xL - offset  # secondary
        
        if self.orbitFlag == 'linear':
            xL0_secondary = offset+self.xL0
            dt_in_years = (t_obs - self.t0) / days_per_year
            xL1 = self.get_lens_astrometry(t_obs)    
            xL2 = xL0_secondary + np.outer(dt_in_years, self.muL_sec) * 1e-3
                
        elif self.orbitFlag == 'acc':
            xL0_secondary = offset+self.xL0
            dt_in_years = (t_obs - self.t0) / days_per_year
            xL1 = self.get_lens_astrometry(t_obs)    
            xL2 = xL0_secondary + np.outer(dt_in_years, self.muL_sec) * 1e-3 + np.outer((0.5*(dt_in_years**2)), self.acc) * 1e-3

        elif self.orbitFlag == 'circular':
                xL1 = np.zeros((len(t_obs), 2), dtype=float)
                xL2 = np.zeros((len(t_obs), 2), dtype=float)
                orb = orbits.Orbit()
                orb.w = self.omega
                orb.o = self.big_omega
                orb.i = self.i
                orb.e = self.e
                orb.p = self.p
                orb.tp = self.tp
                orb.aleph = self.aleph *1e-3
                orb.aleph2 = self.aleph_sec*1e-3
                orb.vx = self.vx
                orb.vy = self.vy
                orb.x0 = self.xL0_E
                orb.y0 = self.xL0_N
                (x, y, x2, y2) = orb.oal2xy(t_obs, self.t0)
                xL1[:, 0] = x
                xL1[:, 1] = y
                xL2[:, 0] = x2
                xL2[:, 1] = y2
                if self.parallaxFlag:
                # Get the parallax vector for each date.
                    parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
                    xL1 += (self.piL * parallax_vec) * 1e-3  # arcsec
        if self.parallaxFlag:
                # Get the parallax vector for each date.
                parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
                xL2 += (self.piL * parallax_vec) * 1e-3  # arcsec
            
        return (xL1, xL2)

    def get_centroid(self, t_obs, ast_filt_idx=0, image_arr=None, amp_arr=None):
        """PSPL: Get the centroid shift (in arcsecs) for a list of
        observation times (in MJD).

        Parameters
        ----------
        t_obs : array or float

        Other Parameters
        ----------------
        ast_filt_idx : int
            Index into the photometry parameter lists for the photometry that
            corresponds to this astrometry data set.
        image_arr : list
            List returned from PSPL get_all_arrays() used to improve efficiency.
        amp_arr : list
            List returned from PSPL get_all_arrays() used to improve efficiency.

        Returns
        -------
        Centroid offset on the plane of the sky in arcseoncds.
        """
        # Observed position in arcsec
        xS_lensed = self.get_astrometry(t_obs,
                                        image_arr=image_arr,
                                        amp_arr=amp_arr,
                                        ast_filt_idx=ast_filt_idx)

        # Unlensed position in arcsec
        xS_unlens = self.get_astrometry_unlensed(t_obs)

        # Centroid offset in arcseconds.
        shift = (xS_lensed - xS_unlens) 

        return shift


    def dexanimate(self, tE, time_steps, frame_time, name, size, zoom, astrometry, loc):
        """ Produces animation of microlensing event. 
        This function takes the PSPL and makes an animation, the input variables are as follows

        Parameters
        ----------

        tE: 
            number of einstein crossings times before/after the peak you want the animation to plot
                e.g tE = 2 => graph will go from -2 tE to 2 tE
        time_steps:
            number of time steps before/after peak, so total number of time steps will 
            be 2 times this value
        frame_time:
            times in ms of each frame in the animation
        name: string
            the animation will be saved as name.html
        size: list
            [horizontal, vertical] cm's
        zoom:
            # of einstein radii plotted in vertical direction
        """
        times = np.array(range(-time_steps, time_steps + 1, 1))
        tau = tE * times / (-times[0])
        t = self.t0 + (tau * self.tE)

        img, amp = self.get_all_arrays(t)

        xL1, xL2 = self.get_resolved_lens_astrometry(t)
        source  = self.get_astrometry_unlensed(t) 
        image  = self.get_astrometry(t, image_arr = img, amp_arr = amp) 
        image_all = self.get_resolved_astrometry(t, image_arr=img, amp_arr=amp)
        #photometry = psbl.get_photometry
    

        fig = plt.figure(figsize=[size[0], size[1] + 0.5])  # sets up the figure
        ax1 = fig.add_subplot(2, 1, 1)
        fig.subplots_adjust(hspace=.5)
        l1_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, label="Primary Lens ", color='purple', linewidth=2)
        l1_line2, = ax1.plot([], '-', markersize=size[0] * 0.3, color='purple', linewidth=2)
        l2_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, label="Secondary Lens ", color='grey', linewidth=2)
        l2_line2, = ax1.plot([], '-', markersize=size[0] * 0.3, color='grey', linewidth=2)
        s1_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, label="Unlensed Source", color='yellow', linewidth=2)
        s1_line2, = ax1.plot([], '-', markersize=size[0] * 0.3, color='yellow', linewidth=2)
        

        if astrometry == "unresolved":
            i1_line1, = ax1.plot([], '.', markersize=size[0] * 1.0, label="Lensed Source (Unresolved)", color='yellow', linewidth=2)
            i1_line2, = ax1.plot([], '.', markersize=size[0] * 0.2, color='yellow', linewidth=2)
            ax1.set_xlabel('RA (")')
            ax1.set_ylabel('Dec (")')
            ax1.set_xlim(zoom, -zoom)
            ax1.set_ylim(-zoom, zoom)
            ax1.legend(fontsize=12, loc=loc)

            line = [l1_line1, l1_line2, l2_line1, l2_line2, s1_line1, s1_line2, i1_line1, i1_line2]
            def update(i, lens1, lens2, source, image, tau, line):
                    # print(str(i) + ", ", end='', flush=True)
                    line[0].set_data(lens1[i, 0], lens1[i, 1])
                    line[1].set_data(lens1[:i + 1, 0], lens1[:i + 1, 1])
                    line[2].set_data(lens2[i, 0], lens2[i, 1])
                    line[3].set_data(lens2[:i + 1, 0], lens2[:i + 1, 1])
                    line[4].set_data(source[i, 0], source[i, 1])
                    line[5].set_data(source[:i + 1, 0], source[:i + 1, 1])
                    line[6].set_data(image[i, 0], image[i, 1])
                    line[7].set_data(image[:i + 1, 0], image[:i + 1, 1])
                    return line
            ani = animation.FuncAnimation(fig, update, len(tau), fargs=[xL1, xL2,source, image, tau, line], blit=True, interval=frame_time)
            ani.save("%s.mp4" % name, writer="ffmpeg")   
        else:
            i2_line1, = ax1.plot([], '.', markersize=size[0] * 1.0, label="Lensed Image", color='yellow', linewidth=2)
            i2_line2, = ax1.plot([], '.', markersize=size[0] * 0.2, color='yellow', linewidth=2)
            i3_line1, = ax1.plot([], '.', markersize=size[0] * 1.0, color='yellow', linewidth=2)
            i3_line2, = ax1.plot([], '.', markersize=size[0] * 0.2, color='yellow', linewidth=2)
            i4_line1, = ax1.plot([], '.', markersize=size[0] * 1.0, color='yellow', linewidth=2)
            i4_line2, = ax1.plot([], '.', markersize=size[0] * 0.2, color='yellow', linewidth=2)
            i5_line1, = ax1.plot([], '.', markersize=size[0] * 1.0, color='yellow', linewidth=2)
            i5_line2, = ax1.plot([], '.', markersize=size[0] * 0.2, color='yellow', linewidth=2)
            i6_line1, = ax1.plot([], '.', markersize=size[0] * 1.0, color='yellow', linewidth=2)
            i6_line2, = ax1.plot([], '.', markersize=size[0] * 0.2, color='yellow', linewidth=2)
            
            ax1.set_xlabel('RA (")')
            ax1.set_ylabel('Dec (")')
            ax1.set_xlim(zoom, -zoom)
            ax1.set_ylim(-zoom, zoom)
            ax1.legend(fontsize=12, loc=loc)

            line = [l1_line1, l1_line2, l2_line1, l2_line2, s1_line1, s1_line2, i2_line1, i2_line2, i3_line1, i3_line2, i4_line1, i4_line2, i5_line1, i5_line2, i6_line1, i6_line2]
            def update(i, lens1, lens2, source, image1, image2, image3, image4, image5, tau, line):
                    line[0].set_data(lens1[i, 0], lens1[i, 1])
                    line[1].set_data(lens1[:i + 1, 0], lens1[:i + 1, 1])
                    line[2].set_data(lens2[i, 0], lens2[i, 1])
                    line[3].set_data(lens2[:i + 1, 0], lens2[:i + 1, 1])
                    line[4].set_data(source[i, 0], source[i, 1])
                    line[5].set_data(source[:i + 1, 0], source[:i + 1, 1])
                    line[6].set_data(image1[i, 0], image1[i, 1])
                    line[7].set_data(image1[:i + 1, 0], image1[:i + 1, 1])
                    line[8].set_data(image2[i, 0], image2[i, 1])
                    line[9].set_data(image2[:i + 1, 0], image2[:i + 1, 1])
                    line[10].set_data(image3[i, 0], image3[i, 1])
                    line[11].set_data(image3[:i + 1, 0], image3[:i + 1, 1])
                    line[12].set_data(image4[i, 0], image4[i, 1])
                    line[13].set_data(image4[:i + 1, 0], image4[:i + 1, 1])
                    line[14].set_data(image5[i, 0], image5[i, 1])
                    line[15].set_data(image5[:i + 1, 0], image5[:i + 1, 1])

                    return line
            ani = animation.FuncAnimation(fig, update, len(tau), fargs=[xL1, xL2,source, image_all[:,0], image_all[:,1], image_all[:,2], image_all[:,3], image_all[:,4], tau, line], blit=True, interval=frame_time)
            ani.save("%s.mp4" % name, writer="ffmpeg")   
    
        return ani
# --------------------------------------------------
#
# Parallax Class Family - PSBL
#
# --------------------------------------------------
class PSBL_Parallax(PSPL_Parallax):
    parallaxFlag = True
    def get_amplification(self, t_obs, amp_arr=None):
        """noParallax: Get the photometric amplification term at a set of times, t.
        
        Parameters
        ----------
        t: 
            Array of times in MJD.DDD
        """
        if amp_arr is None:
            img_arr, amp_arr = self.get_all_arrays(t_obs)

        # Mask invalid values from the amplification array.
        amp_arr_msk = np.ma.masked_invalid(amp_arr)

        # Sum up all the amplifications b/c surface brightness is conserved.
        amp = np.sum(amp_arr_msk, axis=1)

        return amp

class PSBL_noParallax(PSPL_noParallax):
    parallaxFlag = False
    def get_amplification(self, t_obs, amp_arr=None):
        """noParallax: Get the photometric amplification term at a set of times, t.
        
        Parameters
        ----------
        t: 
            Array of times in MJD.DDD
        """
        if amp_arr is None:
            img_arr, amp_arr = self.get_all_arrays(t_obs)

        # Mask invalid values from the amplification array.
        amp_arr_msk = np.ma.masked_invalid(amp_arr)

        # Sum up all the amplifications b/c surface brightness is conserved.
        amp = np.sum(amp_arr_msk, axis=1)

        return amp

# --------------------------------------------------
#
# Parameterization Class Family - PSBL
#
# --------------------------------------------------

class PSBL_PhotAstrom_EllOrbs_Param1(PSPL_Param):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    mLp, mLs : float
        Masses of the lenses (Msun)
    t0_com : float
        Time of closest approach between the source and lens system's COM, as seen from Earth (MJD.DDD)
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    beta: float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    muL_E : float
        Lens System proper motion in the RA direction (mas/yr)
    muL_N : float
        Lens System proper motion in the Dec. direction (mas/yr)
    omega: float
        The argument of periastron of the primary lens's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary lens's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    e: float
        Eccentricity of the System
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary lens in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary lens in mas. 
    muS_E : float
        Source proper motion in the RA direction (mas/yr)
    muS_N : float
        Source proper motion in the Dec. direction (mas/yr)
    dL : float
        Distance from the observer to the lens system (pc)
    dS : float
        Distance from the observer to the source (pc)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['mLp', 'mLs', 't0', 'xS0_E', 'xS0_N',
                          'beta', 'muL_E', 'muL_N', 'omega', 'big_omega', 'i', 'e', 'p', 'tp', 'aleph', 'aleph_sec', 'muS_E', 'muS_N', 'dL', 'dS', 'alpha']
    
    phot_param_names = ['b_sff', 'mag_src']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'

    def __init__(self, mLp, mLs, t0_com, xS0_E, xS0_N,
                 beta_com, muL_E, muL_N, omega, big_omega, i, e, p, tp, aleph, aleph_sec, muS_E, muS_N, dL, dS,
                 alpha, b_sff, mag_src,
                 raL=None, decL=None, root_tol=1e-8):
        self.mLp = mLp  # Msun
        self.mLs = mLs  # Msun
        self.t0 = t0_com
        self.t0_com = t0_com
        self.xS0 = np.array([xS0_E, xS0_N])
        self.beta_com = beta_com
        self.muL = np.array([muL_E, muL_N])
        self.omega = omega
        self.big_omega = big_omega
        self.i = i
        self.e = e
        self.p = p
        self.tp = tp
        self.aleph = aleph
        self.aleph_sec = aleph_sec
        self.muS = np.array([muS_E, muS_N])
        self.dL = dL
        self.dS = dS
        self.sep = self.aleph + self.aleph_sec #Initial separation. Changes as the sources move.
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.b_sff = b_sff
        self.mag_src = mag_src
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol
        self.vx = muL_E * 1e-3
        self.vy = muL_N *1e-3

                     
        # Super handles checking for properly formatted variables.
        super().__init__()

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (dL * units.pc)) - (1.0 / (dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr

        # Calculate the Einstein radius
        # AFAICT, thetaE for binary lenses is calculated from the total lens mass.
        # Checked using Shin+17 (OB160168) and Jung+19 (OB160156)
        self.mL = self.mLp + self.mLs  # Total lens mass
        thetaE = units.rad * np.sqrt((4.0 * const.G * self.mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Calculate the microlensing parallax
        self.piE = (self.piRel / self.thetaE_amp) * self.thetaE_hat

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta_com)
        self.u0_amp = self.beta_com / self.thetaE_amp  # in Einstein units
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)
        self.xL0_E = self.xL0[0]
        self.xL0_N = self.xL0[1]

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat
        self.piE_E, self.piE_N = self.piE

        
        # Calculate the Einstein crossing time. (days)
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year


        
        # Calculate t0 and beta for geometric center of lens
        self.phi_rad = self.alpha_rad - np.arctan2(self.piE_E, self.piE_N)
        self.q = self.mLs/self.mLp
        u0_x_out, u0_y_out, t0_out = fc.convert_u0_t0_psbl(t0_in = self.t0_com, u0_x_in = self.u0[0], u0_y_in = self.u0[1],
                           tE = self.tE, theta_E = self.thetaE_amp, q = self.q, phi = self.phi_rad, sep = self.sep, mu_rel_x = self.muRel[0],
                           mu_rel_y = self.muRel[1], coords_in='COM', coords_out='geom_mid')
        self.u0 = np.array([u0_x_out, u0_y_out])
        self.u0_amp = np.sqrt(self.u0[0]**2 + self.u0[1]**2)
        self.t0 = t0_out
        self.beta = self.u0_amp*self.thetaE_amp

        return

                     
class PSBL_PhotAstrom_CircOrbs_Param1(PSBL_PhotAstrom_EllOrbs_Param1):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    mLp, mLs : float
        Masses of the lenses (Msun)
    t0_com : float
        Time of closest approach between the source and lens system's COM, as seen from Earth (MJD.DDD)
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    beta_com : float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    muL_E : float
        Lens System proper motion in the RA direction (mas/yr)
    muL_N : float
        Lens System proper motion in the Dec. direction (mas/yr)
    omega: float
        The argument of periastron of the primary lens's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary lens's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary lens in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary lens in mas. 
    muS_E : float
        Source proper motion in the RA direction (mas/yr)
    muS_N : float
        Source proper motion in the Dec. direction (mas/yr)
    dL : float
        Distance from the observer to the lens system (pc)
    dS : float
        Distance from the observer to the source (pc)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['mLp', 'mLs', 't0', 'xS0_E', 'xS0_N',
                          'beta', 'muL_E', 'muL_N', 'omega', 'big_omega', 'i', 'p', 'tp', 'aleph',
                          'aleph_sec', 'muS_E', 'muS_N',
                          'dL', 'dS', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'

    def __init__(self, mLp, mLs, t0_com, xS0_E, xS0_N,
                 beta_com, muL_E, muL_N, omega, big_omega, i, p, tp, aleph, aleph_sec, muS_E, muS_N, dL, dS,
                 alpha, b_sff, mag_src,
                 raL=None, decL=None, root_tol=1e-8):
        super().__init__(mLp, mLs, t0_com, xS0_E, xS0_N,
                 beta_com, muL_E, muL_N, omega, big_omega, i, 0, p, tp, aleph, aleph_sec, muS_E, muS_N, dL, dS,
                 alpha, b_sff, mag_src,
                 raL=raL, decL=decL, root_tol=root_tol)
        #self.sep will remain same for circular orbits
                     
        return
class PSBL_PhotAstromParam1(PSPL_Param):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    mLp, mLs : float
        Masses of the lenses (Msun)
    t0 : float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    beta : float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    muL_E : float
        Lens system proper motion in the RA direction (mas/yr)
    muL_N : float
        Lens system proper motion in the Dec. direction (mas/yr)
    muS_E : float
        Source proper motion in the RA direction (mas/yr)
    muS_N : float
        Source proper motion in the Dec. direction (mas/yr)
    dL : float
        Distance from the observer to the lens system (pc)
    dS : float
        Distance from the observer to the source (pc)
    sep : float
        Angular separation of the two lenses (mas)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['mLp', 'mLs', 't0', 'xS0_E', 'xS0_N',
                          'beta', 'muL_E', 'muL_N', 'muS_E', 'muS_N',
                          'dL', 'dS', 'sep', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag=False

    def __init__(self, mLp, mLs, t0, xS0_E, xS0_N,
                 beta, muL_E, muL_N, muS_E, muS_N, dL, dS,
                 sep, alpha, b_sff, mag_src,
                 raL=None, decL=None, root_tol=1e-8):
        self.mLp = mLp  # Msun
        self.mLs = mLs  # Msun
        self.t0 = t0
        self.xS0 = np.array([xS0_E, xS0_N])
        self.beta = beta
        self.muL = np.array([muL_E, muL_N])
        self.muS = np.array([muS_E, muS_N])
        self.dL = dL
        self.dS = dS
        self.sep = sep
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.b_sff = b_sff
        self.mag_src = mag_src
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol

        # Super handles checking for properly formatted variables.
        super().__init__()

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (dL * units.pc)) - (1.0 / (dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr

        # Calculate the Einstein radius
        # AFAICT, thetaE for binary lenses is calculated from the total lens mass.
        # Checked using Shin+17 (OB160168) and Jung+19 (OB160156)
        self.mL = self.mLp + self.mLs  # Total lens mass
        thetaE = units.rad * np.sqrt((4.0 * const.G * self.mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Calculate the microlensing parallax
        self.piE = (self.piRel / self.thetaE_amp) * self.thetaE_hat

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0_amp = self.beta / self.thetaE_amp  # in Einstein units
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat

        # Calculate the Einstein crossing time. (days)
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year

        return


class PSBL_PhotAstromParam2(PSPL_Param):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    t0 : float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can be 
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    q : float
        Mass ratio (M2 / M1)
    sep : float
        Angular separation of the two lenses (mas)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N', 'xS0_E', 'xS0_N', 'muS_E', 'muS_N',
                          'q', 'sep', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag=False

    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N, xS0_E, xS0_N, muS_E, muS_N,
                 q, sep, alpha,
                 b_sff, mag_src,
                 raL=None, decL=None, root_tol=1e-8):
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.thetaE_amp = thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.q = q
        self.sep = sep
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.b_sff = b_sff
        self.mag_src = mag_src
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol

        # Check variable formatting.
        super().__init__()

        # Derived quantities
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)
        self.mLp = self.mL / (1.0 + self.q)
        self.mLs = self.mLp * self.q

        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        inv_dist_diff = (1.0 / dL) - (1.0 / dS)
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        return


class PSBL_PhotAstromParam3(PSPL_Param):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    t0 : float
        Time of photometric peak, as seen from Earth (MJD.DDD)
        FIXME: THIS IS NOT RIGHT
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE : float
        Einstein crossing time (days).
    log10_thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    q : float
        Mass ratio (M2 / M1)
    sep : float
        Angular separation of the two lenses (mas)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
           :math:`b_sff = f_S / (f_S + f_L + f_N)`. 
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_base : numpy array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'log10_thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'xS0_E', 'xS0_N',
                          'muS_E', 'muS_N',
                          'q', 'sep', 'alpha']
    phot_param_names = ['b_sff', 'mag_base']
    additional_param_names = ['thetaE_amp', 'mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N',
                              'mag_src']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = False

    def __init__(self, t0, u0_amp, tE, log10_thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, muS_E, muS_N,
                     q, sep, alpha,
                     b_sff, mag_base,
                     raL=None, decL=None, root_tol=1e-8):
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.log10_thetaE = log10_thetaE
        self.thetaE = 10 ** log10_thetaE
        self.thetaE_amp = 10 ** log10_thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.q = q
        self.sep = sep
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.b_sff = b_sff
        self.mag_base = mag_base
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol

        # Check variable formatting.
        super().__init__()

        # Derived quantities
        self.mag_src = self.mag_base - 2.5 * np.log10(self.b_sff)
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)
        self.mLp = self.mL / (1.0 + self.q)
        self.mLs = self.mLp * self.q
        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        inv_dist_diff = (1.0 / dL) - (1.0 / dS)
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        return

class BSPL_PhotAstrom_Ell_Param1(PSPL_Param):


    
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    mL: float
        Mass of the lens (Msun)
    t0_com: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    beta: float
        Angular distance between the lens and primary source on the
        plane of the sky (mas). Can be
            * positive (u0_amp > 0 when u0_hat[0] < 0) or
            * negative (u0_amp < 0 when u0_hat[0] > 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    omega: float
        The argument of periastron of the primary star's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary star's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    e: float
        Eccentricity of the System
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary source in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary source in mas. 
    muS_system_E: float
        The RA proper motion of the system in mas/yr.
    muS_system_N: float
        The Dec proper motion of the system in mas/yr.
    x0_system_E: float
        The initial center of mass coordinates (RA) of the system in arcsec.
    x0_system_N: float
        The initial center of mass coordinates (Dec) of the system in arcsec.
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_S1 + f_S2) / (f_S1 + f_s2 + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['mL', 't0_com', 'beta', 'dL', 'dL_dS',
                          'x0_system_E', 'x0_system_N',
                          'muL_E', 'muL_N',
                          'muS_system_E', 'muS_system_N',
                          'alpha',
                         'omega', 'big_omega', 'i', 'e',
                          'p', 'tp', 'aleph', 'aleph_sec'
                         ]
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']
    additional_param_names = ['dS', 'tE', 'u0_amp',
                              'thetaE_E', 'thetaE_N',
                              'piE_E', 'piE_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'



    def __init__(self, mL, t0_com, beta, dL, dL_dS,
                 x0_system_E, x0_system_N,
                 muL_E, muL_N,
                 muS_system_E, muS_system_N,
                 alpha, omega, big_omega, i,
                 e, p, tp, aleph, aleph_sec,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=None, decL=None):
        self.t0 = t0_com  # time of closest approach for system=primary pos
        self.mL = mL
        self.xS0 = np.array([x0_system_E, x0_system_N])  # position of source system=primary
        self.beta = beta
        self.muL = np.array([muL_E, muL_N])
        self.muS = np.array([muS_system_E, muS_system_N]) #mas
        self.x0_system = self.xS0
        self.muS_system = self.muS
        self.dL = dL
        self.dL_dS = dL_dS
        self.dS = self.dL / self.dL_dS
        self.b_sff = np.array(b_sff)
        self.raL = raL
        self.decL = decL
        self.w = omega
        self.o = big_omega
        self.omega = self.w
        self.big_omega = self.o
        self.i = i
        self.e = e
        self.p = p
        self.tp = tp
        self.aleph = aleph #mas
        self.aleph_sec = aleph_sec #mas
        self.vx = muS_system_E * 1e-3
        self.vy = muS_system_N * 1e-3
        self.x0 = x0_system_E
        self.y0 = x0_system_N
                     
        self.sep = aleph + aleph_sec

        # Binary source parameters.
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.mag_src_pri = np.array(mag_src_pri)
        self.mag_src_sec = np.array(mag_src_sec)
        self.fratio_bin = 10.0 ** ((self.mag_src_sec - self.mag_src_pri) / -2.5)

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()
                     
        flux_pri = mag2flux(self.mag_src_pri)
        flux_sec = mag2flux(self.mag_src_sec)
        self.mag_base = flux2mag(flux_pri + flux_sec) + 2.5 * np.log10(self.b_sff)

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (self.dL * units.pc)) - (1.0 / (self.dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_E, self.muRel_N = self.muRel
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr

        self.muS_E, self.muS_N = self.muS
        self.muL_E, self.muL_N = self.muL

        # Calculate the Einstein radius
        thetaE = units.rad * np.sqrt(
            (4.0 * const.G * mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.sun = units.M_sun
        self.c = const.c             
        self.G=const.G
        self.inv = inv_dist_diff           #remove later
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat
        self.thetaE_E, self.thetaE_N = self.thetaE

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0_amp = self.beta / self.thetaE_amp  # in Einstein units
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens
        # (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat
        self.piE_E, self.piE_N = self.piE

        # Calculate the Einstein crossing time. (days)
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year

        return

    




class BSPL_PhotAstrom_Ell_Param2(PSPL_Param):
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0_com: float
        Time of photometric peak, as seen from Earth (MJD.DDD) in CoM coordinates. 
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    omega: float
        The argument of periastron of the primary star's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary star's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    e: float
        Eccentricity of the System
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary source in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary source in mas. 
    muS_system_E: float
        The RA proper motion of the system in mas/yr.
    muS_system_N: float
        The Dec proper motion of the system in mas/yr.
    x0_system_E: float
        The initial center of mass coordinates (RA) of the system in arcsec.
    x0_system_N: float
        The initial center of mass coordinates (Dec) of the system in arcsec.
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_{base} = f_{src1} + f_{src2} + f_{blend}`
        such that
            :math:`b_sff = (f_{src1}+ f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['t0_com', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'alpha', 'omega', 'big_omega', 'i', 'e',
                          'p', 'tp', 'aleph', 'aleph_sec', 'muS_system_E', 'muS_system_N',
                          'x0_system_E', 'x0_system_N']
    
    phot_param_names = ['fratio_bin', 'mag_base', 'b_sff']
    
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'


    def __init__(self, t0_com, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 alpha, omega,
                 big_omega, i,
                 e, p, tp,
                 aleph, aleph_sec,
                 muS_system_E, muS_system_N,
                 x0_system_E, x0_system_N,
                 fratio_bin,
                 mag_base, b_sff,
                 raL=None, decL=None):
                     
        self.t0 = t0_com #Makes get_lens_astrometry easier. t0 is t0_com     
        self.u0_amp = u0_amp
        self.tE = tE 
        self.thetaE_amp = thetaE        
        self.piS = piS
        self.piE = np.array([piE_E, piE_N])
        self.xS0 = np.array([x0_system_E, x0_system_N])
        self.muS_E = muS_system_E 
        self.muS_N = muS_system_N 
        self.muS = np.array([muS_system_E, muS_system_N]) #mas
        self.x0_system = self.xS0
        self.muS_system = self.muS
        self.mag_base = np.array(mag_base)
        self.b_sff = np.array(b_sff)      
        self.raL = raL
        self.decL = decL
        self.w = omega
        self.o = big_omega
        self.omega = self.w
        self.big_omega = self.o
        self.i = i
        self.e = e
        self.p = p
        self.tp = tp
        self.aleph = aleph  #mas
        self.aleph_sec = aleph_sec #mas
        self.vx = muS_system_E * 1e-3
        self.vy = muS_system_N * 1e-3
        self.x0 = x0_system_E
        self.y0 = x0_system_N
        self.alpha = alpha
        self.fratio_bin = np.array(fratio_bin)
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.mag_src_pri = mag_base - 2.5 * np.log10(b_sff) + 2.5 * np.log10(1.0 + fratio_bin)
        self.mag_src_sec = mag_base - 2.5 * np.log10(b_sff) + 2.5 * np.log10(1.0 + (1.0 / fratio_bin))

        
        self.sep = aleph + aleph_sec
                     
        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

                     
        # Derived quantities 
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)

        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)
        return 




class BSPL_PhotAstrom_Ell_Param3(PSPL_Param):
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE : float
        Einstein crossing time (days).
    log10_thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    omega: float
        The argument of periastron of the primary star's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary star's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    e: float
        Eccentricity of the System
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary source in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary source in mas. 
    muS_system_E: float
        The RA proper motion of the system in mas/yr.
    muS_system_N: float
        The Dec proper motion of the system in mas/yr.
    x0_system_E: float
        The initial center of mass coordinates (RA) of the system in arcsec.
    x0_system_N: float
        The initial center of mass coordinates (Dec) of the system in arcsec.
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_base = f_{src1{ + f_{src2{ + f_{blend}`
        such that
            :math:`b_sff = (f_{src1} + f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['t0_com', 'u0_amp', 'tE', 'log10_thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'alpha', 'omega', 'big_omega', 'i', 'e', 'p', 
                          'tp', 'aleph', 'aleph_sec',
                          'muS_system_E', 'muS_system_N',
                          'x0_system_E', 'x0_system_N',
                          'muS_E', 'muS_N',
                          ]
    phot_param_names = ['fratio_bin', 'mag_base', 'b_sff']
    additional_param_names = ['thetaE_amp', 'mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'

    

    def __init__(self, t0_com, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 alpha, omega,
                 big_omega, i,
                 e, p, tp,
                 aleph, aleph_sec,
                 muS_system_E, muS_system_N,
                 x0_system_E, x0_system_N,
                 fratio_bin,
                 mag_base, b_sff,
                 raL=None, decL=None):
                     
        self.t0 = t0_com  # time of closest approach for system=primary pos
        self.u0_amp = u0_amp
        self.tE = tE
        self.thetaE_amp = 10 ** log10_thetaE
        self.piS = piS
        self.piE = np.array([piE_E, piE_N])
        self.xS0 = np.array([x0_system_E, x0_system_N])
        self.muS_E = muS_system_E 
        self.muS_N = muS_system_N 
        self.muS = np.array([muS_system_E, muS_system_N]) #mas
        self.x0_system = self.xS0
        self.muS_system = self.muS
        self.mag_base = np.array(mag_base)
        self.b_sff = np.array(b_sff)      
        self.raL = raL
        self.decL = decL
        self.w = omega
        self.o = big_omega
        self.omega = self.w
        self.big_omega = self.o
        self.i = i
        self.e = e
        self.p = p
        self.tp = tp
        self.aleph = aleph #mas
        self.aleph_sec = aleph_sec  #mas
        self.vx = muS_system_E * 1e-3
        self.vy = muS_system_N * 1e-3
        self.x0 = x0_system_E
        self.y0 = x0_system_N
        self.alpha = alpha
        self.fratio_bin = np.array(fratio_bin)
        self.alpha_rad = self.alpha * np.pi / 180.0

        self.mag_src_pri = self.mag_base \
                           - 2.5 * np.log10(self.b_sff) \
                           + 2.5 * np.log10(1.0 + self.fratio_bin)
        self.mag_src_sec = self.mag_base \
                           - 2.5 * np.log10(self.b_sff) \
                           + 2.5 * np.log10(1.0 + (1.0 / self.fratio_bin))

        self.sep = aleph + aleph_sec
        # Must call after setting parameters.
        # This checks for proper parameter formatting.

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

                     
        # Derived quantities
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)

        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens
        # (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

class BSPL_PhotAstrom_Ell_Param4(PSPL_Param):


    
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    mL: float
        Mass of the lens (Msun)
    t0_com: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    beta: float
        Angular distance between the lens and primary source on the
        plane of the sky (mas). Can be
            * positive (u0_amp > 0 when u0_hat[0] < 0) or
            * negative (u0_amp < 0 when u0_hat[0] > 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    omega: float
        The argument of periastron of the primary star's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary star's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary source in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary source in mas. 
    muS_system_E: float
        The RA proper motion of the system in mas/yr.
    muS_system_N: float
        The Dec proper motion of the system in mas/yr.
    x0_system_E: float
        The initial center of mass coordinates (RA) of the system in arcsec.
    x0_system_N: float
        The initial center of mass coordinates (Dec) of the system in arcsec.
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_S1 + f_S2) / (f_S1 + f_s2 + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['mL', 't0_com', 'beta', 'dL', 'dL_dS',
                          'x0_system_E', 'x0_system_N',
                          'muL_E', 'muL_N',
                          'muS_system_E', 'muS_system_N',
                          'alpha',
                         'omega', 'big_omega', 'i', 'e',
                          'p', 'tp', 'aleph', 'aleph_sec'
                         ]
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']
    additional_param_names = ['dS', 'tE', 'u0_amp',
                              'thetaE_E', 'thetaE_N',
                              'piE_E', 'piE_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'



    def __init__(self, mL, t0_com, beta, dL, dL_dS,
                 x0_system_E, x0_system_N,
                 muL_E, muL_N,
                 muS_system_E, muS_system_N,
                 alpha, omega, big_omega, i,
                 e, p, tp, aleph, aleph_sec,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=None, decL=None):
        self.t0 = t0_com  # time of closest approach for system=primary pos
        self.mL = mL
        self.xS0 = np.array([x0_system_E, x0_system_N])  # position of source system=primary
        self.beta = beta
        self.muL = np.array([muL_E, muL_N])
        self.muS = np.array([muS_system_E, muS_system_N]) #mas
        self.x0_system = self.xS0
        self.muS_system = self.muS
        self.dL = dL
        self.dL_dS = dL_dS
        self.dS = self.dL / self.dL_dS
        self.b_sff = np.array(b_sff)
        self.raL = raL
        self.decL = decL
        self.w = omega
        self.o = big_omega
        self.omega = self.w
        self.big_omega = self.o
        self.i = i
        self.e = e
        self.p = p
        self.tp = tp
        self.aleph = aleph/self.dS * 1e-3 #mas
        self.aleph_sec = aleph_sec/self.dS * 1e-3 #mas
        self.vx = muS_system_E * 1e-3
        self.vy = muS_system_N * 1e-3
        self.x0 = x0_system_E
        self.y0 = x0_system_N

        # Binary source parameters.
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.mag_src_pri = np.array(mag_src_pri)
        self.mag_src_sec = np.array(mag_src_sec)
        self.fratio_bin = 10.0 ** ((self.mag_src_sec - self.mag_src_pri) / -2.5)

        self.sep = aleph + aleph_sec
        flux_pri = mag2flux(self.mag_src_pri)
        flux_sec = mag2flux(self.mag_src_sec)
        self.mag_base = flux2mag(flux_pri + flux_sec) + 2.5 * np.log10(self.b_sff)

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (self.dL * units.pc)) - (1.0 / (self.dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_E, self.muRel_N = self.muRel
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr

        self.muS_E, self.muS_N = self.muS
        self.muL_E, self.muL_N = self.muL

        # Calculate the Einstein radius
        thetaE = units.rad * np.sqrt(
            (4.0 * const.G * mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.sun = units.M_sun
        self.c = const.c             
        self.G=const.G
        self.inv = inv_dist_diff           #remove later
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat
        self.thetaE_E, self.thetaE_N = self.thetaE

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0_amp = self.beta / self.thetaE_amp  # in Einstein units
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens
        # (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat
        self.piE_E, self.piE_N = self.piE

        # Calculate the Einstein crossing time. (days)
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year

        return

    





#t0->t0_com
#AU params
class BSPL_PhotAstrom_Circ_Param1(PSPL_Param):


    
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    mL: float
        Mass of the lens (Msun)
    t0_com: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    beta: float
        Angular distance between the lens and primary source on the
        plane of the sky (mas). Can be
            * positive (u0_amp > 0 when u0_hat[0] < 0) or
            * negative (u0_amp < 0 when u0_hat[0] > 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    omega: float
        The argument of periastron of the primary star's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary star's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary source in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary source in mas. 
    muS_system_E: float
        The RA proper motion of the system in mas/yr.
    muS_system_N: float
        The Dec proper motion of the system in mas/yr.
    x0_system_E: float
        The initial center of mass coordinates (RA) of the system in arcsec.
    x0_system_N: float
        The initial center of mass coordinates (Dec) of the system in arcsec.
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_S1 + f_S2) / (f_S1 + f_s2 + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['mL', 't0_com', 'beta', 'dL', 'dL_dS',
                          'x0_system_E', 'x0_system_N',
                          'muL_E', 'muL_N',
                          'muS_system_E', 'muS_system_N',
                          'alpha',
                         'omega', 'big_omega', 'i',
                          'p', 'tp', 'aleph', 'aleph_sec'
                         ]
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']
    additional_param_names = ['dS', 'tE', 'u0_amp',
                              'thetaE_E', 'thetaE_N',
                              'piE_E', 'piE_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'



    def __init__(self, mL, t0_com, beta, dL, dL_dS,
                 x0_system_E, x0_system_N,
                 muL_E, muL_N,
                muS_system_E, muS_system_N,
                 alpha, omega, big_omega, i,
                 p, tp, aleph, aleph_sec,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=None, decL=None):
        self.t0 = t0_com  # time of closest approach for system=primary pos
        self.t0_com = t0_com  # time of closest approach for system=primary pos
        self.mL = mL
        self.xS0 = np.array([x0_system_E, x0_system_N])  # position of source system=primary
        self.beta = beta
        self.muL = np.array([muL_E, muL_N])
        self.muS = np.array([muS_system_E, muS_system_N]) #mas
        self.x0_system = self.xS0
        self.muS_system = self.muS
        self.dL = dL
        self.dL_dS = dL_dS
        self.dS = self.dL / self.dL_dS
        self.b_sff = np.array(b_sff)
        self.raL = raL
        self.decL = decL
        self.w = omega
        self.o = big_omega
        self.omega = self.w
        self.big_omega = self.o
        self.i = i
        self.e = 0
        self.p = p
        self.tp = tp
        self.aleph = aleph #mas
        self.aleph_sec = aleph_sec #mas
        self.vx = muS_system_E * 1e-3
        self.vy = muS_system_N * 1e-3
        self.x0 = x0_system_E
        self.y0 = x0_system_N

        # Binary source parameters.
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.mag_src_pri = np.array(mag_src_pri)
        self.mag_src_sec = np.array(mag_src_sec)
        self.fratio_bin = 10.0 ** ((self.mag_src_sec - self.mag_src_pri) / -2.5)

               
        self.sep = aleph + aleph_sec      
        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        flux_pri = mag2flux(self.mag_src_pri)
        flux_sec = mag2flux(self.mag_src_sec)
        self.mag_base = flux2mag(flux_pri + flux_sec) + 2.5 * np.log10(self.b_sff)

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (self.dL * units.pc)) - (1.0 / (self.dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        #System 
        self.muRel_E, self.muRel_N = self.muRel
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr

        self.muS_E, self.muS_N = self.muS
        self.muL_E, self.muL_N = self.muL

        # Calculate the Einstein radius
        thetaE = units.rad * np.sqrt(
            (4.0 * const.G * mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.sun = units.M_sun
        self.c = const.c             
        self.G=const.G
        self.inv = inv_dist_diff           #remove later
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat
        self.thetaE_E, self.thetaE_N = self.thetaE

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0_amp = self.beta / self.thetaE_amp  # in Einstein units
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens
        # (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat
        self.piE_E, self.piE_N = self.piE

        # Calculate the Einstein crossing time. (days)
        # System tE
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year

        return

    




class BSPL_PhotAstrom_Circ_Param2(PSPL_Param):
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0_com: float
        Time of photometric peak, as seen from Earth (MJD.DDD) in CoM coordinates. 
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE : float
        Einstein crossing time (days). System.
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    omega: float
        The argument of periastron of the primary star's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary star's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary source in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary source in mas. 
    muS_system_E: float
        The RA proper motion of the system in mas/yr.
    muS_system_N: float
        The Dec proper motion of the system in mas/yr.
    x0_system_E: float
        The initial center of mass coordinates (RA) of the system in arcsec.
    x0_system_N: float
        The initial center of mass coordinates (Dec) of the system in arcsec.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_{base} = f_{src1} + f_{src2} + f_{blend}`
        such that
            :math:`b_sff = (f_{src1}+ f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['t0_com', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'alpha', 'omega', 'big_omega', 'i',
                          'p', 'tp', 'aleph', 'aleph_sec', 'muS_system_E', 'muS_system_N',
                          'x0_system_E', 'x0_system_N']
    
    phot_param_names = ['fratio_bin', 'mag_base', 'b_sff']
    
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'


    def __init__(self, t0_com, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 alpha, omega,
                 big_omega, i,
                 p, tp,
                 aleph, aleph_sec,
                 muS_system_E, muS_system_N,
                 x0_system_E, x0_system_N,
                 fratio_bin,
                 mag_base, b_sff,
                 raL=None, decL=None):
        #Get rid of this

        self.t0 = t0_com #Makes get_lens_astrometry easier. t0 is t0_com    
        self.t0_com = t0_com
        self.u0_amp = u0_amp
        self.tE = tE 
        self.thetaE_amp = thetaE        
        self.piS = piS
        self.piE = np.array([piE_E, piE_N])
        self.xS0 = np.array([x0_system_E, x0_system_N])
        self.muS_E = muS_system_E 
        self.muS_N = muS_system_N 
        self.muS = np.array([muS_system_E, muS_system_N]) #mas
        self.x0_system = self.xS0
        self.muS_system = self.muS
        self.mag_base = np.array(mag_base)
        self.b_sff = np.array(b_sff)      
        self.raL = raL
        self.decL = decL
        self.w = omega
        self.o = big_omega
        self.omega = self.w
        self.big_omega = self.o
        self.i = i
        self.e = 0
        self.p = p
        self.tp = tp
        self.aleph = aleph #mas
        self.aleph_sec = aleph_sec #mas
        self.vx = muS_system_E * 1e-3
        self.vy = muS_system_N * 1e-3
        self.x0 = x0_system_E
        self.y0 = x0_system_N
        self.alpha = alpha
        self.fratio_bin =  np.array(fratio_bin)
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.mag_src_pri = mag_base - 2.5 * np.log10(self.b_sff) + 2.5 * np.log10(1.0 + self.fratio_bin)
        self.mag_src_sec = mag_base - 2.5 * np.log10(self.b_sff) + 2.5 * np.log10(1.0 + (1.0 / self.fratio_bin))

                  
        self.sep = aleph + aleph_sec   
        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

                     
        # Derived quantities 
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)

        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)
        return 




class BSPL_PhotAstrom_Circ_Param3(PSPL_Param):
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE : float
        Einstein crossing time (days).
    log10_thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    omega: float
        The argument of periastron of the primary star's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary star's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary source in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary source in mas. 
    muS_system_E: float
        The RA proper motion of the system in mas/yr.
    muS_system_N: float
        The Dec proper motion of the system in mas/yr.
    x0_system_E: float
        The initial center of mass coordinates (RA) of the system in arcsec.
    x0_system_N: float
        The initial center of mass coordinates (Dec) of the system in arcsec.
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_base = f_{src1{ + f_{src2{ + f_{blend}`
        such that
            :math:`b_sff = (f_{src1} + f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['t0_com', 'u0_amp', 'tE', 'log10_thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'alpha', 'omega', 'big_omega', 'i', 'p', 
                          'tp', 'aleph', 'aleph_sec',
                          'muS_system_E', 'muS_system_N',
                          'x0_system_E', 'x0_system_N'
                          ]
    phot_param_names = ['fratio_bin', 'mag_base', 'b_sff']
    additional_param_names = ['thetaE_amp', 'mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'

    

    def __init__(self, t0_com, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 alpha, omega,
                 big_omega, i,
                 p, tp,
                 aleph, aleph_sec,
                 muS_system_E, muS_system_N,
                 x0_system_E, x0_system_N,
                 fratio_bin,
                 mag_base, b_sff,
                 raL=None, decL=None):
                     
        self.t0 = t0_com  # time of closest approach for system=primary pos
        self.t0_com = t0_com
        self.u0_amp = u0_amp
        self.tE = tE
        self.thetaE_amp = 10 ** log10_thetaE
        self.piS = piS
        self.piE = np.array([piE_E, piE_N])
        self.xS0 = np.array([x0_system_E, x0_system_N])
        self.muS_E = muS_system_E 
        self.muS_N = muS_system_N 
        self.muS = np.array([muS_system_E, muS_system_N]) #mas
        self.mag_base = np.array(mag_base)
        self.x0_system = self.xS0
        self.muS_system = self.muS
        self.b_sff = np.array(b_sff)      
        self.raL = raL
        self.decL = decL
        self.w = omega
        self.o = big_omega
        self.omega = self.w
        self.big_omega = self.o
        self.i = i
        self.e = 0
        self.p = p
        self.tp = tp
        self.aleph = aleph #mas
        self.aleph_sec = aleph_sec #mas
        self.vx = muS_system_E * 1e-3
        self.vy = muS_system_N * 1e-3
        self.x0 = x0_system_E
        self.y0 = x0_system_N
        self.alpha = alpha
        self.fratio_bin = np.array(fratio_bin)
        self.alpha_rad = self.alpha * np.pi / 180.0

        self.mag_src_pri = self.mag_base \
                           - 2.5 * np.log10(self.b_sff) \
                           + 2.5 * np.log10(1.0 + self.fratio_bin)
        self.mag_src_sec = self.mag_base \
                           - 2.5 * np.log10(self.b_sff) \
                           + 2.5 * np.log10(1.0 + (1.0 / self.fratio_bin))

        self.sep = aleph + aleph_sec
        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()


        # Derived quantities
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)

        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens
        # (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)
        
        return 


class BSPL_PhotAstrom_Circ_Param4(PSPL_Param):


    
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    mL: float
        Mass of the lens (Msun)
    t0_com: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    beta: float
        Angular distance between the lens and primary source on the
        plane of the sky (mas). Can be
            * positive (u0_amp > 0 when u0_hat[0] < 0) or
            * negative (u0_amp < 0 when u0_hat[0] > 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    omega: float
        The argument of periastron of the primary star's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary star's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary source in AU. 
    aleph_sec: float
        This is the semi-major axis of the secondary source in AU. 
    muS_system_E: float
        The RA proper motion of the system in mas/yr.
    muS_system_N: float
        The Dec proper motion of the system in mas/yr.
    x0_system_E: float
        The initial center of mass coordinates (RA) of the system in arcsec.
    x0_system_N: float
        The initial center of mass coordinates (Dec) of the system in arcsec.
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_S1 + f_S2) / (f_S1 + f_s2 + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['mL', 't0_com', 'beta', 'dL', 'dL_dS',
                          'x0_system_E', 'x0_system_N',
                          'muL_E', 'muL_N',
                          'muS_system_E', 'muS_system_N',
                          'alpha',
                         'omega', 'big_omega', 'i',
                          'p', 'tp', 'aleph', 'aleph_sec'
                         ]
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']
    additional_param_names = ['dS', 'tE', 'u0_amp',
                              'thetaE_E', 'thetaE_N',
                              'piE_E', 'piE_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'



    def __init__(self, mL, t0_com, beta, dL, dL_dS,
                 x0_system_E, x0_system_N,
                 muL_E, muL_N,
                muS_system_E, muS_system_N,
                 alpha, omega, big_omega, i,
                 p, tp, aleph, aleph_sec,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=None, decL=None):
                     
        self.t0 = t0_com  # time of closest approach for system=primary pos
        self.t0_com = t0_com  # time of closest approach for system=primary pos
        self.mL = mL
        self.xS0 = np.array([x0_system_E, x0_system_N])  # position of source system=primary
        self.beta = beta
        self.muL = np.array([muL_E, muL_N])
        self.muS = np.array([muS_system_E, muS_system_N]) #mas
        self.x0_system = self.xS0
        self.muS_system = self.muS
        self.dL = dL
        self.dL_dS = dL_dS
        self.dS = self.dL / self.dL_dS
        self.b_sff = np.array(b_sff)
        self.raL = raL
        self.decL = decL
        self.w = omega
        self.o = big_omega
        self.omega = self.w
        self.big_omega = self.o
        self.i = i
        self.e = 0
        self.p = p
        self.tp = tp
        self.aleph = aleph/self.dS * 1e-3 #mas
        self.aleph_sec = aleph_sec/self.dS *1e-3 #mas
                     
        self.vx = muS_system_E * 1e-3
        self.vy = muS_system_N * 1e-3
        self.x0 = x0_system_E
        self.y0 = x0_system_N

        # Binary source parameters.
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.mag_src_pri = np.array(mag_src_pri)
        self.mag_src_sec = np.array(mag_src_sec)
        self.fratio_bin = 10.0 ** ((self.mag_src_sec - self.mag_src_pri) / -2.5)

        flux_pri = mag2flux(self.mag_src_pri)
        flux_sec = mag2flux(self.mag_src_sec)
        self.mag_base = flux2mag(flux_pri + flux_sec) + 2.5 * np.log10(self.b_sff)

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (self.dL * units.pc)) - (1.0 / (self.dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        #System 
        self.muRel_E, self.muRel_N = self.muRel
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr

        self.muS_E, self.muS_N = self.muS
        self.muL_E, self.muL_N = self.muL

        # Calculate the Einstein radius
        thetaE = units.rad * np.sqrt(
            (4.0 * const.G * mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.sun = units.M_sun
        self.c = const.c             
        self.G=const.G
        self.inv = inv_dist_diff           #remove later
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat
        self.thetaE_E, self.thetaE_N = self.thetaE

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0_amp = self.beta / self.thetaE_amp  # in Einstein units
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens
        # (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat
        self.piE_E, self.piE_N = self.piE

        # Calculate the Einstein crossing time. (days)
        # System tE
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year

        return


class PSBL_PhotAstromParam4(PSPL_Param):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    t0_com : float
        Time of photometric peak, as seen from Earth (MJD.DDD) 
        FIXME: THIS IS NOT RIGHT
    u0_amp_com : float
        Angular distance between the source and the binary lens COM
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    beta : float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). 
    q : float
        Mass ratio (M2 / M1)
    sep : float
        Angular separation of the two lenses (mas)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N', 'xS0_E', 'xS0_N', 'muS_E', 'muS_N',
                          'q', 'sep', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']

    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']
        
    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = False

    def __init__(self, t0_com, u0_amp_com, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, muS_E, muS_N,
                     q, sep, alpha,
                     b_sff, mag_src,
                     raL=None, decL=None, root_tol=1e-8):
        
        self.t0_com = t0_com
        self.u0_amp_com = u0_amp_com
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.thetaE_amp = thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.q = q
        self.sep = sep
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.b_sff = b_sff
        self.mag_src = mag_src
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol

        # Check variable formatting.
        super().__init__()
            
        # Derived quantities
        self.phi_rad = self.alpha_rad - np.arctan2(piE_E, piE_N)
        qeff = (1 - q)/(1 + q)
        self.t0 = self.t0_com - 0.5 * qeff * self.tE * self.sep * np.cos(self.phi_rad) / self.thetaE_amp
        self.u0_amp = self.u0_amp_com - 0.5 * qeff * self.sep * np.sin(self.phi_rad) / self.thetaE_amp

        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)
        self.mLp = self.mL / (1.0 + self.q)
        self.mLs = self.mLp * self.q
        
        
        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        inv_dist_diff = (1.0 / dL) - (1.0 / dS)
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        return

                         
class PSBL_PhotAstrom_EllOrbs_Param4(PSBL_PhotAstromParam4):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    t0_com : float
        Time of photometric peak, as seen from Earth (MJD.DDD) 
        FIXME: THIS IS NOT RIGHT
    u0_amp_com : float
        Angular distance between the source and the binary lens COM
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    omega: float
        The argument of periastron of the primary lens's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary lens's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    e: float
        Eccentricity of the System
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary lens in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary lens in mas. 
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    beta : float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). 
    q : float
        Mass ratio (M2 / M1)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N', 'xS0_E', 'xS0_N','omega', 'big_omega', 'i', 'e', 'p', 'tp', 'aleph', 
                          'aleph_sec', 'muS_E', 'muS_N',
                          'q', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']
        
    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'

    def __init__(self, t0_com, u0_amp_com, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, omega, big_omega, i, e, p, tp, aleph, aleph_sec, muS_E, muS_N,
                     q, alpha,
                     b_sff, mag_src,
                     raL=None, decL=None, root_tol=1e-8):
                         
        super().__init__(t0_com, u0_amp_com, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, muS_E, muS_N,
                     q, aleph+aleph_sec, alpha,
                     b_sff, mag_src,
                     raL=raL, decL=decL, root_tol=root_tol)
                         
        #Orbital parameters
        self.omega = omega
        self.big_omega = big_omega
        self.i = i
        self.p = p
        self.e = e
        self.tp = tp
        self.aleph = aleph
        self.aleph_sec = aleph_sec
        self.vx = self.muL_E * 1e-3
        self.vy = self.muL_N *1e-3
        self.sep = aleph + aleph_sec
        self.xL0_E, self.xL0_N = self.xL0

        PSPL_Phot().__init__()
        return
                         
class PSBL_PhotAstrom_CircOrbs_Param4(PSBL_PhotAstrom_EllOrbs_Param4):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    t0_com : float
        Time of photometric peak, as seen from Earth (MJD.DDD) 
        FIXME: THIS IS NOT RIGHT
    u0_amp_com : float
        Angular distance between the source and the binary lens COM
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    omega: float
        The argument of periastron of the primary lens's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary lens's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary lens in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary lens in mas. 
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    beta : float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). 
    q : float
        Mass ratio (M2 / M1)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N', 'xS0_E', 'xS0_N','omega', 'big_omega', 'i', 'p', 'tp', 'aleph', 
                          'aleph_sec', 'muS_E', 'muS_N',
                          'q', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']
    
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']
        
    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'

    def __init__(self, t0_com, u0_amp_com, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, omega, big_omega, i, p, tp, aleph, aleph_sec, muS_E, muS_N,
                     q, alpha,
                     b_sff, mag_src,
                     raL=None, decL=None, root_tol=1e-8):
                         
        super().__init__(t0_com, u0_amp_com, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, omega, big_omega, i, 0, p, tp, aleph, aleph_sec, muS_E, muS_N,
                     q, alpha,
                     b_sff, mag_src,
                     raL=raL, decL=decL, root_tol=root_tol)

                     
        PSPL_Param().__init__()
        return

    
class PSBL_PhotAstromParam5(PSPL_Param):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    t0_prim : float
        Time of closest approach between source and primary lens (MJD.DDD).
    u0_amp_prim : float
        Angular distance between the source and the PRIMARY lens
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piEN_piEE : float
        The ratio of piE_E / piE_N.
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    beta : float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). 
    q : float
        Mass ratio (M2 / M1)
    sep : float
        Angular separation of the two lenses (mas)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['t0_prim', 'u0_amp_prim', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piEN_piEE', 'xS0_E', 'xS0_N', 'muS_E', 'muS_N',
                          'q', 'sep', 'alpha']
    phot_param_names = ['b_sff', 'mag_base']
    additional_param_names = ['piE_N', 'mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N', 'mag_src']
        
    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = False

    def __init__(self, t0_prim, u0_amp_prim, tE, thetaE, piS,
                     piE_E, piEN_piEE, xS0_E, xS0_N, muS_E, muS_N,
                     q, sep, alpha,
                     b_sff, mag_base,
                     raL=None, decL=None, root_tol=1e-8):
        self.t0_prim = t0_prim
        self.u0_amp_prim = u0_amp_prim
        self.tE = tE
        self.piEN_piEE = piEN_piEE
        piE_N = piEN_piEE * piE_E
        self.piE_N = piE_N
        self.piE = np.array([piE_E, piE_N])
        self.thetaE_amp = thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.q = q
        self.sep = sep
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.b_sff = b_sff
        self.mag_base = mag_base
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol

        # Check variable formatting.
        super().__init__()
            
        # Derived quantities
        self.mag_src = self.mag_base - 2.5*np.log10(self.b_sff)
        self.phi_rad = self.alpha_rad - np.arctan2(piE_E, piE_N)
        self.t0 = self.t0_prim - 0.5 * self.tE * self.sep * np.cos(self.phi_rad) / self.thetaE_amp
        self.u0_amp = self.u0_amp_prim - 0.5 * self.sep * np.sin(self.phi_rad) / self.thetaE_amp

        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)
        self.mLp = self.mL / (1.0 + self.q)
        self.mLs = self.mLp * self.q
        
        
        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        inv_dist_diff = (1.0 / dL) - (1.0 / dS)
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # IS THIS IT? DOES IT WORK?
        
        return


class PSBL_PhotAstromParam6(PSPL_Param):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    t0_prim : float
        Time of photometric peak, as seen from Earth (MJD.DDD) 
        FIXME: THIS IS NOT RIGHT
    u0_amp_prim : float
        Angular distance between the source and the PRIMARY lens
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    beta : float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). 
    q : float
        Mass ratio (M2 / M1)
    sep : float
        Angular separation of the two lenses (mas)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N', 'xS0_E', 'xS0_N', 'muS_E', 'muS_N',
                          'q', 'sep', 'alpha']
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = False


    def __init__(self, t0_prim, u0_amp_prim, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, muS_E, muS_N,
                     q, sep, alpha,
                     b_sff, mag_src,
                     raL=None, decL=None, root_tol=1e-8):
        self.t0_prim = t0_prim
        self.u0_amp_prim = u0_amp_prim
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.thetaE_amp = thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.q = q
        self.sep = sep
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.b_sff = b_sff
        self.mag_src = mag_src
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol

        # Check variable formatting.
        super().__init__()

        # Derived quantities
        self.phi_rad = self.alpha_rad - np.arctan2(piE_E, piE_N)
        self.t0 = self.t0_prim - 0.5 * self.tE * self.sep * np.cos(self.phi_rad) / self.thetaE_amp
        self.u0_amp = self.u0_amp_prim - 0.5 * self.sep * np.sin(self.phi_rad) / self.thetaE_amp

        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)
        self.mLp = self.mL / (1.0 + self.q)
        self.mLs = self.mLp * self.q
        
        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        inv_dist_diff = (1.0 / dL) - (1.0 / dS)
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        return

      
class PSBL_PhotAstrom_AccOrbs_Param6(PSBL_PhotAstromParam6):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    mLp, mLs : float
        Masses of the lenses (Msun)
    t0_p : float
        Time of closest approach between the source and primary lens, as seen from Earth (MJD.DDD)
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    beta : float
        Angular distance between the source and the GEOMETRIC center 
        of the lenses on the plane of the sky (mas). Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    muL_E : float
        Primary Lens System proper motion in the RA direction (mas/yr)
    muL_N : float
        Primary Lens System proper motion in the Dec. direction (mas/yr)
    delta_muLsec_E: float
        Secondary Lens System proper motion in the RA direction (mas/yr)
    delta_muLsec_N: float
        Secondary Lens System proper motion in the Dec. direction (mas/yr)
    acc_E: float
        Secondary Lens System's acceleration in the RA. direction (mas/yr^2)
    acc_N: float
        Secondary Lens System's acceleration in the Dec. direction (mas/yr^2)
    muS_E : float
        Source proper motion in the RA direction (mas/yr)
    muS_N : float
        Source proper motion in the Dec. direction (mas/yr)
    dL : float
        Distance from the observer to the lens system (pc)
    dS : float
        Distance from the observer to the source (pc)
    sep : float
        Initial angular separation of the two lenses (mas)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['mLp', 'mLs', 't0_p', 'xS0_E', 'xS0_N',
                          'beta', 'muL_E', 'muL_N', 'delta_muLsec_E', 'delta_muLsec_N', 'acc_E', 'acc_N', 'muS_E', 'muS_N',
                          'dL', 'dS', 'sep', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'acc'

    def __init__(self, t0_prim, u0_amp_prim, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, delta_muLsec_E, delta_muLsec_N, acc_E, acc_N,  muS_E, muS_N,
                     q, sep, alpha,
                     b_sff, mag_src,
                     raL=None, decL=None, root_tol=1e-8):
        
                     
        super().__init__(t0_prim, u0_amp_prim, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, muS_E, muS_N,
                     q, sep, alpha,
                     b_sff, mag_src,
                     raL=raL, decL=decL, root_tol=1e-8)

           
        self.delta_muL_sec =  np.array([delta_muLsec_E, delta_muLsec_N])
        self.delta_muLsec_E, self.delta_muLsec_N = self.delta_muL_sec
        self.muL_sec = np.array([self.muL_E+delta_muLsec_E, self.muL_N+delta_muLsec_N])
        self.acc = np.array([acc_E, acc_N])

        return
                     
class PSBL_PhotAstrom_LinOrbs_Param6(PSBL_PhotAstrom_AccOrbs_Param6):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    mLp, mLs : float
        Masses of the lenses (Msun)
    t0_p : float
        Time of closest approach between the source and primary lens, as seen from Earth (MJD.DDD)
    xS0_E : float
        R.A. of source position on the sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on the sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    beta : float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    muL_E : float
        Primary Lens System proper motion in the RA direction (mas/yr)
    muL_N : float
        Primary Lens System proper motion in the Dec. direction (mas/yr)
    delta_muLsec_E: float
        Secondary Lens System proper motion in the RA direction (mas/yr)
    delta_muLsec_N: float
        Secondary Lens System proper motion in the Dec. direction (mas/yr)
    muS_E : float
        Source proper motion in the RA direction (mas/yr)
    muS_N : float
        Source proper motion in the Dec. direction (mas/yr)
    dL : float
        Distance from the observer to the lens system (pc)
    dS : float
        Distance from the observer to the source (pc)
    sep : float
        Initial angular separation of the two lenses (mas)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['mLp', 'mLs', 't0_p', 'xS0_E', 'xS0_N',
                          'beta', 'muL_E', 'muL_N', 'delta_muLsec_E', 'delta_muLsec_N', 'muS_E', 'muS_N',
                          'dL', 'dS', 'sep', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'linear'

    def __init__(self, t0_prim, u0_amp_prim, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, delta_muLsec_E, delta_muLsec_N, muS_E, muS_N,
                     q, sep, alpha,
                     b_sff, mag_src,
                     raL=None, decL=None, root_tol=1e-8):
        
                     
        super().__init__(t0_prim, u0_amp_prim, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, delta_muLsec_E, delta_muLsec_N, 0, 0, muS_E, muS_N,
                     q, sep, alpha,
                     b_sff, mag_src,
                     raL=raL, decL=decL, root_tol=1e-8)

                         
        return




class PSBL_PhotAstromParam7(PSPL_Param):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    mLp, mLs : float
        Masses of the lenses (Msun)
    t0_p : float
        Time of closest approach between source and PRIMARY LENS (MJD.DDD)
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    beta_p : float
        Angular distance between the source and the PRIMARY LENS
        of the lenses on the plane of the sky (mas) at t0. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    muL_E : float
        Lens system proper motion in the RA direction (mas/yr)
    muL_N : float
        Lens system proper motion in the Dec. direction (mas/yr)
    muS_E : float
        Source proper motion in the RA direction (mas/yr)
    muS_N : float
        Source proper motion in the Dec. direction (mas/yr)
    dL : float
        Distance from the observer to the lens system (pc)
    dS : float
        Distance from the observer to the source (pc)
    sep : float
        Angular separation of the two lenses (mas)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['mLp', 'mLs', 't0_p', 'xS0_E', 'xS0_N',
                          'beta_p', 'muL_E', 'muL_N', 'muS_E', 'muS_N',
                          'dL', 'dS', 'sep', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = False


    def __init__(self, mLp, mLs, t0_p, xS0_E, xS0_N,
                 beta_p, muL_E, muL_N, muS_E, muS_N, dL, dS,
                 sep, alpha, b_sff, mag_src,
                 raL=None, decL=None, root_tol=1e-8):
        self.mLp = mLp  # Msun
        self.mLs = mLs  # Msun
        self.t0_p = t0_p
        self.xS0 = np.array([xS0_E, xS0_N])
        self.beta_p = beta_p
        self.muL = np.array([muL_E, muL_N])
        self.muS = np.array([muS_E, muS_N])
        self.dL = dL
        self.dS = dS
        self.sep = sep
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.b_sff = b_sff
        self.mag_src = mag_src
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol

        # Super handles checking for properly formatted variables.
        super().__init__()

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (dL * units.pc)) - (1.0 / (dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr
        

        # Calculate the Einstein radius
        # AFAICT, thetaE for binary lenses is calculated from the total lens mass.
        # Checked using Shin+17 (OB160168) and Jung+19 (OB160156)
        self.mL = self.mLp + self.mLs  # Total lens mass
        thetaE = units.rad * np.sqrt((4.0 * const.G * self.mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat
        
        # Calculate the Einstein crossing time. (days)
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year

        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat
        
        # Calculate other angles
        self.phi_piE_rad = np.arctan2(self.piE[0], self.piE[1])
        self.phi_piE = self.phi_piE_rad * 180.0 / np.pi
        self.phi_rad = self.alpha_rad - self.phi_piE_rad
        self.phi = self.phi_rad * 180.0 / np.pi

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat_p = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta_p)
        self.u0_amp_p = self.beta_p / self.thetaE_amp  # in Einstein units
        self.u0_p = np.abs(self.u0_amp_p) * self.u0_hat_p
        
        # Calculate t0 and beta for geometric center of lens
        self.q = self.mLs/self.mLp
        u0_x_out, u0_y_out, t0_out = fc.convert_u0_t0_psbl(t0_in = self.t0_p, u0_x_in = self.u0_p[0], u0_y_in = self.u0_p[1],
                           tE = self.tE, theta_E = self.thetaE_amp, q = self.q, phi = self.phi_rad, sep = self.sep, mu_rel_x = self.muRel[0],
                           mu_rel_y = self.muRel[1], coords_in='prim_center', coords_out='geom_mid')
        self.u0 = np.array([u0_x_out, u0_y_out])
        self.u0_amp = np.sqrt(self.u0[0]**2 + self.u0[1]**2)
        self.t0 = t0_out
        self.beta = self.u0_amp*self.thetaE_amp

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        

        return


                     
class PSBL_PhotAstrom_AccOrbs_Param7(PSBL_PhotAstromParam7):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    mLp, mLs : float
        Masses of the lenses (Msun)
    t0_p : float
        Time of closest approach between the source and primary lens, as seen from Earth (MJD.DDD)
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    beta : float
        Angular distance between the source and the GEOMETRIC center 
        of the lenses on the plane of the sky (mas). Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    muL_E : float
        Primary Lens System proper motion in the RA direction (mas/yr)
    muL_N : float
        Primary Lens System proper motion in the Dec. direction (mas/yr)
    delta_muLsec_E: float
        Secondary Lens System proper motion in the RA direction (mas/yr)
    delta_muLsec_N: float
        Secondary Lens System proper motion in the Dec. direction (mas/yr)
    acc_E: float
        Secondary Lens System's acceleration in the RA. direction (mas/yr^2)
    acc_N: float
        Secondary Lens System's acceleration in the Dec. direction (mas/yr^2)
    muS_E : float
        Source proper motion in the RA direction (mas/yr)
    muS_N : float
        Source proper motion in the Dec. direction (mas/yr)
    dL : float
        Distance from the observer to the lens system (pc)
    dS : float
        Distance from the observer to the source (pc)
    sep : float
        Initial angular separation of the two lenses (mas)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['mLp', 'mLs', 't0_p', 'xS0_E', 'xS0_N',
                          'beta', 'muL_E', 'muL_N', 'delta_muLsec_E', 'delta_muLsec_N', 'acc_E', 'acc_N', 'muS_E', 'muS_N',
                          'dL', 'dS', 'sep', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'acc'

    def __init__(self, mLp, mLs, t0_p, xS0_E, xS0_N,
                 beta_p, muL_E, muL_N, delta_muLsec_E, delta_muLsec_N, acc_E, acc_N, muS_E, muS_N, dL, dS,
                 sep, alpha, b_sff, mag_src,
                 raL=None, decL=None, root_tol=1e-8):
                     
        super().__init__(mLp, mLs, t0_p, xS0_E, xS0_N,
                 beta_p, muL_E, muL_N, muS_E, muS_N, dL, dS,
                 sep, alpha, b_sff, mag_src,
                 raL, decL, root_tol)

           
        self.delta_muL_sec =  np.array([delta_muLsec_E, delta_muLsec_N])
        self.delta_muLsec_E, self.delta_muLsec_N = self.delta_muL_sec
        self.muL_sec = np.array([muL_E+delta_muLsec_E, muL_N+delta_muLsec_N])
        self.acc = np.array([acc_E, acc_N])

        return
                     
class PSBL_PhotAstrom_LinOrbs_Param7(PSBL_PhotAstrom_AccOrbs_Param7):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    mLp, mLs : float
        Masses of the lenses (Msun)
    t0_p : float
        Time of closest approach between the source and primary lens, as seen from Earth (MJD.DDD)
    xS0_E : float
        R.A. of source position on the sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on the sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    beta : float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    muL_E : float
        Primary Lens System proper motion in the RA direction (mas/yr)
    muL_N : float
        Primary Lens System proper motion in the Dec. direction (mas/yr)
    delta_muLsec_E: float
        Secondary Lens System proper motion in the RA direction (mas/yr)
    delta_muLsec_N: float
        Secondary Lens System proper motion in the Dec. direction (mas/yr)
    muS_E : float
        Source proper motion in the RA direction (mas/yr)
    muS_N : float
        Source proper motion in the Dec. direction (mas/yr)
    dL : float
        Distance from the observer to the lens system (pc)
    dS : float
        Distance from the observer to the source (pc)
    sep : float
        Initial angular separation of the two lenses (mas)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['mLp', 'mLs', 't0_p', 'xS0_E', 'xS0_N',
                          'beta', 'muL_E', 'muL_N', 'delta_muLsec_E', 'delta_muLsec_N', 'muS_E', 'muS_N',
                          'dL', 'dS', 'sep', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'linear'

    def __init__(self, mLp, mLs, t0_p, xS0_E, xS0_N,
                 beta, muL_E, muL_N, delta_muLsec_E, delta_muLsec_N, muS_E, muS_N, dL, dS,
                 sep, alpha, b_sff, mag_src,
                 raL=None, decL=None, root_tol=1e-8):
        super().__init__(mLp, mLs, t0_p, xS0_E, xS0_N,
                 beta, muL_E, muL_N, delta_muLsec_E, delta_muLsec_N, 0, 0, muS_E, muS_N, dL, dS,
                 sep, alpha, b_sff, mag_src,
                 raL=raL, decL=decL, root_tol=root_tol)
        return


                     

class PSBL_PhotAstromParam8(PSPL_Param):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    t0_com : float
        Time of photometric peak, as seen from Earth (MJD.DDD) 
        FIXME: THIS IS NOT RIGHT
    u0_amp_com : float
        Angular distance between the source and the binary lens COM
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE : float
        Einstein crossing time (days).
    log10_thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    beta : float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). 
    q : float
        Mass ratio (M2 / M1)
    sep : float
        Angular separation of the two lenses (mas)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'log10_thetaE', 'piS',
                          'piE_E', 'piE_N', 'xS0_E', 'xS0_N', 'muS_E', 'muS_N',
                          'q', 'sep', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']
        
    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = False

    def __init__(self, t0_com, u0_amp_com, tE, log10_thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, muS_E, muS_N,
                     q, sep, alpha,
                     b_sff, mag_src,
                     raL=None, decL=None, root_tol=1e-8):
        
        self.t0_com = t0_com
        self.u0_amp_com = u0_amp_com
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.log10_thetaE = log10_thetaE
        self.thetaE = 10 ** log10_thetaE
        self.thetaE_amp = 10 ** log10_thetaE
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muS = np.array([muS_E, muS_N])
        self.piS = piS
        self.q = q
        self.sep = sep
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.b_sff = b_sff
        self.mag_src = mag_src
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol

        # Check variable formatting.
        super().__init__()
            
        # Derived quantities
        self.phi_rad = self.alpha_rad - np.arctan2(piE_E, piE_N)
        qeff = (1 - q)/(1 + q)
        self.t0 = self.t0_com - 0.5 * qeff * self.tE * self.sep * np.cos(self.phi_rad) / self.thetaE_amp
        self.u0_amp = self.u0_amp_com - 0.5 * qeff * self.sep * np.sin(self.phi_rad) / self.thetaE_amp

        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)
        self.mLp = self.mL / (1.0 + self.q)
        self.mLs = self.mLp * self.q
        
        
        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        inv_dist_diff = (1.0 / dL) - (1.0 / dS)
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        return


class PSBL_PhotAstrom_EllOrbs_Param8(PSBL_PhotAstromParam8):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    t0_com : float
        Time of photometric peak, as seen from Earth (MJD.DDD) 
        FIXME: THIS IS NOT RIGHT
    u0_amp_com : float
        Angular distance between the source and the binary lens COM
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    omega: float
        The argument of periastron of the primary lens's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary lens's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    e: float
        Eccentricity of the System
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary lens in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary lens in mas. 
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    beta : float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). 
    q : float
        Mass ratio (M2 / M1)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N', 'xS0_E', 'xS0_N','omega', 'big_omega', 'i', 'e', 'p', 'tp', 'aleph', 
                          'aleph_sec', 'muS_E', 'muS_N',
                          'q', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']
        
    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'

    def __init__(self, t0_com, u0_amp_com, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, omega, big_omega, i, e, p, tp, aleph, aleph_sec, muS_E, muS_N,
                     q, alpha,
                     b_sff, mag_src,
                     raL=None, decL=None, root_tol=1e-8):
                         
        super().__init__(t0_com, u0_amp_com, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, muS_E, muS_N,
                     q, aleph+aleph_sec, alpha,
                     b_sff, mag_src,
                     raL=raL, decL=raL, root_tol=root_tol)

        self.omega = omega
        self.big_omega = big_omega
        self.i = i
        self.p = p
        self.e = e
        self.tp = tp
        self.aleph = aleph
        self.aleph_sec = aleph_sec
        self.vx = self.muL_E * 1e-3
        self.vy = self.muL_N *1e-3
        self.sep = aleph + aleph_sec
        self.xL0_E, self.xL0_N = self.xL0

        PSPL_Param().__init__()
        return
                         
class PSBL_PhotAstrom_CircOrbs_Param8(PSBL_PhotAstrom_EllOrbs_Param8):
    """
    Point source binary lens.
    It has 3 more parameters than PSPL (additional mass term, separation,
    and angle of approach). Note that this is a STATIC binary lens, i.e.
    there is no orbital motion.

    Attributes
    ----------
    t0_com : float
        Time of photometric peak, as seen from Earth (MJD.DDD) 
        FIXME: THIS IS NOT RIGHT
    u0_amp_com : float
        Angular distance between the source and the binary lens COM
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    omega: float
        The argument of periastron of the primary lens's orbit in degrees. 
        The secondary source will be directly 180 degrees across the primary 
        source's argument of periastron.
    big_omega: float
        The longitude of the ascending node of the secondary lens's orbit
        in degrees. Since the primary and secondary sources share the same 
        inclination angle for their orbital planes, they also share the same
        big_omega
    i: float
        Inclination angle of the system in degrees. 
    p: float
        Orbital period of the system in days.
    tp: float
        This is the time of the periastron of the system in days.
    aleph: float
        This is the semi-major axis of the primary lens in mas. 
    aleph_sec: float
        This is the semi-major axis of the secondary lens in mas.  
    muS_E : float
        RA Source proper motion (mas/yr)
    muS_N : float
        Dec Source proper motion (mas/yr)
    beta : float
        Angular distance between the source and the GEOMETRIC center
        of the lenses on the plane of the sky (mas). 
    q : float
        Mass ratio (M2 / M1)
    alpha : float
        Angle made between the binary axis and North;
        measured in degrees East of North.
    b_sff : numpy array or list
        The ratio of the source flux to the total (source + neighbors + lenses). One
        for each filter.
    mag_src : numpy array or list
        Source magnitude, unlensed. One in each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N', 'xS0_E', 'xS0_N','omega', 'big_omega', 'i', 'p', 'tp', 'aleph', 
                          'aleph_sec', 'muS_E', 'muS_N',
                          'q', 'alpha']
    phot_param_names = ['b_sff', 'mag_src']
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']
        
    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'circular'

    def __init__(self, t0_com, u0_amp_com, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, omega, big_omega, i, p, tp, aleph, aleph_sec, muS_E, muS_N,
                     q, alpha,
                     b_sff, mag_src,
                     raL=None, decL=None, root_tol=1e-8):
                         
        super().__init__(t0_com, u0_amp_com, tE, thetaE, piS,
                     piE_E, piE_N, xS0_E, xS0_N, omega, big_omega, i, 0, p, tp, aleph, aleph_sec, muS_E, muS_N,
                     q, alpha,
                     b_sff, mag_src,
                     raL=raL, decL=decL, root_tol=root_tol)

        PSPL_Param().__init__()
        return

    

class PSBL_PhotParam1(PSPL_Param):
    """
    Point source binary lens, photometry only.

    It has 3 more parameters than PSPL_PhotParam1:
       * mass ratio
       * separation -- in units of thetaE
       * angle of approach 
    Note that this is a STATIC binary lens, i.e. there is no orbital motion.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth [MJD]
    u0_amp: float
        Angular distance between the lens and source on the plane of the
        sky at closest approach in units of thetaE. It can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE: float
        Einstein crossing time based on the system mass. [MJD]
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    q: float
        Mass ratio (low-mass / high-mass)
    sep: float
        Angular separation of the two lenses in units of thetaE where
        thetaE is defined with the total binary mass.
    phi: float
        Angle made between the binary axis and the relative proper motion vector,
        measured in degrees.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = f_S / (f_S + f_L + f_N)`. This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_src:  array or list
        Photometric magnitude of the source. This must be passed in as a
        list or array, with one entry for each photometric filter.

    Other Parameters
    ----------------
    raL: float
        Right ascension of the lens in decimal degrees.
        Required if calculating with parallax
    decL: float
        Declination of the lens in decimal degrees.
        Required if calculating with parallax
    root_tol: float
        | Tolerance in comparing the polynomial roots to the physical solutions. 
        | Default = 0.0
    """
    fitter_param_names = ['t0', 'u0_amp', 'tE', 'piE_E', 'piE_N',
                          'q', 'sep', 'phi']
    phot_param_names = ['b_sff', 'mag_src']

    paramAstromFlag = False
    paramPhotFlag = True

    def __init__(self, t0, u0_amp, tE, piE_E, piE_N, q, sep, phi,
                 b_sff, mag_src,
                 raL=None, decL=None, root_tol=1e-8):
        self.t0 = t0
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.q = q
        self.sep = sep
        self.phi = phi
        self.b_sff = b_sff
        self.mag_src = mag_src

        # Check variable formatting.
        super().__init__()

        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol

        # Calculate the microlensing parallax amplitude
        self.piE_amp = np.linalg.norm(self.piE)

        # Calculate the angle between muRel and the binary axis
        # in radians.
        self.phi_rad = self.phi * np.pi / 180.0

        # For convenience, calculate the vector pointing
        # from the geometric center of the lens binary system
        # to the primary mass. This requires a combination of
        # phi and piE_hat.
        self.phi_piE_rad = np.arctan2(self.piE[0], self.piE[1])
        # Note that phi_rho1 is the same alpha in our astrometry model;
        # however, here we don't have North as a reference.
        self.phi_rho1_rad = self.phi_piE_rad - self.phi_rad
        self.xL1_over_theta = np.array([0.5 * self.sep * np.sin(self.phi_rho1_rad),
                                        0.5 * self.sep * np.cos(self.phi_rho1_rad)])
        self.xL2_over_theta = np.array([-0.5 * self.sep * np.sin(self.phi_rho1_rad),
                                        -0.5 * self.sep * np.cos(self.phi_rho1_rad)])

        # Get thetaE_hat (same direction as piE and muRel)
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.u0_amp)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Calculate m1 and m2 (see PSBL writeup). Note this is no longer the
        # individual einstein radius**2... but rather the fractional
        # einstein radius for each component of the binary. Thus these
        # should only be used in phot models where everything has been
        # divided by the Einstein radius.
        self.m1 = 1.0 / (1.0 + self.q)
        self.m2 = self.q / (1.0 + self.q)

        return


class PSBL_GP_PhotParam1(PSBL_PhotParam1):
    # Optional data-set specific parameters -- handled as dictionaries
    # (with keys on the filter index). Not ever data-set needs these.
    # User indicates which data-sets use these parameters by including or not
    # in the dictionary. This is most useful for noise properties.
    phot_optional_param_names = ['gp_log_sigma', 'gp_log_rho', 'gp_log_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, piE_E, piE_N, q, sep, phi,
                 b_sff, mag_src,
                 gp_log_sigma, gp_log_rho, gp_log_S0, gp_log_omega0,
                 raL=None, decL=None, root_tol=1e-8):
        self.gp_log_sigma = gp_log_sigma
        self.gp_log_rho = gp_log_rho
        self.gp_log_S0 = gp_log_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, tE, piE_E, piE_N, q, sep, phi,
                         b_sff, mag_src, raL=raL, decL=decL)

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return


class PSBL_GP_PhotAstromParam1(PSBL_PhotAstromParam1):
    phot_optional_param_names = ['gp_log_sigma', 'gp_log_rho', 'gp_log_S0', 'gp_log_omega0']

    def __init__(self, mLp, mLs, t0, xS0_E, xS0_N,
                 beta, muL_E, muL_N, muS_E, muS_N, dL, dS,
                 sep, alpha, b_sff, mag_src,
                 gp_log_sigma, gp_log_rho, gp_log_S0, gp_log_omega0,
                 raL=None, decL=None, root_tol=1e-8):
        self.gp_log_sigma = gp_log_sigma
        self.gp_log_rho = gp_log_rho
        self.gp_log_S0 = gp_log_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(mLp, mLs, t0, xS0_E, xS0_N,
                         beta, muL_E, muL_N, muS_E, muS_N, dL, dS,
                         sep, alpha, b_sff, mag_src,
                         raL=raL, decL=decL)

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return


class PSBL_GP_PhotAstromParam2(PSBL_PhotAstromParam2):
    phot_optional_param_names = ['gp_log_sigma', 'gp_log_rho', 'gp_log_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N, xS0_E, xS0_N, muS_E, muS_N,
                 q, sep, alpha,
                 b_sff, mag_src,
                 gp_log_sigma, gp_log_rho, gp_log_S0, gp_log_omega0,
                 raL=None, decL=None, root_tol=1e-8):
        self.gp_log_sigma = gp_log_sigma
        self.gp_log_rho = gp_log_rho
        self.gp_log_S0 = gp_log_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, tE, thetaE, piS,
                         piE_E, piE_N, xS0_E, xS0_N, muS_E, muS_N,
                         q, sep, alpha, b_sff, mag_src,
                         raL=raL, decL=decL)

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return

######################################################
### BINARY SOURCE POINT LENS (BSPL) CLASSES ###
######################################################
# --------------------------------------------------
#
# Data Class Family - BSPL
#
# --------------------------------------------------

class BSPL(PSPL):
    
    
    def get_u(self, t):
        """
        
        Parameters
        ----------
        t : numpy array
            Times at which to evaluate the source-lens separation.

        Returns
        -------
        u : numpy array 
            Shape = `[len(t), 2 sources, 2 directions on sky]`
        """
        # t0 is the time of closest approach, projected on the sky, for the 
        # whole binary source system. By default, we assume a static binary with
        # the primary star being at the origin of the system.

        # Calculate the elapsed time, in units of tE. 
        # This gives the linear motion offset due to muRel in the muRel_hat direction.

        xS_unlensed = self.get_resolved_astrometry_unlensed(t)
        
        xS1_unlens = xS_unlensed[:, 0, :] 
        xS2_unlens = xS_unlensed[:, 1, :] 

        xL = self.get_lens_astrometry(t)

        thetaE_amp = self.thetaE_amp * 1e-3
        u_pri = (xS1_unlens-xL)/thetaE_amp        
        u_sec = (xS2_unlens-xL)/thetaE_amp
        #import pdb
        #pdb.set_trace()
        # Incorporate parallax
        '''
        if self.parallaxFlag:
            parallax_vec = parallax_in_direction(self.raL, self.decL, t)
            u_pri -= (self.piE_amp * parallax_vec) * 1e-3
            u_sec -= (self.piE_amp * parallax_vec) * 1e-3
'''
        u = np.zeros((len(t), 2, 2), dtype=float)
        u[:, 0, :] = u_pri
        u[:, 1, :] = u_sec

        return u

    def get_resolved_amplification(self, t):
        """Parallax: Get the photometric amplification term at a set of times, t for both the
        plus and minus images.

        Parameters
        ----------
        t: Array of times in MJD.DDD

        Returns
        -------
        A_resolved : numpy array 
            [shape = len(t), len(sources), 2]
            
        Notes
        -----

        For each time t and each source, we have:
            * A_plus is the amplification for the plus image.
            * A_minus is the amplification for the minus image.

        In other words, 
            xS[0, 0, 0] returns the amplification of the 
            first source's "plus" image at the first time. 
        Similarly, 
            xS[0, 0, 1] returns the amplification of the 
            first source's "minus" image at the first time. 
        """
        # Get u for the primary and secondary at all times. 
        u_vec = self.get_u(t)

        u_vec1 = u_vec[:, 0, :]
        u_vec2 = u_vec[:, 1, :]

        u1 = np.linalg.norm(u_vec1, axis=1)
        u2 = np.linalg.norm(u_vec2, axis=1)

        A1_plus = 0.5 * ((u1 ** 2 + 2) / (u1 * np.sqrt(u1 ** 2 + 4)) + 1)
        A1_minu = 0.5 * ((u1 ** 2 + 2) / (u1 * np.sqrt(u1 ** 2 + 4)) - 1)

        A2_plus = 0.5 * ((u2 ** 2 + 2) / (u2 * np.sqrt(u2 ** 2 + 4)) + 1)
        A2_minu = 0.5 * ((u2 ** 2 + 2) / (u2 * np.sqrt(u2 ** 2 + 4)) - 1)

        N_sources = 2
        A = np.zeros((len(t), N_sources, 2), dtype=float)
        A[:, 0, 0] = A1_plus
        A[:, 0, 1] = A1_minu
        A[:, 1, 0] = A2_plus
        A[:, 1, 1] = A2_minu

        return A

    def get_amplification(self, t, filt_idx=0):
        """Parallax: Get the photometric amplification term at a set of times, t.

        Note that this is a convenience function that combines amplifications from 
        multiple sources. The returned amplification is

        ..math::
        
            A = (f1 * A1 + f2 * A2) / (f1 + f2)

        where the fluxes are the intrinsic source flux in the specified filter.

        Parameters
        ----------
        t: Array of times in MJD.DDD
        
        Returns
        -------
        A : numpy array
            | Array of combined amplifications in the specified filter. 
            | Shape = [len(t)]
        """
        u_vec = self.get_u(t)

        u_vec1 = u_vec[:, 0, :]
        u_vec2 = u_vec[:, 1, :]

        u1 = np.linalg.norm(u_vec1, axis=1)
        u2 = np.linalg.norm(u_vec2, axis=1)

        A1 = (u1 ** 2 + 2) / (u1 * np.sqrt(u1 ** 2 + 4))
        A2 = (u2 ** 2 + 2) / (u2 * np.sqrt(u2 ** 2 + 4))

        f1 = mag2flux(self.mag_src_pri[filt_idx])
        f2 = mag2flux(self.mag_src_sec[filt_idx])

        A = ((A1 * f1) + (A2 * f2)) / (f1 + f2)

        return A

    def get_photometry(self, t, filt_idx=0, print_warning=True):
        """
        Get the predicted photomety at the specified times for the specified 
        photometric filter or data set. 

        Parameters
        ----------
        t : array_like
            List of times in MJD for the observations. 
        filt_idx : int, optional
            Index of the photometric filter or data set.

        Other Parameters
        ----------------
        print_warning : bool, optional
            Print a warning in the rare case that the magnitude exceeds a 
            zeropoint of 30 and conversions result in NaN returned.

        Returns
        -------
        mag_model : array_like
            Magnitude of the unresolved microlensing event at t_obs.

        """
        # Get u for the primary and secondary at all times. 
        u_vec = self.get_u(t)

        u_vec1 = u_vec[:, 0, :]
        u_vec2 = u_vec[:, 1, :]

        u1 = np.linalg.norm(u_vec1, axis=1)
        u2 = np.linalg.norm(u_vec2, axis=1)

        A1 = (u1 ** 2 + 2) / (u1 * np.sqrt(u1 ** 2 + 4))
        A2 = (u2 ** 2 + 2) / (u2 * np.sqrt(u2 ** 2 + 4))
        
        # mags to fluxes
        # switch nan mags to 0 fluxes
        f1 = np.nan_to_num(mag2flux(self.mag_src_pri[filt_idx]), nan = 0)
        f2 = np.nan_to_num(mag2flux(self.mag_src_sec[filt_idx]), nan = 0)

        # Add linear source flux change. 
        if hasattr(self, 'fdfdt_pri'):
            f1 += f1 * (self.fdfdt_pri / 100) * (t - self.t0_pri)
        if hasattr(self, 'fdfdt_sec'):
            f2 += f2 * (self.fdfdt_sec / 100) * (t - self.t0_pri)

        # Amplify and combine together.
        flux_lensed1 = f1 * A1
        flux_lensed2 = f2 * A2

        flux_srcs = f1 + f2
        flux_lensed = flux_lensed1 + flux_lensed2

        # Account for blending.
        #        try:
        flux_lensed += flux_srcs * (1.0 - self.b_sff[filt_idx]) / self.b_sff[filt_idx]
        #        except AttributeError:
        #            pass

        # Catch the edge case where we exceed the zeropoint.
        bad = np.where(flux_lensed <= 0)[0]
        if len(bad) > 0:
            if print_warning:
                print('Warning: get_photometry: bad flux encountered.')
            flux_lensed[bad] = np.nan

        mag_lensed = flux2mag(flux_lensed)

        return mag_lensed


class BSPL_Phot(BSPL, PSPL_Phot):
    """
    Contains methods for model a BSPL photometry only.
    This is a Data-type class in our hierarchy. It is abstract and should not
    be instantiated. 

    Attributes
    ----------
    t0
    tE
    u0_amp
    u0_E
    u0_N
    piE_E:
        valid only if parallax model
    piE_N:
        valid only if parallax model
    piE_amp
    b_sff[#]
    mag_src[#]:
        add in
    mag_base[#]:
        add in 
    raL:
        if parallax model
    decL:
        if parallax model
    """
    photometryFlag = True
    astrometryFlag = False

    def get_resolved_astrometry_unlensed(self, t):
        """Get the astrometry of the source if the lens didn't exist.
        Note, this is a photometry only model, so units are in Einstein radii.

        Returns
        -------
        xS_resolved_unlensed : numpy array, [shape = len(t_obs), N_sources, 2]
            The unlensed positions of the sources in Einstein radii.

        In other words,
            xS[0, 0, :] returns the 2D sky position of the
            first source at the first time.
        Similarly,
            xS[0, 1, :] returns the 2D sky position of the
            second source at the first time.
        """
        u_unlens = self.get_u(t)

        return u_unlens

    def get_astrometry_unlensed(self, t, ast_filt_idx=0):
        """Get the astrometry of the source if the lens didn't exist.
        Note, this is a photometry only model, so units are in Einstein radii.

        Returns
        -------
        xS_unlensed : numpy array, dtype=float, shape = len(t_obs) x 2
            The unlensed positions of the source in Einstein radii.
        """
        u_unlens_both = self.get_resolved_astrometry_unlensed(t)
        u1_unlens = u_unlens_both[:, 0, :]
        u2_unlens = u_unlens_both[:, 1, :]

        # Calculate un-magnified fluxes. Note, we ignore blended flux entirely.
        f1 = mag2flux(self.mag_src_pri[ast_filt_idx])
        f2 = mag2flux(self.mag_src_sec[ast_filt_idx])

        # Flux-weighted centroid.
        u_unlens = (u1_unlens * f1 + u2_unlens * f2) / (f1 + f2)

        return u_unlens

    def get_resolved_astrometry(self, t):
        '''
        Position of the observed source position in Einstein radii.

        Parameters
        ----------
        t : array_like, shape = [N_times]
            Array of times to model.
            
        Returns
        -------
        model_pos : array_like. shape = [N_times, N_images, 2]
            Array of vector positions of the centroid at each t_obs.
        '''
        u_vec = self.get_u(t)
        u_vec1 = u_vec[:, 0, :]
        u_vec2 = u_vec[:, 1, :]

        u1 = np.linalg.norm(u_vec1, axis=1)
        u2 = np.linalg.norm(u_vec2, axis=1)

        u1_hat = (u_vec1.T / u1).T
        u2_hat = (u_vec2.T / u2).T

        u1_plus = ((u1 + np.sqrt(u1 ** 2 + 4)) / 2.0).reshape(u1.size, 1) * u1_hat
        u1_minu = ((u1 - np.sqrt(u1 ** 2 + 4)) / 2.0).reshape(u1.size, 1) * u1_hat
        u2_plus = ((u2 + np.sqrt(u2 ** 2 + 4)) / 2.0).reshape(u2.size, 1) * u2_hat
        u2_minu = ((u2 - np.sqrt(u2 ** 2 + 4)) / 2.0).reshape(u2.size, 1) * u2_hat

        # Shape = [len(t), N_sources, [+, -], [E, N]]
        N_sources = 2
        u_lensed = np.zeros((len(t), N_sources, 2, 2), dtype=float)

        u_lensed[:, 0, 0, :] = u1_plus
        u_lensed[:, 0, 1, :] = u1_minu
        u_lensed[:, 1, 0, :] = u2_plus
        u_lensed[:, 1, 1, :] = u2_minu

        return u_lensed

    def get_astrometry(self, t, ast_filt_idx=0):
        '''
        Position of the observed (unresolved) source position in Einstein radii.

        Parameters
        ----------
        t: array_like
            Array of times to model.


        Returns
        -------
        model_pos : array_like
            Array of vector positions of the centroid at each t_obs.
        '''
        # Get the lensed positions ang ampflications
        # for all 4 images (2 per source)
        u_lensed4 = self.get_resolved_astrometry(t)
        A_lensed4 = self.get_resolved_amplification(t)

        # Sum over the two (+/-) images for each source.
        u1_lensed = np.sum(u_lensed4[:, 0, :, :] * A_lensed4[:, 0, :, np.newaxis], axis=1)
        u2_lensed = np.sum(u_lensed4[:, 1, :, :] * A_lensed4[:, 1, :, np.newaxis], axis=1)
        u1_lensed /= np.sum(A_lensed4[:, 0, :, np.newaxis], axis=1)
        u2_lensed /= np.sum(A_lensed4[:, 1, :, np.newaxis], axis=1)

        # Calculate un-magnified fluxes. Note, we ignore blended flux entirely.
        f1 = mag2flux(self.mag_src_pri[ast_filt_idx])
        f2 = mag2flux(self.mag_src_sec[ast_filt_idx])

        u_lensed = (u1_lensed * f1 + u2_lensed * f2) / (f1 + f2)

        return u_lensed

    def get_centroid_shift(self, t, ast_filt_idx=0):
        raise RuntimeError(
            "Astrometry is not supported on this object: " +
            str(self.__class__))


class BSPL_PhotAstrom(BSPL, PSPL_PhotAstrom):
    photometryFlag = True
    astrometryFlag = True

    def get_resolved_astrometry_unlensed(self, t):
        """Get the astrometry of the source if the lens didn't exist.

        Returns
        -------
        xS_resolved_unlensed : numpy array, [shape = len(t_obs), N_sources, 2]
            The unlensed positions of the sources in arcseconds.

        In other words,
            xS[0, 0, :] returns the 2D sky position of the
            first source at the first time.
        Similarly,
            xS[0, 1, :] returns the 2D sky position of the
            second source at the first time.
        """
        # Equation of motion for just the background source.
        dt1_in_years = (t - self.t0) / days_per_year
        #dt2_in_years = (t - self.t0_sec) / days_per_year
        #Test. call function. two position vectors on the sky calculated. Subtract the two and make sure that the separation stays the same in the no orbits case. 


        # Calculate position vs. time in arcsec

        
        if self.orbitFlag=='linear':
            xS1_unlens = self.xS0_pri + np.outer(dt1_in_years, self.muS) * 1e-3
            xS2_unlens = self.xS0_sec + np.outer(dt1_in_years, self.muS_sec) * 1e-3
        elif self.orbitFlag == 'accelerated':
            xS1_unlens = self.xS0_pri + np.outer(dt1_in_years, self.muS) * 1e-3
            xS2_unlens = self.xS0_sec + np.outer(dt1_in_years, self.muS_sec) * 1e-3 + np.outer((0.5*(dt1_in_years**2)), self.acc) * 1e-3
        elif self.orbitFlag == 'circular':
            xS1_unlens = np.zeros((len(t), 2), dtype=float)
            xS2_unlens = np.zeros((len(t), 2), dtype=float)
            orb = orbits.Orbit()
            orb.w = self.w
            orb.o = self.o
            orb.i = self.i
            orb.e = self.e
            orb.p = self.p
            orb.tp = self.tp
            orb.aleph = self.aleph *1e-3
            orb.aleph2 = self.aleph_sec*1e-3
            orb.vx = self.vx
            orb.vy = self.vy
            orb.x0 = self.x0
            orb.y0 = self.y0
            (x, y, x2, y2) = orb.oal2xy(t, self.t0)
            xS1_unlens[:, 0] = x
            xS1_unlens[:, 1] = y
            xS2_unlens[:, 0] = x2
            xS2_unlens[:, 1] = y2
        else:    
            xS1_unlens = self.xS0_pri + np.outer(dt1_in_years, self.muS) * 1e-3
            xS2_unlens = self.xS0_sec + np.outer(dt1_in_years, self.muS) * 1e-3

            


    
        N_sources = 2
        xS_unlensed = np.zeros((len(t), N_sources, 2), dtype=float)

        xS_unlensed[:, 0, :] = xS1_unlens
        xS_unlensed[:, 1, :] = xS2_unlens
        

        if self.parallaxFlag:
            parallax_vec = parallax_in_direction(self.raL, self.decL, t)  # mas
            xS_unlensed += (self.piS * parallax_vec[:, np.newaxis, :]) * 1e-3  # arcsec

        return xS_unlensed

    def get_astrometry_unlensed(self, t, ast_filt_idx=0):
        """Get the astrometry of the sources if the lens didn't exist.

        Returns
        -------
        xS_unlensed : numpy array, dtype=float, shape = len(t_obs) x 2 directions
            | The unlensed positions of the combined sources in arcseconds.
            | Shape = [len(t), 2 directions]
        """
        xS_unlens_both = self.get_resolved_astrometry_unlensed(t)
        xS1_unlens = xS_unlens_both[:, 0, :]
        xS2_unlens = xS_unlens_both[:, 1, :]

        # Calculate un-magnified fluxes. Note, we ignore blended flux entirely.
        f1 = mag2flux(self.mag_src_pri[ast_filt_idx])
        f2 = mag2flux(self.mag_src_sec[ast_filt_idx])

        # Flux-weighted centroid.
        xS_unlensed = (xS1_unlens * f1 + xS2_unlens * f2) / (f1 + f2)

        return xS_unlensed

    def get_resolved_astrometry(self, t):
        """Parallax: For each source, get the x, y astrometry for the 
        two lensed source images. For each source, we label the two
        images as plus and minus.

        Returns
        -------
        xS_resolved : numpy array 
            [shape = len(t), len(sources), 2, 2]

        Notes
        -----
        For each time t and each source, we have:
            * xS_plus is the vector position of the plus image.
            * xS_minus is the vector position of the minus image.

        In other words, 
            xS[0, 0, 0, :] returns the 2D sky position of the 
            first source's "plus" image at the first time. 
        Similarly, 
            xS[0, 0, 1, :] returns the 2D sky position of the 
            first source's "minus" image at the first time. 
        """

        # Shape is [len(t), 2]
        xL = self.get_lens_astrometry(t)

        u_vec = self.get_u(t)
        u_vec1 = u_vec[:, 0, :]
        u_vec2 = u_vec[:, 1, :]

        u1 = np.linalg.norm(u_vec1, axis=1)
        u2 = np.linalg.norm(u_vec2, axis=1)

        u1_hat = (u_vec1.T / u1).T
        u2_hat = (u_vec2.T / u2).T

        u1_plus = ((u1 + np.sqrt(u1 ** 2 + 4)) / 2.0).reshape(u1.size, 1) * u1_hat
        u1_minu = ((u1 - np.sqrt(u1 ** 2 + 4)) / 2.0).reshape(u1.size, 1) * u1_hat
        u2_plus = ((u2 + np.sqrt(u2 ** 2 + 4)) / 2.0).reshape(u2.size, 1) * u2_hat
        u2_minu = ((u2 - np.sqrt(u2 ** 2 + 4)) / 2.0).reshape(u2.size, 1) * u2_hat

        # Lensed Source Images - Lens Image
        xSL1_plus = u1_plus * self.thetaE_amp  # in mas
        xSL1_minu = u1_minu * self.thetaE_amp  # in mas
        xSL2_plus = u2_plus * self.thetaE_amp  # in mas
        xSL2_minu = u2_minu * self.thetaE_amp  # in mas

        # Shape = [len(t), N_sources, [+, -], [E, N]]
        N_sources = 2
        xSL = np.zeros((len(t), N_sources, 2, 2), dtype=float)

        xSL[:, 0, 0, :] = xSL1_plus
        xSL[:, 0, 1, :] = xSL1_minu
        xSL[:, 1, 0, :] = xSL2_plus
        xSL[:, 1, 1, :] = xSL2_minu

        # xS = xL + xSL = xL + (xS - xL)
        xS_res = xL[:, np.newaxis, np.newaxis, :] + (xSL * 1e-3)  # arcsec

        return xS_res

    def get_astrometry(self, t, ast_filt_idx=0):
        """Parallax: Get unresolved astrometry for binary source, point lens.

        Parameters
        ----------
        t: 
            Array of times in MJD.DDD

        Returns
        -------
        xS_lensed
            Returns flux-weighted average of lensed source positions.
        """
        xS_unlens_both = self.get_resolved_astrometry_unlensed(t)
        xS1_unlens = xS_unlens_both[:, 0, :]  # shape = [len(t), 2]
        xS2_unlens = xS_unlens_both[:, 1, :]

        # Get u for the primary and secondary at all times. 
        u_vec = self.get_u(t)
        u_vec1 = u_vec[:, 0, :]
        u_vec2 = u_vec[:, 1, :]

        u1 = np.linalg.norm(u_vec1, axis=1)
        u2 = np.linalg.norm(u_vec2, axis=1)

        # Calculate the shifts for each source.
        thetaS1 = u_vec1 * self.thetaE_amp
        thetaS2 = u_vec2 * self.thetaE_amp

        shift1 = thetaS1 / (u1[:, np.newaxis] ** 2 + 2.0)
        shift2 = thetaS2 / (u2[:, np.newaxis] ** 2 + 2.0)

        xS1_lensed = xS1_unlens + (shift1 * 1e-3)
        xS2_lensed = xS2_unlens + (shift2 * 1e-3)

        # Calculate un-magnified fluxes. Note, we ignore blended flux entirely.
        f1 = mag2flux(self.mag_src_pri[ast_filt_idx])
        f2 = mag2flux(self.mag_src_sec[ast_filt_idx])

        xS_lensed = (xS1_lensed * f1 + xS2_lensed * f2) / (f1 + f2)

        return xS_lensed
        
    def get_astrometry_shift(self, t, ast_filt_idx=0):
        """Parallax: Get unresolved centroid shift (due to lensing) for each of the binary source.

        Parameters
        ----------
        t: 
            Array of times in MJD.DDD

        Returns
        -------
        xS_lensed
            Returns flux-weighted average of lensed source positions.
        """
        xS_unlens_both = self.get_resolved_astrometry_unlensed(t)
        xS1_unlens = xS_unlens_both[:, 0, :]  # shape = [len(t), 2]
        xS2_unlens = xS_unlens_both[:, 1, :]

        # Get u for the primary and secondary at all times. 
        u_vec = self.get_u(t)
        u_vec1 = u_vec[:, 0, :]
        u_vec2 = u_vec[:, 1, :]

        u1 = np.linalg.norm(u_vec1, axis=1)
        u2 = np.linalg.norm(u_vec2, axis=1)
 

        # Calculate the shifts for each source.
        thetaS1 = u_vec1 * self.thetaE_amp
        thetaS2 = u_vec2 * self.thetaE_amp

        shift1 = thetaS1 / (u1[:, np.newaxis] ** 2 + 2.0)
        shift2 = thetaS2 / (u2[:, np.newaxis] ** 2 + 2.0)

        xS1_lensed = xS1_unlens + (shift1 *1e-3)
        xS2_lensed = xS2_unlens + (shift2 *1e-3)

        
        N_sources = 2
        xS_lensed = np.zeros((len(t), N_sources, 2), dtype=float)

        xS_lensed[:, 0, :] = xS1_lensed
        xS_lensed[:, 1, :] = xS2_lensed
        

        return xS_lensed

    def get_centroid_shift(self, t, ast_filt_idx=0):
        """Parallax: Get the centroid shift (in arcsec) for a list of
        observation times (in MJD).

        Returns the flux-weighted centroid of all the sources lensed images. 

        Parameters
        ----------
        t: 
            Array of times in MJD.DDD

        Returns
        -------
        centroid_shift : numpy array 
            [shape = len(t), 2]
        """

        xS = self.get_astrometry(t, ast_filt_idx=ast_filt_idx)
        xS_unlensed = self.get_astrometry_unlensed(t, ast_filt_idx=ast_filt_idx)

        shift = xS - xS_unlensed

        return shift

    
    def dexanimate(self, tE, time_steps, frame_time, name, size, zoom,
                astrometry, type, loc):
        """ Produces animation of microlensing event. 
        This function takes the PSPL and makes an animation, the input variables are as follows

        Parameters
        ----------

        tE: 
            number of einstein crossings times before/after the peak you want the animation to plot
                e.g tE = 2 => graph will go from -2 tE to 2 tE
        time_steps:
            number of time steps before/after peak, so total number of time steps will 
            be 2 times this value
        frame_time:
            times in ms of each frame in the animation
        name: string
            the animation will be saved as name.html
        size: list
            [horizontal, vertical] cm's
        zoom:
            # of einstein radii plotted in vertical direction
        """
        times = np.array(range(-time_steps, time_steps + 1, 1))
        tau = tE * times / (-times[0])
        t = self.t0 + (tau * self.tE)

        l = self.get_lens_astrometry(t)
                    
        xS_unlensed = self.get_resolved_astrometry_unlensed(t)
        source1 =xS_unlensed[:, 0, :]
        source2 = xS_unlensed[:, 1, :]  


        xS_lensed = self.get_astrometry_shift(t)
        image1 = xS_lensed[:, 0, :]
        image2 =  xS_lensed[:, 1, :]
                    
    
        xSL = self.get_resolved_astrometry(t)
        xSL1_plus = xSL[:, 0, 0, :] 
        xSL1_minu = xSL[:, 0, 1, :] 
        xSL2_plus = xSL[:, 1, 0, :]
        xSL2_minu = xSL[:, 1, 1, :]

        


        fig = plt.figure(figsize=[size[0], size[1] + 0.5])  # sets up the figure
        ax1 = fig.add_subplot(2, 1, 1)
        fig.subplots_adjust(hspace=.5)

        if type == 'resolved_unlensed':
            s1_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, label="Unlensed Primary Source ", color='grey', linewidth=2)
            s1_line2, = ax1.plot([], '-', markersize=size[0] * 0.3, color='grey', linewidth=2)
            s2_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, label="Unlensed Secondary Source ", color='purple', linewidth=2)
            s2_line2, = ax1.plot([], '-', markersize=size[0] * 0.3, color='purple', linewidth=2)
            
        elif type == 'unresolved_lensed':
            s1_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, label="Lensed Primary Source ", color='grey', linewidth=2)
            s1_line2, = ax1.plot([], '-', markersize=size[0] * 0.3, color='grey', linewidth=2)
            s2_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, label="Lensed Secondary Source ", color='purple', linewidth=2)
            s2_line2, = ax1.plot([], '-', markersize=size[0] * 0.3, color='purple', linewidth=2)
            
        elif type == 'resolved_unlensed_unresolved_lensed':
            s1_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, label="Unlensed Primary Source ", color='grey', linewidth=2)
            s1_line2, = ax1.plot([], '-', markersize=size[0] * 0.3, color='grey', linewidth=2)
            s2_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, label="Unlensed Secondary Source ", color='purple', linewidth=2)
            s2_line2, = ax1.plot([], '-', markersize=size[0] * 0.3, color='purple', linewidth=2)
            i1_line1, = ax1.plot([], '.', markersize=size[0] * 1.0, label="Lensed Primary Source ", color='grey', linewidth=2)
            i1_line2, = ax1.plot([], '--', markersize=size[0] * 0.2, color='grey', linewidth=2)
            i2_line1, = ax1.plot([], '.', markersize=size[0] * 1.0, label="Lensed Secondary Source ", color='purple', linewidth=2)
            i2_line2, = ax1.plot([], '--', markersize=size[0] * 0.2, color='purple', linewidth=2)
            
        elif type == 'resolved_unlensed_resolved_lensed':    
            s1_line1, = ax1.plot([], '.', markersize=size[0] * 1.3,  color='grey', linewidth=2)
            s1_line2, = ax1.plot([], '-', markersize=size[0] * 0.3, label="Unlensed Primary Source ", color='grey', linewidth=2)
            s2_line1, = ax1.plot([], '.', markersize=size[0] * 1.3, color='purple', linewidth=2)
            s2_line2, = ax1.plot([], '-', markersize=size[0] * 0.3, label="Unlensed Secondary Source ", color='purple', linewidth=2)
                        
            i1_plus_line1, = ax1.plot([], '.', markersize=size[0] * 1.0, color='grey', linewidth=2)
            i1_plus_line2, = ax1.plot([], '--', markersize=size[0] * 0.2, label="Lensed Primary Source Major Image ", color='grey', linewidth=2)
            i1_minus_line1, = ax1.plot([], '.', markersize=size[0] * 1.0, color='grey', linewidth=2)
            i1_minus_line2, = ax1.plot([], ':', markersize=size[0] * 0.2, label="Lensed Primary Source Minor Image", color='grey', linewidth=2)
                        
            i2_plus_line1, = ax1.plot([], '.', markersize=size[0] * 1.0, color='purple', linewidth=2)
            i2_plus_line2, = ax1.plot([], '--', markersize=size[0] * 0.2, label="Lensed Secondary Source Major Image ", color='purple', linewidth=2)
            i2_minus_line1, = ax1.plot([], '.', markersize=size[0] * 1.0,  color='purple', linewidth=2)
            i2_minus_line2, = ax1.plot([], ':', markersize=size[0] * 0.2, label="Lensed Secondary Source Minor Image ", color='purple', linewidth=2)    
    


        l_line1, = ax1.plot([], '.', markersize=size[0] * 0.7,  color='k', linewidth=2)
        l_line2, = ax1.plot([], '-', markersize=size[0] * 0.2, label="Lens", color='k', linewidth=2)




        ax1.set_xlabel('RA (")')
        ax1.set_ylabel('Dec (")')
        ax1.set_xlim(zoom, -zoom)
        ax1.set_ylim(-zoom, zoom)
        
        #ax1.set_xlim(zoom, -zoom)
        #ax1.set_ylim(-zoom, zoom)

#        ax1.set_xlim(1.5, 0.25)
 #       ax1.set_ylim(0.1, 1.5)

        
#        ax1.set_xlim(1.1, 0.9)
 #       ax1.set_ylim(0.6, 1.5)
        #plt.gca().set_xlim(1.1, 0.9)
        #plt.gca().set_ylim(0.75, 1.3)

        ax1.legend(fontsize=12, loc=loc)


                    
                        
            


                    
        if astrometry == "yes":
            line = [s1_line1, s1_line2, s2_line1, s2_line2, l_line1, l_line2]
            def update(i, source1, source2, lens, tau, line):
                    # print(str(i) + ", ", end='', flush=True)
                    line[0].set_data(source1[i, 0], source1[i, 1])
                    line[1].set_data(source1[:i + 1, 0], source1[:i + 1, 1])
                    line[2].set_data(source2[i, 0], source2[i, 1])
                    line[3].set_data(source2[:i + 1, 0], source2[:i + 1, 1])
                    line[4].set_data(lens[i, 0], lens[i, 1])
                    line[5].set_data(lens[:i + 1, 0], lens[:i + 1, 1])
                    return line
            if type == 'resolved_unlensed':
    
                """
                FuncAnimation takes in the following arguments
    
                fig = background figure
    
                update = function that is called every frame
    
                len(tau) = the number of frames, so now the first argument passed 
                    into update (i) will be (0,1,2...len(tau))
    
                fargs specifies the other arguments to pass into update
    
                blit being true means that each frame, if there are elements of it 
                    that don't change from the last frame,
                    it won't replot them, so this makes it faster
    
                interval = number of milliseconds between each frame
                alternatively you can specify fps in save after after the file name
    
                """
                ani = animation.FuncAnimation(fig, update, len(tau),
                                              fargs=[source1, source2, l, tau,
                                                     line],
                                              blit=True, interval=frame_time)
                ani.save("%s.mp4" % name, writer="ffmpeg")
                
            elif type == 'unresolved_lensed': 
               
                ani = animation.FuncAnimation(fig, update, len(tau),
                                              fargs=[image1, image2, l, tau,
                                                     line],
                                              blit=True, interval=frame_time)
                ani.save("%s.mp4" % name, writer="ffmpeg")
                
            elif type == 'resolved_unlensed_unresolved_lensed':
                line = [s1_line1, s1_line2, s2_line1, s2_line2,i1_line1, i1_line2, i2_line1, i2_line2, l_line1, l_line2]
                def update(i, source1, source2, image1, image2, lens, tau, line):
                    # print(str(i) + ", ", end='', flush=True)
                    line[0].set_data(source1[i, 0], source1[i, 1])
                    line[1].set_data(source1[:i + 1, 0], source1[:i + 1, 1])
                    line[2].set_data(source2[i, 0], source2[i, 1])
                    line[3].set_data(source2[:i + 1, 0], source2[:i + 1, 1])
                    line[4].set_data(image1[i, 0], image1[i, 1])
                    line[5].set_data(image1[:i + 1, 0], image1[:i + 1, 1])
                    line[6].set_data(image2[i, 0], image2[i, 1])
                    line[7].set_data(image2[:i + 1, 0], image2[:i + 1, 1])
                    line[8].set_data(lens[i, 0], lens[i, 1])
                    line[9].set_data(lens[:i + 1, 0], lens[:i + 1, 1])
                    return line
                ani = animation.FuncAnimation(fig, update, len(tau), fargs=[source1, source2,image1, image2, l, tau, line],
                                              blit=True, interval=frame_time)
                ani.save("%s.mp4" % name, writer="ffmpeg")
                
            elif type == 'resolved_unlensed_resolved_lensed':
                line = [s1_line1, s1_line2, s2_line1, s2_line2, i1_plus_line1, i1_plus_line2, i1_minus_line1, i1_minus_line2, i2_plus_line1, i2_plus_line2, 
                        i2_minus_line1,  i2_minus_line2, l_line1, l_line2]
                def update(i, source1, source2, xSL1_plus, xSL1_minu, xSL2_plus, xSL2_minu, lens, tau, line):
                    # print(str(i) + ", ", end='', flush=True)
                    line[0].set_data(source1[i, 0], source1[i, 1])
                    line[1].set_data(source1[:i + 1, 0], source1[:i + 1, 1])
                    line[2].set_data(source2[i, 0], source2[i, 1])
                    line[3].set_data(source2[:i + 1, 0], source2[:i + 1, 1])
                    line[4].set_data(xSL1_plus[i, 0], xSL1_plus[i, 1])
                    line[5].set_data(xSL1_plus[:i + 1, 0], xSL1_plus[:i + 1, 1])
                    line[6].set_data(xSL1_minu[i, 0], xSL1_minu[i, 1])
                    line[7].set_data(xSL1_minu[:i + 1, 0], xSL1_minu[:i + 1, 1])
                    line[8].set_data(xSL2_plus[i, 0], xSL2_plus[i, 1])
                    line[9].set_data(xSL2_plus[:i + 1, 0], xSL2_plus[:i + 1, 1])
                    line[10].set_data(xSL2_minu[i, 0], xSL2_minu[i, 1])
                    line[11].set_data(xSL2_minu[:i + 1, 0], xSL2_minu[:i + 1, 1])
                    line[12].set_data(lens[i, 0], lens[i, 1])
                    line[13].set_data(lens[:i + 1, 0], lens[:i + 1, 1])
                    return line
                ani = animation.FuncAnimation(fig, update, len(tau), fargs=[source1, source2,xSL1_plus, xSL1_minu, xSL2_plus, xSL2_minu, l, tau, line],
                                              blit=True, interval=frame_time)
                ani.save("%s.mp4" % name, writer="ffmpeg")   
        else:
            print("Only astrometry for now")

        return ani
                

class BSPL_Parallax(PSPL_Parallax):
    parallaxFlag = True


class BSPL_noParallax(PSPL_noParallax):
    parallaxFlag = False


# --------------------------------------------------
#
# Parameterization Class Family - BSPL
#
# --------------------------------------------------

class BSPL_PhotParam1(PSPL_Param):
    """BSPL model for photometry only

    A Binary point Source Point Lens model for microlensing.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float
        Angular distance between the lens and source on the plane of the
        sky at closest approach in units of thetaE. It can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE: float
        Einstein crossing time. [MJD]
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    sep: float
        Angular separation of the source scondary from the
        source primary (in units of thetaE).
    phi: float
        Angle made between the binary axis and the relative proper motion vector,
        measured in degrees.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_S1 + f_S2) / (f_S1 + f_s2 + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.

    """

    fitter_param_names = ['t0', 'u0_amp', 'tE', 'piE_E', 'piE_N',
                          'sep', 'phi']
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']

    paramAstromFlag = False
    paramPhotFlag = True
    orbitFlag = 'none'

    def __init__(self, t0, u0_amp, tE, piE_E, piE_N,
                 sep, phi, mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=None, decL=None):
        self.t0 = t0  # time of closest approach for system=primary pos
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.b_sff = b_sff
        self.raL = raL
        self.decL = decL

        # Binary source parameters.
        self.sep = sep  # mas
        self.phi = phi
        self.mag_src_pri = mag_src_pri
        self.mag_src_sec = mag_src_sec

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        # Calculate the microlensing parallax amplitude
        self.piE_amp = np.linalg.norm(self.piE)
        self.piE_E, self.piE_N = self.piE

        # Baseline magnitude
        self.mag_base = self.mag_src_pri \
                        + self.mag_src_sec \
                        + 2.5 * np.log10(self.b_sff)

        # Calculate the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.u0_amp)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        #####
        # Derived binary source parameters.
        #####
        # Calculate the angle between muRel and the binary axis
        # in radians.
        self.phi_rad = self.phi * np.pi / 180.0

        # For convenience, calculate the vector pointing
        # from the geometric center of the lens binary system
        # to the primary mass. This requires a combination of
        # phi and piE_hat.
        self.phi_piE_rad = np.arctan2(self.piE[0], self.piE[1])

        # Note that phi_rho1 is the same alpha in our astrometry model;
        # however, here we don't have North as a reference.
        self.phi_rho1_rad = self.phi_piE_rad - self.phi_rad

        # Primary -- at origin
        self.t0_pri = self.t0
        self.u0_amp_pri = self.u0_amp
        self.u0_pri = self.u0

        # Secondary
        sep_vec = self.sep * np.array((np.sin(self.phi_rho1_rad),
                                       np.cos(self.phi_rho1_rad)))  # mas

        # Closest approach time
        self.u0_amp_sec = self.u0_amp_pri + np.dot(sep_vec, self.u0_hat)
        self.u0_sec = self.u0_amp_sec * self.u0_hat
        s_murelhat = np.dot(sep_vec, self.muRel_hat)
        self.t0_sec = self.t0_pri - (s_murelhat * self.tE)

        return

    
    
class BSPL_PhotAstromParam1(PSPL_Param):
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    mL: float
        Mass of the lens (Msun)
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    beta: float
        Angular distance between the lens and primary source on the
        plane of the sky (mas). Can be
            * positive (u0_amp > 0 when u0_hat[0] < 0) or
            * negative (u0_amp < 0 when u0_hat[0] > 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
    muL_E: float
        RA Lens proper motion (mas/yr)
    muL_N: float
        Dec Lens proper motion (mas/yr)
    muS_E: float
        RA Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    muS_N: float
        Dec Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_S1 + f_S2) / (f_S1 + f_s2 + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['mL', 't0', 'beta', 'dL', 'dL_dS',
                          'xS0_E', 'xS0_N',
                          'muL_E', 'muL_N',
                          'muS_E', 'muS_N',
                          'sep', 'alpha']
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']
    additional_param_names = ['dS', 'tE', 'u0_amp',
                              'thetaE_E', 'thetaE_N',
                              'piE_E', 'piE_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'none'

    def __init__(self, mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 sep, alpha,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=None, decL=None):
        self.t0 = t0  # time of closest approach for system=primary pos
        self.mL = mL
        self.xS0 = np.array([xS0_E, xS0_N])  # position of source system=primary
        self.beta = beta
        self.muL = np.array([muL_E, muL_N])
        self.muS = np.array([muS_E, muS_N])
        self.dL = dL
        self.dL_dS = dL_dS
        self.dS = self.dL / self.dL_dS
        self.b_sff = np.array(b_sff)
        self.raL = raL
        self.decL = decL

        # Binary source parameters.
        self.sep = sep  # mas
        self.alpha = alpha
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.mag_src_pri = np.array(mag_src_pri)
        self.mag_src_sec = np.array(mag_src_sec)
        self.fratio_bin = 10.0 ** ((self.mag_src_sec - self.mag_src_pri) / -2.5)

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        flux_pri = mag2flux(self.mag_src_pri)
        flux_sec = mag2flux(self.mag_src_sec)
        self.mag_base = flux2mag(flux_pri + flux_sec) + 2.5 * np.log10(self.b_sff)

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (self.dL * units.pc)) - (1.0 / (self.dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_E, self.muRel_N = self.muRel
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr

        self.muS_E, self.muS_N = self.muS
        self.muL_E, self.muL_N = self.muL

        # Calculate the Einstein radius
        thetaE = units.rad * np.sqrt(
            (4.0 * const.G * mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.sun = units.M_sun
        self.c = const.c             
        self.G=const.G
        self.inv = inv_dist_diff           #remove later
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat
        self.thetaE_E, self.thetaE_N = self.thetaE

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0_amp = self.beta / self.thetaE_amp  # in Einstein units
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens
        # (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat
        self.piE_E, self.piE_N = self.piE

        # Calculate the Einstein crossing time. (days)
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year

        #####
        # Derived binary source parameters.
        #####
        # Primary -- at origin
        self.t0_pri = self.t0
        self.xS0_pri = self.xS0
        self.u0_amp_pri = self.u0_amp
        self.u0_pri = self.u0

        # Secondary
        sep_vec = self.sep * np.array((np.sin(self.alpha_rad),
                                       np.cos(self.alpha_rad)))  # mas

        #Delete line later
        self.sep_vec=sep_vec
                     
        # Closest approach time
        self.u0_amp_sec = self.u0_amp_pri + (np.dot(sep_vec, self.u0_hat) / self.thetaE_amp)
        self.u0_sec = self.u0_amp_sec * self.u0_hat
        s_murelhat = np.dot(sep_vec, self.muRel_hat)
        #self.t0_sec = self.t0_pri - (s_murelhat * days_per_year / self.muRel_amp)
        self.xS0_sec = self.xS0_pri + (sep_vec * 1e-3) - (s_murelhat * 1e-3 * self.muRel_hat)

        return



class BSPL_PhotAstromParam2(PSPL_Param):
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame. This should be the position of the source primary.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    muS_N : float
        Dec Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_{base} = f_{src1} + f_{src2} + f_{blend}`
        such that
            :math:`b_sff = (f_{src1}+ f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['t0', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'xS0_E', 'xS0_N',
                          'muS_E', 'muS_N',
                          'sep', 'alpha']
    
    phot_param_names = ['fratio_bin', 'mag_base', 'b_sff']
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'none'


    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 raL=None, decL=None):
        self.t0 = t0  # time of closest approach for system=primary pos
        self.u0_amp = u0_amp
        self.tE = tE
        self.thetaE_amp = thetaE
        self.piS = piS
        self.piE = np.array([piE_E, piE_N])
        self.xS0 = np.array([xS0_E, xS0_N])  # position of source system=primary
        self.muS = np.array([muS_E, muS_N])
        self.mag_base = np.array(mag_base)
        self.b_sff = np.array(b_sff)

        self.raL = raL
        self.decL = decL

        # Binary source parameters.
        self.sep = sep  # mas
        self.alpha = alpha
        self.fratio_bin = np.array(fratio_bin)
        self.alpha_rad = self.alpha * np.pi / 180.0
        self.mag_src_pri = mag_base - 2.5 * np.log10(b_sff) + 2.5 * np.log10(1.0 + fratio_bin)
        self.mag_src_sec = mag_base - 2.5 * np.log10(b_sff) + 2.5 * np.log10(1.0 + (1.0 / fratio_bin))

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        # Derived quantities 
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)

        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens
        # (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        #####
        # Derived binary source parameters.
        #####
        # Primary -- at origin
        self.t0_pri = self.t0
        self.xS0_pri = self.xS0
        self.u0_amp_pri = self.u0_amp
        self.u0_pri = self.u0

        # Secondary
        sep_vec = self.sep * np.array((np.sin(self.alpha_rad),
                                       np.cos(self.alpha_rad)))  # mas

        # Closest approach time and distance
        self.u0_amp_sec = self.u0_amp_pri + (np.dot(sep_vec, self.u0_hat) / self.thetaE_amp)
        self.u0_sec = self.u0_amp_sec * self.u0_hat
        s_murelhat = np.dot(sep_vec, self.muRel_hat)
        #self.t0_sec = self.t0_pri - (s_murelhat * days_per_year / self.muRel_amp)
        self.xS0_sec = self.xS0_pri + (sep_vec * 1e-3) - (s_murelhat * 1e-3 * self.muRel_hat)

        return


class BSPL_PhotAstromParam3(PSPL_Param):
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE : float
        Einstein crossing time (days).
    log10_thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame. This should be the position of the source primary.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    muS_N : float
        Dec Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_base = f_{src1{ + f_{src2{ + f_{blend}`
        such that
            :math:`b_sff = (f_{src1} + f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['t0', 'u0_amp', 'tE', 'log10_thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'xS0_E', 'xS0_N',
                          'muS_E', 'muS_N',
                          'sep', 'alpha']
    phot_param_names = ['fratio_bin', 'mag_base', 'b_sff']
    additional_param_names = ['thetaE_amp', 'mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'none'
    def __init__(self, t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 raL=None, decL=None):
        self.t0 = t0  # time of closest approach for system=primary pos
        self.u0_amp = u0_amp
        self.tE = tE
        self.thetaE_amp = 10 ** log10_thetaE
        self.piS = piS
        self.piE = np.array([piE_E, piE_N])
        self.xS0 = np.array([xS0_E, xS0_N])  # position of source system=primary
        self.muS = np.array([muS_E, muS_N])
        self.mag_base = np.array(mag_base)
        self.b_sff = np.array(b_sff)
        self.raL = raL
        self.decL = decL

        # Binary source parameters.
        self.sep = sep  # mas
        self.alpha = alpha
        self.fratio_bin = np.array(fratio_bin)
        self.alpha_rad = self.alpha * np.pi / 180.0

        self.mag_src_pri = self.mag_base \
                           - 2.5 * np.log10(self.b_sff) \
                           + 2.5 * np.log10(1.0 + self.fratio_bin)
        self.mag_src_sec = self.mag_base \
                           - 2.5 * np.log10(self.b_sff) \
                           + 2.5 * np.log10(1.0 + (1.0 / self.fratio_bin))

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        # Derived quantities
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)

        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens
        # (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        #####
        # Derived binary source parameters.
        #####
        # Primary -- at origin
        self.t0_pri = self.t0
        self.xS0_pri = self.xS0
        self.u0_amp_pri = self.u0_amp
        self.u0_pri = self.u0

        # Secondary
        sep_vec = self.sep * np.array((np.sin(self.alpha_rad),
                                       np.cos(self.alpha_rad)))  # mas

        # Closest approach time and distance
        self.u0_amp_sec = self.u0_amp_pri + (np.dot(sep_vec, self.u0_hat) / self.thetaE_amp)
        self.u0_sec = self.u0_amp_sec * self.u0_hat
        s_murelhat = np.dot(sep_vec, self.muRel_hat)
        #self.t0_sec = self.t0_pri - (s_murelhat * days_per_year / self.muRel_amp)
        self.xS0_sec = self.xS0_pri + (sep_vec * 1e-3) - (s_murelhat * 1e-3 * self.muRel_hat)

        return




class BSPL_GP_PhotParam1(BSPL_PhotParam1):
    """BSPL model for photometry only, with GP.

    A Binary point Source Point Lens model for microlensing.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Parameters
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float
        Angular distance between the lens and source on the plane of the
        sky at closest approach in units of thetaE. It can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE: float
        Einstein crossing time. [MJD]
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    sep: float
        Angular separation of the source scondary from the
        source primary (in units of thetaE).
    phi: float
        Angle made between the binary axis and the relative proper motion vector,
        measured in degrees.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N).`
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    gp_log_sigma: float
        Guassian process :math:`log(\sigma)` for the Matern 3/2 kernel. 
    gp_rho: float
        Guassian process :math:`{\\rho}` for the Matern 3/2 kernel.
    gp_log_omega04_S0: float
        Guassian process :math:`log(\omega_0^4 * S_0)` from the SHO kernel.
    gp_log_omega0: float
        Guassian process :math:`log(\omega_0)` from the SHO kernel.

    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, piE_E, piE_N,
                 sep, phi, mag_src_pri, mag_src_sec,
                 b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        self.gp_log_sigma = gp_log_sigma
        self.gp_rho = gp_rho
        self.gp_log_omega04_S0 = gp_log_omega04_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, tE, piE_E, piE_N,
                 sep, phi, mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=raL, decL=decL)
        
        self.gp_log_rho = {}
        for key, val in self.gp_rho.items():
            self.gp_log_rho[key] = np.log(val)

        self.gp_log_S0 = {}
        for key, val in self.gp_log_omega04_S0.items():
            self.gp_log_S0[key] = self.gp_log_omega04_S0[key] - 4 * self.gp_log_omega0[key]

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True
            
        return


class BSPL_GP_PhotAstromParam1(BSPL_PhotAstromParam1):
    """BSPL model for astrometry and photometry with GP - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    mL: float
        Mass of the lens (Msun)
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    beta: float
        Angular distance between the lens and primary source on the
        plane of the sky (mas). Can be
            * positive (u0_amp > 0 when u0_hat[0] < 0) or
            * negative (u0_amp < 0 when u0_hat[0] > 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
    muL_E: float
        RA Lens proper motion (mas/yr)
    muL_N: float
        Dec Lens proper motion (mas/yr)
    muS_E: float
        RA Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    muS_N: float
        Dec Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    gp_log_sigma: float
        Guassian process :math:`log(\sigma)` for the Matern 3/2 kernel.
    gp_rho: float
        Guassian process :math:`{\\rho}` for the Matern 3/2 kernel.
    gp_log_omega04_S0: float
        Guassian process :math:`log(\omega_0^4 * S_0)` from the SHO kernel.
    gp_log_omega0: float
        Guassian process :math:`log(\omega_0)` from the SHO kernel.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 sep, alpha,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        self.gp_log_sigma = gp_log_sigma
        self.gp_rho = gp_rho
        self.gp_log_omega04_S0 = gp_log_omega04_S0
        self.gp_log_omega0 = gp_log_omega0
        
        super().__init__(mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 sep, alpha,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=raL, decL=decL)
        
        self.gp_log_rho = {}
        for key, val in self.gp_rho.items():
            self.gp_log_rho[key] = np.log(val)

        self.gp_log_S0 = {}
        for key, val in self.gp_log_omega04_S0.items():
            self.gp_log_S0[key] = self.gp_log_omega04_S0[key] - 4 * self.gp_log_omega0[key]

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return


class BSPL_GP_PhotAstromParam2(BSPL_PhotAstromParam2):
    """BSPL model for astrometry and photometry with GP - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame. This should be the position of the source primary.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    muS_N : float
        Dec Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_{base} = f_{src1} + f_{src2} + f_{blend}`
        such that
            :math:`b_{sff} = (f_{src1} + f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:` b_{sff} = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    gp_log_sigma: float
        Guassian process :math:`log(\sigma)` for the Matern 3/2 kernel.
    gp_rho: float
        Guassian process :math:`{\\rho}` for the Matern 3/2 kernel.
    gp_log_omega04_S0: float
        Guassian process :math:`log(\omega_0^4 * S_0)` from the SHO kernel.
    gp_log_omega0: float
        Guassian process :math:`log(\omega_0)` from the SHO kernel.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):
        
        self.gp_log_sigma = gp_log_sigma
        self.gp_rho = gp_rho
        self.gp_log_omega04_S0 = gp_log_omega04_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 raL=raL, decL=decL)

        self.gp_log_rho = {}
        for key, val in self.gp_rho.items():
            self.gp_log_rho[key] = np.log(val)

        self.gp_log_S0 = {}
        for key, val in self.gp_log_omega04_S0.items():
            self.gp_log_S0[key] = self.gp_log_omega04_S0[key] - 4 * self.gp_log_omega0[key]

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True
        
        return

class BSPL_GP_PhotAstromParam3(BSPL_PhotAstromParam3):
    """
    Point Source Point Lens with GP model for microlensing. This model includes
    proper motions of the source and the source position on the sky.
    It is the same as PSPL_PhotAstromParam4 except it fits for log10(thetaE)
    instead of thetaE.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float
        Angular distance between the lens and source on the plane of the
        sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE: float
        Einstein crossing time (days).
    log10_thetaE: float
        log10 of the size of the Einstein radius in (mas).
    piS: float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    muS_E: float
        RA Source proper motion (mas/yr)
    muS_N: float
        Dec Source proper motion (mas/yr)
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_{base} = f_{src1} + f_{src2} + f_{blend}`
        such that
            :math:`b_{sff} = (f_{src1} + f_{src2}) / ( f_{src1} + f_{src2} + f_{blend})`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:` b_{sff} = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    gp_log_sigma: float
        Guassian process :math:`log(\sigma)` for the Matern 3/2 kernel.
    gp_rho: float
        Guassian process :math:`{\\rho}` for the Matern 3/2 kernel.
    gp_log_omega04_S0: float
        Guassian process :math:`log(\omega_0^4 * S_0)` from the SHO kernel.
    gp_log_omega0: float
        Guassian process :math:`log(\omega_0)` from the SHO kernel.

    Other Parameters
    ----------------
    Note: Required if calculating with parallax
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        self.gp_log_sigma = gp_log_sigma
        self.gp_rho = gp_rho
        self.gp_log_omega04_S0 = gp_log_omega04_S0
        self.gp_log_omega0 = gp_log_omega0

        super().__init__(t0, u0_amp, tE, log10_thetaE, piS,
                         piE_E, piE_N,
                         xS0_E, xS0_N,
                         muS_E, muS_N,
                         sep, alpha, fratio_bin,
                         mag_base, b_sff,
                         raL=raL, decL=decL)

        self.gp_log_rho = {}
        for key, val in self.gp_rho.items():
            self.gp_log_rho[key] = np.log(val)

        self.gp_log_S0 = {}
        for key, val in self.gp_log_omega04_S0.items():
            self.gp_log_S0[key] = self.gp_log_omega04_S0[key] - 4 * self.gp_log_omega0[key]

        # Setup a useful "use_phot_gp" flag.
        self.use_gp_phot = np.zeros(len(self.b_sff), dtype='bool')
        for key in self.gp_log_sigma.keys():
            self.use_gp_phot[key] = True

        return

                     
class BSPL_PhotAstrom_LinOrbs_Param1(BSPL_PhotAstromParam1):
    """BSPL model for astrometry and photometry - physical parameterization - for linear orbits.

A Binary Point Source Point Lens model for microlensing. This model uses a parameterization that depends on only physical quantities such as the proper motions of the lens, the primary source and the secondary source. This is a linear orbit model. 

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    mL: float
        Mass of the lens (Msun)
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    t: float
        Time of Observation
    beta: float
        Angular distance between the lens and primary source on the
        plane of the sky (mas). Can be
            * positive (u0_amp > 0 when u0_hat[0] < 0) or
            * negative (u0_amp < 0 when u0_hat[0] > 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
        
    muL_E: float
        RA Lens proper motion (mas/yr)
    muL_N: float
        Dec Lens proper motion (mas/yr)
    muS_E: float
        RA Source proper motion (mas/yr) for primary source.
    muS_N: float
        Dec Source proper motion (mas/yr) for primary source.
    delta_muS_sec_E: float
        RA Source proper motion (mas/yr) for secondary source.
    delta_muS_sec_N: float
        Dec Source proper motion (mas/yr) for secondary source.
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_S1 + f_S2) / (f_S1 + f_s2 + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    
    fitter_param_names = ['mL', 't0', 't', 'beta', 'dL', 'dL_dS',
                          'xS0_E', 'xS0_N',
                          'muL_E', 'muL_N',
                          'muS_E', 'muS_N',
                          'delta_muS_sec_E', 'delta_muS_sec_N',
                          'sep', 'alpha']
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']
    additional_param_names = ['dS', 'tE', 'u0_amp',
                              'thetaE_E', 'thetaE_N',
                              'piE_E', 'piE_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'linear'

    def __init__(self, mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 sep, alpha,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=None, decL=None):

        super().__init__(mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 sep, alpha,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=raL, decL=decL)

        
        self.delta_muS_sec =  np.array([delta_muS_sec_E, delta_muS_sec_N])
        self.delta_muS_sec_E, self.delta_muS_sec_N = self.delta_muS_sec
        self.muS_sec = self.muS + self.delta_muS_sec 

        self.muRel_sec = self.muS + self.delta_muS_sec - self.muL
        self.muRel_sec_E, self.muRel_sec_N = self.muRel_sec
        self.muRel_sec_amp = np.linalg.norm(self.muRel_sec)  # mas/yr
        self.muRel_sec_hat = self.muRel_sec/self.muRel_sec_amp

        self.tE_sec = self.thetaE_amp/self.muRel_sec_amp * days_per_year

        return


class BSPL_PhotAstrom_AccOrbs_Param1(BSPL_PhotAstrom_LinOrbs_Param1):
    """BSPL model for astrometry and photometry - physical parameterization - for linear orbits.

A Binary Point Source Point Lens model for microlensing. This model uses a parameterization that depends on only physical quantities such as the proper motions of the lens, the primary source and the secondary source. This is a linear orbit model. 

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    mL: float
        Mass of the lens (Msun)
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    t: float
        Time of Observation
    beta: float
        Angular distance between the lens and primary source on the
        plane of the sky (mas). Can be
            * positive (u0_amp > 0 when u0_hat[0] < 0) or
            * negative (u0_amp < 0 when u0_hat[0] > 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
        
    muL_E: float
        RA Lens proper motion (mas/yr)
    muL_N: float
        Dec Lens proper motion (mas/yr)
    muS_E: float
        RA Source proper motion (mas/yr) for primary source.
    muS_N: float
        Dec Source proper motion (mas/yr) for primary source.
    acc_E: 
        Acceleration of the secondary source in the direction of RA (mas/yr^2)
    acc_N: 
        Acceleration of the secondary source in the direction of DEC (mas/yr^2)
    delta_muS_sec_E: float
        RA Source proper motion (mas/yr) for secondary source.
    delta_muS_sec_N: float
        Dec Source proper motion (mas/yr) for secondary source.
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_S1 + f_S2) / (f_S1 + f_s2 + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    fitter_param_names = ['mL', 't0', 't', 'beta', 'dL', 'dL_dS',
                          'xS0_E', 'xS0_N',
                          'muL_E', 'muL_N',
                          'muS_E', 'muS_N',
                          'delta_muS_sec_E', 'delta_muS_sec_N',
                          'acc_E', 'acc_N',
                          'sep', 'alpha']
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']
    additional_param_names = ['dS', 'tE', 'u0_amp',
                              'thetaE_E', 'thetaE_N',
                              'piE_E', 'piE_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'accelerated'

    def __init__(self, mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 acc_E, acc_N,
                 sep, alpha,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=None, decL=None):

        super().__init__(mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 sep, alpha,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=raL, decL=decL)
                     
        self.acc = np.array([acc_E, acc_N])
        self.acc_E, self.acc_N = self.acc
        self.acc_amp = np.linalg.norm(self.acc)
        #self.acc_hat = self.muRel_sec_hat
        if self.acc_E==0.0 and self.acc_N==0.0:
            self.acc_hat = np.array([0.0, 0.0])
        else:
            self.acc_hat = self.acc/self.acc_amp
                     
        return
class BSPL_PhotAstrom_LinOrbs_Param2(BSPL_PhotAstromParam2):
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame. This should be the position of the source primary.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    muS_N : float
        Dec Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    delta_muS_sec_E: float
        RA secondary source proper motion with respect to the primary source (mas/yr)
    delta_muS_sec_N: float
        Dec secondary source proper motion with respect to the primary source (mas/yr)
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_{base} = f_{src1} + f_{src2} + f_{blend}`
        such that
            :math:`b_sff = (f_{src1}+ f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['t0', 'u0_amp', 'tE', 'thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'xS0_E', 'xS0_N',
                          'muS_E', 'muS_N',
                          'delta_muS_sec_E', 'delta_muS_sec_N',
                          'sep', 'alpha']
    phot_param_names = ['fratio_bin', 'mag_base', 'b_sff']
    additional_param_names = ['mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'linear'


    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 raL=None, decL=None):

        super().__init__(t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 raL=raL, decL=decL)
                     
        self.delta_muS_sec = np.array([delta_muS_sec_E, delta_muS_sec_N])
        self.delta_muS_sec_E = delta_muS_sec_E
        self.delta_muS_sec_N = delta_muS_sec_N
        self.muS_sec = self.muS + self.delta_muS_sec 

        self.muRel_sec = self.muS_sec - self.muL
        self.muRel_sec_E, self.muRel_sec_N = self.muRel_sec
        self.muRel_sec_amp = np.linalg.norm(self.muRel_sec)  # mas/yr
        self.muRel_sec_hat = self.muRel_sec/self.muRel_sec_amp


        self.tE_sec = self.thetaE_amp/self.muRel_sec_amp * days_per_year


                     
        return


class BSPL_PhotAstrom_AccOrbs_Param2(BSPL_PhotAstrom_LinOrbs_Param2):
    """BSPL model for astrometry and photometry - physical parameterization - for linear orbits.

A Binary Point Source Point Lens model for microlensing. This model uses a parameterization that depends on only physical quantities such as the proper motions of the lens, the primary source and the secondary source. This is a linear orbit model. 

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    mL: float
        Mass of the lens (Msun)
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    t: float
        Time of Observation
    beta: float
        Angular distance between the lens and primary source on the
        plane of the sky (mas). Can be
            * positive (u0_amp > 0 when u0_hat[0] < 0) or
            * negative (u0_amp < 0 when u0_hat[0] > 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
    acc_E: Acceleration of the secondary source in the direction of RA (mas/yr^2)
    acc_N: Acceleration of the secondary source in the direction of DEC (mas/yr^2)
    muL_E: float
        RA Lens proper motion (mas/yr)
    muL_N: float
        Dec Lens proper motion (mas/yr)
    muS_E: float
        RA Source proper motion (mas/yr) for primary source.
    muS_N: float
        Dec Source proper motion (mas/yr) for primary source.
    delta_muS_sec_E: float
        RA secondary source proper motion with respect to the primary source (mas/yr)
    delta_muS_sec_N: float
        Dec secondary source proper motion with respect to the primary source (mas/yr)
    acc_E: 
        Acceleration of the secondary source in the direction of RA (mas/yr^2)
    acc_N: 
        Acceleration of the secondary source in the direction of DEC (mas/yr^2)
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_S1 + f_S2) / (f_S1 + f_s2 + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    
    fitter_param_names = ['mL', 't0', 't', 'beta', 'dL', 'dL_dS',
                          'xS0_E', 'xS0_N',
                          'muL_E', 'muL_N',
                          'muS_E', 'muS_N',
                          'delta_muS_sec_E', 'delta_muS_sec_N',
                          'acc_E', 'acc_N',
                          'sep', 'alpha']
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']
    additional_param_names = ['dS', 'tE', 'u0_amp',
                              'thetaE_E', 'thetaE_N',
                              'piE_E', 'piE_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'accelerated'

    

    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 acc_E, acc_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 raL=None, decL=None):

        super().__init__(t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 raL=raL, decL=decL)
                     
        self.acc = np.array([acc_E, acc_N])
        self.acc_E, self.acc_N = self.acc
        self.acc_amp = np.linalg.norm(self.acc)
        if self.acc_E==0.0 and self.acc_N==0.0:
            self.acc_hat = np.array([0.0, 0.0])
        else:
            self.acc_hat = self.acc/self.acc_amp
                     
        return

class BSPL_PhotAstrom_LinOrbs_Param3(BSPL_PhotAstromParam3):
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE : float
        Einstein crossing time (days).
    log10_thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame. This should be the position of the source primary.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    muS_N : float
        Dec Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    delta_muS_sec_E: float
        RA secondary source proper motion with respect to the primary source (mas/yr)
    delta_muS_sec_N: float
        Dec secondary source proper motion with respect to the primary source (mas/yr)
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_base = f_{src1{ + f_{src2{ + f_{blend}`
        such that
            :math:`b_sff = (f_{src1} + f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['t0', 'u0_amp', 'tE', 'log10_thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'xS0_E', 'xS0_N',
                          'muS_E', 'muS_N',
                          'delta_muS_sec_E', 'delta_muS_sec_N',
                          'sep', 'alpha']
    phot_param_names = ['fratio_bin', 'mag_base', 'b_sff']
    additional_param_names = ['thetaE_amp', 'mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'linear'

    def __init__(self, t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 raL=None, decL=None):

        super().__init__(t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 raL=raL, decL=decL
        )
        self.delta_muS_sec = np.array([delta_muS_sec_E, delta_muS_sec_N])
        self.delta_muS_sec_E = delta_muS_sec_E
        self.delta_muS_sec_N = delta_muS_sec_N
        self.muS_sec = self.muS + self.delta_muS_sec 

        
        self.muRel_sec = self.muRel + self.delta_muS_sec
        self.muRel_sec_E, self.muRel_sec_N = self.muRel_sec
        self.muRel_sec_amp = np.linalg.norm(self.muRel_sec)  # mas/yr
        self.muRel_sec_hat = self.muRel_sec/self.muRel_sec_amp
        self.tE_sec = self.thetaE_amp/self.muRel_sec_amp * days_per_year


        return


class BSPL_PhotAstrom_AccOrbs_Param3(BSPL_PhotAstrom_LinOrbs_Param3):
    """BSPL model for astrometry and photometry - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE : float
        Einstein crossing time (days).
    log10_thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame. This should be the position of the source primary.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    muS_N : float
        Dec Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    delta_muS_sec_E: float
        RA secondary source proper motion with respect to the primary source (mas/yr)
    delta_muS_sec_N: float
        Dec secondary source proper motion with respect to the primary source (mas/yr)
    acc_E: 
        Acceleration of the secondary source in the direction of RA (mas/yr^2)
    acc_N: 
        Acceleration of the secondary source in the direction of DEC (mas/yr^2)
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_base = f_{src1{ + f_{src2{ + f_{blend}`
        such that
            :math:`b_sff = (f_{src1} + f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    fitter_param_names = ['t0', 'u0_amp', 'tE', 'log10_thetaE', 'piS',
                          'piE_E', 'piE_N',
                          'xS0_E', 'xS0_N',
                          'muS_E', 'muS_N',
                          'delta_muS_sec_E', 'delta_muS_sec_N',
                          'acc_E', 'acc_N',
                          'sep', 'alpha']
    phot_param_names = ['fratio_bin', 'mag_base', 'b_sff']
    additional_param_names = ['thetaE_amp', 'mL', 'piL', 'piRel',
                              'muL_E', 'muL_N',
                              'muRel_E', 'muRel_N']

    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag = 'accelerated'

    def __init__(self, t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 acc_E, acc_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 raL=None, decL=None):

        super().__init__(t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 raL=raL, decL=decL
        )
                
        self.acc = np.array([acc_E, acc_N])
        self.acc_E, self.acc_N = self.acc
        self.acc_amp = np.linalg.norm(self.acc)
        if self.acc_E==0.0 and self.acc_N==0.0:
            self.acc_hat = np.array([0.0, 0.0])
        else:
            self.acc_hat = self.acc/self.acc_amp
                     
        return
                     
class BSPL_GP_PhotAstrom_LinOrbs_Param1(BSPL_GP_PhotAstromParam1):
    """BSPL model for astrometry and photometry with GP - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributess
    ----------
    mL: float
        Mass of the lens (Msun)
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    beta: float
        Angular distance between the lens and primary source on the
        plane of the sky (mas). Can be
            * positive (u0_amp > 0 when u0_hat[0] < 0) or
            * negative (u0_amp < 0 when u0_hat[0] > 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
    muL_E: float
        RA Lens proper motion (mas/yr)
    muL_N: float
        Dec Lens proper motion (mas/yr)
    muS_E: float
        RA Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    muS_N: float
        Dec Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    delta_muS_sec_E: float
        RA Source proper motion (mas/yr) for secondary source.
    delta_muS_sec_N: float
        Dec Source proper motion (mas/yr) for secondary source.
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    gp_log_sigma: float
        Guassian process :math:`log(\sigma)` for the Matern 3/2 kernel.
    gp_rho: float
        Guassian process :math:`{\\rho}` for the Matern 3/2 kernel.
    gp_log_omega04_S0: float
        Guassian process :math:`log(\omega_0^4 * S_0)` from the SHO kernel.
    gp_log_omega0: float
        Guassian process :math:`log(\omega_0)` from the SHO kernel.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    orbitFlag='linear'
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 sep, alpha,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        super().__init__(mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 sep, alpha,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=raL, decL=decL)
        
        
        self.delta_muS_sec = np.array([delta_muS_sec_E, delta_muS_sec_N])
        self.delta_muS_sec_E = delta_muS_sec_E
        self.delta_muS_sec_N = delta_muS_sec_N
        
        self.muRel_sec = self.muRel + self.delta_muS_sec
        self.muRel_sec_E, self.muRel_sec_N = self.muRel_sec
        self.muRel_sec_amp = np.linalg.norm(self.muRel_sec)  # mas/yr
        self.muRel_sec_hat = self.muRel_sec/self.muRel_sec_amp
        self.tE_sec = self.thetaE_amp/self.muRel_sec_amp * days_per_year

        return

class BSPL_GP_PhotAstrom_AccOrbs_Param1(BSPL_GP_PhotAstrom_LinOrbs_Param1):
    """BSPL model for astrometry and photometry with GP - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributess
    ----------
    mL: float
        Mass of the lens (Msun)
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    beta: float
        Angular distance between the lens and primary source on the
        plane of the sky (mas). Can be
            * positive (u0_amp > 0 when u0_hat[0] < 0) or
            * negative (u0_amp < 0 when u0_hat[0] > 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
        This should be the position of the source primary.
    muL_E: float
        RA Lens proper motion (mas/yr)
    muL_N: float
        Dec Lens proper motion (mas/yr)
    muS_E: float
        RA Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    muS_N: float
        Dec Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    delta_muS_sec_E: float
        RA Source proper motion (mas/yr) for secondary source.
    delta_muS_sec_N: float
        Dec Source proper motion (mas/yr) for secondary source.
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    gp_log_sigma: float
        Guassian process :math:`log(\sigma)` for the Matern 3/2 kernel.
    gp_rho: float
        Guassian process :math:`{\\rho}` for the Matern 3/2 kernel.
    gp_log_omega04_S0: float
        Guassian process :math:`log(\omega_0^4 * S_0)` from the SHO kernel.
    gp_log_omega0: float
        Guassian process :math:`log(\omega_0)` from the SHO kernel.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """

    orbitFlag='accelerated'
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 acc_E, acc_N,
                 sep, alpha,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        super().__init__(mL, t0, beta, dL, dL_dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 sep, alpha,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=raL, decL=decL)
        
                
        self.acc = np.array([acc_E, acc_N])
        self.acc_E, self.acc_N = self.acc
        self.acc_amp = np.linalg.norm(self.acc)
        if self.acc_E==0.0 and self.acc_N==0.0:
            self.acc_hat = np.array([0.0, 0.0])
        else:
            self.acc_hat = self.acc/self.acc_amp
                     

        return

class BSPL_GP_PhotAstrom_LinOrbs_Param2(BSPL_GP_PhotAstromParam2):
    """BSPL model for astrometry and photometry with GP - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame. This should be the position of the source primary.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    muS_N : float
        Dec Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    delta_muS_sec_E: float
        RA Source proper motion (mas/yr) for secondary source.
    delta_muS_sec_N: float
        Dec Source proper motion (mas/yr) for secondary source.

    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_{base} = f_{src1} + f_{src2} + f_{blend}`
        such that
            :math:`b_{sff} = (f_{src1} + f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:` b_{sff} = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    gp_log_sigma: float
        Guassian process :math:`log(\sigma)` for the Matern 3/2 kernel.
    gp_rho: float
        Guassian process :math:`{\\rho}` for the Matern 3/2 kernel.
    gp_log_omega04_S0: float
        Guassian process :math:`log(\omega_0^4 * S_0)` from the SHO kernel.
    gp_log_omega0: float
        Guassian process :math:`log(\omega_0)` from the SHO kernel.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    orbitFlag = 'linear'
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        super().__init__(t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=raL, decL=decL)

        
        self.delta_muS_sec = np.array([delta_muS_sec_E, delta_muS_sec_N])
        self.delta_muS_sec_E = delta_muS_sec_E
        self.delta_muS_sec_N = delta_muS_sec_N
        
        self.muRel_sec = self.muRel + self.delta_muS_sec
        self.muRel_sec_E, self.muRel_sec_N = self.muRel_sec
        self.muRel_sec_amp = np.linalg.norm(self.muRel_sec)  # mas/yr
        self.muRel_sec_hat = self.muRel_sec/self.muRel_sec_amp
        self.tE_sec = self.thetaE_amp/self.muRel_sec_amp * days_per_year

        return

class BSPL_GP_PhotAstrom_AccOrbs_Param2(BSPL_GP_PhotAstrom_LinOrbs_Param2):
    """BSPL model for astrometry and photometry with GP - physical parameterization.

    A Binary point Source Point Lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the
    lens mass and positions and proper motions of both the lens and source.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp : float
        Angular distance between the source and the GEOMETRIC center of the lenses
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E : float
        R.A. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame. This should be the position of the source primary.
    xS0_N : float
        Dec. of source position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    muS_E : float
        RA Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    muS_N : float
        Dec Source proper motion (mas/yr)
        Identical proper motions are assumed for the source primary and secondary.
    delta_muS_sec_E: float
        RA Source proper motion (mas/yr) for secondary source.
    delta_muS_sec_N: float
        Dec Source proper motion (mas/yr) for secondary source.

    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_{base} = f_{src1} + f_{src2} + f_{blend}`
        such that
            :math:`b_{sff} = (f_{src1} + f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:` b_{sff} = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    gp_log_sigma: float
        Guassian process :math:`log(\sigma)` for the Matern 3/2 kernel.
    gp_rho: float
        Guassian process :math:`{\\rho}` for the Matern 3/2 kernel.
    gp_log_omega04_S0: float
        Guassian process :math:`log(\omega_0^4 * S_0)` from the SHO kernel.
    gp_log_omega0: float
        Guassian process :math:`log(\omega_0)` from the SHO kernel.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    orbitFlag = 'accelerated'
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 acc_E, acc_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        super().__init__(t0, u0_amp, tE, thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=raL, decL=decL)

        
        
        self.acc = np.array([acc_E, acc_N])
        self.acc_E, self.acc_N = self.acc
        self.acc_amp = np.linalg.norm(self.acc)
        if self.acc_E==0.0 and self.acc_N==0.0:
            self.acc_hat = np.array([0.0, 0.0])
        else:
            self.acc_hat = self.acc/self.acc_amp
                     

        return

class BSPL_GP_PhotAstrom_LinOrbs_Param3(BSPL_GP_PhotAstromParam3):
    """
    Point Source Point Lens with GP model for microlensing. This model includes
    proper motions of the source and the source position on the sky.
    It is the same as PSPL_PhotAstromParam4 except it fits for log10(thetaE)
    instead of thetaE.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float
        Angular distance between the lens and source on the plane of the
        sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE: float
        Einstein crossing time (days).
    log10_thetaE: float
        log10 of the size of the Einstein radius in (mas).
    piS: float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    muS_E: float
        RA Source proper motion (mas/yr)
    muS_N: float
        Dec Source proper motion (mas/yr)
    delta_muS_sec_E: float
        RA Source proper motion (mas/yr) for secondary source.
    delta_muS_sec_N: float
        Dec Source proper motion (mas/yr) for secondary source.
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_{base} = f_{src1} + f_{src2} + f_{blend}`
        such that
            :math:`b_{sff} = (f_{src1} + f_{src2}) / ( f_{src1} + f_{src2} + f_{blend})`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:` b_{sff} = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    gp_log_sigma: float
        Guassian process :math:`log(\sigma)` for the Matern 3/2 kernel.
    gp_rho: float
        Guassian process :math:`{\\rho}` for the Matern 3/2 kernel.
    gp_log_omega04_S0: float
        Guassian process :math:`log(\omega_0^4 * S_0)` from the SHO kernel.
    gp_log_omega0: float
        Guassian process :math:`log(\omega_0)` from the SHO kernel.

    Other Parameters
    ----------------
    Note: Required if calculating with parallax
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    orbitFlag='linear'
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        super().__init__(t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=raL, decL=decL)

        
        
        self.delta_muS_sec = np.array([delta_muS_sec_E, delta_muS_sec_N])
        self.delta_muS_sec_E = delta_muS_sec_E
        self.delta_muS_sec_N = delta_muS_sec_N
        
        self.muRel_sec = self.muRel + self.delta_muS_sec
        self.muRel_sec_E, self.muRel_sec_N = self.muRel_sec
        self.muRel_sec_amp = np.linalg.norm(self.muRel_sec)  # mas/yr
        self.muRel_sec_hat = self.muRel_sec/self.muRel_sec_amp

        self.tE_sec = self.thetaE_amp/self.muRel_sec_amp * days_per_year

                     
        return

class BSPL_GP_PhotAstrom_AccOrbs_Param3(BSPL_GP_PhotAstrom_LinOrbs_Param3):
    """
    Point Source Point Lens with GP model for microlensing. This model includes
    proper motions of the source and the source position on the sky.
    It is the same as PSPL_PhotAstromParam4 except it fits for log10(thetaE)
    instead of thetaE.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float
        Angular distance between the lens and source on the plane of the
        sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE: float
        Einstein crossing time (days).
    log10_thetaE: float
        log10 of the size of the Einstein radius in (mas).
    piS: float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    muS_E: float
        RA Source proper motion (mas/yr)
    muS_N: float
        Dec Source proper motion (mas/yr)
    delta_muS_sec_E: float
        RA Source proper motion (mas/yr) for secondary source.
    delta_muS_sec_N: float
        Dec Source proper motion (mas/yr) for secondary source.
    sep: float
        Angular separation of the source scondary from the
        source primary (mas).
    alpha: float
        Angle made between the binary source axis and North;
        measured in degrees East of North.
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_{base} = f_{src1} + f_{src2} + f_{blend}`
        such that
            :math:`b_{sff} = (f_{src1} + f_{src2}) / ( f_{src1} + f_{src2} + f_{blend})`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:` b_{sff} = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    gp_log_sigma: float
        Guassian process :math:`log(\sigma)` for the Matern 3/2 kernel.
    gp_rho: float
        Guassian process :math:`{\\rho}` for the Matern 3/2 kernel.
    gp_log_omega04_S0: float
        Guassian process :math:`log(\omega_0^4 * S_0)` from the SHO kernel.
    gp_log_omega0: float
        Guassian process :math:`log(\omega_0)` from the SHO kernel.

    Other Parameters
    ----------------
    Note: Required if calculating with parallax
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    orbitFlag='accelerated'
    phot_optional_param_names = ['gp_log_sigma', 'gp_rho', 'gp_log_omega04_S0', 'gp_log_omega0']

    def __init__(self, t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 acc_E, acc_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=None, decL=None):

        super().__init__(t0, u0_amp, tE, log10_thetaE, piS,
                 piE_E, piE_N,
                 xS0_E, xS0_N,
                 muS_E, muS_N,
                 delta_muS_sec_E, delta_muS_sec_N,
                 sep, alpha, fratio_bin,
                 mag_base, b_sff,
                 gp_log_sigma, gp_rho, gp_log_omega04_S0, gp_log_omega0,
                 raL=raL, decL=decL)

        
        
        self.acc = np.array([acc_E, acc_N])
        self.acc_E, self.acc_N = self.acc
        self.acc_amp = np.linalg.norm(self.acc)
        if self.acc_E==0.0 and self.acc_N==0.0:
            self.acc_hat = np.array([0.0, 0.0])
        else:
            self.acc_hat = self.acc/self.acc_amp
                     
        return
                     
######################################################
### BINARY SOURCE BINARY LENS (BSBL) CLASSES ###
######################################################
# --------------------------------------------------
#
# Data Class Family - BSBL
#
# --------------------------------------------------
class BSBL(PSBL):
    """
    Contains methods for model a BSBL photometry + astrometry.
    This is a Data-type class in our hierarchy. It is abstract and should not
    be instantiated.
    """

    def get_amp_arr(self, z_arr, z1, z2):
        """Calculations amplification array

        Calculates the amplification A from the Jacobian J, :math:`A = 1/|J|`

        Parameters
        ----------
        z_arr : array_like
            | Complex position of images for both sources.
            | ``Shape = [N_times, N_sources, N_solutions]``
            | where [:, 0, :] is primary and [:, 1, :] is secondary.
            | -- note this could be jagged.

        z1 : array_like
            Complex position(s) of lens 1 (primary). ``Shape = [N_times]``

        z2 : array_like
            Complex position(s) of lens 2 (secondary). ``Shape = [N_times]``

        Returns
        -------
        amp_arr : array_like
            BLEH
        """

        N_times = z1.shape[0]
        N_sources = z_arr.shape[1]

        dwbardz = self.m1 / (z_arr - z1[:, np.newaxis, np.newaxis]) ** 2
        dwbardz += self.m2 / (z_arr - z2[:, np.newaxis, np.newaxis]) ** 2
        jacobian = 1 - np.absolute(dwbardz) ** 2
        amp_arr = 1.0 / np.absolute(jacobian)  # Absolute value of J

        return amp_arr

    def rescale_complex_pos(self, w, z1, z2):
        """
        Make sure everything is roughly centered on the origin
        in a 1 x 1 box.
        """
        m1 = copy.deepcopy(self.m1)
        m2 = copy.deepcopy(self.m2)

        # Note shapes coming in and all complex nubmers (for 2 directions):
        # w = [N_times, N_sources]
        # z1 = [N_times]
        # z2 = [N_times]

        # Put the positions of the source and lenses into
        # an array, so we can calculate the average position
        # and "width" of points at each time, in order to center
        # and scale them.
        # pos shape = [N_times, N_sources + N_lenses]
        pos = np.concatenate([w, z1[:, np.newaxis], z2[:, np.newaxis]], axis=1)

        # Calculate the average position to get the shift.
        shift = np.average(pos, axis=1)
        w -= shift[:, np.newaxis]
        z1 -= shift
        z2 -= shift

        # Calculate the average spread to get the scale.
        xscale = np.max(pos.real, axis=1) - np.min(pos.real, axis=1)
        yscale = np.max(pos.imag, axis=1) - np.min(pos.imag, axis=1)
        xyscale = np.concatenate([xscale, yscale]).reshape(len(xscale), 2)
        scale = 1 / np.max(xyscale, axis=1)
        w *= scale[:, np.newaxis]
        z1 *= scale
        z2 *= scale
        m1 *= scale ** 2
        m2 *= scale ** 2

        return w, z1, z2, m1, m2, scale, shift

    def get_image_pos_arr_old(self, w, z1, z2, check_sols=True):
        """Gets image positions.
        | Solve the fifth-order polynomial and get the image positions.
        | See PSBL writeup for full equations.
        | All angular distances are in arcsec.

        Parameters
        ----------
        w : array_like
            Complex position(s) of the source. Shape = [N_times, N_sources

        z1 : array_like
            Complex position(s) of lens 1 (primary). Shape = [N_times]

        z2 : array_like
            Complex position(s) of lens 2 (secondary). Shape = [N_times]

        check_sols : bool, optional
            If True, calculated roots are checked against the lens equation,
            and output will only contain those within self.root_tol.
            If False, all calculated roots are returned.

        Returns
        -------
        z_arr : array_like
            Position of the lensed source images.
            Rank-1 array of polynomial roots, possibly complex.
            If check_sols = True, only roots solving the lens
            equation are returned.
        """

        # Setup the output array with the right shape.
        N_times = w.shape[0]
        N_sources = w.shape[1]
        z_arr = np.zeros((N_times, N_sources, 5), dtype=np.complex_)

        # Loop through each of the sources and just call the parent class.
        for ss in range(N_sources):
            w_ss = w[:, ss]

            z_arr[:, ss, :] = PSBL.get_image_pos_arr_old(self, w_ss, z1, z2, check_sols=check_sols)

        return z_arr

    def get_image_pos_arr(self, w, z1, z2, m1, m2, check_sols=True):
        """Gets image positions.

        | Solve the fifth-order polynomial and get the image positions.
        | See PSBL writeup for full equations.
        | All angular distances are in arcsec.

        Parameters
        ----------
        w : array_like
            Complex position(s) of the sources. Shape = [N_times, N_sources]

        z1 : array_like
            Complex position(s) of lens 1 (primary). Shape = [N_times]

        z2 : array_like
            Complex position(s) of lens 2 (secondary). Shape = [N_times]

        check_sols : bool, optional
            If True, calculated roots are checked against the lens equation,
            and output will only contain those within self.root_tol.
            If False, all calculated roots are returned.

        Returns
        -------
        z_arr : array_like
            Rank-1 array of polynomial roots, possibly complex.
            If check_sols = True, only roots solving the lens
            equation are returned.
        """
        # Setup the output array with the right shape.
        N_times = w.shape[0]
        N_sources = w.shape[1]
        z_arr = np.zeros((N_times, N_sources, 5), dtype=np.complex_)

        # Loop through each of the sources and just call the parent class.
        for ss in range(N_sources):
            w_ss = w[:, ss]

            z_arr[:, ss, :] = PSBL.get_image_pos_arr(self, w_ss, z1, z2,
                                                     m1, m2,
                                                     check_sols=check_sols)
        return z_arr

    def get_all_arrays(self, t, check_sols=True, rescale=True):
        '''
        Obtain the image and amplitude arrays for each t_obs.

        Parameters
        ----------
        t : array_like
            Array of times to model.

        Returns
        -------
        images : array_like
            Array/tuple of complex positions of each images at each t_obs.
            shape = [N_times, N_sources, 5, 2]

        amp_arr : array_like
            Array/tuple of amplification of each images at each t_obs.
            shape = [N_times, N_sources, 5]
        '''
        kwargs = {'check_sols': check_sols}

        if rescale:
            # Get complex positions (no rescaling).
            _comp = self.get_complex_pos(t)

            # Deepcopy because for some reason in my test it would modify.
            comp = copy.deepcopy(_comp)

            # Rescaled complex positions.
            rcomp = self.rescale_complex_pos(*_comp)

            # Image positions derived from rescale complex positions.
            rimages = self.get_image_pos_arr(*rcomp[0:5], **kwargs)

            # Take the image positions derived from the rescaled complex positions
            # and rescale them to get the images back in the original scale.
            images = (rimages / rcomp[5][:, np.newaxis, np.newaxis]) + rcomp[6][:, np.newaxis, np.newaxis]

            # Get amplifications.
            amps = self.get_amp_arr(images, *comp[1:])

        else:
            comp = self.get_complex_pos(t)
            images = self.get_image_pos_arr_old(*comp)
            amps = self.get_amp_arr(images, *comp[1:])

        return images, amps

    def get_resolved_photometry(self, t_obs, filt_idx=0, amp_arr=None, print_warning=True):
        '''
        Get the photometry for each of the lensed source images.
        Implement with no blending (since we don't support different
        blendings for the different images).

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.

        Other Parameters
        ----------------
        amp_arr : array_like
            Amplifications of each individual image at each time,
            i.e. amp_arr.shape = (len(t_obs), number of images at each t_obs).

            This will over-ride t_obs; but is more efficient when calculating
            both photometry and astrometry. If None, then just use t_obs.
        filt_idx : int
            The filter index (def=0).

        Returns
        -------
        mag_model : array_like
            Magnitude of each lensed image centroid at t_obs.
            Shape = [5, len(t_obs)]
        '''
        if amp_arr is None:
            img_arr, amp_arr = self.get_all_arrays(t_obs)

        # Mask invalid values from the amplification array.
        amp_arr_mskd = np.masked_invalid(amp_arr)
        
        # mags to fluxes
        # switch nan mags to 0 fluxes
        f1 = np.nan_to_num(mag2flux(self.mag_src_pri[filt_idx]), nan = 0)
        f2 = np.nan_to_num(mag2flux(self.mag_src_sec[filt_idx]), nan = 0)

        flux_model = amp_arr_mskd
        flux_model[:, 0, :] *= f1
        flux_model[:, 1, :] *= f2

        # Catch the edge case where we exceed the zeropoint.
        bad = np.where(flux_model <= 0)
        if len(bad[0]) > 0:
            if print_warning:
                pdb.set_trace()
                print('!! ! ! !! Warning: get_photometry: bad flux encountered.')
            flux_model[bad] = np.nan

        mag_model = -2.5 * np.log10(flux_model / flux_zp) + mag_zp

        return mag_model

    def get_photometry(self, t_obs, filt_idx=0, amp_arr=None, print_warning=True):
        '''
        Get the photometry for each of the lensed source images.

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.
        filt_idx : int, optional
            Index of the photometric filter or data set.

        Other Parameters
        ----------------
        amp_arr : array_like
            Amplifications of each individual image at each time,
            i.e. ``amp_arr.shape = (len(t_obs)``, number of images at each t_obs).

            This will over-ride t_obs; but is more efficient when calculating
            both photometry and astrometry. If None, then just use t_obs.
        print_warning : bool, optional
            Print a warning in the rare case that the magnitude exceeds a
            zeropoint of 30 and conversions result in NaN returned.

        Returns
        -------
        mag_model : array_like
            Magnitude of the unresolved microlensing event at t_obs.
        '''
        if amp_arr is None:
            img_arr, amp_arr = self.get_all_arrays(t_obs)

        # Mask invalid values from the amplification array.
        amp_arr_msk = np.ma.masked_invalid(amp_arr)

        # Sum up all the amplifications b/c surface brightness is conserved.
        amp = np.sum(amp_arr_msk, axis=2)
        A1 = amp[:, 0]
        A2 = amp[:, 1]
        
        # mags to fluxes
        # switch nan mags to 0 fluxes
        f1 = np.nan_to_num(mag2flux(self.mag_src_pri[filt_idx]), nan = 0)
        f2 = np.nan_to_num(mag2flux(self.mag_src_sec[filt_idx]), nan = 0)

        f1_lensed = f1 * A1
        f2_lensed = f2 * A2

        flux_srcs = f1 + f2
        flux_model = f1_lensed + f2_lensed

        # Adding flux of neighbors and lenses
        # b_sff = (fS1 + fS2) / (fS1 + fS2 + fN + fL1 + fL2)
        flux_model += flux_srcs * (1.0 - self.b_sff[filt_idx]) / self.b_sff[filt_idx]

        # Catch the edge case where we exceed the zeropoint.
        bad = np.where(flux_model <= 0)[0]
        if len(bad) > 0:
            if print_warning:
                print('!!!!!!!!!! Warning: get_photometry: bad flux encountered.')
                print('')
            flux_model[bad] = np.nan

        mag_model = flux2mag(flux_model)

        # Set the masked values (in the data array) to also be nan.
        bad = np.where(flux_model.mask == True)
        if len(bad) > 0:
            mag_model.data[bad] = np.nan

        return mag_model


class BSBL_Phot(BSBL, PSBL_Phot):
    photometryFlag = True
    astrometryFlag = False

    def get_complex_pos(self, t_obs):
        """ Get the positions of the lenses and source as complex numbers.

        This is needed for further calculations.
        Note that all units are still the same as before, this
        is just rewriting vectors :math:`z = (x,y)` as :math:`z = x + iy`.

        Returns
        -------
        w : complex array
            Source position as an array of complex numbers with
            real = east component, imaginary = north component

        z1 : complex array
            Lens primary component position as an array of complex numbers with
            real = east component, imaginary = north component

        z2 : complex array
            Lens secondary component position as an array of complex numbers with
            real = east component, imaginary = north component
        """
        if not isinstance(t_obs, np.ndarray):
            raise RuntimeError("time must be a 1D numpy array")

        # t0 is the time of closest approach, projected on the sky, for the
        # whole binary source system. By default, we assume a static binary with
        # the primary star being at the origin of the system.

        # Calculate the elapsed time, in units of tE.
        # This gives the linear motion offset due to muRel in the muRel_hat direction.
        tau_pri = (t_obs - self.t0_pri) / self.tE
        tau_sec = (t_obs - self.t0_sec) / self.tE

        # Calculate u due to linear motion of the system.
        u_pri = self.u0_pri[np.newaxis, :] + tau_pri[:, np.newaxis] * self.muRel_hat[np.newaxis, :]
        u_sec = self.u0_sec[np.newaxis, :] + tau_sec[:, np.newaxis] * self.muRel_hat[np.newaxis, :]

        # Some day: implement orbital motion
        # if self.orbitFlag == 'full':
        #     add orbital motion

        # Incorporate parallax
        if self.parallaxFlag:
            parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
            u_pri -= self.piE_amp * parallax_vec
            u_sec -= self.piE_amp * parallax_vec

        # Convert positions to complex coordinates shape: [N_tobs, 2] where
        #   [:, 0] = primary complex number with x and y positions
        #   [:, 1] = secondary complex number with x and y positions
        w = np.zeros((len(t_obs), 2), dtype=float)
        w[:, 0] = u_pri[:, 0] + u_pri[:, 1] * 1j
        w[:, 1] = u_sec[:, 0] + u_sec[:, 1] * 1j

        # Get the position of the lenses (in units of Einstein radii)
        z1 = self.xL1_over_theta[0] + self.xL1_over_theta[1] * 1j
        z2 = self.xL2_over_theta[0] + self.xL2_over_theta[1] * 1j

        z1 = np.repeat(z1, w.shape[0])
        z2 = np.repeat(z2, w.shape[0])

        return w, z1, z2

    def get_resolved_lens_astrometry(self, t_obs):
        """Equation of motion for just the foreground lenses, individually.

        Parameters
        ----------
        t_obs : array_like
            Time (in MJD).

        Notes
        -----
        .. note::
           Note, this is a photometry only model, so units are in Einstein radii.
        """
        # In phot only fits, lens is at rest. So just duplicate to get
        # the right shape.
        xL1 = np.tile(self.xL1_over_theta, (len(t_obs), 1))
        xL2 = np.tile(self.xL2_over_theta, (len(t_obs), 1))

        return (xL1, xL2)

    def get_astrometry_unlensed(self, t_obs):
        """Get the astrometry of the source if the lens didn't exist.
        Note, this is a photometry only model, so units are in Einstein radii.

        Returns
        -------
        xS_unlensed : numpy array, dtype=float, ``shape = len(t_obs) x 2``
            The unlensed positions of the source in Einstein radii.


        Notes
        -----
        .. note::
           Note, this is a photometry only model, so units are in Einstein radii.
        """
        # Calculate the position of the source w.r.t. lens (in Einstein radii)
        # Distance along muRel direction
        tau = (t_obs - self.t0) / self.tE
        tau = tau.reshape(len(tau), 1)

        # Distance along u0 direction -- always constant with time.
        u0 = self.u0.reshape(1, len(self.u0))
        thetaE_hat = self.thetaE_hat.reshape(1, len(self.thetaE_hat))

        # Total distance
        u = u0 + tau * thetaE_hat

        # Incorporate parallax
        if self.parallaxFlag:
            parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
            u -= self.piE_amp * parallax_vec

        return u

    def get_resolved_astrometry(self, t_obs, image_arr=None, amp_arr=None):
        '''
        Position of the observed source position in Einstein radii.

        Parameters
        ----------
        t_obs : array_like, shape = [N_times]
            Array of times to model.

        Other Parameters
        ----------------
        image_arr : array_like
            Array of complex image positions at each t_obs,
            i.e. image_arr.shape = (len(t_obs), number of images at each t_obs).
            Each value in this array is complex
            (real = north component, imaginary = east component)

        amp_arr : array_like
            Array of magnifications of each images.
            Same shape as image_arr.

        Returns
        -------
        model_pos : array_like. shape = [N_times, N_images, 2]
            Array of vector positions of the centroid at each t_obs.
        '''
        if (image_arr is None) or (amp_arr is None):
            image_arr, amp_arr = self.get_all_arrays(t_obs)

        # In units of Einstein radii.
        xS_lensed_pos = image_arr.view('(2,)float')

        return xS_lensed_pos

    def get_astrometry(self, t_obs, image_arr=None, amp_arr=None, ast_filt_idx=0):
        '''
        Position of the observed (unresolved) source position in Einstein radii.

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.

        Other Parameters
        ----------------
        image_arr : array_like
            Array of complex image positions at each t_obs,
            i.e. image_arr.shape = (len(t_obs), number of images at each t_obs).
            Each value in this array is complex
            (real = north component, imaginary = east component)

        amp_arr : array_like
            Array of magnifications of each images.
            Same shape as image_arr.

        Returns
        -------
        model_pos : array_like
            Array of vector positions of the centroid at each t_obs.
        '''
        if (image_arr is None) or (amp_arr is None):
            image_arr, amp_arr = self.get_all_arrays(t_obs)

        # Split back into x, y such that shape = [N_times, N_images, 2]
        xS_lensed_res = image_arr.view('(2,)float')

        # Mask invalid values from the amplification array.
        amp_arr_mskd = np.ma.masked_invalid(amp_arr)
        amp_arr_mskd2 = amp_arr_mskd.reshape((amp_arr_mskd.shape[0], amp_arr_mskd.shape[1], 1))
        xS_lensed_res_mskd = np.ma.masked_invalid(xS_lensed_res)

        xS_lensed_ures = np.sum(xS_lensed_res_mskd * amp_arr_mskd2, axis=1) / np.sum(amp_arr_mskd2, axis=1)

        return xS_lensed_ures.data


class BSBL_PhotAstrom(BSBL, PSBL_PhotAstrom):
    photometryFlag = True
    astrometryFlag = True

    def get_complex_pos(self, t):
        """
        Get the positions of the lenses and sources as
        complex numbers. This is needed for further calculations.
        Note that all units are still the same as before, this
        is just rewriting vectors :math:`z = (x,y)` as :math:`z = x + iy`.

        Parameters
        ----------
        t : array_like
            Array of times to model.

        Returns
        -------
        w : complex array
            Source positions (arcsec) as an array of complex numbers with
            real = east component, imaginary = north component
            shape = [N_times, N_sources].

        z1 : complex array
            Lens primary component position (arcsec) as an array of complex numbers with
            real = east component, imaginary = north component
            shape = [N_times]

        z2 : complex array
            Lens secondary component position (arcsec) as an array of complex numbers with
            real = east component, imaginary = north component
            shape = [N_times]
        """
        if not isinstance(t, np.ndarray):
            raise RuntimeError("time must be a 1D numpy array")

        # Find positions of lenses and sources over t_obs (arcsec)
        xS_vec = self.get_resolved_astrometry_unlensed(t)  # shape = [N_times, N_sources, 2 directions]
        xL1_vec, xL2_vec = self.get_resolved_lens_astrometry(t)

        # Convert positions to complex coordinates
        w = xS_vec[:, :, 0] + xS_vec[:, :, 1] * 1j

        z1 = xL1_vec[:, 0] + xL1_vec[:, 1] * 1j
        z2 = xL2_vec[:, 0] + xL2_vec[:, 1] * 1j

        return w, z1, z2

    
    def get_resolved_astrometry_unlensed(self, t):
        """Get the astrometry of the source if the lens didn't exist.

        Returns
        -------
        xS_resolved_unlensed : numpy array, [shape = len(t_obs), N_sources, 2]
            The unlensed positions of the sources in arcseconds.

        In other words,
            xS[0, 0, :] returns the 2D sky position of the
            first source at the first time.
        Similarly,
            xS[0, 1, :] returns the 2D sky position of the
            second source at the first time.
        """
        # Equation of motion for just the background source.
        dt1_in_years = (t - self.t0) / days_per_year
        #dt2_in_years = (t - self.t0_sec) / days_per_year
        #Test. call function. two position vectors on the sky calculated. Subtract the two and make sure that the separation stays the same in the no orbits case. 


        # Calculate position vs. time in arcsec

        
        if self.orbitFlag == 'circular':
            xS1_unlens = np.zeros((len(t), 2), dtype=float)
            xS2_unlens = np.zeros((len(t), 2), dtype=float)
            orb = orbits.Orbit()
            orb.w = self.omegaS
            orb.o = self.big_omegaS
            orb.i = self.iS
            orb.e = self.eS
            orb.p = self.pS
            orb.tp = self.tpS
            orb.aleph = self.alephS *1e-3
            orb.aleph2 = self.aleph_secS*1e-3
            orb.vx = self.vxS
            orb.vy = self.vyS
            orb.x0 = self.xS0_E
            orb.y0 = self.xS0_N
            (x, y, x2, y2) = orb.oal2xy(t, self.t0)
            xS1_unlens[:, 0] = x
            xS1_unlens[:, 1] = y
            xS2_unlens[:, 0] = x2
            xS2_unlens[:, 1] = y2
        else:    
            xS1_unlens = self.xS0_pri + np.outer(dt1_in_years, self.muS) * 1e-3
            xS2_unlens = self.xS0_sec + np.outer(dt1_in_years, self.muS) * 1e-3

    
        N_sources = 2
        xS_unlensed = np.zeros((len(t), N_sources, 2), dtype=float)

        xS_unlensed[:, 0, :] = xS1_unlens
        xS_unlensed[:, 1, :] = xS2_unlens
        

        if self.parallaxFlag:
            parallax_vec = parallax_in_direction(self.raL, self.decL, t)  # mas
            xS_unlensed += (self.piS * parallax_vec[:, np.newaxis, :]) * 1e-3  # arcsec

        return xS_unlensed #arcsex
        

    def get_astrometry_unlensed(self, t, ast_filt_idx=0):
        """Get the astrometry of the combined, flux-weighted sources if the
        lens didn't exist.

        Returns
        -------
        xS_unlensed : numpy array, dtype=float
            | The unlensed positions of the combined sources in arcseconds.
            | Shape = [len(t), 2 directions]
        """
        xS_unlens_both = self.get_resolved_astrometry_unlensed(t)
        xS1_unlens = xS_unlens_both[:, 0, :]
        xS2_unlens = xS_unlens_both[:, 1, :]

        # Calculate un-magnified fluxes. Note, we ignore blended flux entirely.
        f1 = mag2flux(self.mag_src_pri[ast_filt_idx])
        f2 = mag2flux(self.mag_src_sec[ast_filt_idx])

        # Flux-weighted centroid.
        xS_unlensed = (xS1_unlens * f1 + xS2_unlens * f2) / (f1 + f2)

        return xS_unlensed

    def get_lens_astrometry(self, t):
        """Equation of motion for just the foreground lens system.
        Note that this returns the position of the geometric center of
        the lens system on the sky as a function of time.

        Parameters
        ----------
        t : array_like
            Time (in MJD).

        Return
        ------
        xL : array_like, shape = [N_times, 2 directions]
            Position of the lens system (geometric center) over time.
        """
        dt_in_years = (t - self.t0) / days_per_year
        xL = self.xL0 + np.outer(dt_in_years, self.muL) * 1e-3

        if self.parallaxFlag:
            # Get the parallax vector for each date.
            parallax_vec = parallax_in_direction(self.raL, self.decL, t)
            xL += (self.piL * parallax_vec) * 1e-3  # arcsec

        return xL


    def get_resolved_lens_astrometry(self, t_obs):
        """Equation of motion for just the foreground lenses, individually.

        Parameters
        ----------
        t_obs : array_like
            Time (in MJD).

        Return
        ------
        xL1 : array_like, shape = [N_times, 2 directions]
            Position of the lens primary
        xL2 : array_like, shape = [N_times, 2 directions]
            Position of the lens secondary
        """
        
        xL = self.get_lens_astrometry(t_obs)

        offset = 0.5 * self.sepL * np.array([np.sin(self.alphaL_rad),
                                            np.cos(self.alphaL_rad)])
        offset *= 1e-3  # convert to arcsec

        xL1 = xL + offset  # primary
        xL2 = xL - offset  # secondary
        if self.orbitFlag == 'circular':
                xL1 = np.zeros((len(t_obs), 2), dtype=float)
                xL2 = np.zeros((len(t_obs), 2), dtype=float)
                orb = orbits.Orbit()
                orb.w = self.omegaL
                orb.o = self.big_omegaL
                orb.i = self.iL
                orb.e = self.eL
                orb.p = self.pL
                orb.tp = self.tpL
                orb.aleph = self.alephL *1e-3
                orb.aleph2 = self.aleph_secL*1e-3
                orb.vx = self.vxL
                orb.vy = self.vyL
                orb.x0 = self.xL0_E
                orb.y0 = self.xL0_N
                (x, y, x2, y2) = orb.oal2xy(t_obs, self.t0)
                xL1[:, 0] = x
                xL1[:, 1] = y
                xL2[:, 0] = x2
                xL2[:, 1] = y2
                if self.parallaxFlag:
                # Get the parallax vector for each date.
                    parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
                    xL1 += (self.piL * parallax_vec) * 1e-3  # arcsec
        if self.parallaxFlag:
                # Get the parallax vector for each date.
                parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
                xL2 += (self.piL * parallax_vec) * 1e-3  # arcsec
            
        return (xL1, xL2)


    def get_resolved_astrometry(self, t, image_arr=None, amp_arr=None):
        """Parallax: For each source, get the x, y astrometry for the
        multiple lensed source images. For each source, there can be up
        to 5 images. Sometimes there will be 3 and sometimes 5.

        Parameters
        ----------
        t : array_like, shape = [N_times]
            Array of times to model.

        Other Parameters
        ----------------
        image_arr : array_like
            Array of complex image positions at each t_obs,
            i.e. `image_arr.shape = [len(t_obs)`, N_sources, N_images at each t_obs].
            Each value in this array is complex
            (real = north component, imaginary = east component)

        amp_arr : array_like
            Array of magnifications of each images.
            Same shape as image_arr.

        Returns
        -------
        xS_resolved : numpy array
            [shape = len(t), N_sources, N_images, 2]
            Array of vector positions of each lensed image.

        Notes
        -----
        For each time t and each source, we have 3 or 5 images as a result
        of the binary lens.

        In other words,
            xS[0, 0, 0, :] returns the 2D sky position of the
            first source's first lensed image at the first time.
        Similarly,
            xS[0, 0, 1, :] returns the 2D sky position of the
            first source's second lensed image at the first time.
        """
        if (image_arr is None) or (amp_arr is None):
            image_arr, amp_arr = self.get_all_arrays(t)

        xS_lensed_pos = image_arr.view('(2,)float')

        return xS_lensed_pos

    def get_astrometry(self, t, image_arr=None, amp_arr=None, ast_filt_idx=0):
        """Position of the observed (unresolved) source position in arcsec.

        Parameters
        ----------
        t : array_like
            Array of times in MJD.DDD

        Other Parameters
        ----------------
        image_arr : array_like
            Array of complex image positions at each t_obs,
            i.e. image_arr.shape = (N_times, N_sources, N_images).
            Each value in this array is complex
            (real = north component, imaginary = east component)

        amp_arr : array_like
            Array of magnifications of each images.
            Same shape as image_arr.

        ast_filt_idx : int
            The filter index for the astrometry.

        Returns
        -------
        xS_lensed
            Returns flux-weighted average of lensed source positions.
        """
        if (image_arr is None) or (amp_arr is None):
            image_arr, amp_arr = self.get_all_arrays(t)

        # Split back into x, y such that shape = [N_times, N_sources, N_images, 2]
        xS_lensed_res = image_arr.view('(2,)float')

        # Mask invalid values from the amplification array.
        amp_arr_mskd = np.ma.masked_invalid(amp_arr)
        xS_lensed_res_mskd = np.ma.masked_invalid(xS_lensed_res)

        xS_lensed_ures = np.sum(xS_lensed_res_mskd * amp_arr_mskd[:,:,:,np.newaxis], axis=(1,2))
        xS_lensed_ures /= np.sum(amp_arr_mskd[:,:,:,np.newaxis], axis=(1,2))

        return xS_lensed_ures.data

    def get_centroid_shift(self, t, ast_filt_idx=0):
        """Parallax: Get the centroid shift (in arcsec) for a list of
        observation times (in MJD).

        Returns the flux-weighted centroid of all the sources lensed images.

        Parameters
        ----------
        t:
            Array of times in MJD.DDD

        Returns
        -------
        centroid_shift : numpy array
            [shape = len(t), 2]
        """

        xS = self.get_astrometry(t, ast_filt_idx=ast_filt_idx)
        xS_unlensed = self.get_astrometry_unlensed(t, ast_filt_idx=ast_filt_idx)

        shift = xS - xS_unlensed

        return shift
    
    def dexanimate(self, tE, time_steps, frame_time, name, size, zoom, astrometry, loc):
            """ Produces animation of microlensing event. 
            This function takes the PSPL and makes an animation, the input variables are as follows
    
            Parameters
            ----------
    
            tE: 
                number of einstein crossings times before/after the peak you want the animation to plot
                    e.g tE = 2 => graph will go from -2 tE to 2 tE
            time_steps:
                number of time steps before/after peak, so total number of time steps will 
                be 2 times this value
            frame_time:
                times in ms of each frame in the animation
            name: string
                the animation will be saved as name.html
            size: list
                [horizontal, vertical] cm's
            zoom:
                # of einstein radii plotted in vertical direction
            """
    
            times = np.array(range(-time_steps, time_steps + 1, 1))
            tau = tE * times / (-times[0])
            t = self.t0 + (tau * self.tE)
    
            img, amp = self.get_all_arrays(t)
    
            xL1, xL2 = self.get_resolved_lens_astrometry(t)
            source  = self.get_resolved_astrometry_unlensed(t) 
            source_primary = source[:, 0, :]
            source_secondary = source[:, 1, :]
    
            xS_resolved = self.get_resolved_astrometry(t, image_arr = img)
            img_pri = xS_resolved[:, 0, :, :] 
            img_sec = xS_resolved[:, 1, :, :] 
    
            fig = plt.figure(figsize=[size[0], size[1] + 0.5])  # sets up the figure
            ax1 = fig.add_subplot(2, 1, 1)
            fig.subplots_adjust(hspace=.5)
    
            markersize = 1
            linewidth = 1
            l1_line1, = ax1.plot([], '.', markersize = size[0] * 1.3, label="Primary Lens ", color='#FF8C00', linewidth=2)
            l1_line2, = ax1.plot([], '-', markersize = size[0] * .3, color='#FF8C00', linewidth=2)
            l2_line1, = ax1.plot([], '.', markersize = size[0] * 1.3, label="Secondary Lens ", color='#FF218C', linewidth=2)
            l2_line2, = ax1.plot([], '-', markersize = size[0] * .3, color='#FF218C',linewidth=2 )
        
            s1_line1, = ax1.plot([], '.', markersize = size[0] * 1.3, label="Unlensed Primary Source", color='#21B1FF', linewidth=2)
            s1_line2, = ax1.plot([], '-', markersize = size[0] * .3, color='#21B1FF', linewidth=2)
        
            s2_line1, = ax1.plot([], '.', markersize = size[0] * 1.3,label="Unlensed Secondary Source", color='springgreen', linewidth=2)
            s2_line2, = ax1.plot([], '-', markersize = size[0] * .3, color='springgreen', linewidth=2)
    
    
    
        
            i11_line1, = ax1.plot([], '.', markersize = size[0] * 1.0, label="Lensed Primary Image", color='#21B1FF')
            i11_line2, = ax1.plot([], '.', markersize = size[0] * .2, color='#21B1FF')
            i12_line1, = ax1.plot([], '.', markersize = size[0] * 1.0, color='#21B1FF')
            i12_line2, = ax1.plot([], '.', markersize = size[0] * .2, color='#21B1FF')
            i13_line1, = ax1.plot([], '.', markersize = size[0] * 1.0, color='#21B1FF')
            i13_line2, = ax1.plot([], '.', markersize = size[0] * .2, color='#21B1FF')
            i14_line1, = ax1.plot([], '.', markersize = size[0] * 1.0, color='#21B1FF')
            i14_line2, = ax1.plot([], '.', markersize = size[0] * .2, color='#21B1FF')
            i15_line1, = ax1.plot([], '.', markersize = size[0] * 1.0, color='#21B1FF')
            i15_line2, = ax1.plot([], '.', markersize = size[0] * .2, color='#21B1FF')
        
            i21_line1, = ax1.plot([], '.', markersize = size[0] * 1.0, label="Lensed Secondary Image", color='springgreen')
            i21_line2, = ax1.plot([], '.', markersize = size[0] * .2, color='springgreen')
            i22_line1, = ax1.plot([], '.', markersize = size[0] * 1.0, color='springgreen')
            i22_line2, = ax1.plot([], '.', markersize = size[0] * .2, color='springgreen')
            i23_line1, = ax1.plot([], '.', markersize = size[0] * 1.0, color='springgreen')
            i23_line2, = ax1.plot([], '.', markersize = size[0] * .2, color='springgreen')
            i24_line1, = ax1.plot([], '.', markersize = size[0] * 1.0, color='springgreen')
            i24_line2, = ax1.plot([], '.', markersize = size[0] * .2, color='springgreen')
            i25_line1, = ax1.plot([], '.', markersize = size[0] * 1.0, color='springgreen')
            i25_line2, = ax1.plot([], '.', markersize = size[0] * .2, color='springgreen')
    
        
            
            ax1.set_xlabel('RA (")')
            ax1.set_ylabel('Dec (")')
            ax1.set_xlim(zoom, -zoom)
            ax1.set_ylim(-zoom, zoom)
            ax1.legend(fontsize=12, loc=loc)
    
            line = [l1_line1, l1_line2, l2_line1, l2_line2, s1_line1, s1_line2, s2_line1, s2_line2, i11_line1, i11_line2, i12_line1, i12_line2, i13_line1, i13_line2, i14_line1, i14_line2, i15_line1, i15_line2,i21_line1, i21_line2, i22_line1, i22_line2, i23_line1, i23_line2, i24_line1, i24_line2, i25_line1, i25_line2]
        
            def update(i, lens1, lens2, source1, source2, image1, image2, image3, image4, image5,image6, image7, image8, image9, image10, tau, line):
                    line[0].set_data(lens1[i, 0], lens1[i, 1])
                    line[1].set_data(lens1[:i + 1, 0], lens1[:i + 1, 1])
                    line[2].set_data(lens2[i, 0], lens2[i, 1])
                    line[3].set_data(lens2[:i + 1, 0], lens2[:i + 1, 1])
                    line[4].set_data(source1[i, 0], source1[i, 1])
                    line[5].set_data(source1[:i + 1, 0], source1[:i + 1, 1])
                    line[6].set_data(source2[i, 0], source2[i, 1])
                    line[7].set_data(source2[:i + 1, 0], source2[:i + 1, 1])
                
                    line[8].set_data(image1[i, 0], image1[i, 1])
                    line[9].set_data(image1[:i + 1, 0], image1[:i + 1, 1])
                    line[10].set_data(image2[i, 0], image2[i, 1])
                    line[11].set_data(image2[:i + 1, 0], image2[:i + 1, 1])
                    line[12].set_data(image3[i, 0], image3[i, 1])
                    line[13].set_data(image3[:i + 1, 0], image3[:i + 1, 1])
                    line[14].set_data(image4[i, 0], image4[i, 1])
                    line[15].set_data(image4[:i + 1, 0], image4[:i + 1, 1])
                    line[16].set_data(image5[i, 0], image5[i, 1])
                    line[17].set_data(image5[:i + 1, 0], image5[:i + 1, 1])
                    line[18].set_data(image6[i, 0], image6[i, 1])
                    line[19].set_data(image6[:i + 1, 0], image6[:i + 1, 1])
                    line[20].set_data(image7[i, 0], image7[i, 1])
                    line[21].set_data(image7[:i + 1, 0], image7[:i + 1, 1])
                    line[22].set_data(image8[i, 0], image8[i, 1])
                    line[23].set_data(image8[:i + 1, 0], image8[:i + 1, 1])
                    line[24].set_data(image9[i, 0], image9[i, 1])
                    line[25].set_data(image9[:i + 1, 0], image9[:i + 1, 1])
                    line[26].set_data(image10[i, 0], image10[i, 1])
                    line[27].set_data(image10[:i + 1, 0], image10[:i + 1, 1])
                    return line
            ani = animation.FuncAnimation(fig, update, len(tau), fargs=[xL1, xL2,source_primary, source_secondary, img_pri[:,0], img_pri[:,1], img_pri[:,2], img_pri[:,3], img_pri[:,4],img_sec[:,0], img_sec[:,1], img_sec[:,2], img_sec[:,3], img_sec[:,4], tau, line], blit=True, interval=frame_time)
            ani.save("%s.mp4" % name, writer="ffmpeg")   
        
            return ani


class BSBL_Parallax(PSPL_Parallax):
    parallaxFlag = True


class BSBL_noParallax(PSPL_noParallax):
    parallaxFlag = False

# --------------------------------------------------
#
# Parameterization Class Family - BSBL
#
# --------------------------------------------------

# Phot class not done yet.
class BSBL_PhotParam1(PSPL_Param):
    """BSBL model for photometry only

    A Binary point Source Binary Lens model for microlensing.

    Note the attributes, RA (raL) and Dec (decL) are required
    if you are calculating a model with parallax.

    Attributes
    ----------
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    u0_amp: float
        Angular distance between the lens and source on the plane of the
        sky at closest approach in units of thetaE. It can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or
          * negative (u0_amp < 0 when u0_hat[0] < 0).
        Note, since this is a binary source, we are expressing the
        nominal source position as that of the primary star in the source
        binary system.
    tE: float
        Einstein crossing time based on the lens system mass. [MJD]
    piE_E: float
        The microlensing parallax in the East direction in units of thetaE
    piE_N: float
        The microlensing parallax in the North direction in units of thetaE
    q: float
        Mass ratio of the lens binary (low-mass / high-mass)
    sep_SL: float
        Angular separation of the source system position (defined as position of
        primary) from the lens system geometric center (in units of thetaE).
    sepS: float
        Angular separation between the two sources in units of thetaE where
        thetaE is defined with the total binary lens mass.
    phiS: float
        Angle made between the binary source axis and the relative proper motion vector,
        measured in degrees.
    sepL: float
        Angular separation between the two lenses in units of theteaE where
        thetaE is defined with the total binary lens mass.
    phiL: float
        Angle made between the binary lens axis and the relative proper motion vector,
        measured in degrees.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_S1 + f_S2) / (f_S1 + f_s2 + f_L1 + f_L2 + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.

    """

    fitter_param_names = ['t0', 'u0_amp', 'tE', 'piE_E', 'piE_N',
                          'sep_SL', 'sep_S', 'phi_S', 'sep_L', 'phi_L']
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']

    paramAstromFlag = False
    paramPhotFlag = True

    def __init__(self, t0, u0_amp, tE, piE_E, piE_N,
                 sep_SL, sep_S, phi_S, sep_L, phi_L,
                 mag_src_pri, mag_src_sec,
                 b_sff,
                 raL=None, decL=None):
        self.t0 = t0  # time of closest approach for system=primary pos
        self.u0_amp = u0_amp
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.b_sff = b_sff
        self.raL = raL
        self.decL = decL

        # Separation between source and lens
        self.sep_SL = sep # mas

        # Binary source parameters.
        self.sep_S = sep_S  # mas
        self.phi_S = phi_S
        self.mag_src_pri = mag_src_pri
        self.mag_src_sec = mag_src_sec

        # Binary lens parameters.
        self.sep_L = sep_L
        self.phi_L = phi_L

        # Must call after setting parameters.
        # This checks for proper parameter formatting.
        super().__init__()

        # Calculate the microlensing parallax amplitude
        self.piE_amp = np.linalg.norm(self.piE)
        self.piE_E, self.piE_N = self.piE

        # Baseline magnitude
        self.mag_base = self.mag_src_pri \
                        + self.mag_src_sec \
                        + 2.5 * np.log10(self.b_sff)

        # Calculate the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.u0_amp)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        #####
        # Derived binary source and binary lens parameters.
        #####
        # Calculate the angle between muRel and the binary axis
        # in radians.
        self.phi_S_rad = self.phi_S * np.pi / 180.0
        self.phi_L_rad = self.phi_L * np.pi / 180.0

        # Calculate the position angle of the piE vector, East of North (CHECK).
        self.phi_piE_rad = np.arctan2(self.piE[0], self.piE[1])

        # Note that phi_rho1 is the same alpha in our astrometry model;
        # however, here we don't have North as a reference.
        self.phi_S_rho1_rad = self.phi_piE_rad - self.phi_S_rad
        self.phi_L_rho1_rad = self.phi_piE_rad - self.phi_L_rad

        # Position of the two lenses.
        self.xL1_over_theta = np.array([0.5 * self.sep_L * np.sin(self.phi_L_rho1_rad),
                                        0.5 * self.sep_L * np.cos(self.phi_L_rho1_rad)])
        self.xL2_over_theta = np.array([-0.5 * self.sep_L * np.sin(self.phi_L_rho1_rad),
                                        -0.5 * self.sep_L * np.cos(self.phi_L_rho1_rad)])


        # ON NO>>> TWO DIFFERENT CONVENTIONS.
        # Primary -- at origin
        self.t0_pri = self.t0
        self.u0_amp_pri = self.u0_amp
        self.u0_pri = self.u0

        # Secondary
        sep_vec = self.sep * np.array((np.sin(self.phi_rho1_rad),
                                       np.cos(self.phi_rho1_rad)))  # mas

        # Closest approach time
        self.u0_amp_sec = self.u0_amp_pri + np.dot(sep_vec, self.u0_hat)
        self.u0_sec = self.u0_amp_sec * self.u0_hat
        s_murelhat = np.dot(sep_vec, self.muRel_hat)
        self.t0_sec = self.t0_pri - (s_murelhat * self.tE)

        return

class BSBL_PhotAstromParam1(PSPL_Param):
    """BSBL model for astrometry and photometry - physical parameterization.
    
    A binary source binary lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the 
    lens masses and positions and proper motions. 
    Note that this is a STATIC binary lens, i.e. there is no orbital motion.

    Attributes
    ----------
    mLp, mLs : float
        Masses of the lenses (Msun)
    t0 : float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    xS0_E : float
        R.A. of source primary position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source primary position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    beta : float
        Angular distance between the source PRIMARY position
        and the GEOMETRIC center of the lenses on the plane of the sky (mas). 
        Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    muL_E : float
        Lens system proper motion in the RA direction (mas/yr)
    muL_N : float
        Lens system proper motion in the Dec. direction (mas/yr)
    muS_E : float
        Source system proper motion in the RA direction (mas/yr)
    muS_N : float
        Source system proper motion in the Dec. direction (mas/yr)
    dL : float
        Distance from the observer to the lens system (pc)
    dS : float
        Distance from the observer to the source (pc)
    sepL : float
        Angular separation of the lens secondary from the lens primary (mas).
    alphaL : float
        Angle made between the binary lens axis and North; 
        measured in degrees East of North. 
        Example, alphaL = 90 will place the lens primary to the East 
        and the lens secondary to the West.
    sepS : float
        Angular separation of the source secondary from the source primary (mas).
    alphaS : float
        Angle made between the binary source axis and North; 
        measured in degrees East of North.
        Example, alphaL = 90 will place the source primary to the East 
        and the source secondary to the West.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff : numpy array or list
        The ratio of the combined source flux to the total (sources + neighbors + lenses). One
        for each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    fitter_param_names = ['mLp', 'mLs', 't0', 'xS0_E', 'xS0_N',
                          'beta', 'muL_E', 'muL_N', 'muS_E', 'muS_N',
                          'dL', 'dS', 'sepL', 'alphaL', 'sepS', 'alphaS']
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']

    paramAstromFlag = True
    paramPhotFlag = True

    def __init__(self, mLp, mLs, t0, xS0_E, xS0_N,
                 beta, muL_E, muL_N, muS_E, muS_N, dL, dS,
                 sepL, alphaL, sepS, alphaS, 
                 mag_src_pri, mag_src_sec, b_sff,
                 raL=None, decL=None, root_tol=1e-8):
        self.mLp = mLp  # Msun
        self.mLs = mLs  # Msun
        self.t0 = t0
        self.xS0 = np.array([xS0_E, xS0_N])
        self.beta = beta
        self.muL = np.array([muL_E, muL_N])
        self.muS = np.array([muS_E, muS_N])
        self.dL = dL
        self.dS = dS
        self.sepL = sepL
        self.alphaL = alphaL
        self.alphaL_rad = self.alphaL * np.pi / 180.0
        self.sepS = sepS
        self.alphaS = alphaS
        self.alphaS_rad = self.alphaS * np.pi / 180.0
        self.mag_src_pri = mag_src_pri
        self.mag_src_sec = mag_src_sec
        self.b_sff = b_sff
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol

        # Super handles checking for properly formatted variables.
        super().__init__()

        flux_pri = mag2flux(self.mag_src_pri)
        flux_sec = mag2flux(self.mag_src_sec)
        self.mag_base = flux2mag(flux_pri + flux_sec) + 2.5 * np.log10(self.b_sff)

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (dL * units.pc)) - (1.0 / (dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_E, self.muRel_N = self.muRel
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr

        self.muS_E, self.muS_N = self.muS
        self.muL_E, self.muL_N = self.muL

        # Calculate the Einstein radius
        # AFAICT, thetaE for binary lenses is calculated from the total lens mass.
        # Checked using Shin+17 (OB160168) and Jung+19 (OB160156)
        self.mL = self.mLp + self.mLs  # Total lens mass
        thetaE = units.rad * np.sqrt((4.0 * const.G * self.mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Calculate the microlensing parallax
        self.piE = (self.piRel / self.thetaE_amp) * self.thetaE_hat

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0_amp = self.beta / self.thetaE_amp  # in Einstein units
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source).
        # Note this is the angle between the geometric center of the lens and the
        # source primary position. Apologies for the mixed systems, but this is best
        # for fitting purposes.
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat
        self.piE_E, self.piE_N = self.piE

        # Calculate the Einstein crossing time. (days)
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year

        #####
        # Derived binary source parameters. Origin is at the primary.
        #####
        # Primary -- at origin
        self.t0_pri = self.t0
        self.xS0_pri = self.xS0
        self.u0_amp_pri = self.u0_amp
        self.u0_pri = self.u0

        # Secondary
        sepS_vec = self.sepS * np.array((np.sin(self.alphaS_rad),
                                         np.cos(self.alphaS_rad)))  # mas

        # Closest approach time
        self.u0_amp_sec = self.u0_amp_pri + (np.dot(sepS_vec, self.u0_hat) / self.thetaE_amp)
        self.u0_sec = self.u0_amp_sec * self.u0_hat
        s_murelhat = np.dot(sepS_vec, self.muRel_hat)
        self.t0_sec = self.t0_pri - (s_murelhat * days_per_year / self.muRel_amp)
        self.xS0_sec = self.xS0_pri + (sepS_vec * 1e-3) - (s_murelhat * 1e-3 * self.muRel_hat)

        return
    
class BSBL_PhotAstromParam2(PSPL_Param):
    """BSBL model for astrometry and photometry - physical parameterization
    with primary lens and primary source center.
    
    A binary source binary lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the 
    lens masses and positions and proper motions. 
    Note that this is a STATIC binary lens, i.e. there is no orbital motion.

    Attributes
    ----------
    mLp, mLs : float
        Masses of the lenses (Msun)
    t0_p : float
        Time of closest approach between source and PRIMARY LENS (MJD.DDD)
    xS0_E : float
        R.A. of source primary position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source primary position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    beta_p : float
        Angular distance between the source PRIMARY position
        and the PRIMARY center of the lenses on the plane of the sky (mas). 
        Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    muL_E : float
        Lens system proper motion in the RA direction (mas/yr)
    muL_N : float
        Lens system proper motion in the Dec. direction (mas/yr)
    muS_E : float
        Source system proper motion in the RA direction (mas/yr)
    muS_N : float
        Source system proper motion in the Dec. direction (mas/yr)
    dL : float
        Distance from the observer to the lens system (pc)
    dS : float
        Distance from the observer to the source (pc)
    sepL : float
        Angular separation of the lens secondary from the lens primary (mas).
    alphaL : float
        Angle made between the binary lens axis and North; 
        measured in degrees East of North. 
        Example, alphaL = 90 will place the lens primary to the East 
        and the lens secondary to the West.
    sepS : float
        Angular separation of the source secondary from the source primary (mas).
    alphaS : float
        Angle made between the binary source axis and North; 
        measured in degrees East of North.
        Example, alphaL = 90 will place the source primary to the East 
        and the source secondary to the West.
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff : numpy array or list
        The ratio of the combined source flux to the total (sources + neighbors + lenses). One
        for each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    fitter_param_names = ['mLp', 'mLs', 't0_p', 'xS0_E', 'xS0_N',
                          'beta_p', 'muL_E', 'muL_N', 'muS_E', 'muS_N',
                          'dL', 'dS', 'sepL', 'alphaL', 'sepS', 'alphaS']
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']

    paramAstromFlag = True
    paramPhotFlag = True

    def __init__(self, mLp, mLs, t0_p, xS0_E, xS0_N,
                 beta_p, muL_E, muL_N, muS_E, muS_N, dL, dS,
                 sepL, alphaL, sepS, alphaS, 
                 mag_src_pri, mag_src_sec, b_sff,
                 raL=None, decL=None, root_tol=1e-8):
        self.mLp = mLp  # Msun
        self.mLs = mLs  # Msun
        self.t0_p = t0_p
        self.xS0 = np.array([xS0_E, xS0_N])
        self.beta_p = beta_p
        self.muL = np.array([muL_E, muL_N])
        self.muS = np.array([muS_E, muS_N])
        self.dL = dL
        self.dS = dS
        self.sepL = sepL
        self.alphaL = alphaL
        self.alphaL_rad = self.alphaL * np.pi / 180.0
        self.sepS = sepS
        self.alphaS = alphaS
        self.alphaS_rad = self.alphaS * np.pi / 180.0
        self.mag_src_pri = mag_src_pri
        self.mag_src_sec = mag_src_sec
        self.b_sff = b_sff
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol

        # Super handles checking for properly formatted variables.
        super().__init__()

        flux_pri = mag2flux(self.mag_src_pri)
        flux_sec = mag2flux(self.mag_src_sec)
        self.mag_base = flux2mag(flux_pri + flux_sec) + 2.5 * np.log10(self.b_sff)

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (dL * units.pc)) - (1.0 / (dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_E, self.muRel_N = self.muRel
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr

        self.muS_E, self.muS_N = self.muS
        self.muL_E, self.muL_N = self.muL

        # Calculate the Einstein radius
        # AFAICT, thetaE for binary lenses is calculated from the total lens mass.
        # Checked using Shin+17 (OB160168) and Jung+19 (OB160156)
        self.mL = self.mLp + self.mLs  # Total lens mass
        thetaE = units.rad * np.sqrt((4.0 * const.G * self.mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Calculate the microlensing parallax
        self.piE = (self.piRel / self.thetaE_amp) * self.thetaE_hat
        
        # Calculate other angles
        self.phi_piE_rad = np.arctan2(self.piE[0], self.piE[1])
        self.phi_piE = self.phi_piE_rad * 180.0 / np.pi
        self.phiL_rad = self.alphaL_rad - self.phi_piE_rad
        self.phiL = self.phiL_rad * 180.0 / np.pi

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat_p = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta_p)
        self.u0_amp_p = self.beta_p / self.thetaE_amp  # in Einstein units
        self.u0_p = np.abs(self.u0_amp_p) * self.u0_hat_p

        # Calculate the Einstein crossing time. (days)
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year
        
        # Calculate t0 and beta for geometric center of lens
        self.q = self.mLs/self.mLp
        u0_x_out, u0_y_out, t0_out = fc.convert_u0_t0_psbl(t0_in = self.t0_p, u0_x_in = self.u0_p[0], u0_y_in = self.u0_p[1],
                           tE = self.tE, theta_E = self.thetaE_amp, q = self.q, phi = self.phiL_rad, sep = self.sepL, mu_rel_x = self.muRel[0],
                           mu_rel_y = self.muRel[1], coords_in='prim_center', coords_out='geom_mid')
        self.u0 = np.array([u0_x_out, u0_y_out])
        self.u0_amp = np.sqrt(self.u0[0]**2 + self.u0[1]**2)
        self.u0_hat = self.u0/self.u0_amp
        self.t0 = t0_out
        self.beta = self.u0_amp*self.thetaE_amp
        
        # Angular separation vector between source and lens (vector from lens to source).
        # Note this is the angle between the geometric center of the lens and the
        # source primary position. Apologies for the mixed systems, but this is best
        # for fitting purposes.
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat
        self.piE_E, self.piE_N = self.piE


        #####
        # Derived binary source parameters. Origin is at the primary.
        #####
        # Primary -- at origin
        self.t0_pri = self.t0
        self.xS0_pri = self.xS0
        self.u0_amp_pri = self.u0_amp
        self.u0_pri = self.u0

        # Secondary
        sepS_vec = self.sepS * np.array((np.sin(self.alphaS_rad),
                                         np.cos(self.alphaS_rad)))  # mas

        # Closest approach time
        self.u0_amp_sec = self.u0_amp_pri + (np.dot(sepS_vec, self.u0_hat) / self.thetaE_amp)
        self.u0_sec = self.u0_amp_sec * self.u0_hat
        s_murelhat = np.dot(sepS_vec, self.muRel_hat)
        self.t0_sec = self.t0_pri - (s_murelhat * days_per_year / self.muRel_amp)
        self.xS0_sec = self.xS0_pri + (sepS_vec * 1e-3) - (s_murelhat * 1e-3 * self.muRel_hat)

        return
        return
    
class BSBL_PhotAstrom_EllOrbs_Param1(PSPL_Param):
    """BSBL model for astrometry and photometry - physical parameterization.
    
    A binary source binary lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the 
    lens masses and positions and proper motions. 
    Note that this is a STATIC binary lens, i.e. there is no orbital motion.

    Attributes
    ----------
    mLp, mLs : float
        Masses of the lenses (Msun)
    t0 : float
        Time of closest approach between source COM and Lens COM (MJD.DDD)
    xS0_E : float
        R.A. of source COM position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source COM position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    beta : float
        Angular distance between the source COM position
        and the GEOMETRIC center of the lenses on the plane of the sky (mas). 
        Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    muL_E : float
        Lens system proper motion in the RA direction (mas/yr)
    muL_N : float
        Lens system proper motion in the Dec. direction (mas/yr)
    muS_E : float
        Source system proper motion in the RA direction (mas/yr)
    muS_N : float
        Source system proper motion in the Dec. direction (mas/yr)
    dL : float
        Distance from the observer to the lens system (pc)
    dS : float
        Distance from the observer to the source (pc)
    alphaL : float
        Angle made between the binary lens axis and North; 
        measured in degrees East of North. 
        Example, alphaL = 90 will place the lens primary to the East 
        and the lens secondary to the West.
    alphaS : float
        Angle made between the binary source axis and North; 
        measured in degrees East of North.
        Example, alphaL = 90 will place the source primary to the East 
        and the source secondary to the West.
    omegaL: float
        The argument of periastron of the primary lens's orbit in degrees. 
    big_omegaL: float
        The longitude of the ascending node of the secondary lens's orbit
        in degrees. 
    iL: float
        Inclination angle of the system in degrees. 
    eL: float
        Eccentricity of the System
    pL: float
        Orbital period of the system in days.
    tpL: float
        This is the time of the periastron of the system in days.
    alephL: float
        This is the semi-major axis of the primary lens in mas. 
    aleph_secL: float
        This is the semi-major axis of the secondary lens in mas. 
    omegaS: float
        The argument of periastron of the primary source's orbit in degrees. 
    big_omegaS: float
        The longitude of the ascending node of the secondary source's orbit
        in degrees. 
    is: float
        Inclination angle of the system in degrees. 
    eS: float
        Eccentricity of the System
    pS: float
        Orbital period of the system in days.
    tpS: float
        This is the time of the periastron of the system in days.
    alephS: float
        This is the semi-major axis of the primary source in mas. 
    aleph_secS: float
        This is the semi-major axis of the secondary source in mas. 
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff : numpy array or list
        The ratio of the combined source flux to the total (sources + neighbors + lenses). One
        for each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    fitter_param_names = ['mLp', 'mLs', 't0', 'xS0_E', 'xS0_N',
                          'beta', 'muL_E', 'muL_N', 'muS_E', 'muS_N',
                          'dL', 'dS', 'alphaL', 'alphaS',  'omegaL', 'big_omegaL', 'iL', 'eL', 'pL', 'tpL', 'alephL', 'aleph_secL', 
                 'omegaS', 'big_omegaS', 'iS', 'eS', 'pS', 'tpS', 'alephS', 'aleph_secS']
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']
    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag='circular'

    def __init__(self, mLp, mLs, t0_com, xS0_E, xS0_N,
                 beta_com, muL_E, muL_N, muS_E, muS_N, dL, dS,
                alphaL, alphaS, omegaL, big_omegaL, iL, eL, pL, tpL, alephL, aleph_secL, 
                 omegaS, big_omegaS, iS, eS, pS, tpS, alephS, aleph_secS, 
                 mag_src_pri, mag_src_sec, b_sff,
                 raL=None, decL=None, root_tol=1e-8):
        self.mLp = mLp  # Msun
        self.mLs = mLs  # Msun
        self.t0_com = t0_com
        self.xS0 = np.array([xS0_E, xS0_N])
        self.xS0_E = xS0_E
        self.xS0_N = xS0_N
        self.beta_com = beta_com
        self.muL = np.array([muL_E, muL_N])
        self.muS = np.array([muS_E, muS_N])
        self.dL = dL
        self.dS = dS
        self.sepL = alephL+aleph_secL
        self.sepS = alephS+aleph_secS

        self.alphaL = alphaL
        self.alphaL_rad = self.alphaL * np.pi / 180.0
        self.alphaS = alephS + aleph_secS
        self.alphaS_rad = self.alphaS * np.pi / 180.0
        self.omegaL = omegaL
        self.big_omegaL = big_omegaL
        self.iL = iL
        self.eL = eL
        self.pL = pL
        self.tpL = tpL
        self.alephL = alephL
        self.aleph_secL = aleph_secL
                     
        self.omegaS = omegaS
        self.big_omegaS = big_omegaS
        self.iS = iS
        self.eS = eS
        self.pS = pS
        self.tpS = tpS
        self.alephS = alephS
        self.aleph_secS = aleph_secS
                     
        self.mag_src_pri = mag_src_pri
        self.mag_src_sec = mag_src_sec
        self.b_sff = b_sff
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol
                     
        self.vxS = muS_E * 1e-3
        self.vyS = muS_N *1e-3

        self.vxL = muL_E * 1e-3
        self.vyL = muL_N *1e-3

                     
        # Super handles checking for properly formatted variables.
        super().__init__()

        flux_pri = mag2flux(self.mag_src_pri)
        flux_sec = mag2flux(self.mag_src_sec)
        self.mag_base = flux2mag(flux_pri + flux_sec) + 2.5 * np.log10(self.b_sff)

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (dL * units.pc)) - (1.0 / (dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_E, self.muRel_N = self.muRel
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr

        self.muS_E, self.muS_N = self.muS
        self.muL_E, self.muL_N = self.muL

        # Calculate the Einstein radius
        # AFAICT, thetaE for binary lenses is calculated from the total lens mass.
        # Checked using Shin+17 (OB160168) and Jung+19 (OB160156)
        self.mL = self.mLp + self.mLs  # Total lens mass
        thetaE = units.rad * np.sqrt((4.0 * const.G * self.mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Calculate the microlensing parallax
        self.piE = (self.piRel / self.thetaE_amp) * self.thetaE_hat

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta_com)
        self.u0_amp = self.beta_com / self.thetaE_amp  # in Einstein units
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source).
        # Note this is the angle between the geometric center of the lens and the
        # source primary position. Apologies for the mixed systems, but this is best
        # for fitting purposes.
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)
        self.xL0_E, self.xL0_N = self.xL0

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat
        self.piE_E, self.piE_N = self.piE

        # Calculate the Einstein crossing time. (days)
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year

        #####

        
        # Calculate t0 and beta for geometric center of lens
        self.phi_rad = self.alphaL_rad - np.arctan2(self.piE_E, self.piE_N)
        self.q = self.mLs/self.mLp
        u0_x_out, u0_y_out, t0_out = fc.convert_u0_t0_psbl(t0_in = self.t0_com, u0_x_in = self.u0[0], u0_y_in = self.u0[1],
                           tE = self.tE, theta_E = self.thetaE_amp, q = self.q, phi = self.phi_rad, sep = self.sepL, mu_rel_x = self.muRel[0],
                           mu_rel_y = self.muRel[1], coords_in='COM', coords_out='geom_mid')                     
        self.u0 = np.array([u0_x_out, u0_y_out])
        self.u0_amp = np.sqrt(self.u0[0]**2 + self.u0[1]**2)
        self.t0 = t0_out
        self.beta = self.u0_amp*self.thetaE_amp



                     
        return




    
class BSBL_PhotAstrom_CircOrbs_Param1(BSBL_PhotAstrom_EllOrbs_Param1):
    """BSBL model for astrometry and photometry - physical parameterization.
    
    A binary source binary lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the 
    lens masses and positions and proper motions. 
    Note that this is a STATIC binary lens, i.e. there is no orbital motion.

    Attributes
    ----------
    mLp, mLs : float
        Masses of the lenses (Msun)
    t0 : float
        Time of closest approach between source COM and Lens COM (MJD.DDD)
    xS0_E : float
        R.A. of source COM position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source COM position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    beta : float
        Angular distance between the source COM position
        and the GEOMETRIC center of the lenses on the plane of the sky (mas). 
        Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    muL_E : float
        Lens system proper motion in the RA direction (mas/yr)
    muL_N : float
        Lens system proper motion in the Dec. direction (mas/yr)
    muS_E : float
        Source system proper motion in the RA direction (mas/yr)
    muS_N : float
        Source system proper motion in the Dec. direction (mas/yr)
    dL : float
        Distance from the observer to the lens system (pc)
    dS : float
        Distance from the observer to the source (pc)
    alphaL : float
        Angle made between the binary lens axis and North; 
        measured in degrees East of North. 
        Example, alphaL = 90 will place the lens primary to the East 
        and the lens secondary to the West.
    alphaS : float
        Angle made between the binary source axis and North; 
        measured in degrees East of North.
        Example, alphaL = 90 will place the source primary to the East 
        and the source secondary to the West.
    omegaL: float
        The argument of periastron of the primary lens's orbit in degrees. 
    big_omegaL: float
        The longitude of the ascending node of the secondary lens's orbit
        in degrees. 
    iL: float
        Inclination angle of the system in degrees. 
    pL: float
        Orbital period of the system in days.
    tpL: float
        This is the time of the periastron of the system in days.
    alephL: float
        This is the semi-major axis of the primary lens in mas. 
    aleph_secL: float
        This is the semi-major axis of the secondary lens in mas. 
    omegaS: float
        The argument of periastron of the primary source's orbit in degrees. 
    big_omegaS: float
        The longitude of the ascending node of the secondary source's orbit
        in degrees. 
    is: float
        Inclination angle of the system in degrees. 
    pS: float
        Orbital period of the system in days.
    tpS: float
        This is the time of the periastron of the system in days.
    alephS: float
        This is the semi-major axis of the primary source in mas. 
    aleph_secS: float
        This is the semi-major axis of the secondary source in mas. 
    mag_src_pri: array or list
        Photometric magnitude of the first (primary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    mag_src_sec: array or list
        Photometric magnitude of the second (secondary) source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    b_sff : numpy array or list
        The ratio of the combined source flux to the total (sources + neighbors + lenses). One
        for each filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    fitter_param_names = ['mLp', 'mLs', 't0', 'xS0_E', 'xS0_N',
                          'beta', 'muL_E', 'muL_N', 'muS_E', 'muS_N',
                          'dL', 'dS', 'alphaL', 'alphaS',  'omegaL', 'big_omegaL', 'iL',  'pL', 'tpL', 'alephL', 'aleph_secL', 
                 'omegaS', 'big_omegaS', 'iS', 'pS', 'tpS', 'alephS', 'aleph_secS']
    phot_param_names = ['mag_src_pri', 'mag_src_sec', 'b_sff']
    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag='circular'

    def __init__(self, mLp, mLs, t0_com, xS0_E, xS0_N,
                 beta_com, muL_E, muL_N, muS_E, muS_N, dL, dS,
                alphaL, alphaS, omegaL, big_omegaL, iL, pL, tpL, alephL, aleph_secL, 
                 omegaS, big_omegaS, iS, pS, tpS, alephS, aleph_secS, 
                 mag_src_pri, mag_src_sec, b_sff,
                 raL=None, decL=None, root_tol=1e-8):
        super().__init__( mLp, mLs, t0_com, xS0_E, xS0_N,
                 beta_com, muL_E, muL_N, muS_E, muS_N, dL, dS,
                alphaL, alphaS, omegaL, big_omegaL, iL, 0, pL, tpL, alephL, aleph_secL, 
                 omegaS, big_omegaS, iS, 0, pS, tpS, alephS, aleph_secS, 
                 mag_src_pri, mag_src_sec, b_sff,
                 raL=raL, decL=decL, root_tol=root_tol)
                     
        return



    

class BSBL_PhotAstrom_EllOrbs_Param2(PSPL_Param):
    """BSBL model for astrometry and photometry - physical parameterization.
    
    A binary source binary lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the 
    lens masses and positions and proper motions. 
    Note that this is a STATIC binary lens, i.e. there is no orbital motion.

    Attributes
    ----------
    t0_com : float
        Time of photometric peak, as seen from Earth (MJD.DDD) 
        FIXME: THIS IS NOT RIGHT
    u0_amp_com : float
        Angular distance between the source and the binary lens COM
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    q : float
            Mass ratio (M2 / M1)
    xS0_E : float
        R.A. of source COM position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source COM position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.

        
    alphaL : float
        Angle made between the binary lens axis and North; 
        measured in degrees East of North. 
        Example, alphaL = 90 will place the lens primary to the East 
        and the lens secondary to the West.
    alphaS : float
        Angle made between the binary source axis and North; 
        measured in degrees East of North.
        Example, alphaL = 90 will place the source primary to the East 
        and the source secondary to the West.
    omegaL: float
        The argument of periastron of the primary lens's orbit in degrees. 
    big_omegaL: float
        The longitude of the ascending node of the secondary lens's orbit
        in degrees. 
    iL: float
        Inclination angle of the system in degrees. 
    eL: float
        Eccentricity of the System
    pL: float
        Orbital period of the system in days.
    tpL: float
        This is the time of the periastron of the system in days.
    alephL: float
        This is the semi-major axis of the primary lens in mas. 
    aleph_secL: float
        This is the semi-major axis of the secondary lens in mas. 
    omegaS: float
        The argument of periastron of the primary source's orbit in degrees. 
    big_omegaS: float
        The longitude of the ascending node of the secondary source's orbit
        in degrees. 
    is: float
        Inclination angle of the system in degrees. 
    eS: float
        Eccentricity of the System
    pS: float
        Orbital period of the system in days.
    tpS: float
        This is the time of the periastron of the system in days.
    alephS: float
        This is the semi-major axis of the primary source in mas. 
    aleph_secS: float
        This is the semi-major axis of the secondary source in mas. 

        
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_{base} = f_{src1} + f_{src2} + f_{blend}`
        such that
            :math:`b_sff = (f_{src1}+ f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    fitter_param_names = ['t0_com', 'u0_amp_com', 'tE', 'thetaE', 'piS', 'piE_E', 'piE_N', 'q', 'xS0_E', 'xS0_N', 'muS_E', 'muS_N',
                'alphaL', 'alphaS',  'omegaL', 'big_omegaL', 'iL',  'pL', 'tpL', 'alephL', 'aleph_secL', 
                 'omegaS', 'big_omegaS', 'iS', 'pS', 'tpS', 'alephS', 'aleph_secS']
    phot_param_names = ['fratio_bin', 'mag_base', 'b_sff']
    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag='circular'

    def __init__(self, t0_com, u0_amp_com, tE, thetaE, piS, piE_E, piE_N, q, xS0_E, xS0_N, muS_E, muS_N,
                alphaL, alphaS, omegaL, big_omegaL, iL, eL, pL, tpL, alephL, aleph_secL, 
                 omegaS, big_omegaS, iS, eS, pS, tpS, alephS, aleph_secS, fratio_bin, mag_base, b_sff,
                 raL=None, decL=None, root_tol=1e-8):
                     
        self.t0_com = t0_com
        self.u0_amp_com = u0_amp_com
        self.tE = tE
        self.piE = np.array([piE_E, piE_N])
        self.thetaE_amp = thetaE
        self.piS = piS
        self.q = q
        self.sepL = alephL+aleph_secL
        self.sepS = alephS+aleph_secS
        self.muS = np.array([muS_E, muS_N])
        self.muS_E, self.muS_N = self.muS
        self.xS0 = np.array([xS0_E, xS0_N])
        self.xS0_E = xS0_E
        self.xS0_N = xS0_N

                     
        self.alphaL = alphaL
        self.alphaL_rad = self.alphaL * np.pi / 180.0
        self.alphaS = alephS + aleph_secS
        self.alphaS_rad = self.alphaS * np.pi / 180.0
        self.omegaL = omegaL
        self.big_omegaL = big_omegaL
        self.iL = iL
        self.eL = eL
        self.pL = pL
        self.tpL = tpL
        self.alephL = alephL
        self.aleph_secL = aleph_secL
                     
        self.omegaS = omegaS
        self.big_omegaS = big_omegaS
        self.iS = iS
        self.eS = eS
        self.pS = pS
        self.tpS = tpS
        self.alephS = alephS
        self.aleph_secS = aleph_secS


        self.mag_base = np.array([mag_base])
        self.b_sff = np.array([b_sff])      
        self.fratio_bin =  np.array([fratio_bin])
                     
        self.mag_src_pri = self.mag_base - 2.5 * np.log10(self.b_sff) + 2.5 * np.log10(1.0 + self.fratio_bin)
        self.mag_src_sec = self.mag_base - 2.5 * np.log10(self.b_sff) + 2.5 * np.log10(1.0 + (1.0 / self.fratio_bin))
        self.raL = raL
        self.decL = decL
        self.root_tol = root_tol
                     
        self.vxS = muS_E * 1e-3
        self.vyS = muS_N *1e-3


                     
        # Super handles checking for properly formatted variables.
        super().__init__()


        
        # Derived quantities
        self.phi_radL = self.alphaL_rad - np.arctan2(piE_E, piE_N)
        qeff = (1 - q)/(1 + q)
        self.t0 = self.t0_com - 0.5 * qeff * self.tE * self.sepL * np.cos(self.phi_radL) / self.thetaE_amp
        self.u0_amp = self.u0_amp_com - 0.5 * qeff * self.sepL * np.sin(self.phi_radL) / self.thetaE_amp

                     
        self.beta = self.u0_amp * self.thetaE_amp
        self.piE_amp = np.linalg.norm(self.piE)
        self.piRel = self.piE_amp * self.thetaE_amp
        self.muRel_amp = self.thetaE_amp / (self.tE / days_per_year)
        self.piL = self.piRel + self.piS

        kappa_tmp = 4.0 * const.G / (const.c ** 2 * units.AU)
        kappa = kappa_tmp.to(units.mas / units.Msun,
                             equivalencies=units.dimensionless_angles()).value
        self.mL = self.thetaE_amp ** 2 / (self.piRel * kappa)
        self.mLp = self.mL / (1.0 + self.q)
        self.mLs = self.mLp * self.q
        
        
        # Calculate the distance to source and lens.
        dL = (self.piL * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        dS = (self.piS * units.mas).to(units.parsec,
                                       equivalencies=units.parallax())
        self.dL = dL.to('pc').value
        self.dS = dS.to('pc').value

        # Get the directional vectors.
        self.thetaE_hat = self.piE / self.piE_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat

        # Calculate the relative velocity vector. Note that this will be in the
        # direction of theta_hat
        self.muRel = self.muRel_amp * self.thetaE_hat
        self.muRel_E, self.muRel_N = self.muRel
        self.muL = self.muS - self.muRel
        self.muL_E, self.muL_N = self.muL

        self.vxL = self.muL_E * 1e-3
        self.vyL = self.muL_N *1e-3
                     
        # Calculate m1 and m2 (see PSBL writeup) -- note these are the individual Einstein radii**2
        inv_dist_diff = (1.0 / dL) - (1.0 / dS)
        m1 = units.rad**2 * (4 * const.G * self.mLp * units.Msun / const.c ** 2) * inv_dist_diff
        m2 = units.rad**2 * (4 * const.G * self.mLs * units.Msun / const.c ** 2) * inv_dist_diff
        self.m1 = m1.to(units.arcsec ** 2).value  # arcsec^2
        self.m2 = m2.to(units.arcsec ** 2).value

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)
        self.xL0_E, self.xL0_N = self.xL0


                     
        return



class BSBL_PhotAstrom_CircOrbs_Param2(BSBL_PhotAstrom_EllOrbs_Param2):
    """BSBL model for astrometry and photometry - physical parameterization.
    
    A binary source binary lens model for microlensing. This model uses a
    parameterization that depends on only physical quantities such as the 
    lens masses and positions and proper motions. 
    Note that this is a STATIC binary lens, i.e. there is no orbital motion.

    Attributes
    ----------
    t0_com : float
        Time of photometric peak, as seen from Earth (MJD.DDD) 
        FIXME: THIS IS NOT RIGHT
    u0_amp_com : float
        Angular distance between the source and the binary lens COM
        on the plane of the sky at closest approach in units of thetaE. Can be
          * positive (u0_amp > 0 when u0_hat[0] > 0) or 
          * negative (u0_amp < 0 when u0_hat[0] < 0).
    tE : float
        Einstein crossing time (days).
    thetaE : float
        The size of the Einstein radius in (mas).
    piS : float
        Amplitude of the parallax (1AU/dS) of the source. (mas)
    piE_E : float
        The microlensing parallax in the East direction in units of thetaE
    piE_N : float
        The microlensing parallax in the North direction in units of thetaE
    q : float
            Mass ratio (M2 / M1)
    xS0_E : float
        R.A. of source COM position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.
    xS0_N : float
        Dec. of source COM position on sky at t = t0 (arcsec) in an
        arbitrary ref. frame.

        
    alphaL : float
        Angle made between the binary lens axis and North; 
        measured in degrees East of North. 
        Example, alphaL = 90 will place the lens primary to the East 
        and the lens secondary to the West.
    alphaS : float
        Angle made between the binary source axis and North; 
        measured in degrees East of North.
        Example, alphaL = 90 will place the source primary to the East 
        and the source secondary to the West.
    omegaL: float
        The argument of periastron of the primary lens's orbit in degrees. 
    big_omegaL: float
        The longitude of the ascending node of the secondary lens's orbit
        in degrees. 
    iL: float
        Inclination angle of the system in degrees. 
    pL: float
        Orbital period of the system in days.
    tpL: float
        This is the time of the periastron of the system in days.
    alephL: float
        This is the semi-major axis of the primary lens in mas. 
    aleph_secL: float
        This is the semi-major axis of the secondary lens in mas. 
    omegaS: float
        The argument of periastron of the primary source's orbit in degrees. 
    big_omegaS: float
        The longitude of the ascending node of the secondary source's orbit
        in degrees. 
    is: float
        Inclination angle of the system in degrees. 
    pS: float
        Orbital period of the system in days.
    tpS: float
        This is the time of the periastron of the system in days.
    alephS: float
        This is the semi-major axis of the primary source in mas. 
    aleph_secS: float
        This is the semi-major axis of the secondary source in mas. 

        
    fratio_bin: float
        Flux ratio of secondary flux / primary flux.
    mag_base : array or list
        Photometric magnitude of the base. This must be passed in as a
        list or array, with one entry for each photometric filter.
        Note that
            :math:`flux_{base} = f_{src1} + f_{src2} + f_{blend}`
        such that
            :math:`b_sff = (f_{src1}+ f_{src2}) / ( f_{src1} + f_{src2} + f_{blend} )`
    b_sff: array or list
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = (f_{S1} + f_{S2}) / (f_{S1} + f_{s2} + f_L + f_N)`.
        This must be passed in as a list or
        array, with one entry for each photometric filter.
    root_tol : float
        Tolerance in comparing the polynomial roots to the physical solutions. Default = 1e-8
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    fitter_param_names = ['t0_com', 'u0_amp_com', 'tE', 'thetaE', 'piS', 'piE_E', 'piE_N', 'q', 'xS0_E', 'xS0_N', 'muS_E', 'muS_N',
                'alphaL', 'alphaS',  'omegaL', 'big_omegaL', 'iL',  'pL', 'tpL', 'alephL', 'aleph_secL', 
                 'omegaS', 'big_omegaS', 'iS', 'pS', 'tpS', 'alephS', 'aleph_secS']
    phot_param_names = ['fratio_bin', 'mag_base', 'b_sff']
    paramAstromFlag = True
    paramPhotFlag = True
    orbitFlag='circular'

    def __init__(self, t0_com, u0_amp_com, tE, thetaE, piS, piE_E, piE_N, q, xS0_E, xS0_N, muS_E, muS_N,
                alphaL, alphaS, omegaL, big_omegaL, iL, pL, tpL, alephL, aleph_secL, 
                 omegaS, big_omegaS, iS, pS, tpS, alephS, aleph_secS, fratio_bin, mag_base, b_sff,
                 raL=None, decL=None, root_tol=1e-8):
                     
        super().__init__(t0_com, u0_amp_com, tE, thetaE, piS, piE_E, piE_N, q, xS0_E, xS0_N, muS_E, muS_N,
                alphaL, alphaS, omegaL, big_omegaL, iL, 0, pL, tpL, alephL, aleph_secL, 
                 omegaS, big_omegaS, iS, 0, pS, tpS, alephS, aleph_secS, fratio_bin, mag_base, b_sff,
                 raL=raL, decL=decL, root_tol=1e-8)

                     
        # Super handles checking for properly formatted variables.
        PSPL_Param().__init__()


        return






                     

# ==================================================
# FSPL Models
# Finite-Source Point Lens Models.
# NOT DONE YET... place holders
# DO NOT USE
# ==================================================
class FSPL(PSPL):
    def get_source_outline_astrometry(self, r, n, center):
        """Return astrometric points that outline the outer circumference of the
        source star. 

        | The outline is described as a circle of radius
          self.radius and is evaluated at self.n_outline number of points. 
        
        | takes in the radius of the circle, centre position and number of points we are 
          approximating the circle by and returns a numpy array of positions
            
        | e.g: ``( ((1,0), (0,1), (-1,0), (0,-1)) )`` if n = 4 and radius = 1
        
        Returns
        -------
        source_points : numpy array
            Returns an array of ``shape = [2, self.n_outline, len(time)]``

        """
        sourcepos = []
        for i in range(n):
            # (rcosa, rsina), # n positions of the boundary of the star equally spaced
            sourcepos.append([center[0] + self.radius * np.cos((2 * i * np.pi) / (n)),
                              center[1] + self.radius * np.sin((2 * i * np.pi) / (n))])
        return np.array(sourcepos)



    def get_photometry(self, t_obs, filt_idx=0, amp_arr=None, print_warning=True):
        '''
        Get the photometry for each of the lensed source images.

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.
        filt_idx : int, optional
            Index of the photometric filter or data set.

        Other Parameters
        ----------------
        amp_arr : array_like
            Amplifications of each individual image at each time,
            i.e. amp_arr.shape = (len(t_obs), number of images at each t_obs).

            This will over-ride t_obs; but is more efficient when calculating
            both photometry and astrometry. If None, then just use t_obs.
        print_warning : bool, optional
            Print a warning in the rare case that the magnitude exceeds a 
            zeropoint of 30 and conversions result in NaN returned.

        Returns
        -------
        mag_model : array_like
            Magnitude of the unresolved microlensing event at t_obs.

        '''
        mag_zp = 30.0  # arbitrary but allows for negative blend fractions.
        flux_zp = 1.0

        if amp_arr is None:
            amp_arr, img_arr = self.get_centroids(t_obs, self.radius)
            amp = amp_arr

            # CHECK THIS STUFF
        #         # Do we need this masked stuff?
        #         # Mask invalid values from the amplification array.
        #         amp_arr_msk = np.ma.masked_invalid(amp_arr)
        #
        #         # Sum up all the amplifications b/c surface brightness is conserved.
        #         amp = np.sum(amp_arr_msk, axis=1)

        flux_src = flux_zp * 10 ** ((self.mag_src[filt_idx] - mag_zp) / -2.5)
        flux_model = flux_src * amp

        # Account for blending, if necessary.
        try:
            # Adding flux of neighbors and lenses
            # b_sff = fS / (fS + fN + fL)
            flux_model += flux_src * (1.0 - self.b_sff[filt_idx]) / self.b_sff[filt_idx]
        except AttributeError:
            pass
        
        # Catch the edge case where we exceed the zeropoint.
        bad = np.where(flux_model <= 0)[0]
        if len(bad) > 0:
            if print_warning:
                print('!!!!!!! Warning: get_photometry: bad flux encountered.')
            flux_model[bad] = np.nan

        mag_model = -2.5 * np.log10(flux_model / flux_zp) + mag_zp

        return mag_model


class FSPL_PhotAstrom(FSPL, PSPL_PhotAstrom):
    """
    Contains methods for model a PSPL photometry + astrometry.
    This is a Data-type class in our hierarchy. It is abstract and should not
    be instantiated. 

    Attributes
    ----------
    Available class variables that should be defined.

    t0
    tE
    u0_amp
    u0_E
    u0_N
    beta
    piE_E - valid only if parallax model
    piE_N - valid only if parallax model
    piE_amp
    mL
    thetaE_amp
    thetaE_E
    thetaE_N
    xS0_E
    xS0_N
    xL0_E
    xL0_N
    muS_E
    muS_N
    muL_E
    muL_N
    muRel_E
    muRel_N
    muRel_amp
    piS
    piL
    dL
    dS
    dL_dS (dL over dS)
    radius
    n
    b_sff[#]
    mag_src[#] -- add in
    mag_base[#] -- add in 
    raL - if parallax model
    decL - if parallax model

    """
    photometryFlag = True
    astrometryFlag = True

    # Inhertied functions include:
    # get_lens_astrometry

    def get_lens_astrometry(self, t):
        # returns the position of the lens in the sky at a list of times, t in units of einstein time
        # t is an np array of numbers
        t_yrs = (t - self.t0) / days_per_year
        xl = self.xL0 + np.outer(t_yrs,
                                 self.muL) * 1e-3  # this is just x = x0 + vt for a constant velocity object
        return xl

    # Analogous to get_all_arrays for PSBL
    # Do we want to rename/reorder to be consistent with PSBL or nah?
    def get_centroids(self, t, r):
        """Calculates the magnification at a list of times.
        
        | List of times t, are in units of einstein time.
        | Implements an algorithm where we can use green's theorem to change an area integral of the images/source
          into a path integral around the outside.
        | We then do a contour plot and approximate this integral.

        """
        images = self.get_resolved_astrometry(t)
        plus = images[0]
        minus = images[1]
        # positions of the images
        area_plus = []
        area_minus = []
        centroid_plus = []
        centroid_minus = []

        for i in range(len(t)):
            Aplus = 0
            Aminus = 0
            Cplus1 = 0
            Cplus2 = 0
            Cminus1 = 0
            Cminus2 = 0
            for j in range(len(plus[0])):
                if j == len(plus[
                                0]) - 1:  # this is the last element, so index j+1 is out of bounds. so we use index 1
                    Aplus += plus[i, j, 0] * (plus[i, 0, 1] - plus[i, j, 1]) - \
                             plus[i, j, 1] * (plus[i, 0, 0] - plus[i, j, 0])
                    Aminus += minus[i, j, 0] * (
                            minus[i, 0, 1] - minus[i, j, 1]) - minus[
                                  i, j, 1] * (minus[i, 0, 0] - minus[i, j, 0])
                    Cplus1 += (plus[i, j, 0] ** 2 + plus[i, 0, 0] ** 2) * (
                            plus[i, 0, 1] - plus[i, j, 1]) + (
                                      plus[i, j, 0] ** 2 - plus[
                                  i, 0, 0] ** 2) * (
                                      plus[i, 0, 1] + plus[i, j, 1])
                    Cplus2 += (plus[i, j, 1] ** 2 - plus[i, 0, 1] ** 2) * (
                            plus[i, 0, 0] + plus[i, j, 0]) + (
                                      plus[i, j, 1] ** 2 + plus[
                                  i, 0, 1] ** 2) * (
                                      plus[i, 0, 0] - plus[i, j, 0])
                    Cminus1 += (minus[i, j, 0] ** 2 + minus[i, 0, 0] ** 2) * (
                            minus[i, 0, 1] - minus[i, j, 1]) + (
                                       minus[i, j, 0] ** 2 - minus[
                                   i, 0, 0] ** 2) * (
                                       minus[i, 0, 1] + minus[i, j, 1])
                    Cminus2 += (minus[i, j, 1] ** 2 - minus[i, 0, 1] ** 2) * (
                            minus[i, 0, 0] + minus[i, j, 0]) + (
                                       minus[i, j, 1] ** 2 + minus[
                                   i, 0, 1] ** 2) * (
                                       minus[i, 0, 0] - minus[i, j, 0])

                else:
                    Aplus += plus[i, j, 0] * (
                            plus[i, j + 1, 1] - plus[i, j, 1]) - plus[
                                 i, j, 1] * (plus[i, j + 1, 0] - plus[i, j, 0])
                    Aminus += minus[i, j, 0] * (
                            minus[i, j + 1, 1] - minus[i, j, 1]) - minus[
                                  i, j, 1] * (
                                      minus[i, j + 1, 0] - minus[i, j, 0])
                    Cplus1 += (plus[i, j, 0] ** 2 + plus[i, j + 1, 0] ** 2) * (
                            plus[i, j + 1, 1] - plus[i, j, 1]) + (
                                      plus[i, j, 0] ** 2 - plus[
                                  i, j + 1, 0] ** 2) * (
                                      plus[i, j + 1, 1] + plus[i, j, 1])
                    Cplus2 += (plus[i, j, 1] ** 2 - plus[i, j + 1, 1] ** 2) * (
                            plus[i, j + 1, 0] + plus[i, j, 0]) + (
                                      plus[i, j, 1] ** 2 + plus[
                                  i, j + 1, 1] ** 2) * (
                                      plus[i, j + 1, 0] - plus[i, j, 0])
                    Cminus1 += (minus[i, j, 0] ** 2 + minus[
                        i, j + 1, 0] ** 2) * (minus[i, j + 1, 1] - minus[
                        i, j, 1]) + (minus[i, j, 0] ** 2 - minus[
                        i, j + 1, 0] ** 2) * (
                                       minus[i, j + 1, 1] + minus[i, j, 1])
                    Cminus2 += (minus[i, j, 1] ** 2 - minus[
                        i, j + 1, 1] ** 2) * (minus[i, j + 1, 0] + minus[
                        i, j, 0]) + (minus[i, j, 1] ** 2 + minus[
                        i, j + 1, 1] ** 2) * (
                                       minus[i, j + 1, 0] - minus[i, j, 0])

            area_plus.append(abs(Aplus))
            area_minus.append(abs(Aminus))
            centroid_plus.append(
                [(1 / (4 * Aplus)) * Cplus1, (-1 / (4 * Aplus)) * Cplus2])
            centroid_minus.append(
                [(1 / (4 * Aminus)) * Cminus1, (-1 / (4 * Aminus)) * Cminus2])
        area_plus = np.array(area_plus)
        area_minus = np.array(area_minus)
        amplification = (1 / (2 * np.pi * (self.radius) ** 2)) * (
                area_plus + area_minus)
        centroid_plus = np.array(centroid_plus)
        centroid_minus = np.array(centroid_minus)
        centroid = centroid_plus * (
                area_plus / (area_plus + area_minus)).reshape(
            area_plus.size, 1) + centroid_minus * (
                           area_minus / (area_plus + area_minus)).reshape(
            area_plus.size, 1)

        return (amplification, centroid)  # area of images/area of source

    # FIXME: I don't think this works... need to check what the shape of amp_array is.
    #     def get_resolved_photometry(self, t_obs, filt_idx=0, amp_arr=None):
    #         '''
    #         Get the photometry for each of the lensed source images.
    #         Implement with no blending (since we don't support different
    #         blendings for the different images).
    #
    #         Parameters
    #         ----------
    #         t_obs : array_like
    #             Array of times to model.
    #
    #         Other Parameters
    #         ----------------
    #         amp_arr : array_like
    #             Amplifications of each individual image at each time,
    #             i.e. amp_arr.shape = (len(t_obs), number of images at each t_obs).
    #
    #             This will over-ride t_obs; but is more efficient when calculating
    #             both photometry and astrometry. If None, then just use t_obs.
    #         filt_idx : int
    #             The filter index (def=0).
    #
    #         Returns
    #         -------
    #         mag_model : array_like
    #             Magnitude of each lensed image centroid at t_obs.
    #             Shape = FIXME
    #         '''
    #         mag_zp = 30.0  # arbitrary but allows for negative blend fractions.
    #         flux_zp = 1.0
    #
    #         if amp_arr is None:
    #             amp_arr, img_arr = self.get_centroids(t_obs, self.radius)
    #
    #         # Do we need this masked stuff?
    #         # Mask invalid values from the amplification array.
    #         amp_arr_mskd = np.masked_invalid(amp_arr)
    #
    #         flux_src = flux_zp * 10 ** ((self.mag_src[filt_idx] - mag_zp) / -2.5)
    #         flux_model = flux_src * amp_arr_mskd
    #
    #         # Account for blending, if necessary.
    #         try:
    #             # Adding flux of neighbors and lens
    #             # b_sff = fS / (fS + fN + fL)
    #             flux_model += flux_src * (1.0 - self.b_sff[filt_idx]) / \
    #                           self.b_sff[filt_idx]
    #         except AttributeError:
    #             pass
    #
    #         # Catch the edge case where we exceed the zeropoint.
    #         bad = np.where(flux_model <= 0)
    #         if len(bad[0]) > 0:
    #             print('Warning: get_photometry: bad flux encountered.')
    #             flux_model[bad] = np.nan
    #
    #         mag_model = -2.5 * np.log10(flux_model / flux_zp) + mag_zp
    #
    #         return mag_model
    
    def get_source_lens_separation_unlensed(self, t_obs):
        """
        Get the separation vector, \vec{u}(t), which is the unlensed
        source - lens separation vector on the plane of the sky 
        in units of \theta_E.

        Input Parameters
        ----------------
        t_obs : array, float
            Times in MJD at which to evaluate the separation.

        Return
        -------
        u : array, float, shape = [len(t_obs), 2]
            Separation vector in East, North on the sky in units of \theta_E.
        """
        dt_in_years = (t_obs - self.t0) / days_per_year
            
        # Equation of motion for the relative angular separation
        # between the background source and lens. (Source - Lens)
        thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel)  # mas

        if self.parallaxFlag:
            parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
            thetaS -= (self.piRel * parallax_vec)  # mas
        
        u_vec = thetaS / self.thetaE_amp

        # # Second way to calculate, valid for photometry-only fits.
        # # Get the parallax vector for each date.
        # parallax_vec = parallax_in_direction_old(self.raL, self.decL, t)
        # tau = (t - self.t0) / self.tE

        # # Convert to matrices for more efficient operations.
        # # Matrix shapes below are:
        # #  u0, thetaE_hat: [1, 2]
        # #  tau:      [N_times, 1]
        # u0 = self.u0.reshape(1, len(self.u0))

        # # In direction of muRel_hat
        # thetaE_hat = self.thetaE_hat.reshape(1, len(self.thetaE_hat))
        # tau = tau.reshape(len(tau), 1)

        # # Shape of u: [N_times, 2]
        # u_vec = u0 + tau * thetaE_hat
        # u_vec -= self.piE_amp * parallax_vec

        return u_vec

    def get_source_outline_lens_separation_unlensed(self, t_obs):
        """
        Get the separation vector, \vec{u}(t), which is the unlensed
        source - lens separation vector for each point of the source 
        outline. Positions are  on the plane of the sky in units of \theta_E.

        Input Parameters
        ----------------
        t_obs : array, float
            Times in MJD at which to evaluate the separation.

        Return
        -------
        u : array, float, shape = [len(t_obs), 2]
            Separation vector in East, North on the sky in units of \theta_E.
        """
        u_vec = self.get_source_lens_separation_unlensed(t_obs)

        # Now expand and do this for all the outline points.
        u_vec_outline = np.zeros((len(t_obs), self.n_outline, 2), dtype=float)
        
        # The angles of the points equally spaced around the source circumference.
        angles = (np.arange(self.n_outline) / self.n_outline) * 2 * np.pi  # radians
        rho = self.radius / self.thetaE_amp

        dux = rho * np.cos(angles) 
        duy = rho * np.sin(angles) 

        # This could be faster with repeat, etc. Get rid of the for loop.
        for n in range(self.n_outline):
            u_vec_outline[:, n, 0] = u_vec[:, 0] + dux[n]
            u_vec_outline[:, n, 1] = u_vec[:, 1] + duy[n]

        return u_vec_outline
    

    def get_astrometry_outline_unlensed(self, t_obs):
        """Get the astrometry of the source outline if the lens didn't exist.

        Return
        -------
        xS_unlensed : numpy array, dtype=float, shape = [len(t_obs), self.n_outline, 2]
            The unlensed positions of the source outline points in arcseconds.
            The source outline is described by a list of points along the circumference
            of the circular source. The last axis contains East/North positions.
        """
        xS_unlensed_center = self.get_astrometry_unlensed(t_obs) # arcsec

        xS_unlensed_outline = np.zeros((len(t_obs), self.n_outline, 2), dtype=float)

        # The angles of the points equally spaced around the source circumference.
        angles = (np.arange(self.n_outline) / self.n_outline) * 2 * np.pi  # radians
        dx = self.radius * 1e-3 * np.cos(angles) # arcsec
        dy = self.radius * 1e-3 * np.sin(angles) # arcsec

        # This could be faster with repeat, etc. Get rid of the for loop.
        for n in range(self.n_outline):
            xS_unlensed_outline[:, n, 0] = xS_unlensed_center[:, 0] + dx[n]
            xS_unlensed_outline[:, n, 1] = xS_unlensed_center[:, 1] + dy[n]
        
        return xS_unlensed_outline
    
    
    def get_resolved_shift_outline(self, t_obs):
        """
        Get the astrometric microlensing shift of each
        point in the source outline for each of the multiple
        lensed images. 

        Note this is actually the source position w.r.t. 
        the lens. In other words, it is the source - lens separation,
        not just the astrometric microlensing shift.
        Sorry it is poorly named.

        Input Parameters
        ----------------
        t_obs : array, float
            Times in MJD at which to evaluate the separation.

        Return
        -------
        pos_plus : array, float, shape = [len(t_obs), 2]
            Relative astrometric position of the plus image in East, North 
            w.r.t. the lens in units of milli-arcseconds.
        pos_minus : array, float, shape = [len(t_obs), 2]
            Relative astrometric position of the minus image in East, North 
            w.r.t. the lens in units of milli-arcseconds.
        """
        
        # Shape = [len(t_obs), self.n_outline, 2]
        u_vec = self.get_source_outline_lens_separation_unlensed(t_obs)

        # Shape = [len(t_obs), self.n_outline]
        u_amp = np.linalg.norm(u_vec, axis=2)
        u_hat = u_vec / u_amp[:, :, np.newaxis]
            
        # Shape = [len(t_obs), self.n_outline, 2]
        u_obs_amp_plus  = ((u_amp + np.sqrt(u_amp ** 2 + 4)) / 2.0)
        u_obs_amp_minus = ((u_amp - np.sqrt(u_amp ** 2 + 4)) / 2.0)
        u_obs_vec_plus  = u_obs_amp_plus[:, :, np.newaxis] * u_hat
        u_obs_vec_minus = u_obs_amp_minus[:, :, np.newaxis] * u_hat

        pos_plus  = u_obs_vec_plus  * self.thetaE_amp  # in mas
        pos_minus = u_obs_vec_minus * self.thetaE_amp  # in mas

        return (pos_plus, pos_minus)

    
    def get_resolved_astrometry_outline(self, t_obs):
        """Get the x, y astrometry for each of the two lensed source images
        and all the associated outline points. The two lensed source images
        are labeled plus and minus.

        These are actual positions on the sky.

        Returns
        -------
        [xS_plus, xS_minus] : list of numpy arrays
            xS_plus is the vector position of the plus image.
            xS_minus is the vector position of the plus image.
            Each shape = [len(t_obs), self.n_outline, 2]
            where the last axis contains East and North positions.
        """
        
        # Things we will need.
        dt_in_years = (t_obs - self.t0) / days_per_year

        xL = self.get_lens_astrometry(t_obs) # arcsec

        shift_plus, shift_minus = self.get_resolved_shift_outline(t_obs) # S - L position in mas

        # Lensed outline positions
        xS_plus  = xL[:, np.newaxis, :] + (shift_plus * 1e-3)  # arcsec
        xS_minus = xL[:, np.newaxis, :] + (shift_minus * 1e-3)  # arcsec

        return (xS_plus, xS_minus)
        
    def get_all_arrays(self, t_obs):
        """
        Obtain the image and amplitude arrays for each t_obs. These arrays
        contain the positions for each point in the outline for each lensed image.
        
        Parameters
        ----------
        t_obs : array_like
            Array of times to model.

        Returns
        -------
        images : array_like
            Array/tuple of positions of each lensed image at each t_obs.
            Shape = [len(t_obs), n_images=2, 2]
            The last axis contains East and North positions on the sky
            in arcseconds.

        amp_arr : array_like
            Array/tuple of amplification of each lensed image at each t_obs.
            Shape = [len(t_obs), n_images=2]

        Notes
        -----
        The algorithm uses green's theorem to change an area integral of the 
        image of the source into a path integral around the outside outline.
        We perform a first-order contour integral to approximate the area.
        """
        # Lensed positions of each outline point for both plus/minus images.
        # Note these are positions on the sky. in arcsec
        images = self.get_resolved_astrometry_outline(t_obs)

        angles = (np.arange(self.n_outline) / self.n_outline) * 2 * np.pi  # radians
        # d_angles = np.diff(np.append(angles, angles[0:1]))
        d_angles = np.diff(angles)
        
        # Shape of plus array:  [len(times), self.n_outline, 2] where 
        # the last dimension is E and N on the sky.
        plus = images[0]
        minus = images[1]

        # Temporarily duplicate the first point as the last point
        # to speed up our contour integrals.
        # Shape of plus array: [len(times), self.n_outline + 1, 2] where
        plus = np.append(plus, plus[:,0:1,:], axis=1)
        minus = np.append(minus, minus[:,0:1,:], axis=1)

        # Pre-calculate squared versions...
        # we use these a lot in the calculations below.
        plus2 = plus**2
        minus2 = minus**2

        # First derivatives (len = n_outline)
        d1_plus = np.diff(plus, axis=1)
        d1_minus = np.diff(minus, axis=1)

        # Second derivatives (len = n_outline)
        d2_plus = np.diff( np.append(d1_plus, d1_plus[:, 0:1, :], axis=1), axis=1)
        d2_minus = np.diff( np.append(d1_minus, d1_minus[:, 0:1, :], axis=1), axis=1)

        # 2 element box addition
        b2_plus = plus[:, :-1, :] + plus[:, 1:, :]
        b2_minus = minus[:, :-1, :] + minus[:, 1:, :]

        def wedge_product(aa, bb):
            foo = aa[:, :, 0] * bb[:, :, 1] - aa[:, :, 1] * bb[:, :, 0]

            return foo
        
        # Do the contour integrals.
        # Equations from Bozza+ 2021 (Eq 8 and 9)
        # Aplus  =  0.5 * wedge_product( plus[:, :-1, :], plus[:, 1:, :] )
        Aplus  =  0.25 * (b2_plus[:, :, 0] * d1_plus[:, :, 1] - b2_plus[:, :, 1] * d1_plus[:, :, 0])
        Aplus = np.sum( Aplus, axis=1 )
        Aplus += np.sum( (d_angles**3 / 24.) * (  wedge_product( d1_plus[:, :-1, :], d2_plus[:, :-1, :])
                                                 + wedge_product( d1_plus[:, 1:, :],  d2_plus[:, 1:, :]) ), axis=1)

        # Aminus  =  0.5 * wedge_product( minus[:, :-1, :], minus[:, 1:, :] )
        Aminus  =  0.25 * (b2_minus[:, :, 0] * d1_minus[:, :, 1] - b2_minus[:, :, 1] * d1_minus[:, :, 0])
        Aminus = np.sum( Aminus, axis=1 )
        Aminus += np.sum( (d_angles**3 / 24.) * (  wedge_product( d1_minus[:, :-1, :], d2_minus[:, :-1, :])
                                                 + wedge_product( d1_minus[:, 1:, :],  d2_minus[:, 1:, :]) ), axis=1)

        Cplus_x = -(1. / 8.0) * np.sum( d1_plus[:, :, 1]  * b2_plus[:, :, 0]**2, axis=1 )
        Cplus_y =  (1. / 8.0) * np.sum( d1_plus[:, :, 0]  * b2_plus[:, :, 1]**2, axis=1 )

        Cminus_x =  (1. / 8.0) * np.sum( d1_minus[:, :, 1]  * b2_minus[:, :, 0]**2, axis=1 )
        Cminus_y = -(1. / 8.0) * np.sum( d1_minus[:, :, 0]  * b2_minus[:, :, 1]**2, axis=1 )

        amp_plus  = np.abs(Aplus)  / (np.pi * (self.radius * 1e-3)**2)
        amp_minus = np.abs(Aminus) / (np.pi * (self.radius * 1e-3)**2)
        img_pos_plus  = np.array([Cplus_x / np.abs(Aplus),  Cplus_y / np.abs(Aplus)])
        img_pos_minus = np.array([Cminus_x / np.abs(Aminus), Cminus_y / np.abs(Aminus)])
        

        # # Original Broadberry equations.
        # Aplus  = 0.5 * np.sum(  plus[:, :-1, 0]  * d1_plus[:, :, 1]
        #                       - plus[:, :-1, 1]  * d1_plus[:, :, 0], axis=1 )
        
        # Aminus = 0.5 * np.sum(  minus[:, :-1, 0] * d1_minus[:, :, 1]
        #                       - minus[:, :-1, 1] * d1_minus[:, :, 0], axis=1 )

        # Cplus1 = np.sum( np.diff(plus[:, :, 1], axis=1)  * (plus2[:, :-1, 0] + plus2[:, 1:, 0]) 
        #                - np.diff(plus2[:, :, 0], axis=1) * (plus[:, 1:, 1]   + plus[:, :-1, 1]), axis=1 )

        # Cplus2 = np.sum( np.diff(plus[:, :, 0], axis=1)  * (plus2[:, :-1, 1] + plus2[:, 1:, 1])
        #                - np.diff(plus2[:, :, 1], axis=1) * (plus[:, 1:, 0]   + plus[:, :-1, 0]), axis=1 )

        # Cminus1  = np.sum( np.diff(minus[:, :, 1], axis=1)  * (minus2[:, :-1, 0] + minus2[:, 1:, 0])
        #                  - np.diff(minus2[:, :, 0], axis=1) * (minus[:, 1:, 1]   + minus[:, :-1, 1]), axis=1 )

        # Cminus2 = np.sum( np.diff(minus[:, :, 0], axis=1)  * (minus2[:, :-1, 1] + minus2[:, 1:, 1])
        #                 - np.diff(minus2[:, :, 1], axis=1) * (minus[:, 1:, 0]   + minus[:, :-1, 0]), axis=1 )

        # # Above arrays should have shape of len(t_obs).
        # area_plus = np.abs(Aplus)
        # area_minus = np.abs(Aminus)
        
        # amp_plus  = area_plus  / (np.pi * (self.radius * 1e-3)**2)
        # amp_minus = area_minus / (np.pi * (self.radius * 1e-3)**2)
        # img_pos_plus  = np.array([(1 / (2 * Aplus))  * Cplus1,  (1 / (2 * Aplus))  * Cplus2])
        # img_pos_minus = np.array([(1 / (2 * Aminus)) * Cminus1,  (-1 / (2 * Aminus)) * Cminus2])


        print('imag_pos_plus.shape = ', img_pos_plus.shape)
        print('Aplus.shape = ', Aplus.shape)
        print('plus.shape = ', plus.shape)

        images = np.array((img_pos_plus, img_pos_minus)).T  # arcsec
        amps = np.array((amp_plus, amp_minus)).T  # amplifications

        print(amp_plus[:5])
        print(amp_minus[:5])
        print(amps[:5].sum(axis=1))
        
        pdb.set_trace()
        
        return images, amps

    def get_resolved_astrometry(self, t_obs, image_arr=None, amp_arr=None):
        """
        Position of the observed (lensed) source position on the sky.

        Parameters
        ----------
        t_obs : array_like, shape = [N_times]
            Array of times to model.

        Other Parameters
        ----------------
        image_arr : array_like
            Array of complex image positions at each t_obs,
            i.e. image_arr.shape = (len(t_obs), number of images at each t_obs).
            Each value in this array is complex
            (real = north component, imaginary = east component)

        amp_arr : array_like
            Array of magnifications of each images.
            Same shape as image_arr.

        Returns
        -------
        model_pos : array_like. shape = [N_times, N_images=2, 2]
            Array of vector positions of the centroid at each t_obs.
            Last axis contains East/North positions.
        """
        if (image_arr is None) or (amp_arr is None):
            image_arr, amp_arr = self.get_all_arrays(t_obs)

        xS_lensed_pos = np.swapaxes(image_arr, 0, 1)[::-1, :, :]

        return xS_lensed_pos
    
    def get_resolved_amplification(self, t_obs, filt_idx=0, amp_arr=None):
        """Get the photometric amplification term at a set of times, t for both the
        plus and minus images.

        Inputs
        ----------
        t: Array of times in MJD.DDD
        """
        if amp_arr is None:
            img_arr, amp_arr = self.get_all_arrays(t_obs)

        return np.swapaxis(amp_arr, 0, 1)

    
    def get_resolved_photometry(self, t_obs, filt_idx=0, amp_arr=None):
        '''
        Get the photometry for each of the lensed source images.
        Implement with no blending (since we don't support different
        blendings for the different images).

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.

        Other Parameters
        ----------------
        amp_arr : array_like
            Amplifications of each individual image and each outline point
            for that image at each time,
            i.e. amp_arr.shape = (len(t_obs), self.n_outline, number of images at each t_obs).

            This will over-ride t_obs; but is more efficient when calculating
            both photometry and astrometry. If None, then just use t_obs.
        filt_idx : int
            The filter index (def=0).

        Returns
        -------
        mag_model : array_like
            Magnitude of each lensed image centroid at t_obs.
            Shape = [2, len(t_obs)]
        '''
        mag_zp = 30.0  # arbitrary but allows for negative blend fractions.
        flux_zp = 1.0

        if amp_arr is None:
            img_arr, amp_arr = self.get_all_arrays(t_obs)

        # Mask invalid values from the amplification array.
        amp_arr_mskd = np.ma.masked_invalid(amp_arr)

        flux_src = flux_zp * 10 ** ((self.mag_src[filt_idx] - mag_zp) / -2.5)
        flux_model = flux_src * amp_arr_mskd.T

        # Account for blending, if necessary.
        try:
            # Adding flux of neighbors and lens
            # b_sff = fS / (fS + fN + fL)
            flux_model += flux_src * (1.0 - self.b_sff[filt_idx]) / \
                          self.b_sff[filt_idx]
        except AttributeError:
            pass
        
        # Catch the edge case where we exceed the zeropoint.
        bad = np.where(flux_model <= 0)
        if len(bad[0]) > 0:
            if print_warning:
                print('Warning: get_photometry: bad flux encountered.')
            flux_model[bad] = np.nan

        mag_model = -2.5 * np.log10(flux_model / flux_zp) + mag_zp

        return mag_model
        
    def get_photometry(self, t_obs, filt_idx=0, amp_arr=None, print_warning=True):
        '''
        Get the photometry for the combined source images. 

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.

        Other Parameters
        ----------------
        amp_arr : array_like
            Amplifications of each individual image at each time,
            i.e. amp_arr.shape = (len(t_obs), number of images at each t_obs).

            This will over-ride t_obs; but is more efficient when calculating
            both photometry and astrometry. If None, then just use t_obs.

        Returns
        -------
        mag_model : array_like
            Magnitude of the centroid at t_obs.
        '''
        mag_zp = 30.0  # arbitrary but allows for negative blend fractions.
        flux_zp = 1.0

        if amp_arr is None:
            img_arr, amp_arr = self.get_all_arrays(t_obs)

        # Mask invalid values from the amplification array.
        # amp_arr_mskd = np.ma.masked_invalid(amp_arr)
        amp_arr_mskd = amp_arr

        amp = np.sum(amp_arr_mskd, axis=1)

        flux_src = flux_zp * 10 ** ((self.mag_src[filt_idx] - mag_zp) / -2.5)
        flux_model = flux_src * amp

        # Account for blending, if necessary.
        try:
            # Adding flux of neighbors and lens
            # b_sff = fS / (fS + fN + fL)
            flux_model += flux_src * (1.0 - self.b_sff[filt_idx]) / \
                          self.b_sff[filt_idx]
        except AttributeError:
            pass
        
        # Catch the edge case where we exceed the zeropoint.
        bad = np.where(flux_model <= 0)
        if len(bad[0]) > 0:
            if print_warning:
                print('Warning: get_photometry: bad flux encountered.')
            flux_model[bad] = np.nan

        mag_model = -2.5 * np.log10(flux_model / flux_zp) + mag_zp

        return mag_model

    def get_astrometry(self, t_obs, image_arr=None, amp_arr=None, ast_filt_idx=0):
        '''
        Position of the observed (unresolved) source position in arcsec.

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.

        Other Parameters
        ----------------
        image_arr : array_like
            Array of complex image positions at each t_obs,
            i.e. image_arr.shape = (len(t_obs), number of images at each t_obs).
            Each value in this array is complex
            (real = north component, imaginary = east component)

        amp_arr : array_like
            Array of magnifications of each images.
            Same shape as image_arr.

        Returns
        -------
        model_pos : array_like
            Array of vector positions of the centroid at each t_obs.
        '''
        if (image_arr is None) or (amp_arr is None):
            image_arr, amp_arr = self.get_all_arrays(t_obs)

        # image_arr shape = [N_times, N_images, 2]

        xS_lensed = np.sum(image_arr * amp_arr[:, np.newaxis, :], axis=1)
        xS_lensed /= np.sum(amp_arr, axis=1)[:, np.newaxis]

        return xS_lensed

    def animate(self, crossings, time_steps, frame_time, name, size, zoom,
                astrometry):
        # creates the animation html, given an instance of the Uniformly_bright class and a list of times

        times = np.array(range(-time_steps, time_steps + 1, 1))
        tau = crossings * times / (-times[0])
        t = (tau * self.tE) + self.t0

        rs = self.get_astrometry_unlensed(t)  # position of source
        rl = self.get_lens_astrometry(t)  # position of lens
        images = self.get_resolved_astrometry(t)  # positions of images
        plus = images[0]  # plus image
        minus = images[1]  # minus image
        C = self.get_centroids(t, self.radius)
        A = C[0]  # magnification
        times = range(
            time_steps)  # gets time in units of the einstein crossing time

        fig = plt.figure(
            figsize=[size[0], size[1] + 0.5])  # sets up the figure
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        fig.subplots_adjust(hspace=.5)
        # creates 5 different plts (source, lens, image1, image2, magnification)
        line1, = ax1.plot(rl[:, 0], rl[:, 1], markersize=4, c='b', marker='.',
                          label="Lens")
        line2, = ax1.plot([], 'g.', markersize=0.5, label="Source")
        line3, = ax1.plot([], 'r.', markersize=2., label="Image")
        line4, = ax1.plot([], 'r.', markersize=2.)
        ax1.set_xlabel("RA")
        ax1.set_ylabel("Dec")

        ax1.set_xlim(
            (rl[0][0] + rl[-1][0]) / 2 - 2 * (size[0]) / (2 * size[1]) * (
                    rl[-1][1] - rl[0][
                1] + 2 * zoom * self.thetaEamp * 1e-3),
            (rl[0][0] + rl[-1][0]) / 2 + 2 * (size[0]) / (2 * size[1]) * (
                    rl[-1][1] - rl[0][
                1] + 2 * zoom * self.thetaEamp * 1e-3))
        ax1.set_ylim(rl[0, 1] - zoom * self.thetaEamp * 0.001,
                     rl[-1, 1] + zoom * self.thetaEamp * 0.001)
        a = C[1]
        line5, = ax1.plot([], 'm.', markersize=size[0] * 0.5,
                          label="Image Centroid")
        line6, = ax2.plot(tau, A)
        ax1.legend()
        ax2.set_xlabel("Time(tE)")
        ax2.set_ylabel("Magnification")

        line = [line1, line2, line3, line4, line5, line6]

        # this function is called at every frame,
        # with i being the number of the frame that it's currently on
        def update(i, rs, rl, line, plus, minus, astrometry, tau, A):
            print(i)
            line[0].set_data(rl[i, 0], rl[i, 1])
            line[1].set_data(rs[i, :, 0], rs[i, :, 1])
            line[2].set_data(plus[i, :, 0], plus[i, :, 1])
            line[3].set_data(minus[i, :, 0], minus[i, :, 1])
            line[4].set_data(astrometry[:i, 0], astrometry[:i, 1])
            line[5].set_data(tau[:i], A[:i])
            return line

        """
        FuncAnimation takes in lots of arguments

        fig = background figure

        update = function that is called every frame

        len(tau) = the number of frames, so now the first argument
        passed into update (i) will be (0,1,2...len(tau))

        fargs specifies the other arguments to pass into update

        blit being true means that each frame, if there are elements
        of it that don't change from the last frame,
        it won't replot them, so this makes it faster

        interval = number of milliseconds between each frame
        alternatively you can specify fps in save after after the file name

        """
        ani = animation.FuncAnimation(fig, update, len(tau),
                                      fargs=[rs, rl, line, plus, minus, a, tau,
                                             A],
                                      blit=True, interval=frame_time)
        ani.save("%s.mp4" % name, writer="ffmpeg", dpi=600)
        
        return ani

    
    

class FSPL_Phot(FSPL):
    pass

class FSPL_noParallax(PSPL_noParallax):
    parallaxFlag = False


class FSPL_Parallax(PSPL_Parallax):
    parallaxFlag = True


class FSPL_PhotAstromParam1(PSPL_Param):
    """PSPL model for astrometry and photometry - physical parameterization.

    A Point Source Point Lens model for microlensing. This model uses a 
    parameterization that depends on only physical quantities such as the 
    lens mass and positions and proper motions of both the lens and source. 

    Note the attributes, RA (raL) and Dec (decL) are required 
    if you are calculating a model with parallax. 

    Parameters
    ----------
    mL: float
        Mass of the lens (Msun)
    t0: float
        Time of photometric peak, as seen from Earth (MJD.DDD)
    beta: float
        Angular distance between the lens and source on the plane of the sky (mas). Can be
        
        * positive (u0_amp > 0 when u0_hat[0] (East component) < 0) or 
        * negative (u0_amp < 0 when u0_hat[0] (East component) > 0).
        
    dL: float
        Distance from the observer to the lens (pc)
    dL_dS: float
        Ratio of Distance from the obersver to the lens to
        Distance from the observer to the source
    xS0_E: float
        RA Source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    xS0_N: float
        Dec source position on sky at t = t0 (arcsec) in an arbitrary ref. frame.
    muL_E: float
        RA Lens proper motion (mas/yr)
    muL_N: float
        Dec Lens proper motion (mas/yr)
    muS_E: float
        RA Source proper motion (mas/yr)
    muS_N: float
        Dec Source proper motion (mas/yr)
    radius: float
        Projected radius of the star in arcsec on the sky plane.
    b_sff: float
        The ratio of the source flux to the total (source + neighbors + lens)
        :math:`b_sff = f_S / (f_S + f_L + f_N)`. This must be passed in as a list or
        array, with one entry for each photometric filter.
    mag_src: float
        Photometric magnitude of the source. This must be passed in as a
        list or array, with one entry for each photometric filter.
    n_outline: int
        Number of boundary points to use when approximating the source outline.
        Calculation time scales approximately linearly with 'n_outline'.
    raL: float, optional
        Right ascension of the lens in decimal degrees.
    decL: float, optional
        Declination of the lens in decimal degrees.
    """
    fitter_param_names = ['mL', 't0', 'xS0_E', 'xS0_N', 'beta', 'muL_N', 'muL_E',
                          'muS_N', 'muS_E', 'dL', 'dS', 'radius']
    phot_param_names = ['mag_src', 'b_sff']

    paramAstromFlag = True
    paramPhotFlag = True

    def __init__(self, mL, t0, beta, dL, dS,
                 xS0_E, xS0_N,
                 muL_E, muL_N,
                 muS_E, muS_N,
                 radius,
                 b_sff, mag_src,
                 n_outline=10,
                 raL=None, decL=None):
        # Initialised variables
        self.t0 = t0
        self.mL = mL
        self.beta = beta
        self.dL = dL
        self.dS = dS
        self.xS0 = np.array([xS0_E, xS0_N])
        self.muL = np.array([muL_E, muL_N])
        self.muS = np.array([muS_E, muS_N])
        self.n_outline = n_outline
        self.radius = (radius * meter_per_Rsun / meter_per_AU) / dS
        self.mag_src = mag_src
        self.b_sff = b_sff
        self.raL = raL
        self.decL = decL

        # Check variable formatting.
        super().__init__()

        self.mag_base = self.mag_src + 2.5 * np.log10(self.b_sff)

        # Calculate the relative parallax
        inv_dist_diff = (1.0 / (self.dL * units.pc)) - (1.0 / (self.dS * units.pc))
        piRel = units.rad * units.au * inv_dist_diff
        self.piRel = piRel.to('mas').value

        # Calculate the individual parallax
        piS = (1.0 / self.dS) * (units.rad * units.au / units.pc)
        piL = (1.0 / self.dL) * (units.rad * units.au / units.pc)
        self.piS = piS.to('mas').value
        self.piL = piL.to('mas').value

        # Calculate the relative proper motion vector.
        # Note that this will be in the direction of theta_hat
        self.muRel = self.muS - self.muL
        self.muRel_E, self.muRel_N = self.muRel
        self.muRel_amp = np.linalg.norm(self.muRel)  # mas/yr

        self.muS_E, self.muS_N = self.muS
        self.muL_E, self.muL_N = self.muL

        # Calculate the Einstein radius
        thetaE = units.rad * np.sqrt(
            (4.0 * const.G * mL * units.M_sun / const.c ** 2) * inv_dist_diff)
        self.thetaE_amp = thetaE.to('mas').value  # mas
        self.thetaE_hat = self.muRel / self.muRel_amp
        self.muRel_hat = self.thetaE_hat
        self.thetaE = self.thetaE_amp * self.thetaE_hat
        self.thetaE_E, self.thetaE_N = self.thetaE

        # Comment on sign conventions:
        # thetaS0 = xS0 - xL0
        # (difference in positions on sky, heliocentric, at t0)
        # u0 = thetaS0 / thetaE -- so u0 is source - lens position vector
        # if u0_E > 0 then the Source is to the East of the lens
        # if u0_E < 0 then the source is to the West of the lens
        # We adopt the following sign convention (same as Gould:2004):
        #    u0_amp > 0 means u0_E > 0
        #    u0_amp < 0 means u0_E < 0
        # Note that we assume beta = u0_amp (with same signs).

        # Calculate the closest approach vector. Define beta sign convention
        # same as of Andy Gould does with beta > 0 means u0_E > 0
        # (lens passes to the right of the source as seen from Earth or Sun).
        # The function u0_hat_from_thetaE_hat is programmed to use thetaE_hat and beta, but
        # the sign of beta is always the same as the sign of u0_amp. Therefore this
        # usage of the function with u0_amp works exactly the same.
        self.u0_hat = u0_hat_from_thetaE_hat(self.thetaE_hat, self.beta)
        self.u0_amp = self.beta / self.thetaE_amp  # in Einstein units
        self.u0 = np.abs(self.u0_amp) * self.u0_hat

        # Angular separation vector between source and lens
        # (vector from lens to source)
        self.thetaS0 = self.u0 * self.thetaE_amp  # mas

        # Calculate the position of the lens on the sky at time, t0
        self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)

        # Calculate the microlensing parallax
        self.piE_amp = self.piRel / self.thetaE_amp
        self.piE = self.piE_amp * self.thetaE_hat
        self.piE_E, self.piE_N = self.piE

        # Calculate the Einstein crossing time. (days)
        self.tE = (self.thetaE_amp / self.muRel_amp) * days_per_year

        return


class FSPL_PhotAstrom(FSPL, PSPL_PhotAstrom):
    """
    Contains methods for model a PSPL photometry + astrometry.
    This is a Data-type class in our hierarchy. It is abstract and should not
    be instantiated. 

    Attributes
    ----------
    Available class variables that should be defined.

    t0
    tE
    u0_amp
    u0_E
    u0_N
    beta
    piE_E:
        valid only if parallax model
    piE_N:
        valid only if parallax model
    piE_amp
    mL
    thetaE_amp
    thetaE_E
    thetaE_N
    xS0_E
    xS0_N
    xL0_E
    xL0_N
    muS_E
    muS_N
    muL_E
    muL_N
    muRel_E
    muRel_N
    muRel_amp
    piS
    piL
    dL
    dS
    dL_dS (dL over dS)
    radius
    n
    b_sff[#]
    mag_src[#] -- add in
    mag_base[#] -- add in 
    raL:
        if parallax model
    decL:
        if parallax model


    """
    photometryFlag = True
    astrometryFlag = True

    # Shouldn't get_lens_astrometry be inherited?
    # Why is parallax stuff different than PSPL?
    def get_lens_astrometry(self, t_obs):
        # returns the position of the lens in the sky at a list of times, t in units of einstein time
        # t is an np array of numbers
        t_yrs = (t_obs - self.t0) / days_per_year
        xL = self.xL0 + np.outer(t_yrs,
                                 self.muL) * 1e-3  # this is just x = x0 + vt for a constant velocity object

        if self.parallaxFlag:
            # Get the parallax vector for each date.
            parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
            xL += (self.piL * parallax_vec) * 1e-3  # arcsec

        return xL

    def get_resolved_astrometry(self, t):  # r is not actually used anywhere I think
        # Returns the position of the 2 images at a list of times and the associated fluxes
        source_pos = self.get_astrometry_unlensed(t)  # position of the source
        lens_pos = self.get_lens_astrometry(t)  # position of the lens

        thetas = get_thetas(source_pos,
                            lens_pos)  # angular position of the source relative to the lens over time
        thetasamp = get_amplitudes(thetas)  # list of amplituds of theta over time
        thetahats = get_unit_vectors(thetas)  # list of unit vectors for each theta
        # These functions get the angular position of both of the images over time
        plus = get_plus(thetasamp, thetahats, source_pos, lens_pos,
                        self.thetaE_amp * 1e-3)
        minus = get_minus(thetasamp, thetahats, source_pos, lens_pos,
                          self.thetaE_amp * 1e-3)
        # returns a tuple of numpy arrays
        return (plus, minus)

    #    def get_astrometry_unlensed(self, t, r):
    def get_astrometry_unlensed(self, t):  # r is not actually used anywhere I think
        """ Outputs position of source unlensed.
        
        Input a list of times and it will output the position of the source had it not been lensed at each of the
        times in the list

        | e.g if ``n = 4``, and say ``v = [1,0]`` & the times are ``[0,1,2]`` in years.
        | This will return
        | ``((( (1,0),(0,1),(-1,0),(0,-1) ), ( (2,0),(1,1),(0,0),(1,-1) ), ( (3,0),(2,1),(1,0),(2,-1) ))...``
        | =  (positions at t=0), (positions at t=1), (positions at t=2)

        so ``np.array(positions)`` is an array which contains an array for each time step with the positions of all the
        points on the boundary of the source.
        """
        t_yrs = (t - self.t0) / 365.5
        deltax = np.outer(t_yrs, self.muS * 1e-3)
        positions = []
        for i in deltax:
            positions.append(
                self.get_source(self.radius, self.n, self.xS0) + i)

        # FIXME NEED TO PUT PARALLAX IN HERE!!!!
        # NEED TO CHECK UNITS
        #        if self.parallaxFlag:
        #            # Get the parallax vector for each date.
        #            parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
        #            xS_unlensed += (self.piS * parallax_vec) * 1e-3  # arcsec

        return np.array(positions)


class FSPL_Limb(FSPL):
    def F(self, r):
        return 2 / (1 - self.utilde / 3) * (
                (1 / 2) * (1 - self.utilde) * r ** 2 - (
                self.utilde / 3) * (1 - r ** 2) ** (
                        3 / 2) + self.utilde / 3)

    def get_photometry(self, t_obs, filt_idx=0, amp_arr=None, print_warning=True):
        '''
        Get the photometry for each of the lensed source images.

        Parameters
        ----------
        t_obs : array_like
            Array of times to model.
        filt_idx : int, optional
            Index of the photometric filter or data set.

        Other Parameters
        ----------------
        amp_arr : array_like
            Amplifications of each individual image at each time,
            i.e. amp_arr.shape = (len(t_obs), number of images at each t_obs).

            This will over-ride t_obs; but is more efficient when calculating
            both photometry and astrometry. If None, then just use t_obs.
        print_warning : bool, optional
            Print a warning in the rare case that the magnitude exceeds a 
            zeropoint of 30 and conversions result in NaN returned.

        Returns
        -------
        mag_model : array_like
            Magnitude of the unresolved microlensing event at t_obs.
        '''
        mag_zp = 30.0  # arbitrary but allows for negative blend fractions.
        flux_zp = 1.0

        if amp_arr is None:
            amp_arr, img_arr = self.get_centroids(t_obs, self.radius)
            amp = self.get_amplification(t_obs)

            # CHECK
        #        # Do we need this masked stuff?
        #        # Mask invalid values from the amplification array.
        #        amp_arr_msk = np.ma.masked_invalid(amp_arr)
        #
        #        # Sum up all the amplifications b/c surface brightness is conserved.
        #        amp = np.sum(amp_arr_msk, axis=1)

        flux_src = flux_zp * 10 ** ((self.mag_src[filt_idx] - mag_zp) / -2.5)
        flux_model = flux_src * amp

        # Account for blending, if necessary.
        try:
            # Adding flux of neighbors and lenses
            # b_sff = fS / (fS + fN + fL)
            flux_model += flux_src * (1.0 - self.b_sff[filt_idx]) / self.b_sff[filt_idx]
        except AttributeError:
            pass

        # Catch the edge case where we exceed the zeropoint.
        bad = np.where(flux_model <= 0)[0]
        if len(bad) > 0:
            if print_warning:
                print('Warning: get_photometry: bad flux encountered.')
            flux_model[bad] = np.nan

        mag_model = -2.5 * np.log10(flux_model / flux_zp) + mag_zp

        return mag_model

    def get_amplification(self, t):
        radii = np.array(range(1, self.nr + 1, 1)) / self.nr
        amplifications = []
        centroids = []
        Fs = []
        for i in range(len(radii)):
            Ci = self.get_centroids(t, self.radius * radii[i])
            # If this is fixed length, can speed up by making it an array
            #    and overwriting.
            amplifications.append(Ci[0])
            centroids.append(Ci[1])
            Fs.append(self.F(radii[i]))
        amplification = []
        for j in range(len(t)):
            M = 0
            for k in range(len(radii)):
                if k == 0:
                    Mk = Fs[k] * amplifications[k][j]
                else:
                    fk = (Fs[k] - Fs[k - 1]) / (
                            radii[k] ** 2 - radii[k - 1] ** 2)
                    Mk = fk * (amplifications[k][j] * (radii[k] ** 2) -
                               amplifications[k - 1][j] * (radii[k - 1]) ** 2)
                M += Mk
            amplification.append(M)
        return amplification

    def animate(self, crossings, time_steps, frame_time, name, size, zoom):
        # creates the animation html, given an instance of the Uniformly_bright class and a list of times

        times = np.array(range(-time_steps, time_steps + 1, 1))
        tau = crossings * times / (-times[0])
        t = tau * self.tE

        rs = self.get_astrometry_unlensed(t, self.radius)  # position of source
        rl = self.get_lens_astrometry(t)  # position of lens
        images = self.get_resolved_astrometry(t, self.radius)  # positions of images
        plus = images[0]  # plus image
        minus = images[1]  # minus image
        C = self.get_amplification(t)
        A = C

        fig = plt.figure(
            figsize=[size[0], size[1] + 0.5])  # sets up the figure
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        fig.subplots_adjust(hspace=.5)
        # creates 5 different plts (source, lens, image1, image2, magnification)
        line1, = ax1.plot(rl[:, 0], rl[:, 1], 'b.', markersize=20.,
                          label="Lens")
        line2, = ax1.plot([], 'g.', markersize=5, label="Source")
        line3, = ax1.plot([], 'r.', markersize=5, label="Images")
        line4, = ax1.plot([], 'r.', markersize=5)
        ax1.set_xlabel("RA")
        ax1.set_ylabel("Dec")
        ax1.set_xlim(
            (rl[0][0] + rl[-1][0]) / 2 - 2 * (size[0]) / (2 * size[1]) * (
                    rl[-1][1] - rl[0][
                1] + 2 * zoom * self.thetaE_amp * 1e-3),
            (rl[0][0] + rl[-1][0]) / 2 + 2 * (size[0]) / (2 * size[1]) * (
                    rl[-1][1] - rl[0][
                1] + 2 * zoom * self.thetaE_amp * 1e-3))
        ax1.set_ylim(rl[0, 1] - zoom * self.thetaE_amp * 0.001,
                     rl[-1, 1] + zoom * self.thetaE_amp * 0.001)
        a = self.get_centroids(t, self.radius)[1]
        xcent = a[:, 0]
        ycent = a[:, 1]
        line5, = ax1.plot([], 'm', markersize=5, label="Image Centroid")
        line6, = ax2.plot(t, A)
        ax1.legend(fontsize=25, markerscale=3)
        ax2.set_xlabel("Time (days)", fontsize=40)
        ax2.set_ylabel("Magnification", fontsize=40)

        line = [line1, line2, line3, line4, line5, line6]

        # this function is called at every frame, with i being the number of the frame that it's currently on
        def update(i, rs, rl, line, plus, minus, xcent, ycent, tau, A):
            line[0].set_data(rl[i, 0], rl[i, 1])
            line[1].set_data(rs[i, :, 0], rs[i, :, 1])
            line[2].set_data(plus[i, :, 0], plus[i, :, 1])
            line[3].set_data(minus[i, :, 0], minus[i, :, 1])
            line[4].set_data(xcent[:i], ycent[:i])
            line[5].set_data(tau[:i], A[:i])
            return line

        ani = animation.FuncAnimation(fig, update, len(tau),
                                      fargs=[rs, rl, line, plus, minus, xcent,
                                             ycent, t, A], blit=True,
                                      interval=frame_time)
        ani.save("%s.mp4" % name, writer="ffmpeg", dpi=600)

        return ani


class FSPL_Limb_noParallax(FSPL_noParallax):
    parallaxFlag = False


class FSPL_Limb_Parallax(FSPL_Parallax):
    parallaxFlag = True


# FIXME: Use super here
class FSPL_Limb_PhotAstromParam1(PSPL_Param):
    def __init__(self, lens_mass, t0, xS0, beta, muL, muS, dL, dS, n, radius,
                 utilde, nr, mag_src):
        """
        DO NOT USE -- in progress

        """
        # Initialised variables
        """
        The only new parameters here that aren't in the uniformly birhgt source is:
            mu = a parameter that determined how uniform the star is and 0 =< mu =< 1
            n_int = number of points in each direction to approximate the flux integral (see below)
        """
        self.utilde = utilde  # tells you how limbdarkended the source is
        self.lens_mass = lens_mass  # Mass of the lens in solar masses
        self.xS0 = np.array(
            xS0)  # Position of centroid at peak [[Ra, Dec],flux]
        self.beta = beta  # Angular distance between lens and centroid (mas)
        self.muL = np.array(muL)  # Lens proper motion (mas/yr) [Ra, Dec]
        self.muS = np.array(muS)  # Source proper motion "   "   "   "   "
        self.dL = dL  # Distance from observer to lens (pc)
        self.dS = dS  # Distance from observer to source (pc)
        self.n = n  # No. of boundary points approximating the source
        self.radius = (radius * 6.96e8 / 1.496e11) / dS  # Radius of star on the sky (as)
        self.source = self.get_source(self.radius, n, xS0)  # Positions of the centroid + points on the boundary
        self.nr = nr  # sets the precision on the integrals
        self.t0 = t0
        self.mag_src = mag_src

        # Check variable formatting.
        super().__init__()

        # Variables that need to be calculated
        self.muRel = self.muS - self.muL  # Source-lens relative proper motion
        self.thetaE_hat = get_unit_vector(
            self.muRel)  # unit vector in direction of thetaE
        self.muRel_hat = self.thetaE_hat
        self.thetaE_amp = get_angular_einstein_radius(self.lens_mass, self.dL,
                                                      self.dS)
        self.thetaE = self.thetaE_amp * self.thetaE_hat  # vector version of the einstein radius
        self.u0 = get_u0(self.thetaE_hat, self.beta,
                         self.thetaE_amp)  # closest approach vector
        self.thetas0 = self.u0 * self.thetaE_amp  # [RA,Dec] position of the source at peak
        self.xL0 = self.xS0[0] - self.thetas0 * 1e-3  # [RA, Dec] position of the lens at peak
        self.tE = get_einstein_time(self.thetaE_amp, self.muRel,
                                    365.25)  # Einstein crossing time


#############################################
### POINT SOURCE POINT LENS (PSPL) MODELS ###
#############################################

def inheritdocstring(cls):
    for base in inspect.getmro(cls)[::-1]:
        if base != object and base != ABC and base.__doc__ is not None:
            cls.__doc__ = base.__doc__
            break
    return cls


def startbases(self):
    for base in self.__class__.__bases__:
        if hasattr(base, 'start'):
            base.start(self)


def checkconflicts(self):
    hasDataClass = False
    hasParallaxClass = False
    hasParamClass = False
    for base in self.__class__.__bases__:

        if (hasattr(base, 'astrometryFlag') or hasattr(base, 'photometryFlag')):
            if not hasDataClass:
                hasDataClass = True
            else:
                raise RuntimeError('Multiple Data Classes '
                                   'in this model')

        if hasattr(base, 'parallaxFlag'):
            if not hasParallaxClass:
                hasParallaxClass = True
            else:
                raise RuntimeError('Multiple Parallax Classes '
                                   'in this model')

        if (hasattr(base, 'paramAstromFlag') or hasattr(base, 'paramPhotFlag')):
            if not hasParamClass:
                hasParamClass = True
            else:
                raise RuntimeError('Multiple Parametrization Classes '
                                   'in this model')

    if not hasDataClass:
        raise RuntimeError('Model missing a Data Class')

    if not hasParallaxClass:
        raise RuntimeError('Model missing a Parallax Class')

    if not hasParamClass:
        raise RuntimeError('Model missing a Parameterization Class')

    if self.paramAstromFlag and not self.astrometryFlag:
        raise RuntimeError(str(self.__class__) + " uses an astrometry "
                                                 "Parametrization Class without "
                                                 "an astrometry Data Class. "
                                                 "See model.py docstring for "
                                                 "more details.")

    if not self.paramAstromFlag and self.astrometryFlag:
        raise RuntimeError(str(self.__class__) + " uses an astrometry Data class"
                                                 "without an astrometry "
                                                 "Parametrization class. "
                                                 "See model.py docstring for "
                                                 "more details.")

    if self.paramPhotFlag and not self.photometryFlag:
        raise RuntimeError(str(self.__class__) + " uses a photometry "
                                                 "Parametrization Class without "
                                                 "a photometry Data Class. "
                                                 "See model.py docstring for "
                                                 "more details.")

    if not self.paramPhotFlag and self.photometryFlag:
        raise RuntimeError(str(self.__class__) + " uses a photometry Data class"
                                                 "without a photometry "
                                                 "Parametrization Class. "
                                                 "See model.py docstring for "
                                                 "more details.")


# --------------------------------------------------
#
# Model Class Family
#
# --------------------------------------------------
# Abstract base class for Model objects (end-user uses these).
class ModelClassABC(ABC):
    pass

# PSPL
@inheritdocstring
class PSPL_PhotAstrom_noPar_Param1(ModelClassABC,
                                   PSPL_PhotAstrom,
                                   PSPL_noParallax,
                                   PSPL_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSPL_parallax
@inheritdocstring
class PSPL_PhotAstrom_Par_Param1(ModelClassABC,
                                 PSPL_PhotAstrom,
                                 PSPL_Parallax,
                                 PSPL_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSPL_parallax
@inheritdocstring
class PSPL_PhotAstrom_LumLens_Par_Param1(ModelClassABC,
                                         PSPL_PhotAstrom,
                                         PSPL_Parallax_LumLens,
                                         PSPL_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSPL_parallax
@inheritdocstring
class PSPL_PhotAstrom_LumLens_Par_Param2(ModelClassABC,
                                         PSPL_PhotAstrom,
                                         PSPL_Parallax_LumLens,
                                         PSPL_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSPL_parallax
@inheritdocstring
class PSPL_PhotAstrom_LumLens_Par_Param4(ModelClassABC,
                                         PSPL_PhotAstrom,
                                         PSPL_Parallax_LumLens,
                                         PSPL_PhotAstromParam4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSPL_parallax2 / PSPL_multiphot_parallax
@inheritdocstring
class PSPL_PhotAstrom_Par_Param2(ModelClassABC,
                                 PSPL_PhotAstrom,
                                 PSPL_Parallax,
                                 PSPL_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSPL_parallax2 / PSPL_multiphot_parallax
@inheritdocstring
class PSPL_PhotAstrom_noPar_Param2(ModelClassABC,
                                   PSPL_PhotAstrom,
                                   PSPL_noParallax,
                                   PSPL_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_PhotAstrom_noPar_Param3(ModelClassABC,
                                   PSPL_PhotAstrom,
                                   PSPL_noParallax,
                                   PSPL_PhotAstromParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)
        
@inheritdocstring
class PSPL_PhotAstrom_Par_Param3(ModelClassABC,
                                 PSPL_PhotAstrom,
                                 PSPL_Parallax,
                                 PSPL_PhotAstromParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_PhotAstrom_Par_Param4(ModelClassABC,
                                 PSPL_PhotAstrom,
                                 PSPL_Parallax,
                                 PSPL_PhotAstromParam4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_PhotAstrom_Par_Param4_geoproj(ModelClassABC,
                                         PSPL_PhotAstrom,
                                         PSPL_Parallax_geoproj,
                                         PSPL_PhotAstromParam4_geoproj):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_PhotAstrom_Par_Param5(ModelClassABC,
                                 PSPL_PhotAstrom,
                                 PSPL_Parallax,
                                 PSPL_PhotAstromParam5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

@inheritdocstring
class PSPL_PhotAstrom_noPar_Param4(ModelClassABC,
                                   PSPL_PhotAstrom,
                                   PSPL_noParallax,
                                   PSPL_PhotAstromParam4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_Astrom_Par_Param4(ModelClassABC,
                             PSPL_Astrom,
                             PSPL_Parallax,
                             PSPL_AstromParam4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_Astrom_Par_Param3(ModelClassABC,
                             PSPL_Astrom,
                             PSPL_Parallax,
                             PSPL_AstromParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSPL_phot
@inheritdocstring
class PSPL_Phot_noPar_Param1(ModelClassABC,
                             PSPL_Phot,
                             PSPL_noParallax,
                             PSPL_PhotParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_Phot_noPar_Param2(ModelClassABC,
                             PSPL_Phot,
                             PSPL_noParallax,
                             PSPL_PhotParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

@inheritdocstring
class PSPL_Phot_noPar_Param3(ModelClassABC,
                             PSPL_Phot,
                             PSPL_noParallax,
                             PSPL_PhotParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)
        

# PSPL_phot_parallax / PSPL_phot_multiphot_parallax
@inheritdocstring
class PSPL_Phot_Par_Param1(ModelClassABC,
                           PSPL_Phot,
                           PSPL_Parallax,
                           PSPL_PhotParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

@inheritdocstring
class PSPL_Phot_Par_Param1_geoproj(ModelClassABC,
                                   PSPL_Phot,
                                   PSPL_Parallax_geoproj,
                                   PSPL_PhotParam1_geoproj):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_Phot_Par_Param2(ModelClassABC,
                           PSPL_Phot,
                           PSPL_Parallax,
                           PSPL_PhotParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

@inheritdocstring
class PSPL_Phot_Par_Param3(ModelClassABC,
                           PSPL_Phot,
                           PSPL_Parallax,
                           PSPL_PhotParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)
        

# PSPL Phot parallax with GP
@inheritdocstring
class PSPL_Phot_Par_GP_Param1(ModelClassABC,
                              PSPL_GP,
                              PSPL_Phot,
                              PSPL_Parallax,
                              PSPL_GP_PhotParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_Phot_Par_GP_Param2(ModelClassABC,
                              PSPL_GP,
                              PSPL_Phot,
                              PSPL_Parallax,
                              PSPL_GP_PhotParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_Phot_Par_GP_Param1_2(ModelClassABC,
                                PSPL_GP,
                                PSPL_Phot,
                                PSPL_Parallax,
                                PSPL_GP_PhotParam1_2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_Phot_Par_GP_Param2_2(ModelClassABC,
                                PSPL_GP,
                                PSPL_Phot,
                                PSPL_Parallax,
                                PSPL_GP_PhotParam2_2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

@inheritdocstring
class PSPL_Phot_Par_GP_Param3(ModelClassABC,
                              PSPL_GP,
                              PSPL_Phot,
                              PSPL_Parallax,
                              PSPL_GP_PhotParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        

# PSPL Phot, no parallax with GP
@inheritdocstring
class PSPL_Phot_noPar_GP_Param1(ModelClassABC,
                                PSPL_GP,
                                PSPL_Phot,
                                PSPL_noParallax,
                                PSPL_GP_PhotParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_Phot_noPar_GP_Param2(ModelClassABC,
                                PSPL_GP,
                                PSPL_Phot,
                                PSPL_noParallax,
                                PSPL_GP_PhotParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

@inheritdocstring
class PSPL_Phot_noPar_GP_Param3(ModelClassABC,
                                PSPL_GP,
                                PSPL_Phot,
                                PSPL_noParallax,
                                PSPL_GP_PhotParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSPL PhotAstrom, parallax with GP
@inheritdocstring
class PSPL_PhotAstrom_Par_GP_Param1(ModelClassABC,
                                    PSPL_GP,
                                    PSPL_PhotAstrom,
                                    PSPL_Parallax,
                                    PSPL_GP_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_PhotAstrom_Par_GP_Param2(ModelClassABC,
                                    PSPL_GP,
                                    PSPL_PhotAstrom,
                                    PSPL_Parallax,
                                    PSPL_GP_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_PhotAstrom_Par_GP_Param3(ModelClassABC,
                                    PSPL_GP,
                                    PSPL_PhotAstrom,
                                    PSPL_Parallax,
                                    PSPL_GP_PhotAstromParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_PhotAstrom_Par_GP_Param4(ModelClassABC,
                                    PSPL_GP,
                                    PSPL_PhotAstrom,
                                    PSPL_Parallax,
                                    PSPL_GP_PhotAstromParam4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_PhotAstrom_Par_LumLens_GP_Param1(ModelClassABC,
                                            PSPL_GP,
                                            PSPL_PhotAstrom,
                                            PSPL_Parallax_LumLens,
                                            PSPL_GP_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_PhotAstrom_Par_LumLens_GP_Param2(ModelClassABC,
                                            PSPL_GP,
                                            PSPL_PhotAstrom,
                                            PSPL_Parallax_LumLens,
                                            PSPL_GP_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_PhotAstrom_Par_LumLens_GP_Param3(ModelClassABC,
                                            PSPL_GP,
                                            PSPL_PhotAstrom,
                                            PSPL_Parallax_LumLens,
                                            PSPL_GP_PhotAstromParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_PhotAstrom_Par_LumLens_GP_Param4(ModelClassABC,
                                            PSPL_GP,
                                            PSPL_PhotAstrom,
                                            PSPL_Parallax_LumLens,
                                            PSPL_GP_PhotAstromParam4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSPL PhotAstrom, parallax with GP
@inheritdocstring
class PSPL_PhotAstrom_noPar_GP_Param1(ModelClassABC,
                                      PSPL_GP,
                                      PSPL_PhotAstrom,
                                      PSPL_noParallax,
                                      PSPL_GP_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSPL_PhotAstrom_noPar_GP_Param2(ModelClassABC,
                                      PSPL_GP,
                                      PSPL_PhotAstrom,
                                      PSPL_noParallax,
                                      PSPL_GP_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# =====
# PSBL Model Classes
# =====
# PSBL



@inheritdocstring
class PSBL_PhotAstrom_EllOrbs_noPar_Param1(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_noParallax,
                                   PSBL_PhotAstrom_EllOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_CircOrbs_noPar_Param1(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_noParallax,
                                   PSBL_PhotAstrom_CircOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_AccOrbs_noPar_Param7(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_noParallax,
                                   PSBL_PhotAstrom_AccOrbs_Param7):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



@inheritdocstring
class PSBL_PhotAstrom_AccOrbs_noPar_Param6(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_noParallax,
                                   PSBL_PhotAstrom_AccOrbs_Param6):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)




@inheritdocstring
class PSBL_PhotAstrom_LinOrbs_noPar_Param7(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_noParallax,
                                   PSBL_PhotAstrom_LinOrbs_Param7):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_LinOrbs_noPar_Param6(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_noParallax,
                                   PSBL_PhotAstrom_LinOrbs_Param6):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



@inheritdocstring
class PSBL_PhotAstrom_EllOrbs_Par_Param1(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_Parallax,
                                   PSBL_PhotAstrom_EllOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_CircOrbs_Par_Param1(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_Parallax,
                                   PSBL_PhotAstrom_CircOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_AccOrbs_Par_Param7(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_Parallax,
                                   PSBL_PhotAstrom_AccOrbs_Param7):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_AccOrbs_Par_Param6(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_Parallax,
                                   PSBL_PhotAstrom_AccOrbs_Param6):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

@inheritdocstring
class PSBL_PhotAstrom_LinOrbs_Par_Param7(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_Parallax,
                                   PSBL_PhotAstrom_LinOrbs_Param7):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)




@inheritdocstring
class PSBL_PhotAstrom_LinOrbs_Par_Param6(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_Parallax,
                                   PSBL_PhotAstrom_LinOrbs_Param6):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_noPar_Param1(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_noParallax,
                                   PSBL_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSBL_parallax
@inheritdocstring
class PSBL_PhotAstrom_Par_Param1(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_Parallax,
                                 PSBL_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSBL
@inheritdocstring
class PSBL_PhotAstrom_noPar_Param2(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_noParallax,
                                   PSBL_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSBL_parallax
@inheritdocstring
class PSBL_PhotAstrom_Par_Param2(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_Parallax,
                                 PSBL_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_Par_Param3(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_Parallax,
                                 PSBL_PhotAstromParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_noPar_Param3(ModelClassABC,
                                   PSBL_PhotAstrom,
                                   PSBL_noParallax,
                                   PSBL_PhotAstromParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



@inheritdocstring
class PSBL_PhotAstrom_EllOrbs_noPar_Param4(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_noParallax,
                                 PSBL_PhotAstrom_EllOrbs_Param4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_EllOrbs_Par_Param4(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_Parallax,
                                 PSBL_PhotAstrom_EllOrbs_Param4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_CircOrbs_noPar_Param4(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_noParallax,
                                 PSBL_PhotAstrom_CircOrbs_Param4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_CircOrbs_Par_Param4(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_Parallax,
                                 PSBL_PhotAstrom_CircOrbs_Param4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



@inheritdocstring
class PSBL_PhotAstrom_Par_Param4(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_Parallax,
                                 PSBL_PhotAstromParam4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)




@inheritdocstring
class PSBL_PhotAstrom_Par_Param5(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_Parallax,
                                 PSBL_PhotAstromParam5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

@inheritdocstring
class PSBL_PhotAstrom_Par_Param7(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_Parallax,
                                 PSBL_PhotAstromParam7):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        

@inheritdocstring
class PSBL_PhotAstrom_EllOrbs_noPar_Param8(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_noParallax,
                                 PSBL_PhotAstrom_EllOrbs_Param8):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_EllOrbs_Par_Param8(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_Parallax,
                                 PSBL_PhotAstrom_EllOrbs_Param8):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_CircOrbs_noPar_Param8(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_noParallax,
                                 PSBL_PhotAstrom_CircOrbs_Param8):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


@inheritdocstring
class PSBL_PhotAstrom_CircOrbs_Par_Param8(ModelClassABC,
                                 PSBL_PhotAstrom,
                                 PSBL_Parallax,
                                 PSBL_PhotAstrom_CircOrbs_Param8):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)
        
# PSBL_phot
@inheritdocstring
class PSBL_Phot_noPar_Param1(ModelClassABC,
                             PSBL_Phot,
                             PSBL_noParallax,
                             PSBL_PhotParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSBL_phot_parallax
@inheritdocstring
class PSBL_Phot_Par_Param1(ModelClassABC,
                           PSBL_Phot,
                           PSBL_Parallax,
                           PSBL_PhotParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSBL phot+astrom, no parallax , with GP
@inheritdocstring
class PSBL_PhotAstrom_noPar_GP_Param1(ModelClassABC,
                                      PSPL_GP,
                                      PSBL_PhotAstrom,
                                      PSBL_noParallax,
                                      PSBL_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSBL_parallax
@inheritdocstring
class PSBL_PhotAstrom_Par_GP_Param1(ModelClassABC,
                                    PSPL_GP,
                                    PSBL_PhotAstrom,
                                    PSBL_Parallax,
                                    PSBL_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSBL
@inheritdocstring
class PSBL_PhotAstrom_noPar_GP_Param2(ModelClassABC,
                                      PSPL_GP,
                                      PSBL_PhotAstrom,
                                      PSBL_noParallax,
                                      PSBL_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSBL_parallax
@inheritdocstring
class PSBL_PhotAstrom_Par_GP_Param2(ModelClassABC,
                                    PSPL_GP,
                                    PSBL_PhotAstrom,
                                    PSBL_Parallax,
                                    PSBL_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSBL_phot
@inheritdocstring
class PSBL_Phot_noPar_GP_Param1(ModelClassABC,
                                PSPL_GP,
                                PSBL_Phot,
                                PSBL_noParallax,
                                PSBL_PhotParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# PSBL_phot_parallax
@inheritdocstring
class PSBL_Phot_Par_GP_Param1(ModelClassABC,
                              PSPL_GP,
                              PSBL_Phot,
                              PSBL_Parallax,
                              PSBL_PhotParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# =====
# BSPL Model Classes
# =====
# BSPL no parallax
@inheritdocstring
class BSPL_Phot_noPar_Param1(ModelClassABC,
                             BSPL_Phot,
                             BSPL_noParallax,
                             BSPL_PhotParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL_parallax
@inheritdocstring
class BSPL_Phot_Par_Param1(ModelClassABC,
                           BSPL_Phot,
                           BSPL_Parallax,
                           BSPL_PhotParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL no parallax
@inheritdocstring
class BSPL_PhotAstrom_noPar_Param1(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



# BSPL_parallax and linear orbits
@inheritdocstring
class BSPL_PhotAstrom_Par_Param1(ModelClassABC,
                                 BSPL_PhotAstrom,
                                 BSPL_Parallax,
                                 BSPL_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


    
# BSPL no parallax
@inheritdocstring
class BSPL_PhotAstrom_noPar_Param2(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


        

# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_Param2(ModelClassABC,
                                 BSPL_PhotAstrom,
                                 BSPL_Parallax,
                                 BSPL_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL no parallax
@inheritdocstring
class BSPL_PhotAstrom_noPar_Param3(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstromParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_Param3(ModelClassABC,
                                 BSPL_PhotAstrom,
                                 BSPL_Parallax,
                                 BSPL_PhotAstromParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# =====
# BSPL + Linear Orbits Model Classes
# =====

# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_AccOrbs_Param1(ModelClassABC,
                                 BSPL_PhotAstrom,
                                 BSPL_Parallax,
                                 BSPL_PhotAstrom_AccOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL no parallax 
@inheritdocstring
class BSPL_PhotAstrom_noPar_AccOrbs_Param1(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_AccOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        


# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_AccOrbs_Param2(ModelClassABC,
                                 BSPL_PhotAstrom,
                                 BSPL_Parallax,
                                 BSPL_PhotAstrom_AccOrbs_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL no parallax 
@inheritdocstring
class BSPL_PhotAstrom_noPar_AccOrbs_Param2(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_AccOrbs_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        
# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_AccOrbs_Param3(ModelClassABC,
                                 BSPL_PhotAstrom,
                                 BSPL_Parallax,
                                 BSPL_PhotAstrom_AccOrbs_Param3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL no parallax 
@inheritdocstring
class BSPL_PhotAstrom_noPar_AccOrbs_Param3(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_AccOrbs_Param3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        

# BSPL no parallax
@inheritdocstring
class BSPL_PhotAstrom_noPar_GP_AccOrbs_Param1(ModelClassABC,
                                      PSPL_GP,
                                      BSPL_PhotAstrom,
                                      BSPL_noParallax,
                                      BSPL_GP_PhotAstrom_AccOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_GP_AccOrbs_Param1(ModelClassABC,
                                    PSPL_GP,
                                    BSPL_PhotAstrom,
                                    BSPL_Parallax,
                                    BSPL_GP_PhotAstrom_AccOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL no parallax
@inheritdocstring
class BSPL_PhotAstrom_noPar_GP_AccOrbs_Param2(ModelClassABC,
                                      PSPL_GP,
                                      BSPL_PhotAstrom,
                                      BSPL_noParallax,
                                      BSPL_GP_PhotAstrom_AccOrbs_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_GP_AccOrbs_Param2(ModelClassABC,
                                    PSPL_GP,
                                    BSPL_PhotAstrom,
                                    BSPL_Parallax,
                                    BSPL_GP_PhotAstrom_AccOrbs_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        
# BSPL no parallax
@inheritdocstring
class BSPL_PhotAstrom_noPar_GP_AccOrbs_Param3(ModelClassABC,
                                      PSPL_GP,
                                      BSPL_PhotAstrom,
                                      BSPL_noParallax,
                                      BSPL_GP_PhotAstrom_AccOrbs_Param3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_GP_AccOrbs_Param3(ModelClassABC,
                                    PSPL_GP,
                                    BSPL_PhotAstrom,
                                    BSPL_Parallax,
                                    BSPL_GP_PhotAstrom_AccOrbs_Param3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        
        
# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_LinOrbs_Param1(ModelClassABC,
                                 BSPL_PhotAstrom,
                                 BSPL_Parallax,
                                 BSPL_PhotAstrom_LinOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL no parallax 
@inheritdocstring
class BSPL_PhotAstrom_noPar_LinOrbs_Param1(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_LinOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_LinOrbs_Param2(ModelClassABC,
                                 BSPL_PhotAstrom,
                                 BSPL_Parallax,
                                 BSPL_PhotAstrom_LinOrbs_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL no parallax 
@inheritdocstring
class BSPL_PhotAstrom_noPar_LinOrbs_Param2(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_LinOrbs_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



# BSPL no parallax 
@inheritdocstring
class BSPL_PhotAstrom_noPar_LinOrbs_Param3(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_LinOrbs_Param3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL_parallax 
@inheritdocstring
class BSPL_PhotAstrom_Par_LinOrbs_Param3(ModelClassABC,
                                 BSPL_PhotAstrom,
                                 BSPL_Parallax,
                                 BSPL_PhotAstrom_LinOrbs_Param3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        
        

# BSPL no parallax
@inheritdocstring
class BSPL_PhotAstrom_noPar_GP_LinOrbs_Param1(ModelClassABC,
                                      PSPL_GP,
                                      BSPL_PhotAstrom,
                                      BSPL_noParallax,
                                      BSPL_GP_PhotAstrom_LinOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_GP_LinOrbs_Param1(ModelClassABC,
                                    PSPL_GP,
                                    BSPL_PhotAstrom,
                                    BSPL_Parallax,
                                    BSPL_GP_PhotAstrom_LinOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL no parallax
@inheritdocstring
class BSPL_PhotAstrom_noPar_GP_LinOrbs_Param2(ModelClassABC,
                                      PSPL_GP,
                                      BSPL_PhotAstrom,
                                      BSPL_noParallax,
                                      BSPL_GP_PhotAstrom_LinOrbs_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_GP_LinOrbs_Param2(ModelClassABC,
                                    PSPL_GP,
                                    BSPL_PhotAstrom,
                                    BSPL_Parallax,
                                    BSPL_GP_PhotAstrom_LinOrbs_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        
# BSPL no parallax
@inheritdocstring
class BSPL_PhotAstrom_noPar_GP_LinOrbs_Param3(ModelClassABC,
                                      PSPL_GP,
                                      BSPL_PhotAstrom,
                                      BSPL_noParallax,
                                      BSPL_GP_PhotAstrom_LinOrbs_Param3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_GP_LinOrbs_Param3(ModelClassABC,
                                    PSPL_GP,
                                    BSPL_PhotAstrom,
                                    BSPL_Parallax,
                                    BSPL_GP_PhotAstrom_LinOrbs_Param3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)





class BSPL_PhotAstrom_Par_CircOrbs_Param1(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_Parallax,
                                   BSPL_PhotAstrom_Circ_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



class BSPL_PhotAstrom_noPar_CircOrbs_Param1(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_Circ_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)




class BSPL_PhotAstrom_Par_CircOrbs_Param4(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_Parallax,
                                   BSPL_PhotAstrom_Circ_Param4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



class BSPL_PhotAstrom_noPar_CircOrbs_Param4(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_Circ_Param4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)






class BSPL_PhotAstrom_Par_CircOrbs_Param2(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_Parallax,
                                   BSPL_PhotAstrom_Circ_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



class BSPL_PhotAstrom_noPar_CircOrbs_Param2(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_Circ_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


        


class BSPL_PhotAstrom_Par_CircOrbs_Param3(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_Parallax,
                                   BSPL_PhotAstrom_Circ_Param3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



class BSPL_PhotAstrom_noPar_CircOrbs_Param3(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_Circ_Param3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)





class BSPL_PhotAstrom_Par_EllOrbs_Param1(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_Parallax,
                                   BSPL_PhotAstrom_Ell_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



class BSPL_PhotAstrom_noPar_EllOrbs_Param1(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_Ell_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)





class BSPL_PhotAstrom_Par_EllOrbs_Param2(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_Parallax,
                                   BSPL_PhotAstrom_Ell_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



class BSPL_PhotAstrom_noPar_EllOrbs_Param2(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_Ell_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


        


class BSPL_PhotAstrom_Par_EllOrbs_Param3(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_Parallax,
                                   BSPL_PhotAstrom_Ell_Param3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



class BSPL_PhotAstrom_noPar_EllOrbs_Param3(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_Ell_Param3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


class BSPL_PhotAstrom_Par_EllOrbs_Param4(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_Parallax,
                                   BSPL_PhotAstrom_Ell_Param4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



class BSPL_PhotAstrom_noPar_EllOrbs_Param4(ModelClassABC,
                                   BSPL_PhotAstrom,
                                   BSPL_noParallax,
                                   BSPL_PhotAstrom_Ell_Param4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


        
        
# =====
# BSPL + GP Model Classes
# =====
# BSPL no parallax
@inheritdocstring
class BSPL_Phot_noPar_GP_Param1(ModelClassABC,
                                PSPL_GP,
                                BSPL_Phot,
                                BSPL_noParallax,
                                BSPL_GP_PhotParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL_parallax
@inheritdocstring
class BSPL_Phot_Par_GP_Param1(ModelClassABC,
                              PSPL_GP,
                              BSPL_Phot,
                              BSPL_Parallax,
                              BSPL_GP_PhotParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL no parallax
@inheritdocstring
class BSPL_PhotAstrom_noPar_GP_Param1(ModelClassABC,
                                      PSPL_GP,
                                      BSPL_PhotAstrom,
                                      BSPL_noParallax,
                                      BSPL_GP_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_GP_Param1(ModelClassABC,
                                    PSPL_GP,
                                    BSPL_PhotAstrom,
                                    BSPL_Parallax,
                                    BSPL_GP_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL no parallax
@inheritdocstring
class BSPL_PhotAstrom_noPar_GP_Param2(ModelClassABC,
                                      PSPL_GP,
                                      BSPL_PhotAstrom,
                                      BSPL_noParallax,
                                      BSPL_GP_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_GP_Param2(ModelClassABC,
                                    PSPL_GP,
                                    BSPL_PhotAstrom,
                                    BSPL_Parallax,
                                    BSPL_GP_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL no parallax
@inheritdocstring
class BSPL_PhotAstrom_noPar_GP_Param3(ModelClassABC,
                                      PSPL_GP,
                                      BSPL_PhotAstrom,
                                      BSPL_noParallax,
                                      BSPL_GP_PhotAstromParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSPL_parallax
@inheritdocstring
class BSPL_PhotAstrom_Par_GP_Param3(ModelClassABC,
                                    PSPL_GP,
                                    BSPL_PhotAstrom,
                                    BSPL_Parallax,
                                    BSPL_GP_PhotAstromParam3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

# =====
# PSBL Model Classes
# =====
# BSBL
@inheritdocstring
class BSBL_PhotAstrom_noPar_Param1(ModelClassABC,
                                   BSBL_PhotAstrom,
                                   BSBL_noParallax,
                                   BSBL_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSBL_parallax
@inheritdocstring
class BSBL_PhotAstrom_Par_EllOrbs_Param1(ModelClassABC,
                                 BSBL_PhotAstrom,
                                 BSBL_Parallax,
                                 BSBL_PhotAstrom_EllOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        
@inheritdocstring
class BSBL_PhotAstrom_noPar_EllOrbs_Param1(ModelClassABC,
                                   BSBL_PhotAstrom,
                                   BSBL_noParallax,
                                   BSBL_PhotAstrom_EllOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


# BSBL_parallax
@inheritdocstring
class BSBL_PhotAstrom_Par_EllOrbs_Param2(ModelClassABC,
                                 BSBL_PhotAstrom,
                                 BSBL_Parallax,
                                 BSBL_PhotAstrom_EllOrbs_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        
@inheritdocstring
class BSBL_PhotAstrom_noPar_EllOrbs_Param2(ModelClassABC,
                                   BSBL_PhotAstrom,
                                   BSBL_noParallax,
                                   BSBL_PhotAstrom_EllOrbs_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


        
# BSBL_parallax
@inheritdocstring
class BSBL_PhotAstrom_Par_CircOrbs_Param1(ModelClassABC,
                                 BSBL_PhotAstrom,
                                 BSBL_Parallax,
                                 BSBL_PhotAstrom_CircOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        
@inheritdocstring
class BSBL_PhotAstrom_noPar_CircOrbs_Param1(ModelClassABC,
                                   BSBL_PhotAstrom,
                                   BSBL_noParallax,
                                   BSBL_PhotAstrom_CircOrbs_Param1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        
# BSBL_parallax
@inheritdocstring
class BSBL_PhotAstrom_Par_CircOrbs_Param2(ModelClassABC,
                                 BSBL_PhotAstrom,
                                 BSBL_Parallax,
                                 BSBL_PhotAstrom_CircOrbs_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)

        
@inheritdocstring
class BSBL_PhotAstrom_noPar_CircOrbs_Param2(ModelClassABC,
                                   BSBL_PhotAstrom,
                                   BSBL_noParallax,
                                   BSBL_PhotAstrom_CircOrbs_Param2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


        

# BSBL_parallax
@inheritdocstring
class BSBL_PhotAstrom_Par_Param1(ModelClassABC,
                                 BSBL_PhotAstrom,
                                 BSBL_Parallax,
                                 BSBL_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


        
# BSBL_noparallax
@inheritdocstring
class BSBL_PhotAstrom_noPar_Param2(ModelClassABC,
                                 BSBL_PhotAstrom,
                                 BSBL_noParallax,
                                 BSBL_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)
        
# BSBL_parallax
@inheritdocstring
class BSBL_PhotAstrom_Par_Param2(ModelClassABC,
                                 BSBL_PhotAstrom,
                                 BSBL_Parallax,
                                 BSBL_PhotAstromParam2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)



# =====
# FSPL Model
# =====
# FSPL_parallax
@inheritdocstring
class FSPL_PhotAstrom_Par_Param1(ModelClassABC,
                                 FSPL_PhotAstrom,
                                 FSPL_Parallax,
                                 FSPL_PhotAstromParam1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        startbases(self)
        checkconflicts(self)


########################################
### GENERAL USE AND SHARED FUNCTIONS ###
########################################

# Define some constants & conversion factors
kappa = 4.0 * const.G * units.rad / (const.c ** 2 * units.au)
kappa = kappa.to(units.mas / units.solMass)

days_per_year = 365.25
meter_per_AU = 1.496e11
meter_per_Rsun = 6.96e8

    
def get_model(model_class, params, params_fixed):
    """Helper function to get a BAGLE model based on the
    model_class and input dictionaries for the
    variable and the fixed parameters.

    Parameters
    ----------
    model_class : a BAGLE model class
        A BAGLE model class object uninstantiated.

    params : dict
        A dictionary of parameters for the input model
        class. The returned BAGLE model instance will
        be instantiated with these parameters.

    params_fixed : dict or None
        Input fixed model parameters such as raL, decL
        (for parallax) in a model.
    """

    if model_class.fixed_param_names is not None:
        mod = model_class(*params.values(), **params_fixed)
    else:
        mod = model_class(*params.values())

    return mod


def mag2flux(mag):
    mag_zp = 30.0  # arbitrary but allows for negative blend fractions.
    flux_zp = 1.0

    flux = flux_zp * 10 ** ((mag - mag_zp) / -2.5)

    return flux


def flux2mag(flux):
    mag_zp = 30.0  # arbitrary but allows for negative blend fractions.
    flux_zp = 1.0

    mag = -2.5 * np.log10(flux / flux_zp) + mag_zp

    return mag


def u0_hat_from_thetaE_hat(thetaE_hat, beta):
    """
    Calculate the closest approach vector direction. Define the beta sign convention
    as Andy Gould does with 
    
        * beta > 0 means u0_E > 0
        * u0_amp > 0 mean u0_E > 0 

    See `Gould 2004, pg 320, bottom right`
    
    u0 > 0 --> lens passes to the right side of the source as seen from Earth   

    :math:`thetaX0 = xS0 - xL0 =  u0 * thetaE`

    which implies that:

        * u0_E > 0 for u0 > 0
        * u0_E < 0 for u0 < 0

    which is what we use.
    """
    u0_hat = np.zeros(2, dtype=float)
    if beta > 0:
        u0_hat[0] = np.abs(thetaE_hat[1])

        if np.sign(thetaE_hat).prod() > 0:
            u0_hat[1] = -np.abs(thetaE_hat[0])
        else:
            u0_hat[1] = np.abs(thetaE_hat[0])
    else:
        u0_hat[0] = -np.abs(thetaE_hat[1])

        if np.sign(thetaE_hat).prod() > 0:
            u0_hat[1] = np.abs(thetaE_hat[0])
        else:
            u0_hat[1] = -np.abs(thetaE_hat[0])

    return u0_hat


@cache_memory.cache()
def parallax_in_direction(RA, Dec, mjd):
    """
    | R.A. in degrees. (J2000)
    | Dec. in degrees. (J2000)
    | MJD
    
    Equations following MulensModel.
    """
    print('parallax_in_direction: len(t) = ', len(mjd))

    # Munge inputs into astropy format.
    times = Time(mjd + 2400000.5, format='jd', scale='tdb')
    coord = SkyCoord(RA, Dec, unit=(units.deg, units.deg))

    direction = coord.cartesian.xyz.value
    north = np.array([0., 0., 1.])
    _east_projected = np.cross(north, direction) / np.linalg.norm(np.cross(north, direction))
    _north_projected = np.cross(direction, _east_projected) / np.linalg.norm(np.cross(direction, _east_projected))
    sun_earth_pos = get_body_barycentric(body='sun', time=times) - get_body_barycentric(body='earth', time=times)
    pos = sun_earth_pos.xyz.T.to(units.au)

    e = np.dot(pos, _east_projected)
    n = np.dot(pos, _north_projected)

    pvec = np.array([e.value, n.value]).T

    return pvec


def dparallax_dt_in_direction(RA, Dec, mjd):
    """
    R.A. in degrees. (J2000)
    Dec. in degrees. (J2000)
    MJD
    
    Equations following MulensModel.
    Time derivative --> units are yr^-1
    
    """
    print('parallax_in_direction: len(t) = ', len(mjd))
    # Munge inputs into astropy format.
    times = Time(mjd + 2400000.5, format='jd', scale='tdb')
    coord = SkyCoord(RA, Dec, unit=(units.deg, units.deg))

    direction = coord.cartesian.xyz.value
    north = np.array([0., 0., 1.])
    _east_projected = np.cross(north, direction) / np.linalg.norm(np.cross(north, direction))
    _north_projected = np.cross(direction, _east_projected) / np.linalg.norm(np.cross(direction, _east_projected))
    sun_earth_vel = get_body_barycentric_posvel('Sun', times)[1] - get_body_barycentric_posvel('Earth', times)[1]
    vel = sun_earth_vel.xyz.T.to(units.au / units.year)

    e = np.dot(vel, _east_projected)
    n = np.dot(vel, _north_projected)

    dpvec_dt = np.array([e.value, n.value]).T

    return dpvec_dt


def sun_position(mjd, radians=False):
    """

    NAME:
          SUNPOS
          
    PURPOSE:
          To compute the RA and Dec of the Sun at a given date.
          
    INPUTS:
          mjd    - The modified Julian date of the day (and time), scalar or vector

    OUTPUTS:
          ra:
              | The right ascension of the sun at that date in DEGREES
              | double precision, same number of elements as jd
          dec:
              The declination of the sun at that date in DEGREES
          elong:
              Ecliptic longitude of the sun at that date in DEGREES.
          obliquity:
              the obliquity of the ecliptic, in DEGREES

    OPTIONAL INPUT KEYWORD:
          RADIAN [def=False] - If this keyword is set to True, then all output variables
          are given in Radians rather than Degrees

    NOTES:
          Patrick Wallace (Rutherford Appleton Laboratory, UK) has tested the
          accuracy of a C adaptation of the sunpos.pro code and found the
          following results.   From 1900-2100 SUNPOS  gave 7.3 arcsec maximum
          error, 2.6 arcsec RMS.  Over the shorter interval 1950-2050 the figures
          were 6.4 arcsec max, 2.2 arcsec RMS.

          The returned RA and Dec are in the given date's equinox.

          Procedure was extensively revised in May 1996, and the new calling
          sequence is incompatible with the old one.
    METHOD:
          Uses a truncated version of Newcomb's Sun.    Adapted from the IDL
          routine SUN_POS by CD Pike, which was adapted from a FORTRAN routine
          by B. Emerson (RGO).
    EXAMPLE:
          (1) Find the apparent RA and Dec of the Sun on May 1, 1982

          | IDL> jdcnv, 1982, 5, 1,0 ,jd      ;Find Julian date jd = 2445090.5
          | IDL> sunpos, jd, ra, dec
          | IDL> print,adstring(ra,dec,2) 
          | 02 31 32.61  +14 54 34.9

          The Astronomical Almanac gives 02 31 32.58 +14 54 34.9 so the error
          in SUNPOS for this case is < 0.5".

          (2) Find the apparent RA and Dec of the Sun for every day in 1997

          | IDL> jdcnv, 1997,1,1,0, jd                ;Julian date on Jan 1, 1997
          | IDL> sunpos, jd+ dindgen(365), ra, dec    ;RA and Dec for each day

    MODIFICATION HISTORY:
    
          * Written by Michael R. Greason, STX, 28 October 1988.
          * Accept vector arguments, W. Landsman -     April,1989
          * Eliminated negative right ascensions - MRG, Hughes STX, 6 May 1992.
          * Rewritten using the 1993 Almanac.  Keywords added.  MRG, HSTX, 10 February 1994.
          * Major rewrite, improved accuracy, always return values in degrees - W. Landsman May, 1996
          * Added /RADIAN keyword; W. Landsman; August, 1997
          * Converted to IDL V5.0; W. Landsman; September 1997
          * Converted to python; J. R. Lu; August 2016
    """
    #  form time in Julian centuries from 1900.0
    t_obj = Time(mjd, format='mjd')
    t = (t_obj.jd - 2415020.0) / 36525.0

    #  form sun's mean longitude
    l = (279.696678 + ((36000.768925 * t) % 360.0)) * 3600.0

    #  allow for ellipticity of the orbit (equation of centre)
    #  using the Earth's mean anomaly ME
    me = 358.475844 + ((35999.049750 * t) % 360.0)
    ellcor = (6910.1 - 17.2 * t) * np.sin(np.radians(me)) + 72.3 * np.sin(
        np.radians(2.0 * me))
    l = l + ellcor

    # allow for the Venus perturbations using the mean anomaly of Venus MV
    mv = 212.603219 + ((58517.803875 * t) % 360.0)
    vencorr = 4.8 * np.cos(np.radians(299.1017 + mv - me)) + \
              5.5 * np.cos(np.radians(148.3133 + 2.0 * mv - 2.0 * me)) + \
              2.5 * np.cos(np.radians(315.9433 + 2.0 * mv - 3.0 * me)) + \
              1.6 * np.cos(np.radians(345.2533 + 3.0 * mv - 4.0 * me)) + \
              1.0 * np.cos(np.radians(318.1500 + 3.0 * mv - 5.0 * me))
    l += vencorr

    #  Allow for the Mars perturbations using the mean anomaly of Mars MM
    mm = 319.529425 + ((19139.858500 * t) % 360.0)
    marscorr = 2.0 * np.cos(np.radians(343.8883 - 2.0 * mm + 2.0 * me)) + \
               1.8 * np.cos(np.radians(200.4017 - 2.0 * mm + me))
    l += marscorr

    # Allow for the Jupiter perturbations using the mean anomaly of Jupiter MJ
    mj = 225.328328 + ((3034.6920239 * t) % 360.0)
    jupcorr = 7.2 * np.cos(np.radians(179.5317 - mj + me)) + \
              2.6 * np.cos(np.radians(263.2167 - mj)) + \
              2.7 * np.cos(np.radians(87.1450 - 2.0 * mj + 2.0 * me)) + \
              1.6 * np.cos(np.radians(109.4933 - 2.0 * mj + me))
    l += jupcorr

    # Allow for the Moons perturbations using the mean elongation of
    # the Moon from the Sun D
    d = 350.7376814 + ((445267.11422 * t) % 360.0)
    mooncorr = 6.5 * np.sin(np.radians(d))
    l += mooncorr

    # Allow for long period terms
    longterm = + 6.4 * np.sin(np.radians(231.19 + 20.20 * t))
    l += longterm
    l = (l + 2592000.0) % 1296000.0
    longmed = l / 3600.0

    # Allow for Aberration
    l -= 20.5

    # Allow for Nutation using the longitude of the Moons mean node OMEGA
    omega = 259.183275 - ((1934.142008 * t) % 360.0)
    l -= 17.2 * np.sin(np.radians(omega))

    # Form the True Obliquity
    oblt = 23.452294 - 0.0130125 * t + (
            9.2 * np.cos(np.radians(omega))) / 3600.0

    # Form Right Ascension and Declination
    l = l / 3600.0
    l_rad = np.radians(l)
    oblt_rad = np.radians(oblt)
    ra = np.arctan2(np.sin(l_rad) * np.cos(oblt_rad), np.cos(l_rad))

    if (len(ra) > 1):
        neg = np.where(ra < 0.0)[0]
        ra[neg] = ra[neg] + 2.0 * math.pi

    dec = np.arcsin(np.sin(l_rad) * np.sin(oblt_rad))

    if radians:
        oblt = oblt_rad
        longmed = np.radians(longmed)
    else:
        ra = np.degrees(ra)
        dec = np.degrees(dec)

    return ra, dec, longmed, oblt


def get_angular_einstein_radius(m, d1, d2):
    # given the mass of the lens and the distance to source/lens we can calculate the einstein radius
    a = (1.0 / (d1 * units.pc)) - (1.0 / (d2 * units.pc))
    A = units.rad * np.sqrt(
        (4.0 * const.G * m * units.M_sun / const.c ** 2) * a)
    return A.to('mas').value


def get_unit_vector(x):
    # takes in a vector, gives out the unit vector
    return np.array(x) / np.linalg.norm(x)


def get_u0(thetaE_hat, beta, thetaE_amp):
    # gives the closest approach vector
    u0hat = get_uhat(thetaE_hat, beta)  # unit vector
    u0amp = beta / thetaE_amp
    return np.abs(u0amp) * u0hat


def get_uhat(thetaE_hat, beta):
    # given beta gives back the closest approach unit vector
    # inherited from the pspl
    u0_hat = np.zeros(2, dtype=float)
    if beta > 0:
        u0_hat[0] = -np.abs(thetaE_hat[1])
        if np.sign(thetaE_hat).prod() > 0:
            u0_hat[1] = np.abs(thetaE_hat[0])
        else:
            u0_hat[1] = -np.abs(thetaE_hat[0])
    else:
        u0_hat[0] = np.abs(thetaE_hat[1])
        if np.sign(thetaE_hat).prod() > 0:
            u0_hat[1] = -np.abs(thetaE_hat[0])
        else:
            u0_hat[1] = np.abs(thetaE_hat[0])
    return u0_hat


def get_einstein_time(theta, v, days):
    # returns the einstein crossing time
    return (theta / np.linalg.norm(v)) * days


def get_thetas(source, lens):
    # returns a list of the relative angular positions of the source relative to the lens
    thetas = []
    for i in range(len(source)):
        thetas.append(source[i] - lens[i])
    return np.array(thetas)


def get_amplitudes(vectors):
    # takes in a list of vectors and returns a list of the amplitudes of those vectors
    amplitudes = []
    for i in vectors:
        amplitudes.append(np.linalg.norm(i, axis=1))
    return np.array(amplitudes)


def get_unit_vectors(vectors):
    # takes in a list of a list vectors, outputs a list of a list unit vectors
    unit_vectors = []
    for i in vectors:
        times = []
        for j in i:
            times.append(get_unit_vector(j))
        unit_vectors.append(times)
    return np.array(unit_vectors)


def get_plus(amps, hats, pos, lens, radius):
    # given the positions of the source returns the positions of the positive image
    plus = []
    for i in range(len(amps)):
        times = []
        for j in range(len(amps[0])):
            times.append(((0.5 * (amps[i][j] + np.sqrt(
                (amps[i][j]) ** 2 + 4 * radius ** 2)) * hats[i][j])) + lens[i])
        plus.append(times)
    return np.array(plus)


def get_minus(amps, hats, pos, lens, radius):
    # given the positions of the source returns the positions of the negative image
    minus = []
    for i in range(len(amps)):
        times = []
        for j in range(len(amps[0])):
            times.append(((0.5 * (amps[i][j] - np.sqrt(
                (amps[i][j]) ** 2 + 4 * radius ** 2)) * hats[i][j])) + lens[i])
        minus.append(times)
    return np.array(minus)


def oned_int(centre, function1, function2, ymax, ymin, n, x, middle, centres):
    integral = 0
    hy = (ymax - ymin) / n
    for i in range(n + 1):
        sqdistance = (x - middle[0]) ** 2 + (i * hy + ymin - middle[1]) ** 2
        mindistance = 10000
        for j in range(len(centres)):
            distance = (x - centres[j][0]) ** 2 + (
                    i * hy + ymin - centres[j][1]) ** 2
            if distance < mindistance:
                mindistance = distance
        if mindistance == sqdistance:
            if i == 0 or i == n:
                integral += hy / 2 * function1(
                    function2([x, i * hy + ymin]) - centre)
            else:
                integral += hy * function1(
                    function2([x, i * hy + ymin]) - centre)
    return integral


def twod_int(centre, function1, function2, xmax, xmin, ymax, ymin, nx, ny,
             middle, centres):
    integral = 0
    hx = (xmax - xmin) / nx
    for i in range(nx + 1):
        x = i * hx
        if i == 0 or i == nx:
            integral += hx / 2 * oned_int(centre, function1, function2, ymax,
                                          ymin, ny, xmin + x, middle, centres)
        else:
            integral += hx * oned_int(centre, function1, function2, ymax, ymin,
                                      ny, xmin + x, middle, centres)
    return integral


def twod_cent_x_int(centre, function1, function2, xmax, xmin, ymax, ymin, nx,
                    ny, middle, centres):
    integral = 0
    hx = (xmax - xmin) / nx
    for i in range(nx + 1):
        x = i * hx
        if i == 0 or i == nx:
            integral += hx / 2 * oned_x_int(centre, function1, function2, ymax,
                                            ymin, ny, xmin + x, middle,
                                            centres)
        else:
            integral += hx * oned_x_int(centre, function1, function2, ymax,
                                        ymin, ny, xmin + x, middle, centres)
    return integral


def oned_x_int(centre, function1, function2, ymax, ymin, n, x, middle,
               centres):
    integral = 0
    hy = (ymax - ymin) / n
    for i in range(n + 1):
        sqdistance = (x - middle[0]) ** 2 + (i * hy + ymin - middle[1]) ** 2
        mindistance = 10000
        for j in range(len(centres)):
            distance = (x - centres[j][0]) ** 2 + (
                    i * hy + ymin - centres[j][1]) ** 2
            if distance < mindistance:
                mindistance = distance
        if mindistance == sqdistance:
            if i == 0 or i == n:
                integral += hy / 2 * x * function1(
                    function2([x, i * hy + ymin]) - centre)
            else:
                integral += hy * x * function1(
                    function2([x, i * hy + ymin]) - centre)
    return integral


def twod_cent_y_int(centre, function1, function2, xmax, xmin, ymax, ymin, nx,
                    ny, middle, centres):
    integral = 0
    hx = (xmax - xmin) / nx
    for i in range(nx + 1):
        x = i * hx
        if i == 0 or i == nx:
            integral += hx / 2 * oned_y_int(centre, function1, function2, ymax,
                                            ymin, ny, xmin + x, middle,
                                            centres)
        else:
            integral += hx * oned_y_int(centre, function1, function2, ymax,
                                        ymin, ny, xmin + x, middle, centres)
    return integral


def oned_y_int(centre, function1, function2, ymax, ymin, n, x, middle,
               centres):
    integral = 0
    hy = (ymax - ymin) / n
    for i in range(n + 1):
        sqdistance = (x - middle[0]) ** 2 + (i * hy + ymin - middle[1]) ** 2
        mindistance = 10000
        for j in range(len(centres)):
            distance = (x - centres[j][0]) ** 2 + (
                    i * hy + ymin - centres[j][1]) ** 2
            if distance < mindistance:
                mindistance = distance
        if mindistance == sqdistance:
            if i == 0 or i == n:
                integral += hy / 2 * (i * hy + ymin) * function1(
                    function2([x, i * hy + ymin]) - centre)
            else:
                integral += hy * (i * hy + ymin) * function1(
                    function2([x, i * hy + ymin]) - centre)
    return integral


def get_image(y0, m1, d, R):
    """ Function to find the images of the star
    
    Parameters
    ----------
    y0:
        position of the cente of the source star, in units of anguler Einstein radius
    m1:
        Mass of rightmost lens divided by the total mass
    d:
        separation of the lenses in angular Einstein radii
    R:
        angular radius of the source in angular Einstein radii
    """
    # print("y0 = (%f, %f), m1 = %f, d = %f\n" %(y0[0],y0[1],m1,d))
    """ These 2 arrays give make up of the contour grid at each step of the iteration
            n = the number of grid points (n x n) centred on each previously saved point at a given step of the iteration
            precision = the radius of star whos images may be found with the resolution of grid at this step
    """
    n = (76, 5, 5, 5, 5, 5, 5, 5, 5, 5)
    precision = (
        0.9, 0.4, 0.04, 0.009, 0.0015, 0.0004, 0.00008, 0.00005, 0.00001)

    def source(x, m1, d):
        # Given an image point, this function tells you where the source is
        y1 = x[0] - m1 * (x[0] - (1 - m1) * d) / (
                (x[0] - (1 - m1) * d) ** 2 + x[1] ** 2) - (1 - m1) * (
                     x[0] + m1 * d) / ((x[0] + m1 * d) ** 2 + x[1] ** 2)
        y2 = x[1] - m1 * (x[1]) / ((x[0] - (1 - m1) * d) ** 2 + x[1] ** 2) - (
                1 - m1) * (x[1]) / ((x[0] + m1 * d) ** 2 + x[1] ** 2)
        return np.array([y1, y2])

    """ This block creates an n[0] x n[0] grid from -3 to 3 in both coordinates, and saves all the image points that correspond to
    source locations closer than precision[0] to the centre of the source
    """
    interesting_points = []
    for i in range(n[0]):
        for j in range(n[0]):
            x = (-3 + i / (n[0] - 1) * 6, -3 + j / (n[0] - 1) * 6)
            y = source(x, m1, d)
            s2 = (y0[0] - y[0]) ** 2 + (y0[1] - y[1]) ** 2
            if np.sqrt(s2) < precision[0]:
                interesting_points.append(x)
    # Note that after this step, the distance between points on the contour grid is 6/(n[0]-1)
    if 1.2 * R > precision[0]:
        return np.array(interesting_points)

    resolution = 6 / (n[0] - 1)
    for i in range(len(precision) - 1):
        image = []
        resolution = resolution * 1 / (n[i + 1])
        if 1.2 * R > precision[i + 1]:
            for j in interesting_points:
                for k in range(n[i + 1]):
                    for l in range(n[i + 1]):
                        x = (j[0] + (k - (n[i + 1] - 1) / 2) * resolution,
                             j[1] + (l - (n[i + 1] - 1) / 2) * resolution)
                        y = source(x, m1, d)
                        s2 = (y0[0] - y[0]) ** 2 + (y0[1] - y[1]) ** 2
                        if np.sqrt(s2) < 1.1 * R:
                            image.append(x)
            return np.array(image)
        else:
            for j in interesting_points:
                for k in range(n[i + 1]):
                    for l in range(n[i + 1]):
                        x = (j[0] + (k - (n[i + 1] - 1) / 2) * resolution,
                             j[1] + (l - (n[i + 1] - 1) / 2) * resolution)
                        y = source(x, m1, d)
                        s2 = (y0[0] - y[0]) ** 2 + (y0[1] - y[1]) ** 2
                        if np.sqrt(s2) < precision[i + 1]:
                            image.append(x)
            interesting_points = image
    return np.array(image)


def cluster(image, R):
    """Split the image points up into a number of clusters
    
    At first there are 10 clusters
            1. The centre of each cluster are set to be equally
                spaced out in angle, at a radius of 1
            2. Each point is looped through and assigned to the
                cluster with the closest centre
            3. The centre of the clusters is set to be the mean
                position of all the points in the cluster.
                If a cluster has no points, then it's centre is set
                to (100,100), effectively deleting the cluster
            4. Each point is then reassigned to the nearest cluster
    
    Parameters
    ----------
    image:
        a list of points, all of which are inside some image of the star
    R:
        radius of the star

    """
    no_of_clusters = 10
    centres = []
    clusters = []
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    cluster5 = []
    cluster6 = []
    cluster7 = []
    cluster8 = []
    cluster9 = []
    cluster10 = []
    for i in range(no_of_clusters):
        centres.append((np.cos(i * 2 * np.pi / no_of_clusters),
                        np.sin(i * 2 * np.pi / no_of_clusters)))
    loops = 0
    total = 0
    while loops < 2:
        changed = False
        for point in image:
            shortestsqdistance = 10000000
            shortestindex = 0
            for j in range(len(centres)):
                distance = (point[0] - centres[j][0]) ** 2 + (
                        point[1] - centres[j][1]) ** 2
                if distance < shortestsqdistance:
                    shortestsqdistance = distance
                    shortestindex = j
            if shortestindex == 0:
                cluster1.append(point)
            elif shortestindex == 1:
                cluster2.append(point)
            elif shortestindex == 2:
                cluster3.append(point)
            elif shortestindex == 3:
                cluster4.append(point)
            elif shortestindex == 4:
                cluster5.append(point)
            elif shortestindex == 5:
                cluster6.append(point)
            elif shortestindex == 6:
                cluster7.append(point)
            elif shortestindex == 7:
                cluster8.append(point)
            elif shortestindex == 8:
                cluster9.append(point)
            elif shortestindex == 9:
                cluster10.append(point)
        cluster1 = np.array(cluster1)
        cluster2 = np.array(cluster2)
        cluster3 = np.array(cluster3)
        cluster4 = np.array(cluster4)
        cluster5 = np.array(cluster5)
        cluster6 = np.array(cluster6)
        cluster7 = np.array(cluster7)
        cluster8 = np.array(cluster8)
        cluster9 = np.array(cluster9)
        cluster10 = np.array(cluster10)
        clusters = []
        clusters.append(cluster1)
        clusters.append(cluster2)
        clusters.append(cluster3)
        clusters.append(cluster4)
        clusters.append(cluster5)
        clusters.append(cluster6)
        clusters.append(cluster7)
        clusters.append(cluster8)
        clusters.append(cluster9)
        clusters.append(cluster10)
        centres = []
        for k in range(no_of_clusters):
            if len(clusters[k]) != 0:
                centres.append((sum(clusters[k][:, 0]) / len(clusters[k]),
                                sum(clusters[k][:, 1]) / len(clusters[k])))
            else:
                centres.append((100, 100))
        if loops < 1:
            cluster1 = []
            cluster2 = []
            cluster3 = []
            cluster4 = []
            cluster5 = []
            cluster6 = []
            cluster7 = []
            cluster8 = []
            cluster9 = []
            cluster10 = []
            loops += 1
        else:
            if changed == False and total < 10:
                """Now if the maximum distance from a point in a cluster to the centre of the cluster is either greater than 3
                standard deviations, or 25 R, then it creates a new cluster centre on the further point, and goes back through
                the loop again.
                """
                for k in range(no_of_clusters):
                    maxdistance = 0
                    maxindex = 0
                    totalsqdistance = 0
                    std = 0
                    if len(clusters[k]) > 1:
                        for i in range(len(clusters[k])):
                            distance = (centres[k][0] - clusters[k][i][
                                0]) ** 2 + (centres[k][1] - clusters[k][i][
                                1]) ** 2
                            totalsqdistance += distance
                            if np.sqrt(distance) > maxdistance:
                                maxdistance = np.sqrt(distance)
                                maxindex = i
                        std = np.sqrt(totalsqdistance / (len(clusters[k]) - 1))
                        if maxdistance > 3 * std:
                            done = False
                            for j in range(5):
                                if centres[j][0] == 100 and done == False:
                                    centres[j] = clusters[k][maxindex]
                                    done = True
                                    cluster1 = []
                                    cluster2 = []
                                    cluster3 = []
                                    cluster4 = []
                                    cluster5 = []
                                    cluster6 = []
                                    cluster7 = []
                                    cluster8 = []
                                    cluster9 = []
                                    cluster10 = []
                                    loops -= 1
                                    changed = True

            if changed == False and total < 8:
                """Now if 2 clusters are too close together, it makes them one cluster assuming this to be 1 image,
                    and goes through the loop again
                        However, if an infinite loop is found where at cluster is continually broken up then merged
                        together (c.f paper), the total < 8 here, when compared with the total < 10 on the previous step
                        ensure that the algorithm errs on the side of splitting them up.
                """
                for k in range(no_of_clusters):
                    for j in range(no_of_clusters - k - 1):
                        if np.sqrt((centres[k][0] - centres[j + k + 1][
                            0]) ** 2 + (centres[k][1] - centres[j + k + 1][
                            1]) ** 2) < 0.4 and (
                                centres[k][0] and centres[j + k + 1][
                            0]) != 100:
                            centres[k] = (100, 100)
                            cluster1 = []
                            cluster2 = []
                            cluster3 = []
                            cluster4 = []
                            cluster5 = []
                            cluster6 = []
                            cluster7 = []
                            cluster8 = []
                            cluster9 = []
                            cluster10 = []
                            loops -= 1
                            changed = True

            loops += 1
            total += 1
    return (np.array(clusters), np.array(centres))

# ###############################################
# ### FINITE SOURCE BINARY LENS (FSBL) MODELS ###
# ###############################################
#
# class FSBL(ABC):
#     """
#     INPUTS:
#     ###############################################################################
#     lens_mass1, lens_mass2: Masses of the lenses (Msun)
#     t0: Time of photometric peak, as seen from Earth (MJD.DDD)
#     xS0: vector [RA, Dec] Source position on sky at t = t0 (arcsec) in an
#     arbitrary ref. frame.
#     beta: Angular distance between the source and the geometric center of the lens
#         on the plane of the sky (mas). Can
#          positive (u0_amp > 0 when u0_hat[0] < 0) or 
#          negative (u0_amp < 0 when u0_hat[0] > 0).
#     muL: vector [RA, Dec] Lens system proper motion (mas/yr)
#     muS: vector [RA, Dec] Source proper motion (mas/yr)
#     dL: Distance from the observer to the lens system (pc)
#     dS: Distance from the observer to the source (pc)
#     Radius: Radius of star in solar radii
#     separation: Angular separation of the two lenses in units (mas)
#     angle: Angle between binary axis (m2 -> m1) and negative RA axis.
#     utilde: Limb darkening coeff, value chosen from [0, 1], 0 corresponds to
#     uniformly bright source
#     mag_src: Source magnitude in a single filter.
#     b_sff: The ratio of the source flux to the total (source + neighbors + lens)
#     b_sff = f_S / (f_S + f_L + f_N)
#     ###############################################################################
#     """
#
#     # CYL : Maybe we should put in the units using astropy.
#     """
#     """
#
#     def __init__(self, lens_mass1, lens_mass2, t0, xS0,
#                  beta, muL, muS, dL, dS, radius,
#                  separation, angle, utilde, mag_src, b_sff):
#         self.M = lens_mass1 + lens_mass2
#         self.m1 = lens_mass1 / self.M  # m1 is NOT lens mass 1!
#         self.t0 = t0
#         self.xS0 = np.array(xS0)
#         self.beta = beta
#         self.muL = np.array(muL)
#         self.muS = np.array(muS)
#         self.dL = dL
#         self.dS = dS
#         self.separation = separation
#         self.utilde = utilde
#         self.angle = angle / 180 * np.pi  # angle in radians
#         self.radius = (
#                                   radius * meter_per_Rsun / meter_per_AU) / dS  # Radius of star in arcsec
#         self.b_sff = b_sff
#         self.mag_src = mag_src
#
#         # Calculate the relative proper motion vector.
#         # Note that this will be in the direction of theta_hat
#         self.muRel = self.muS - self.muL  # mas/yr
#
#         # Calculate the Einstein radius (using the effective lens mass)
#         self.thetaEhat = get_unit_vector(
#             self.muRel)  # unit vector in direction of thetaE
#         self.thetaEamp = get_angular_einstein_radius(self.M, self.dL,
#                                                      self.dS)  # mas
#         self.thetaE = self.thetaEamp * self.thetaEhat  # vector version of the einstein radius
#
#         # Defined from the other quantities
#         self.u0 = get_u0(self.thetaEhat, self.beta,
#                          self.thetaEamp)  # closest approach vector
#
#         # Angular separation vector between source and lens (vector from lens to source)
#         self.thetaS0 = self.u0 * self.thetaEamp
#
#         # Position of the lens on sky at time t0
#         self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)
#
#         self.tE = get_einstein_time(self.thetaEamp, self.muRel,
#                                     days_per_year)  # Einstein crossing time
#
#         # Angular separation between lenses in units of thetaE
#         self.d = (self.separation / (self.thetaEamp * 1e-3))
#
#     def srce(self, x):
#         """
#         Lens equation, centered on the COM of the lens objects.
#         Equation (15) in Dominik 1999, but with m1 instead of q.
#         Note conversion between q and m1 given in Equation (10).
#         """
#         y1 = x[0]
#         y1 -= self.m1 * (x[0] - (1 - self.m1) * self.d) / (
#                     (x[0] - (1 - self.m1) * self.d) ** 2 + x[1] ** 2)
#         y1 -= (1 - self.m1) * (x[0] + self.m1 * self.d) / (
#                     (x[0] + self.m1 * self.d) ** 2 + x[1] ** 2)
#         y2 = x[1]
#         y2 -= self.m1 * (x[1]) / (
#                     (x[0] - (1 - self.m1) * self.d) ** 2 + x[1] ** 2)
#         y2 -= (1 - self.m1) * (x[1]) / (
#                     (x[0] + self.m1 * self.d) ** 2 + x[1] ** 2)
#
#         return np.array([y1, y2])
#
#     def get_lens_astrometry(self, t):
#         """
#         Given a list of times, this returns the positions of both
#         the lenses
#         """
#         t_yrs = (t - self.t0) / days_per_year
#         xl = self.xL0 + np.outer(t_yrs, self.muL) * 1e-3
#         return (xl, (xl + (1 - self.m1) * self.separation * np.array(
#             [-np.cos(self.angle), np.sin(self.angle)]),
#                      xl - self.m1 * self.separation * np.array(
#                          [-np.cos(self.angle), np.sin(self.angle)])))
#
#     def get_astrometry_unlensed(self, t):
#         """ Given a list of times, this returns the position of
#         the centre of the source
#         """
#         dt_in_years = (t - self.t0) / days_per_year
#         return self.xS0 + np.outer(dt_in_years, self.muS * 1e-3)
#
#     def get_caustic(self, t):
#         """ This functions finds the position of the caustics at a list of times """
#
#         n = (51, 9,
#              9)  # number of points in the grid for each step of the iteration (c.f paper)
#         precision = (0.7, 0.3,
#                      0.01)  # Max abs value of det to be considered a point on the caustic at each iteration
#
#         def source(x, m1, d):
#             """ The inverse lens equation """
#             y1 = x[0] - m1 * (x[0] - (1 - m1) * d) / (
#                         (x[0] - (1 - m1) * d) ** 2 + x[1] ** 2) - (1 - m1) * (
#                              x[0] + m1 * d) / (
#                              (x[0] + m1 * d) ** 2 + x[1] ** 2)
#             y2 = x[1] - m1 * (x[1]) / (
#                         (x[0] - (1 - m1) * d) ** 2 + x[1] ** 2) - (1 - m1) * (
#                  x[1]) / ((x[0] + m1 * d) ** 2 + x[1] ** 2)
#             return np.array([y1, y2])
#
#         def caust(x, m1, d):
#             """ This returns the value of the determinant given a point x """
#             J11 = 1 - m1 / ((x[0] - (1 - m1) * d) ** 2 + x[1] ** 2) + (
#                         2 * m1 * (x[0] - (1 - m1) * d) ** 2) / (((x[0] - (
#                         1 - m1) * d) ** 2 + x[1] ** 2) ** 2) - (1 - m1) / (
#                               (x[0] + m1 * d) ** 2 + x[1] ** 2) + (
#                               2 * (1 - m1) * (x[0] + m1 * d) ** 2) / (
#                               ((x[0] + m1 * d) ** 2 + x[1] ** 2) ** 2)
#             J22 = 1 - m1 / (
#                         (x[0] - (1 - m1) * d) ** 2 + x[1] ** 2) + 2 * m1 * (
#                   x[1]) ** 2 / (((x[0] - (1 - m1) * d) ** 2 + x[
#                 1] ** 2) ** 2) - (1 - m1) / (
#                               (x[0] + m1 * d) ** 2 + x[1] ** 2) + 2 * (
#                               1 - m1) * (x[1]) ** 2 / (
#                               ((x[0] + m1 * d) ** 2 + x[1] ** 2) ** 2)
#             J12 = 2 * m1 * x[1] * (x[0] - (1 - m1) * d) / (
#                         ((x[0] - (1 - m1) * d) ** 2 + x[1] ** 2) ** 2) + 2 * (
#                               1 - m1) * x[1] * (x[0] + m1 * d) / (
#                               ((x[0] + m1 * d) ** 2 + x[1] ** 2) ** 2)
#             J = [[J11, J12], [J12, J22]]
#             return np.linalg.det(J)
#
#         """ The first step of the iteration:
#                 1. A grid of size 51x51 is created in the image plane
#                 2. At each point on the grid the determinant is calculated
#                 3. If the determinant is less than 0.7 that point is saved
#         """
#         interesting = []
#         for i in range(n[0]):
#             for j in range(n[0]):
#                 x = (-2 + i / (n[0]) * 4, -2 + j / (n[0]) * 4)
#                 det = caust(x, self.m1, self.d)
#                 if det < precision[0] and det > -precision[0]:
#                     interesting.append(x)
#
#         """The next steps in the iteration
#                 1. A grid of size 9x9 is created in the image plane around each point that was saved in the previous step
#                 2. At each point the determinant is calculated
#                 3. If it has a determinant lower than the precision at this step, then it is saved.
#         """
#         image = []
#         factor = 4 / ((n[0] - 1))
#         for c in range(len(precision) - 1):
#             image = []
#             factor = factor / (n[c + 1] - 1)
#             for i in range(len(interesting)):
#                 for j in range(n[c + 1]):
#                     for k in range(n[c + 1]):
#                         x = (
#                         interesting[i][0] + (j - (n[c + 1] - 1) / 2) * factor,
#                         interesting[i][1] + (k - (n[c + 1] - 1) / 2) * factor)
#                         det = caust(x, self.m1, self.d)
#                         if det < precision[c + 1] and det > - precision[c + 1]:
#                             image.append(x)
#             interesting = image
#         image = []
#
#         for j in interesting:
#             # Each point is now converted from the image plane into the lens place
#             image.append(source(j, self.m1, self.d))
#
#         """Converts from angular einstein coordinates into arcseconds, including rotating in line with the
#             angle of the lens relative to the x-axis """
#         caustic = self.thetaEamp * 1e-3 * np.array(image)
#         lens = self.get_lens_astrometry(t)[0]
#         rot = np.matrix([[-np.cos(self.angle), np.sin(self.angle)],
#                          [np.sin(self.angle), np.cos(self.angle)]])
#         caustic = np.array(caustic.dot(rot))
#         caustics = []
#         for i in range(len(t)):
#             caustics.append(lens[i] + caustic)
#         return np.array(caustics)
#
#     def get_resolved_astrometry(self, t):
#         """ This function returns the position of the images for a list of times """
#
#         """This block sets up the parameters to be fed into the various functions:
#             1. rot is the rotation/reflection matrix to convert to and from the centre of mass frame
#                with the lenses along the x-axis, the frame in which the lens equation is calculated,
#                to right ascension and declination
#             2. COnverts the list of the times in the right units, and gets a list with the positions
#                of the centre of the source and the centre of mass of the lens at each point in time
#         """
#         rot = np.matrix([[-np.cos(self.angle), np.sin(self.angle)],
#                          [np.sin(self.angle), np.cos(self.angle)]])
#         t_yrs = (t - self.t0) / 365.25
#         source = self.get_astrometry_unlensed(t)
#         lens = self.get_lens_astrometry(t)[0]
#
#         images = []
#         for i in range(len(t_yrs)):
#             """This block finds the positions of the images for a given time
#                 1. Gets the position of the centre of the source in the centre of mass frame
#                    in angular einstein units.
#                 2. Finds the position of the images (c.f get_image())
#                 3. Converts it back into arcseconds
#             """
#             source_pos = ((source[i] - lens[i]) / (self.thetaEamp * 1e-3))
#             rot_source = np.array(source_pos.dot(rot))[0]
#             positions = get_image(rot_source, self.m1, self.d,
#                                   self.radius / (self.thetaEamp * 1e-3))
#             image = (self.thetaEamp * 1e-3 * positions)
#             image = np.array(image.dot(rot))
#             images.append(lens[i] + image)
#         return np.array(images)
#
#     def get_single_amplification(self, rot, R, args):
#         """This functions find the amplification of the source at a given point,
#             and is needed for multiprocessing to work
#                 rot = rotation matrix
#                 R = radius of the source in angular einstein units
#                 args = (source, lens):
#                     source = position of the centre of the source
#                     lens = position of the centre of mass of the lens
#         """
#
#         def F(y):
#             """ This function gives the intensity of a point on the star, relative to the maximum intensity """
#             r = np.sqrt(y[0] ** 2 + y[1] ** 2) / R
#             if r > 1:
#                 return 0
#             else:
#                 return 1 / (1 - self.utilde / 3) * (
#                             1 - self.utilde + self.utilde * np.sqrt(
#                         1 - r ** 2))
#
#         # Converts the source position into the right units/CoM frame, and
#         # creates a variable for storing the flux_weighted area.
#         source_pos = ((args[0] - args[1]) / (self.thetaEamp * 1e-3))
#         rot_source = np.array(source_pos.dot(rot))[0]
#         flux_weighted_area = 0
#
#         """ Calculates the amplification:
#             1. Finds the images (c.f get_image())
#             2. Splits the list of image points up into several clusters (c.f cluster())
#             3. Creates a grid and integrates over each cluster (c.f twod_int()) to find the
#                flux weighted area
#             4. Returns the amplification by dividing by pi R^2
#         """
#         positions = get_image(rot_source, self.m1, self.d, R)
#         points = cluster(positions, R)
#         clusters = points[0]
#         centres = points[1]
#         for j in range(len(clusters)):
#             if len(clusters[j]) != 0:
#                 middle = centres[j]
#                 x = clusters[j][:, 0]
#                 xmin = min(x)
#                 xmax = max(x)
#                 y = clusters[j][:, 1]
#                 ymin = min(y)
#                 ymax = max(y)
#                 flux_weighted_area += twod_int(rot_source, F, self.srce, xmax,
#                                                xmin, ymax, ymin, 50, 50,
#                                                middle, centres)
#         return flux_weighted_area / (np.pi * R ** 2)
#
#     def get_amplification(self, t):
#         """ Given a list of times, returns the amplifications at each point in time """
#
#         """ Sets up a bunch of needed parameters
#                 R = radius of the source in angular einstein units
#                 invrot and rot = the rotations matrices to convert to and from the frame
#                     in which the lens equation was calculated
#                 t_yrs = the times in the right units
#                 source, lens = position of the source, lens at the list of times
#                 arguments = an array of arguments that will be passed into get_single_amplification
#         """
#         R = self.radius / (self.thetaEamp * 1e-3)
#         rot = np.matrix([[-np.cos(self.angle), np.sin(self.angle)],
#                          [np.sin(self.angle), np.cos(self.angle)]])
#         t_yrs = (t - self.t0) / 365.25
#         source = self.get_astrometry_unlensed(t)
#         lens = self.get_lens_astrometry(t)[0]
#         arguments = []
#
#         for i in range(len(t_yrs)):
#             arguments.append((source[i], lens[i]))
#
#         """ Calculates the amplifications
#                 1. Partial takes a function and creates a new function with fewer parameters:
#                         So this takes get_single_amplification and creates a new function in which rot, and R
#                         are set and arguments is the only parameter that needs to be entered
#                 2. Pool creates 4 separate processes, i.e my laptop has 4 cpu cores, so it will calculate 4 amplifications
#                         at the same time, 1 in each core. To check how many cores you have, call cpu_count()
#                 3. p.map will pass the list of arguments into the function f and put the results in the variable result,
#                         calculating 4 points at a time
#         """
#         f = partial(self.get_single_amplification, rot, R)
#         p = Pool(processes=4)
#         result = p.map(f, arguments)
#         return np.array(result)
#
#     def get_photometry(self, t_obs):
#         mag_zp = 30.0  # arbitrary but allows for negative blend fractions.
#         flux_zp = 1.0
#
#         points = self.get_animation_points(t_obs)
#
#         amplifications = points[0]
#         finitesource = points[1]
#         images = points[2]
#         centroids = points[3]
#
#         flux_src = flux_zp * 10 ** ((self.mag_src - mag_zp) / -2.5)
#         flux_model = flux_src * amplifications
#         # CYL : FIXME-- check whether this is right.
#         # I THINK you just sum up all the amplifications b/c surface brightness is conserved.
#         # ALSO: Do we need a way to distinguish the possiblities that the different lenses
#         # can have different fluxes?
#
#         # Account for blending, if necessary.
#         try:
#             # Adding flux of neighbors and lenses
#             # b_sff = fS / (fS + fN + fL)
#             flux_model += flux_src * (1.0 - self.b_sff) / self.b_sff
#         except AttributeError:
#             pass
#
#         # Catch the edge case where we exceed the zeropoint.
#         bad = np.where(flux_model <= 0)[0]
#         if len(bad) > 0:
#             print('Warning: get_photometry: bad flux encountered.')
#             flux_model[bad] = np.nan
#
#         mag_model = -2.5 * np.log10(flux_model / flux_zp) + mag_zp
#
#         return mag_model
#
#     # CYL : FIXME-- I HAVE NO IDEA IF THIS WILL WORK
#     def get_astrometry(self, t_obs):
#         '''
#         Position of the observed source position in (arcsec???)
#         '''
#         points = self.get_animation_points(t_obs)
#
#         amplifications = points[0]
#         finitesource = points[1]
#         images = points[2]
#         centroids = points[3]
#
#         srce_pos_model = self.xS0 + np.outer((t_obs - self.t0) / days_per_year,
#                                              self.muS) * 1e-3
#         pos_model = srce_pos_model + centroids
#
#         return pos_model
#
#     def log_likely_photometry(self, t_obs, mag_obs, mag_err_obs):
#         mag_model = self.get_photometry(t_obs)
#
#         lnL_term1 = -0.5 * ((mag_obs - mag_model) / mag_err_obs) ** 2
#         lnL_term2 = -0.5 * np.log(2.0 * math.pi * mag_err_obs ** 2)
#         lnL = lnL_term1 + lnL_term2
#
#         return lnL
#
#     def log_likely_astrometry(self, t_obs, x_obs, y_obs, x_err_obs, y_err_obs):
#         pos_model = self.get_astrometry(t_obs)
#
#         lnL_x_t1 = -0.5 * ((x_obs - pos_model[:, 0]) / x_err_obs) ** 2
#         lnL_x_t2 = -0.5 * np.log(2.0 * math.pi * x_err_obs ** 2)
#         lnL_y_t1 = -0.5 * ((y_obs - pos_model[:, 1]) / y_err_obs) ** 2
#         lnL_y_t2 = -0.5 * np.log(2.0 * math.pi * y_err_obs ** 2)
#
#         lnL = lnL_x_t1 + lnL_x_t2 + lnL_y_t1 + lnL_y_t2
#
#         return lnL
#
#     def get_animation_point(self, rot, R, args):
#         """This functions find the amplification, position of images, and centroid at
#         a given point and is needed for multiprocessing to work
#             rot = rotation matrix
#             invrot = inverse rotation matric
#             R = radius of the source in angular einstein units
#             args = (source, lens):
#                 source = position of the centre of the source
#                 lens = position of the centre of mass of the lens
#         """
#
#         def F(y):
#             """ This function gives the intensity of a point on the star, relative to the maximum intensity """
#             r = np.sqrt(y[0] ** 2 + y[1] ** 2) / R
#             if r > 1:
#                 return 0
#             else:
#                 return 1 / (1 - self.utilde / 3) * (
#                             1 - self.utilde + self.utilde * np.sqrt(
#                         1 - r ** 2))
#
#         """
#         Converts the source position into the right units/CoM frame, and
#         creates a variable for storing the flux_weighted area, and the flux weighted area
#         multiplied by x and y to calculate the position of the centre of the light
#         """
#         source_pos = ((args[0] - args[1]) / (self.thetaEamp * 1e-3))
#         rot_source = np.array(source_pos.dot(rot))[0]
#         area = 0
#         x_int = 0
#         y_int = 0
#
#         """ Calculates the relevant quantities:
#             1. Finds the images (c.f get_image())
#             2. Splits the list of image points up into several clusters (c.f cluster())
#             3. Creates a grid and integrates over each cluster (c.f twod_int()) to find the
#                flux weighted area, as well as the centroid integrals
#             4. Returns the amplification by dividing by pi R^2, the images, and the centroid position
#         """
#         positions = get_image(rot_source, self.m1, self.d, R)
#         image = (self.thetaEamp * 1e-3 * positions)
#         image = np.array(image.dot(rot))
#         image = args[1] + image
#         points = cluster(positions, R)
#         clusters = points[0]
#         centres = points[1]
#         for j in range(len(clusters)):
#             if len(clusters[j]) != 0:
#                 middle = centres[j]
#                 x = clusters[j][:, 0]
#                 xmin = min(x)
#                 xmax = max(x)
#                 y = clusters[j][:, 1]
#                 ymin = min(y)
#                 ymax = max(y)
#                 area += twod_int(rot_source, F, self.srce, xmax, xmin, ymax,
#                                  ymin, 50, 50, middle, centres)
#                 x_int += twod_cent_x_int(rot_source, F, self.srce, xmax, xmin,
#                                          ymax, ymin, 50, 50, middle, centres)
#                 y_int += twod_cent_y_int(rot_source, F, self.srce, xmax, xmin,
#                                          ymax, ymin, 50, 50, middle, centres)
#         centroid = np.array([x_int / area, y_int / area])
#         centroid = np.array(centroid.dot(rot))[0]
#         centroid = args[1] + centroid * self.thetaEamp * 1e-3
#
#         return (area / (np.pi * R ** 2), image, centroid)
#
#     def get_animation_points(self, t):
#         """ Calculates the amplifications, image positions, and centroid positions"""
#         """ Sets up a bunch of needed parameters
#                 R = radius of the source in angular einstein units
#                 invrot and rot = the rotations matrices to convert to and from the frame
#                     in which the lens equation was calculated
#                 t_yrs = the times in the right units
#                 source, lens = position of the source, lens at the list of times
#                 arguments = an array of arguments that will be passed into get_single_amplification
#                 amplifications, images, centroids = arrays to store the results
#         """
#         R = self.radius / (self.thetaEamp * 1e-3)
#         rot = np.matrix([[-np.cos(self.angle), np.sin(self.angle)],
#                          [np.sin(self.angle), np.cos(self.angle)]])
#         t_yrs = (t - self.t0) / 365.25
#         source = self.get_astrometry_unlensed(t)
#         lens = self.get_lens_astrometry(t)[0]
#         amplifications = []
#         images = []
#         centroids = []
#         arguments = []
#
#         for i in range(len(t_yrs)):
#             arguments.append((source[i], lens[i]))
#
#         """ Calculates the relevant quantities:
#                 1. Partial takes a function and creates a new function with fewer parameters:
#                         So this takes get_single_amplification and creates a new function in which rot, invrot, and R
#                         are set and arguments is the only parameter that needs to be entered
#                 2. Pool creates 4 separate processes, i.e my laptop has 4 cpu cores, so it will calculate 4 amplifications
#                         at the same time, 1 in each core. To check how many cores you have, call cpu_count()
#                 3. p.map will pass the list of arguments into the function f and put the results in the variable result,
#                         calculating 4 points at a time
#         """
#         f = partial(self.get_animation_point, rot, R)
#         p = Pool(processes=4)
#         result = p.map(f, arguments)
#
#         for i in range(len(t_yrs)):
#             amplifications.append(result[i][0])
#             images.append(result[i][1])
#             centroids.append(result[i][2])
#         return (
#         np.array(amplifications), np.array(images), np.array(centroids))
#
#     def animate(self, crossings, time_steps, frame_time, name, size, zoom):
#         """Creates an mp4 animation of the event:
#                 crossings = The number of einstein crossing times to animate
#                 time_steps = The number of time_steps either side of the peak
#                 frame_time = The time in ms for each frame
#                 name = the save name of the file
#                 size = [horizontal, vertical] cm's
#                 zoom = # of einstein radii plotted in vertical direction
#         """
#         times = np.array(range(-time_steps, time_steps + 1, 1))
#         tau = crossings * times / (-times[0])
#         t = (tau * self.tE) + self.t0
#         points = self.get_animation_points(t)
#         caustic = self.get_caustic(t)
#         rl = self.get_lens_astrometry(t)[1]
#         rs = self.get_astrometry_unlensed(t)
#         lens1 = rl[0]
#         lens2 = rl[1]
#         image = points[1]
#         A = points[0]
#         centroids = points[2]
#         xcentroids = centroids[:, 0]
#         ycentroids = centroids[:, 1]
#         lens = self.get_lens_astrometry(t)[0]
#         fig = plt.figure(figsize=[size[0], size[1]])
#         matplotlib.rc('xtick', labelsize=25)
#         matplotlib.rc('ytick', labelsize=25)
#         ax1 = fig.add_subplot(2, 1, 1)
#         ax2 = fig.add_subplot(2, 1, 2)
#         patch = plt.Circle(rs[0], self.radius, color='green')
#         line1, = ax1.plot([], 'b.', markersize=20, label="Lens")
#         line2, = ax1.plot([], 'g.', markersize=1, label="Source")
#         line3, = ax1.plot([], 'r.', markersize=5, label="Image")
#         line4, = ax1.plot([], 'y.', markersize=5, label="Caustic")
#         line5, = ax1.plot([], 'b.', markersize=20)
#         line6, = ax1.plot(xcentroids, ycentroids, 'm', linewidth=3,
#                           label="Image Centroid")
#         line7, = ax2.plot(t, A, linewidth=4)
#         ax1.set_xlabel("RA (arcsec)", fontsize=40)
#         ax1.set_ylabel("Dec (arcsec)", fontsize=40)
#         ax1.add_patch(patch)
#         ax1.set_xlim(
#             (lens[0][0] + lens[-1][0]) / 2 - zoom * self.thetaEamp * 1e-3 * 2 *
#             size[0] / size[1],
#             (lens[0][0] + lens[-1][0]) / 2 + zoom * self.thetaEamp * 1e-3 * 2 *
#             size[0] / size[1])
#         ax1.set_ylim(
#             (lens[0][1] + lens[-1][1]) / 2 - zoom * self.thetaEamp * 1e-3,
#             (lens[0][1] + lens[-1][1]) / 2 + zoom * self.thetaEamp * 1e-3)
#         ax1.legend(fontsize=25, markerscale=3)
#         ax1.invert_xaxis()
#         ax2.set_xlabel("Time(days)", fontsize=40)
#         ax2.set_ylabel("Amplification", fontsize=40)
#         line = [line1, line2, line3, line4, line5, line6, line7]
#
#         # this function is called at every frame, with i being the number of the frame that it's currently on
#         def update(i, rs, lens1, line, image, caustic, lens2, tau,
#                    magnification, xcents, ycents):
#             line[0].set_data(lens1[i, 0], lens1[i, 1])
#             line[1].set_data(rs[i][0], rs[i][1])
#             line[2].set_data(image[i][:, 0], image[i][:, 1])
#             line[3].set_data(caustic[i][:, 0], caustic[i][:, 1])
#             line[4].set_data(lens2[i, 0], lens2[i, 1])
#             line[5].set_data(xcents[:i], ycents[:i])
#             line[6].set_data(tau[:i], magnification[:i])
#             patch.center = rs[i]
#             return line
#             """
#             FuncAnimation takes in arguments:
#             fig = background figure
#             update = function that is called every frame
#             len(tau) = the number of frames, so now the first argument passed into update (i) will be (0,1,2...len(tau))
#             fargs specifies the other arguments to pass into update
#             blit being true means that each frame, if there are elements of it that don't change from the last frame,
#             it won't replot them, so this makes it faster
#             interval = number of milliseconds between each frame
#             alternatively you can specify fps in save after the file name
#             """
#
#         ani = animation.FuncAnimation(fig, update, len(tau),
#                                       fargs=[rs, lens1, line, image, caustic,
#                                              lens2, t, A, xcentroids,
#                                              ycentroids], blit=True,
#                                       interval=frame_time)
#         ani.save("%s.mp4" % name, writer="ffmpeg")
#         return
#
#
# class FSBL_parallax(FSBL):
#     """
#     DESCRIPTION:
#     Finite Source (Static) Binary Lens model for microlensing. This model includes
#     proper motions of both the lens and source AND parallax (both the
#     microlensing parallax effects on the photometry and astrometry.
#     """
#
#     def __init__(self, raL, decL, lens_mass1, lens_mass2, t0, xS0,
#                  beta, muL, muS, dL, dS, radius,
#                  separation, angle, utilde, mag_src, b_sff):
#         """
#         INPUTS:
#         ###############################################################################
#         raL: Right ascension of the lens in decimal degrees.
#         decL: Declination of the lens in decimal degrees.
#         Rest same as FSBL.
#         ###############################################################################
#         """
#         self.raL = raL
#         self.decL = decL
#         self.piS = 1.0 / dS
#         self.piL = 1.0 / dL
#         super(FSBL_parallax, self).__init__(lens_mass1, lens_mass2, t0, xS0,
#                                             beta, muL, muS, dL, dS, radius,
#                                             separation, angle, utilde, mag_src,
#                                             b_sff)
#
#         # parallax0 = parallax_in_direction(self.raL, self.decL, np.array([self.t0]))
#         # self.xL0 = self.xL0 - (self.piL - self.piS) * parallax0
#         # This function needs to be fixed
#         # self.calc_piE_ecliptic()
#
#         return
#
#     def get_centroid_shift(self, t):
#         """
#         Needs to be changed from the PSPL_parallax
#         """
#         tau = (t - self.t0) / self.tE
#
#         # Lens-induced astrometric shift of the sum of all source images (in mas)
#         numer = (np.outer(tau, self.thetaE_hat) + self.u0) * self.thetaE_amp
#         denom = (tau ** 2.0 + self.u0_amp ** 2.0 + 2.0).reshape(numer.shape[0],
#                                                                 1)
#         shift = numer / denom
#
#         return shift
#
#     def get_astrometry_unlensed(self, t):
#         """ Given a list of times, this returns the position of the centre of the source """
#         parallax_vec = parallax_in_direction(self.raL, self.decL, t)
#         dt_in_years = (t - self.t0) / days_per_year
#
#         xS = self.xS0 + np.outer(dt_in_years, self.muS * 1e-3) + (
#                     self.piS * parallax_vec)
#
#         return xS
#
#     def get_lens_astrometry(self, t):
#         """
#         Given a list of times, this returns the positions of both the lenses
#         """
#         parallax_vec = parallax_in_direction(self.raL, self.decL, t)
#         t_yrs = (t - self.t0) / days_per_year
#
#         xL_sys = self.xL0 + np.outer(t_yrs, self.muL * 1e-3) + (
#                     self.piL * parallax_vec)
#
#         cosa = np.cos(self.angle)
#         sina = np.sin(self.angle)
#
#         xL1 = xL_sys + (1 - self.m1) * self.separation * np.array(
#             [-cosa, sina])
#         xL2 = xL_sys - self.m1 * self.separation * np.array([-cosa, sina])
#         return (xL_sys, (xL1, xL2))
#
#     def get_astrometry(self, t_obs):
#         """
#         Must be changed form PSPL_Parallax
#         """
#         # Things we will need.
#         dt_in_years = (t_obs - self.t0) / days_per_year
#
#         # Get the parallax vector for each date.
#         parallax_vec = parallax_in_direction(self.raL, self.decL, t_obs)
#
#         # Equation of motion for just the background source.
#         xS_unlensed = self.xS0 + np.outer(dt_in_years, self.muS) * 1e-3
#         xS_unlensed += (self.piS * parallax_vec) * 1e-3  # arcsec
#
#         # Equation of motion for the relative angular separation between the background source and lens.
#         thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel)  # mas
#         thetaS -= (self.piRel * parallax_vec)  # mas
#         u_vec = thetaS / self.thetaE_amp
#         u_amp = np.linalg.norm(u_vec, axis=1)
#
#         denom = u_amp ** 2 + 2.0
#
#         shift = thetaS / denom.reshape((len(u_amp), 1))  # mas
#
#         xS = xS_unlensed + (shift * 1e-3)  # arcsec
#
#         return xS
#
#     def get_resolved_amplification(self, t):
#         """
#         Must be changed from PSPL_Parallax
#         """
#         """Get the photometric amplification term at a set of times, t for both the
#         plus and minus images.
#
#         Inputs
#         ----------
#         t: Array of times in MJD.DDD
#         """
#         # Get the parallax vector for each date.
#         parallax_vec = parallax_in_direction(self.raL, self.decL, t)
#
#         # Equation of relative motion (angular on sky) Eq. 16 from Hog+ 1995
#         dt_in_years = (t - self.t0) / days_per_year
#         thetaS = self.thetaS0 + np.outer(dt_in_years, self.muRel) - (
#                     self.piRel * parallax_vec)  # mas
#         u = thetaS / self.thetaE_amp
#         u_amp = np.linalg.norm(u, axis=1)
#
#         A_plus = 0.5 * (
#                     (u_amp ** 2 + 2) / (u_amp * np.sqrt(u_amp ** 2 + 4)) + 1)
#         A_minus = 0.5 * (
#                     (u_amp ** 2 + 2) / (u_amp * np.sqrt(u_amp ** 2 + 4)) - 1)
#
#         return (A_plus, A_minus)
#
#     def calc_piE_ecliptic(self):
#         """
#         Must be changed from PSPL_parallax
#         """
#         # Project the microlensing parallax into parallel and perpendicular
#         # w.r.t. the ecliptic... useful quantities.
#         parallax_vec_at_t0 = \
#         parallax_in_direction(self.raL, self.decL, np.array([self.t0]))[0]
#
#         # Unit vector parallel to the ecliptic
#         par_hat = parallax_vec_at_t0 / np.linalg.norm(parallax_vec_at_t0)
#
#         # Unit vector perpendicular to the ecliptic
#         # Cross product with -z vector (where z increases along the line of sight) in order
#         # to define the perpindicular to the ecliptic. Ideally, this would point to the North
#         # Galactic pole; but I am not sure if it does.
#         perp_hat = np.cross(np.append(par_hat, [0]), np.array([0, 0, -1]))[0:2]
#
#         # Project piE_EN onto piE_parallel_perpendicular
#         proj_piE_eclipt = np.dot(self.piE, par_hat)
#         proj_piE_eclipt_ortho = np.dot(self.piE, perp_hat)
#
#         # Save into a new vector
#         self.piE_eclipt = np.array([proj_piE_eclipt, proj_piE_eclipt_ortho])
#
#         return
#
#
#
#
# class FSBL_orbital_parallax(FSBL_parallax):
#     def __init__(self, a, e, i, Omega, arg_peri, P, t_peri, alpha, raL, decL,
#                  lens_mass1, lens_mass2, t0, xS0,
#                  beta, muL, muS, dL, dS, radius,
#                  utilde, mag_src, b_sff):
#         """
#         INPUTS:
#         ###############################################################################
#         a = semi-major axis (AU)
#         e = eccentricity
#         i = inclination (deg)
#         Omega = ascending node (deg)
#         arg_peri = argument of periapsis (deg)
#         P = Orbital period (yrs)
#         t_peri = time of periapsis (MJD)
#         alpha = angle between lens-source trajectory and x-axis
#         Rest same as FSBL_parallax.
#         ###############################################################################
#         """
#         separation = 0
#         angle = 0
#         super(FSBL_orbital_parallax, self).__init__(raL, decL, lens_mass1,
#                                                     lens_mass2, t0, xS0,
#                                                     beta, muL, muS, dL, dS,
#                                                     radius, separation, angle,
#                                                     utilde, mag_src, b_sff)
#
#         self.a = a / dL  # Converts a from AU to arcsec
#         self.e = e
#         self.i = i / 180 * np.pi  # Converts the Euler angles into radians
#         self.Omega = Omega / 180 * np.pi
#         self.arg_peri = arg_peri / 180 * np.pi
#         self.P = P * 365.25
#         self.t_peri = t_peri
#         self.alpha = alpha
#         return
#
#     def get_lens_astrometry(self, t):
#         """Gets the positions of the lenses at various points in times"""
#
#         def get_orbit(a, e, tperi, P, m1, m2, CoMs, times, i, O, w, alpha, v):
#             # Function WIP, need to add the angle parameters.
#             def get_CoM_orbit(a, e, tperi, P, times, i, O, w):
#                 """ This function takes in kepler coordinates and a list of times:
#                         a = semi-major axis
#                         e = eccentricity
#                         t0 = time of closest approach
#                         P = orbital period
#                         times = numpy array of times
#                         i = inclination
#                         O = Ascending node
#                         w = argument of periapsis
#                     """
#
#                 def get_kep_orbit(a, e, zi, i, O, w):
#                     """ Given kepler times gives orbit in CoM frame """
#                     xzi = a * (np.cos(zi) - e)
#                     yzi = a * np.sqrt(1 - e ** 2) * np.sin(zi)
#
#                     xproj = - xzi * (
#                                 np.cos(O) * np.cos(w) - np.sin(O) * np.cos(
#                             i) * np.sin(w)) + yzi * (
#                                         np.cos(O) * np.sin(w) + np.sin(
#                                     O) * np.cos(i) * np.cos(w))
#                     yproj = xzi * (np.sin(O) * np.cos(w) + np.cos(O) * np.cos(
#                         i) * np.sin(w)) - yzi * (
#                                         np.sin(O) * np.sin(w) - np.cos(
#                                     O) * np.cos(i) * np.cos(w))
#
#                     if v[0] > 0:
#                         phi = alpha - (np.pi + np.arctan(v[1] / v[0]))
#                     else:
#                         phi = alpha - (2 * np.pi + np.arctan(v[1] / v[0]))
#
#                     xsky = xproj * np.cos(phi) + yproj * np.sin(phi)
#                     ysky = -xproj * np.sin(phi) + yproj * np.cos(phi)
#
#                     return (xsky, ysky)
#
#                 def get_zi(a, e, t, tperi):
#                     """Uses Newton-raphson method to convert time into kepler time"""
#                     zi = 0
#
#                     def f(x):
#                         return 2 * np.pi * ((t - tperi) / P - math.floor(
#                             (t - tperi) / P)) - (x - e * np.sin(x))
#
#                     def f1(x):
#                         return - (1 - e * np.cos(x))
#
#                     for i in range(5):
#                         zi = zi - f(zi) / f1(zi)
#                     return zi
#
#                 """Creates the list kepler times"""
#                 zis = []
#                 for j in times:
#                     zis.append(get_zi(a, e, j, tperi))
#                 zis = np.array(zis)
#
#                 return get_kep_orbit(a, e, zis, i, O, w)
#
#             positions = get_CoM_orbit(a, e, tperi, P, times, i, O, w)
#             x1 = CoMs[:, 0] + m2 / (m1 + m2) * positions[0]
#             y1 = CoMs[:, 1] + m2 / (m1 + m2) * positions[1]
#             x2 = CoMs[:, 0] - m1 / (m1 + m2) * positions[0]
#             y2 = CoMs[:, 1] - m1 / (m1 + m2) * positions[1]
#
#             return (x1, y1, x2, y2)
#
#         parallax_vec = parallax_in_direction(self.raL, self.decL, t)
#         t_yrs = (t - self.t0) / 365.25
#         xl = self.xL0 + np.outer(t_yrs, self.muL) * 1e-3
#         xl += self.piL * parallax_vec
#         return get_orbit(self.a, self.e, self.t_peri, self.P, self.m1,
#                          1 - self.m1, xl, t, self.i, self.Omega, self.arg_peri,
#                          self.alpha, self.muL - self.muS)
#
#     def get_image(self, y0, m1, d, R):
#         """ Function to find the images of the star given the input parameters:
#                 y0 = position of the cente of the source star, in units of anguler Einstein radius
#                 m1 = Mass of rightmost lens divided by the total mass
#                 d = separation of the lenses in angular Einstein radii
#                 R = angular radius of the source in angular Einstein radii
#         """
#         # print("y0 = (%f, %f), m1 = %f, d = %f\n" %(y0[0],y0[1],m1,d))
#
#         """ These 2 arrays give make up of the contour grid at each step of the iteration
#                 n = the number of grid points (n x n) centred on each previously saved point at a given step of the iteration
#                 precision = the radius of star whos images may be found with the resolution of grid at this step
#         """
#         n = (76, 5, 5, 5, 5, 5, 5, 5, 5, 5)
#         precision = (
#         0.9, 0.4, 0.04, 0.009, 0.0015, 0.0004, 0.00008, 0.00005, 0.00001)
#
#         def source(x, m1, d):
#             # Given an image point, this function tells you where the source is
#             y1 = x[0] - m1 * (x[0] - (1 - m1) * d) / (
#                         (x[0] - (1 - m1) * d) ** 2 + x[1] ** 2) - (1 - m1) * (
#                              x[0] + m1 * d) / (
#                              (x[0] + m1 * d) ** 2 + x[1] ** 2)
#             y2 = x[1] - m1 * (x[1]) / (
#                         (x[0] - (1 - m1) * d) ** 2 + x[1] ** 2) - (1 - m1) * (
#                  x[1]) / ((x[0] + m1 * d) ** 2 + x[1] ** 2)
#             return np.array([y1, y2])
#
#         """ This block creates an n[0] x n[0] grid from -3 to 3 in both coordinates, and saves all the image points that correspond to
#         source locations closer than precision[0] to the centre of the source
#         """
#         interesting_points = []
#         for i in range(n[0]):
#             for j in range(n[0]):
#                 x = (-3 + i / (n[0] - 1) * 6, -3 + j / (n[0] - 1) * 6)
#                 y = source(x, m1, d)
#                 s2 = (y0[0] - y[0]) ** 2 + (y0[1] - y[1]) ** 2
#                 if np.sqrt(s2) < precision[0]:
#                     interesting_points.append(x)
#         # Note that after this step, the distance between points on the contour grid is 6/(n[0]-1)
#         if 1.2 * R > precision[0]:
#             return np.array(interesting_points)
#
#         resolution = 6 / (n[0] - 1)
#         for i in range(len(precision) - 1):
#             image = []
#             resolution = resolution * 1 / (n[i + 1])
#             if 1.2 * R > precision[i + 1]:
#                 for j in interesting_points:
#                     for k in range(n[i + 1]):
#                         for l in range(n[i + 1]):
#                             x = (j[0] + (k - (n[i + 1] - 1) / 2) * resolution,
#                                  j[1] + (l - (n[i + 1] - 1) / 2) * resolution)
#                             y = source(x, m1, d)
#                             s2 = (y0[0] - y[0]) ** 2 + (y0[1] - y[1]) ** 2
#                             if np.sqrt(s2) < 1.1 * R:
#                                 image.append(x)
#                 return np.array(image)
#             else:
#                 for j in interesting_points:
#                     for k in range(n[i + 1]):
#                         for l in range(n[i + 1]):
#                             x = (j[0] + (k - (n[i + 1] - 1) / 2) * resolution,
#                                  j[1] + (l - (n[i + 1] - 1) / 2) * resolution)
#                             y = source(x, m1, d)
#                             s2 = (y0[0] - y[0]) ** 2 + (y0[1] - y[1]) ** 2
#                             if np.sqrt(s2) < precision[i + 1]:
#                                 image.append(x)
#                 interesting_points = image
#         return np.array(image)
#
#     def get_animation_point(self, R, args):
#         """This functions find the amplification, position of images, and centroid at
#         a given point and is needed for multiprocessing to work
#             rot = rotation matrix
#             invrot = inverse rotation matric
#             R = radius of the source in angular einstein units
#             args = (source, lens, d, angle):
#                 source = position of the centre of the source
#                 lens = position of the centre of mass of the lens
#                 d = normalised saparation
#                 angle = angle in radians
#         """
#
#         def srce(x):
#             """
#             Lens equation, centered on the COM of the lens objects.
#             Equation (15) in Dominik 1999, but with m1 instead of q.
#             Note conversion between q and m1 given in Equation (10).
#             """
#             y1 = x[0] - self.m1 * (x[0] - (1 - self.m1) * args[2]) / (
#                         (x[0] - (1 - self.m1) * args[2]) ** 2 + x[1] ** 2) - (
#                              1 - self.m1) * (x[0] + self.m1 * args[2]) / (
#                              (x[0] + self.m1 * args[2]) ** 2 + x[1] ** 2)
#             y2 = x[1] - self.m1 * (x[1]) / (
#                         (x[0] - (1 - self.m1) * args[2]) ** 2 + x[1] ** 2) - (
#                              1 - self.m1) * (x[1]) / (
#                              (x[0] + self.m1 * args[2]) ** 2 + x[1] ** 2)
#             return np.array([y1, y2])
#
#         def F(y):
#             """ This function gives the intensity of a point on the star, relative to the maximum intensity """
#             r = np.sqrt(y[0] ** 2 + y[1] ** 2) / R
#             if r > 1:
#                 return 0
#             else:
#                 return 1 / (1 - self.utilde / 3) * (
#                             1 - self.utilde + self.utilde * np.sqrt(
#                         1 - r ** 2))
#
#         """
#         Converts the source position into the right units/CoM frame, and
#         creates a variable for storing the flux_weighted area, and the flux weighted area
#         multiplied by x and y to calculate the position of the centre of the light
#         """
#         rot = np.matrix([[-np.cos(args[3]), np.sin(args[3])],
#                          [np.sin(args[3]), np.cos(args[3])]])
#         source_pos = ((args[0] - args[1]) / (self.thetaEamp * 1e-3))
#         source_pos = source_pos
#         rot_source = np.array(source_pos.dot(rot))[0]
#         area = 0
#         x_int = 0
#         y_int = 0
#
#         """ Calculates the relevant quantities:
#             1. Finds the images (c.f get_image())
#             2. Splits the list of image points up into several clusters (c.f cluster())
#             3. Creates a grid and integrates over each cluster (c.f twod_int()) to find the
#                flux weighted area, as well as the centroid integrals
#             4. Returns the amplification by dividing by pi R^2, the images, and the centroid position
#         """
#         positions = self.get_image(rot_source, self.m1, args[2], R)
#         points = cluster(positions, R)
#         image = (self.thetaEamp * 1e-3 * positions)
#         image = np.array(image.dot(rot))
#         image = args[1] + image
#         clusters = points[0]
#         centres = points[1]
#         for j in range(len(clusters)):
#             if len(clusters[j]) != 0:
#                 middle = centres[j]
#                 x = clusters[j][:, 0]
#                 xmin = min(x)
#                 xmax = max(x)
#                 y = clusters[j][:, 1]
#                 ymin = min(y)
#                 ymax = max(y)
#                 area += twod_int(rot_source, F, srce, xmax, xmin, ymax, ymin,
#                                  50, 50, middle, centres)
#                 x_int += twod_cent_x_int(rot_source, F, srce, xmax, xmin, ymax,
#                                          ymin, 50, 50, middle, centres)
#                 y_int += twod_cent_y_int(rot_source, F, srce, xmax, xmin, ymax,
#                                          ymin, 50, 50, middle, centres)
#         centroid = np.array([x_int / area, y_int / area])
#         centroid = np.array(centroid.dot(rot))[0]
#         centroid = args[1] + centroid * self.thetaEamp * 1e-3
#
#         return (area / (np.pi * R ** 2), image, centroid)
#
#     def get_animation_points(self, t):
#         """ Calculates the amplifications, image positions, and centroid positions"""
#         """ Sets up a bunch of needed parameters
#                 R = radius of the source in angular einstein units
#                 invrot and rot = the rotations matrices to convert to and from the frame
#                     in which the lens equation was calculated
#                 t_yrs = the times in the right units
#                 source, lens = position of the source, lens at the list of times
#                 arguments = an array of arguments that will be passed into get_single_amplification
#                 amplifications, images, centroids = arrays to store the results
#         """
#         R = self.radius / (self.thetaEamp * 1e-3)
#         t_yrs = (t - self.t0) / 365.25
#         source = self.get_astrometry_unlensed(t)
#         lens = self.get_lens_astrometry(t)
#         parallax_vec = parallax_in_direction(self.raL, self.decL, t)
#         xl = self.xL0 + np.outer(t_yrs, self.muL) * 1e-3
#         xl += self.piL * parallax_vec
#         amplifications = []
#         images = []
#         centroids = []
#         arguments = []
#
#         for i in range(len(t_yrs)):
#             d = np.sqrt((lens[3][i] - lens[1][i]) ** 2 + (
#                         lens[2][i] - lens[0][i]) ** 2) / (
#                             self.thetaEamp * 1e-3)
#
#             if lens[0][i] - lens[2][i] < 0:
#                 angle = np.arctan(
#                     (lens[1][i] - lens[3][i]) / (lens[2][i] - lens[0][i]))
#             else:
#                 angle = np.pi + np.arctan(
#                     (lens[1][i] - lens[3][i]) / (lens[2][i] - lens[0][i]))
#             arguments.append((source[i], xl[i], d, angle))
#
#         """ Calculates the relevant quantities:
#                 1. Partial takes a function and creates a new function with fewer parameters:
#                         So this takes get_single_amplification and creates a new function in which rot, invrot, and R
#                         are set and arguments is the only parameter that needs to be entered
#                 2. Pool creates 4 separate processes, i.e my laptop has 4 cpu cores, so it will calculate 4 amplifications
#                         at the same time, 1 in each core. To check how many cores you have, call cpu_count()
#                 3. p.map will pass the list of arguments into the function f and put the results in the variable result,
#                         calculating 4 points at a time
#         """
#         f = partial(self.get_animation_point, R)
#         p = Pool(processes=4)
#         result = p.map(f, arguments)
#
#         for i in range(len(t_yrs)):
#             amplifications.append(result[i][0])
#             images.append(result[i][1])
#             centroids.append(result[i][2])
#         return (
#         np.array(amplifications), np.array(images), np.array(centroids))
#
#     def animate(self, crossings, time_steps, frame_time, name, size, zoom):
#         """Creates an mp4 animation of the event:
#                 crossings = The number of einstein crossing times to animate
#                 time_steps = The number of time_steps either side of the peak
#                 frame_time = The time in ms for each frame
#                 name = the save name of the file
#                 size = [horizontal, vertical] cm's
#                 zoom = # of einstein radii plotted in vertical direction
#         """
#         times = np.array(range(2 * time_steps + 1)) / (2 * time_steps)
#         tau = (crossings[1] - crossings[0]) * (times) + crossings[0]
#         t = (tau * self.tE) + self.t0
#         points = self.get_animation_points(t)
#         rl = self.get_lens_astrometry(t)
#         rs = self.get_astrometry_unlensed(t)
#         image = points[1]
#         A = points[0]
#         centroids = points[2]
#         xcentroids = centroids[:, 0]
#         ycentroids = centroids[:, 1]
#         t_yrs = (t - self.t0) / 365.25
#         parallax_vec = parallax_in_direction(self.raL, self.decL, t)
#         lens = self.xL0 + np.outer(t_yrs, self.muL) * 1e-3
#         lens += self.piL * parallax_vec
#         fig = plt.figure(figsize=[size[0], size[1]])
#         matplotlib.rc('xtick', labelsize=25)
#         matplotlib.rc('ytick', labelsize=25)
#         ax1 = fig.add_subplot(2, 1, 1)
#         ax2 = fig.add_subplot(2, 1, 2)
#         patch = plt.Circle(rs[0], self.radius, color='green')
#         line1, = ax1.plot([], 'y.', markersize=15, label="Lens1")
#         line2, = ax1.plot([], 'g.', markersize=1, label="Source")
#         line3, = ax1.plot([], 'r.', markersize=5, label="Image")
#         line4, = ax1.plot([], 'b.', markersize=15, label="Lens2")
#         line5, = ax1.plot([], 'm', linewidth=3, label="Image Centroid")
#         line6, = ax2.plot(t, A, linewidth=4)
#         ax1.set_xlabel("RA (arcsec)", fontsize=40)
#         ax1.set_ylabel("Dec (arcsec)", fontsize=40)
#         ax1.add_patch(patch)
#         ax1.axis("equal")
#         ax1.set_ylim(
#             (lens[0][1] + lens[-1][1]) / 2 - zoom * self.thetaEamp * 1e-3,
#             (lens[0][1] + lens[-1][1]) / 2 + zoom * self.thetaEamp * 1e-3)
#         ax1.legend(fontsize=25, markerscale=3)
#         ax1.invert_xaxis()
#         ax2.set_xlabel("Time(MJD)", fontsize=40)
#         ax2.set_ylabel("Amplification", fontsize=40)
#         line = [line1, line2, line3, line4, line5, line6]
#         plt.tight_layout()
#
#         # this function is called at every frame, with i being the number of the frame that it's currently on
#         def update(i, rs, rl, line, image, tau, magnification, xcents, ycents):
#             line[0].set_data(rl[0][i], rl[1][i])
#             line[1].set_data(rs[i][0], rs[i][1])
#             line[2].set_data(image[i][:, 0], image[i][:, 1])
#             line[3].set_data(rl[2][i], rl[3][i])
#             line[4].set_data(xcents[:i], ycents[:i])
#             line[5].set_data(tau[:i], magnification[:i])
#             patch.center = rs[i]
#             return line
#             """
#             FuncAnimation takes in arguments:
#             fig = background figure
#             update = function that is called every frame
#             len(tau) = the number of frames, so now the first argument passed into update (i) will be (0,1,2...len(tau))
#             fargs specifies the other arguments to pass into update
#             blit being true means that each frame, if there are elements of it that don't change from the last frame,
#             it won't replot them, so this makes it faster
#             interval = number of milliseconds between each frame
#             alternatively you can specify fps in save after the file name
#             """
#
#         ani = animation.FuncAnimation(fig, update, len(tau),
#                                       fargs=[rs, rl, line, image, t, A,
#                                              xcentroids, ycentroids],
#                                       blit=True, interval=frame_time)
#         ani.save("%s.mp4" % name, writer="ffmpeg")
#         return
