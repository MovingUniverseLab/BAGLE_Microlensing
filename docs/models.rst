==================
Using BAGLE Models
==================

``bagle.model`` is a module that contains a set of classes
and functions that allow the user to construct microlensing
models. The available classes for instantiating a microlensing
event are shown in the list below. See the API documentation
for each class for details.

Example: A Microlens Model Event
================================

To instantiate a model:

.. code-block:: python
                
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

For example, ``PSPL_Phot_noPar_Param2`` has a data and model class type of ``PSPL_Phot``,
which contains a point-source, point-lens event with photometry only. The model
has no parallax, no GP and uses parameterization #2.

List of Available Models
========================

The complete list of instantiable model classes is: 

Point source, point lens, photometry only
-----------------------------------------

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

Point source, point lens, photometry and astrometry
---------------------------------------------------

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
-----------------------------------------

    - PSPL_Astrom_Par_Param4
    - PSPL_Astrom_Par_Param3

Point soruce, binary lens, photometry only
------------------------------------------

    - PSBL_Phot_noPar_Param1
    - PSBL_Phot_Par_Param1
    - PSBL_Phot_noPar_GP_Param1
    - PSBL_Phot_Par_GP_Param1

Point source, binary lens, photometry and astrometry
----------------------------------------------------
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

Binary source, point lens, photometry only
----------------------------------------------

    - BSPL_Phot_noPar_Param1
    - BSPL_Phot_Par_Param1
    - BSPL_Phot_noPar_GP_Param1
    - BSPL_Phot_Par_GP_Param1

Binary source, point lens, photometry and astrometry
----------------------------------------------------

    - BSPL_PhotAstrom_noPar_LinOrbs_Param1
    - BSPL_PhotAstrom_noPar_AccOrbs_Param1
    - BSPL_PhotAstrom_noPar_LinOrbs_Param2
    - BSPL_PhotAstrom_noPar_AccOrbs_Param2
    - BSPL_PhotAstrom_noPar_LinOrbs_Param3
    - BSPL_PhotAstrom_noPar_AccOrbs_Param3
    - BSPL_PhotAstrom_noPar_Param1
    - BSPL_PhotAstrom_noPar_Param2
    - BSPL_PhotAstrom_noPar_Param3
    - BSPL_PhotAstrom_Par_LinOrbs_Param1
    - BSPL_PhotAstrom_Par_AccOrbs_Param1
    - BSPL_PhotAstrom_Par_LinOrbs_Param2
    - BSPL_PhotAstrom_Par_AccOrbs_Param2
    - BSPL_PhotAstrom_Par_LinOrbs_Param3
    - BSPL_PhotAstrom_Par_AccOrbs_Param3
    - BSPL_PhotAstrom_Par_Param1
    - BSPL_PhotAstrom_Par_Param2
    - BSPL_PhotAstrom_Par_Param3
    - BSPL_PhotAstrom_noPar_GP_LinOrbs_Param1
    - BSPL_PhotAstrom_noPar_GP_LinOrbs_Param2
    - BSPL_PhotAstrom_noPar_GP_LinOrbs_Param3
    - BSPL_PhotAstrom_noPar_GP_Param1
    - BSPL_PhotAstrom_noPar_GP_Param2
    - BSPL_PhotAstrom_noPar_GP_Param3
    - BSPL_PhotAstrom_Par_GP_LinOrbs_Param1
    - BSPL_PhotAstrom_Par_GP_LinOrbs_Param2
    - BSPL_PhotAstrom_Par_GP_LinOrbs_Param3
    - BSPL_PhotAstrom_Par_GP_Param1
    - BSPL_PhotAstrom_Par_GP_Param2
    - BSPL_PhotAstrom_Par_GP_Param3

Binary source, binary lens, photometry and astrometry
----------------------------------------------------

    - BSBL_PhotAstrom_noPar_Param1
    - BSBL_PhotAstrom_noPar_Param2
    - BSBL_PhotAstrom_Par_Param1
    - BSBL_PhotAstrom_Par_Param2

Finite source, point lens, photometry and astrometry (broken)
-------------------------------------------------------------

    - FSPL_PhotAstrom_Par_Param1


      
For a more detailed explanation of how BAGLE models are structured and
how you can add your own new models and parameterizations, see
:ref:`Making Your Own Microlens Models <new_models>`.
