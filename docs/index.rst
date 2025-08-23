==========================================================================
BAGLE - Bayesian Analysis of Gravitational Lensing Events
==========================================================================

.. raw:: html 
	
	<video controls src="_static/pspl.mp4"></video>

BAGLE is a python package used to model gravitational microlensing
events both photometrically and astrometrically. Supported
microlensing models include:

* PSPL: point-source, point-lens with and without parallax

* PSBL: point-source, binary-lens

  * static lens secondary
  * moving lens secondary with linear, accelerating, circular, or elliptical orbital motion.

* BSPL: binary-source, point lens

  * static source secondary
  * moving source secondary with linear, accelerating, circular or elliptical orbital motion.

* BSBL: binary-source, binary lens

  * static lens and source secondary
  * moving source or lens secondary with linear, accelerating, circular or elliptical orbital motion.

* FSPL: finite-source, point-lens


All models support fitting data with single or multi-band photometry
only, astrometry only, or joint fitting of photometry and astrometry
(recommended).


.. toctree::
   :maxdepth: 2

   installation
   getting_started
   models
   fitting
   new_models
   api_docs
   citation


