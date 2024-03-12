==========================================================================
BAGLE - Bayesian Analysis of Gravitational Lensing Events
==========================================================================

.. raw:: html 
	
	<video controls src="_static/pspl.mp4"></video>

BAGLE is a python package used to model gravitational microlensing
events both photometrically and astrometrically. Supported
microlensing models include:

* PSPL: point-source, point-lens
* PSBL: point-source, binary-lens
    * static lens secondary
* BSPL: binary-source, point lens
    * static source secondary
    * moving secondary sources with linear, accelerating, and circular or elliptical orbits.
* BSBL: binary-source, binary lens
    * static lens and source secondary
* FSPL: finite-source, point-lens
* Parallax for all of the above

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


