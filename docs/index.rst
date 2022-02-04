==========================================================================
BAGLE - Bayesian Analysis of Gravitational Lensing Events
==========================================================================

.. video:: images/psplparallax.mp4

.. raw:: html 
	
	<video controls src="static/pspl.mp4"></video>

BAGLE allows modeling of gravitational microlensing events both photometrically and astrometrically. Supported microlensing models include:

* PSPL: point-source, point-lens with parallax
* PSBL: point-source, binary-lens
* FSPL: finite-source, point-lens (currently testing further)

All models support fitting data with single or multi-band photometry only, astrometry only, or joint fitting of photometry and astrometry (recommended).

.. toctree::
   :maxdepth: 8 

   intro.rst
   model3.rst
   PSPL.rst
   PSBL.rst
   FSPL.rst
   UserClass.rst
   Gen_Use.rst
