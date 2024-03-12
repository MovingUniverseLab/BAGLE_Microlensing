.. _PSPL Details:
######################################################
Point-Source Point-Lens (PSPL) Implementation
######################################################

PSPL models involve a single lens moving in front of a single star
on the plane of the sky. The majority of PSPL models in BAGLE have parameterizations
with input parameters in heliocentric coordinates, unless otherwise specified.
The heliocentric coordinates are most useful in when modeling astrometry and
photometry jointly, which we will build up to.

Photometry-Only
===============
The simplest form of microlens model is one used for generating a photometric
light curve only. In these events, the absolute position of the lens and source
on the sky are not known. Instead, only the relative separation is known and there
is some information about the relative orientation of their separation and
relative proper motion vectors on the sky at closest approach.

For classes with photometry only, input parameters typically include

=================  ================  ========================================================
Parameter          Units             Description
=================  ================  ========================================================
:math:`t_0`        MJD (day)         Time of closest approach in heliocentric coordinates.
:math:`u_0`        :math:`\theta_E`  Closest approach distance.
:math:`t_E`        (day)             Einstein crossing time.
:math:`\pi_{E,E}`                    Microlensing parallax in the East direction.
:math:`\pi_{E,N}`                    Microlensing parallax in the North direction.
:math:`b_{sff}`                      Blending source flux fraction [0-1].
:math:`m_{src}`    (mag)             Source magnitude.
=================  ================  ========================================================

For microlensing events where parallax should be modeled, the following
additional parameters are needed. Note these parameters are typically fixed
and won't need to be vary when we fit data with these models.

=================  ================  ========================================================
Parameter          Units             Description
=================  ================  ========================================================
:math:`\alpha_L`   (deg)             R.A. (for parallax)
:math:`\delta_L`   (deg)             Dec. (for parallax)
obsLocation        str               `earth` or satellite name (for parallax)
=================  ================  ========================================================

Finally, we occasionally need models that have underlying noise in them
as well (often in excess of the noise from measurements alone). This is
particular useful to model events where the source star is stochastically variable
from spots or activity. But this noise model can also capture systematic red noise
from atmospheric and instrumentation sources.

We utilize a Gaussian Process to model this additional noise using the
``celerite`` package. The default GP kernel includes two temporally
correlated noise terms: a Matern-3/2 and a
damped simple harmonic oscillator. We also include a white-noise jitter term.
The GP classes require additional parameters to specify the noise
as shown in the table below.

=========================  ================  ========================================================
Parameter                  Units             Description
=========================  ================  ========================================================
:math:`\log \sigma_{GP}`
:math:`\log \rho_{GP}`
:math:`\log S_0`
:math:`log \omega_{0,GP}`
=========================  ================  ========================================================

NEEDS FIXING BELOW HERE
=======================
.. toctree::
   :maxdepth: 4 
   
   PSPL_Param.rst
   PSPL_Data.rst
   PSPL_GP.rst
   PSPL_Parallax.rst
