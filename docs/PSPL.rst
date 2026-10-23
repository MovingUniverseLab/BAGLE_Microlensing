.. _PSPL Details:
######################################################
Point-Source Point-Lens (PSPL) Models
######################################################

PSPL models involve a single lens moving in front of a single star
on the plane of the sky. The majority of PSPL models in BAGLE have parameterizations
with input parameters in Solar System barycentric coordinates, unless otherwise specified.
The Solar System barycentric (SSB) coordinates are most useful in when modeling astrometry and
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
:math:`t_0`        MJD (day)         Time of closest approach in SSB coordinates.
:math:`u_0`        :math:`\theta_E`  Closest approach distance.
:math:`t_E`        (day)             Einstein crossing time.
:math:`\pi_{E,E}`                    Microlensing parallax in the East direction.
:math:`\pi_{E,N}`                    Microlensing parallax in the North direction.
:math:`b_{sff}`                      Blending source flux fraction [0-1].
:math:`m_{src}`    (mag)             Source magnitude.
=================  ================  ========================================================

For microlensing events where parallax should be modeled, the following
additional parameters are needed. Note these parameters are typically fixed
and won't need to be varied when we fit data with these models.

=================  ================  ========================================================
Parameter          Units             Description
=================  ================  ========================================================
:math:`\alpha_L`   (deg)             R.A. (for parallax)
:math:`\delta_L`   (deg)             Dec. (for parallax)
obsLocation        str               `earth` or satellite name (for parallax)
=================  ================  ========================================================


Photometry and Astrometry
=========================

BAGLE's strength is jointly modeling or fitting photometric and astrometric data sets.
In these events, the absolution position of the lens and source on the sky are known
and controlled by additional parameters in the model. All orientations of position and velocity
vectors in these models are with respect to North and East as defined by the Earth's equator,
even if models are SSB or observed from some other satellite (e.g. Roman).

Parameterizations for photometric+astrometric models are much more varied. First, we can
start with an expansion approach where we start with photometric parameters and add more
parameters to describe the astrometry. An example of this parameterization is
``PSPL_PhotAstrom_Par_Param2`` with the following parameters:

=================  ================  ========================================================
Parameter          Units             Description
=================  ================  ========================================================
:math:`t_0`        MJD (day)         Time of closest approach in SSB coordinates.
:math:`u_0`        :math:`\theta_E`  Closest approach distance.
:math:`t_E`        (day)             Einstein crossing time.
:math:`\theta_E`   (mas)             Einstein radius.
:math:`\pi_S`      (mas)             Parallax of the source.
:math:`\pi_{E,E}`                    Microlensing parallax in the East direction.
:math:`\pi_{E,N}`                    Microlensing parallax in the North direction.
:math:`X_{S_0,E}`  (arcsec)          R.A. source position on sky at :math:`t = t_0`.
:math:`X_{S_0,N}`  (arcsec)          Dec. source position on sky at :math:`t = t_0`.
:math:`\mu_{S,E}`  (mas/yr)          Source proper motion in R.A. direction.
:math:`\mu_{S,N}`  (mas/yr)          Source proper motion in Dec. direction.
:math:`b_{sff}`                      Blending source flux fraction [0-1].
:math:`m_{src}`    (mag)             Source magnitude.
=================  ================  ========================================================

along with the fixed parameters:

=================  ================  ========================================================
Parameter          Units             Description
=================  ================  ========================================================
:math:`\alpha_L`   (deg)             R.A. (for parallax)
:math:`\delta_L`   (deg)             Dec. (for parallax)
obsLocation        str               `earth` or satellite name (for parallax)
=================  ================  ========================================================

Note that the :math:`X_{S_0}` source positions are in an arbitrary reference frame and are
designed for relative astrometric measurements in a tangential plane projection
(i.e. small proper motions, not large proper motions where spherical coordinates
are important and the tangential plane projection is no longer valid).

Alternatively, parameterizations can be expressed entirely in physical quantities as is
the case for the ``PSPL_PhotAstrom_Par_Param1`` model class with the following parameters:

=================  =================  ========================================================
Parameter          Units              Description
=================  =================  ========================================================
:math:`m_L`        (:math:`M_\odot`)  Lens mass.“
:math:`t_0`        MJD (day)          Time of closest approach in SSB coordinates.
:math:`\beta`      (mas)              Closest approach distance, projected on sky.
:math:`d_L`        (pc)               Distance to the lens.
:math:`d_L/d_S`                       Ratio of lens distance to source distance.
:math:`X_{S_0,E}`  (arcsec)           R.A. source position on sky at t = t0.
:math:`X_{S_0,N}`  (arcsec)           Dec. source position on sky at t = t0.
:math:`\mu_{L,E}`  (mas/yr)           Lens proper motion in R.A. direction.
:math:`\mu_{L,N}`  (mas/yr)           Lens proper motion in Dec. direction.
:math:`\mu_{S,E}`  (mas/yr)           Source proper motion in R.A. direction.
:math:`\mu_{S,N}`  (mas/yr)           Source proper motion in Dec. direction.
:math:`b_{sff}`                       Blending source flux fraction [0-1].
:math:`m_{src}`    (mag)              Source magnitude.
=================  =================  ========================================================


:math:`u_0` Orientation Conventions
-----------------------------------

In both photometric and astrometric models, the :math:`u_0` paramater quantifies both
the amplitude of the :math:`\vec{u}_0` and has a :math:`\pm` sign convention used to break
the degeneracy between the lens passing to the East or the West of the source.
FILL IN MORE. GOULD REFERENCE.


Gaussian Process Noise
======================

We occasionally need models that have underlying noise in them
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


.. toctree::
   :maxdepth: 4 
   
   PSPL_Model.rst

PSPL Developer Classes
======================
.. toctree::
   :maxdepth: 2

   PSPL_Param.rst
   PSPL_Data.rst
   PSPL_Parallax.rst
   PSPL_GP.rst
