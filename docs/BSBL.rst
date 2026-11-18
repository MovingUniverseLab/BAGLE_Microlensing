
######################################################
Binary Source Binary Lens (BSBL) Models
######################################################

.. toctree::
   :maxdepth: 4 

   BSBL_Model.rst

BSBL Developer Classes
=======================
.. toctree::
   :maxdepth: 2

   BSBL_Param.rst
   BSBL_Data.rst
   BSBL_Parallax.rst

Reference points
================
BSBL / BFSBL parameterizations take one of three input pairs, named by
suffix: ``_com`` (source CoM to lens CoM: ``t0_com``, ``beta_com`` /
``u0_amp_com``, ``xS0_com_E`` / ``xS0_com_N``; Keplerian Param1 / Param2),
``_p`` (primary source to primary lens: ``t0_p``, ``beta_p``, ``xS0_p_E`` /
``xS0_p_N``; static Param2 and Keplerian Param3), or the static
midpoint pair (``t0``, ``beta``, ``xS0_E`` / ``xS0_N``; static Param1).
Internally, static and linear / accelerated models store the primary
source to lens-midpoint pair in ``t0`` / ``u0_amp`` / ``xL0``; Keplerian
models store the CoM-CoM pair there and compute the midpoint (``t0_mid``
...) and primary (``t0_p`` ...) pairs on first access. Every model also
stores ``t0_com`` / ``u0_amp_com`` / ``xS0_com`` / ``xL0_com`` and the
source mass ratio ``qS = mSs / mSp`` (a static Param2 input). See
:ref:`binary-reference-frames`.

Example notebooks
=================
Static geometry plots of BSBL models with Keplerian orbits and
parallax, and with parallax but no orbital motion
(``plot_BSBL_static``), are in `example_plot_PSBL_BSBL.ipynb
<https://github.com/MovingUniverseLab/BAGLE_Microlensing/blob/dev/docs/notebooks/example_plot_PSBL_BSBL.ipynb>`_.
