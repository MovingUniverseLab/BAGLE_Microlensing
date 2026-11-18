
######################################################
Point Source Binary Lens (PSBL) Models
######################################################

.. toctree::
   :maxdepth: 4

   PSBL_Model.rst

PSBL Developer Classes
=======================
.. toctree::
   :maxdepth: 2

   PSBL_Param.rst
   PSBL_Data.rst
   PSBL_Parallax.rst

Reference points
================
PSBL / FSBL parameterizations take their closest approach about the lens
geometric midpoint (``t0``, ``u0_amp`` / ``beta``, ``xS0_E`` / ``xS0_N``),
the lens center of mass (``t0_com``, ``u0_amp_com`` / ``beta_com``,
``xS0_com_E`` / ``xS0_com_N``) or the primary lens (``t0_p`` / ``t0_prim``,
``beta_p`` / ``u0_amp_prim``, ``xS0_p_E`` / ``xS0_p_N``). Internally, static,
linear and accelerated models store the midpoint pair in ``t0`` /
``u0_amp`` / ``xL0``; Keplerian (``EllOrbs`` / ``CircOrbs``) models store the
CoM pair there and compute ``t0_mid`` / ``u0_amp_mid`` / ``xL0_mid`` (and
``t0_p`` ...) on first access, except Param2 / Param3, which take and keep
the midpoint pair. See :ref:`binary-reference-frames`.

Example notebooks
=================
Static geometry plots of PSBL models with Keplerian orbits and
parallax, and with parallax but no orbital motion
(``plot_PSBL_static``), are in `example_plot_PSBL_BSBL.ipynb
<https://github.com/MovingUniverseLab/BAGLE_Microlensing/blob/dev/docs/notebooks/example_plot_PSBL_BSBL.ipynb>`_.
