
######################################################
Binary Source Point Lens (BSPL) Models
######################################################

.. toctree::
   :maxdepth: 4 

   BSPL_Model.rst

BSPL Developers
===============
.. toctree::
   :maxdepth: 2

   BSPL_Param.rst
   BSPL_Data.rst
   BSPL_Parallax.rst

Reference points
================
BSPL parameterizations take the closest approach of the primary source
(``t0``, ``u0_amp`` / ``beta``) and the primary source position ``xS0``;
the secondary is placed at ``sep`` / ``alpha`` (or by the orbit) from it.
Source masses are ``mS``, ``mSp``, ``mSs`` (Msun) with ``qS = mSs / mSp``.
The ``_com`` / ``_p`` / ``_mid`` naming and the internal-frame rules shared
by all binary models are in :ref:`binary-reference-frames`.

Example notebooks
=================
Static geometry plots of BSPL models with parallax but no orbital
motion (``plot_BSPL_static``) are in `example_plot_PSBL_BSBL.ipynb
<https://github.com/MovingUniverseLab/BAGLE_Microlensing/blob/dev/docs/notebooks/example_plot_PSBL_BSBL.ipynb>`_.
