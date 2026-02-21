For details, see Chapter 6 (starting on page 221) at https://www.proquest.com/docview/2866322369

convert_bagle_mulens_psbl_phot
==============================
Convert between the native fit parameters of BAGLE and MulensModel for a point source binary lens photometry model.  

convert_helio_geo_ast
=====================
Convert between Solar System barycentric and geocentric-projected parameters for a point source point lens model.
This converts both the photometric and the astrometric parameters.
[NOT YET TESTED OR DOCUMENTED]

convert_helio_geo_phot
======================
Convert between Solar System barycentric and geocentric-projected parameters for a point source point lens model.
This converts only the subset of parameters in photometry fits (t0, u0, tE, piEE, piEN). 

convert_u0vec_t0
================
Convert the values of u0 vector and t0 between the Solar System barycentric and geocentric projected frames.

convert_piEvec_tE
=================
Convert the values of piE vector and tE between the Solar System barycentric and geocentric projected frames.

v_Earth_proj
============
Calculate the Earth-Sun vector direction.

plot_conversion_diagram
=======================

convert_u0_t0_psbl
==================
Natasha wrote this one (according to git blame)

General conventions between the codes:
======================================

Time
----
BAGLE requires times to be in MJD.
MulensModel and pyLIMA required times to be in (H?)JD.
MJD and JD differ by a factor of 2400000.5.

Coordinate conventions
----------------------
BAGLE works in E-N convention.
MulensModel and pyLIMA work in the tau-beta convention.

Relative proper motion
----------------------
BAGLE defined the relative proper motion as source-lens.
MulensModel and pyLIMA defined relative proper motion as lens-source.

Angles
------
There are too many to list/check here, but some that come to mind are:
Primary vs. secondary.
Binary axis angle.
Relative proper motion/binary axis angle.
