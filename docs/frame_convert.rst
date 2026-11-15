General conventions between the codes:
======================================
For details, see Chapter 6 (starting on page 221) at https://www.proquest.com/docview/2866322369

Time and position in BAGLE
--------------------------
BAGLE model times are Modified Julian Date on the Barycentric Dynamical
Time (TDB) scale::

    JD_TDB = MJD + 2400000.5

``t0`` is the closest approach of the rectilinear trajectory as seen from
the Solar System barycenter, on that TDB clock. It is the SSB closest
approach, which differs from the peak time recorded by an observer on
Earth or a satellite. Pass observation times through to TDB MJD before
calling a model. A UTC MJD labeled as TDB shifts the JPL ephemeris by
TDB−UTC, which is 66.2 s at MJD 55775 and 69.2 s at MJD 60000 (32.184 s
plus leap seconds).

On-sky coordinates are ICRS right ascension and declination, in degrees,
J2000. Vectors on the sky (source and lens positions, proper motions,
:math:`u_0`, :math:`\pi_E`) are local East and North in the ICRS tangent
plane. North is the ICRS pole projected orthogonal to the line of sight;
East completes the right-handed pair. Relative proper motion is
source minus lens.

The parallax displacement is the barycentric position of the observer
(Earth, or another JPL Horizons body), in AU, pointing from the observer
toward the Solar System barycenter and then projected onto that East/North
basis. ``parallax_in_direction`` evaluates the JPL ephemeris at the TDB
instant above. This is the SSB vector, in the ICRS equatorial frame.

Time in other packages
----------------------
MulensModel and pyLIMA take Julian Date. Adding 2400000.5 to a BAGLE
MJD produces a TDB JD. VBMicrolensing's numeric clock is
``JD - 2450000``. Its bundled ``SunEphemeris.txt`` is a Horizons
geocentric Sun table whose header stamps the rows as UT (DE441, 1-day
steps). The same number is therefore not the same instant in BAGLE and
in VBM.

``mjd_prime_to_vbm_time`` converts a BAGLE TDB MJD to that VBM clock:
the instant is relabeled UTC, then

.. math::

    t_\mathrm{VBM} = \mathrm{MJD_{UTC}} - 50000 + 0.5
                   = \mathrm{JD_{UTC}} - 2450000.

Use this for ``t0``, ``t0_par``, and the light-curve time grid on any
VBM call that reads the Sun table. Paths without parallax do not read
the table; leave those times in TDB MJD on both sides so ``t - t0`` is
unchanged.

VBM's default ``t_in_HJD=True`` still subtracts a heliocentric
light-travel time when it indexes the Sun table, and it leaves
``t - t0`` on the input clock. ``mjd_prime_to_vbm_time`` does not apply
a light-travel correction. The previous version added a Keck
heliocentric correction before the zero point. That term depends on
direction and on the time of year, so it warped ``t - t0`` by up to
several minutes across an event, and it does not match the geocentric
UT table VBM interpolates. The function name is unchanged; the
light-travel step is gone.

Two further VBM ephemeris differences are outside this clock
conversion. The table's equatorial basis is tilted by the obliquity
(~23.4°), and the table is linearly interpolated at 1-day steps.
BAGLE queries JPL in ICRS at the requested TDB instant.

Coordinate conventions
----------------------
BAGLE works in the ICRS East-North convention described above.
MulensModel and pyLIMA work in the tau-beta convention.

Relative proper motion
----------------------
BAGLE defined the relative proper motion as source-lens.
MulensModel and pyLIMA defined relative proper motion as lens-source.

Angles
------
FILL IN LATER


Frame Conversions (bagle.frame_convert)
=======================================

.. automodule:: bagle.frame_convert
     :memebers:
