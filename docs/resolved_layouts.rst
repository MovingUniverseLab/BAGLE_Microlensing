Resolved array layouts and migration guide
==========================================

BAGLE models expose two kinds of output. **Unresolved** methods
(``get_photometry``, ``get_astrometry``, ``get_amplification``,
``get_centroid_shift``) return what a telescope that cannot separate the
lensed images would see. Their shapes are ``(N,)`` or ``(N, 2)`` and **have
not changed**, so model fitting and likelihoods are unaffected.

**Resolved** methods return one value per lensed image (and per source, for
binary sources). Before this change, each model family used its own axis
order: some were image-first ``(2, N)``, some time-first ``(N, 5)``, some
had a source axis and some did not, and ``get_all_arrays`` returned real
arrays in some families, complex arrays in others, and a 4-tuple in FSBL.
Code that worked for one family broke on the next.

Every family now uses the same **time-first** layout, with an explicit
source axis even for single sources. This page describes the new layout
and how to update code written against the old one.

.. note::

   This change is being rolled out family by family on the
   ``fsbl_ld_jax`` branch, in both ``bagle.model`` and
   ``bagle.model_jax``. Converted so far: **PSPL, PSBL, PSTL, BSPL, BSBL,
   FSPL**. The rest (FSBL, BFSPL, BFSBL) and the lens-astrometry change
   follow. This note is removed when the rollout is complete.


The new layout
--------------

Axis names used below:

* ``N``: number of times (the ``t`` you pass in).
* ``S``: number of sources (1 for single-source models, 2 for binary
  sources; primary first).
* ``I``: number of image slots per source.
* ``L``: number of lenses.

.. list-table:: Return shapes for every model family
   :header-rows: 1
   :widths: 35 30 35

   * - Method
     - Shape
     - Notes
   * - ``get_all_arrays(t)``
     - ``(images, amps)``, each ``(N, S, I)``
     - Always a 2-tuple. ``images`` is **complex**, East + i North.
       Missing images are NaN in both arrays.
   * - ``get_resolved_amplification(t)``
     - ``(N, S, I)``
     - Identical to the ``amps`` from ``get_all_arrays``.
   * - ``get_amplification(t)``
     - ``(N,)``
     - Unchanged. Sum over images, flux-weighted over sources.
   * - ``get_resolved_photometry(t)``
     - ``(N, S, I)``
     - Magnitude of each image, **without blend flux** (see below).
   * - ``get_resolved_astrometry(t)``
     - ``(N, S, I, 2)``
     - Real ``[East, North]``. Arcsec for ``*_PhotAstrom`` classes,
       Einstein radii relative to the lens for ``*_Phot`` classes.
   * - ``get_resolved_lens_astrometry(t)``
     - ``(N, L, 2)``
     - Real ``[East, North]``, one row per lens.

Number of sources, images and lenses per family:

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15

   * - Family
     - S
     - I
     - L
   * - PSPL (point source, point lens)
     - 1
     - 2 (plus, minus)
     - —
   * - PSBL (point source, binary lens)
     - 1
     - 5
     - 2
   * - PSTL (point source, triple lens)
     - 1
     - 10
     - 3
   * - BSPL (binary source, point lens)
     - 2
     - 2
     - —
   * - BSBL (binary source, binary lens)
     - 2
     - 5
     - 2
   * - FSPL (finite source, point lens)
     - 1
     - 2 (plus, minus)
     - —
   * - FSBL (finite source, binary lens)
     - 1
     - 1 (combined image)
     - 2
   * - BFSPL (binary finite source, point lens)
     - 2
     - 2
     - —
   * - BFSBL (binary finite source, binary lens)
     - 2
     - 1 (combined image)
     - 2

For the binary-lens families, the number of real images changes with time
(3 or 5 for PSBL). The array always has ``I`` slots and the unused slots
are NaN. Use ``np.nansum`` or ``np.isfinite`` masks rather than plain
``sum``. FSBL and BFSBL integrate over the source disk, so they report one
combined image (its total amplification and its centroid); use
``get_resolved_astrometry_outline`` when you need the individual image
outlines.

Image coordinates from ``get_all_arrays``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``images`` is always complex, ``East + 1j * North``, and is the input the
model's own ``get_astrometry`` and ``get_resolved_astrometry`` expect back
through ``image_arr=``. In every family it is in the same frame and units
as ``get_resolved_astrometry``:

* ``*_PhotAstrom`` (and ``*_Astrom``) classes: arcsec on the sky.
* ``*_Phot`` classes: Einstein radii, relative to the lens.

So for any model::

    images, amps = mod.get_all_arrays(t)
    xS = mod.get_resolved_astrometry(t)
    np.allclose(xS[..., 0], images.real, equal_nan=True)   # True
    np.allclose(xS[..., 1], images.imag, equal_nan=True)   # True

For lens-relative positions in Einstein radii from an astrometric model,
use ``get_u`` (the source) or a ``*_Phot`` model instead.

Passing arrays back in
~~~~~~~~~~~~~~~~~~~~~~

Methods that accept ``amp_arr=`` / ``image_arr=`` (``get_photometry``,
``get_astrometry``, ``get_centroid_shift``, ``get_resolved_astrometry``,
``get_resolved_amplification``, ``get_resolved_photometry``) expect the
``(N, S, I)`` arrays returned by ``get_all_arrays``. Computing them once and
passing them to several methods remains the fast path::

    images, amps = mod.get_all_arrays(t)
    mag = mod.get_photometry(t, amp_arr=amps)
    pos = mod.get_astrometry(t, image_arr=images, amp_arr=amps)


What changed, method by method
------------------------------

``get_all_arrays``
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Family
     - Before
     - After
   * - PSPL
     - did not exist
     - new: complex ``(N, 1, 2)``, ``(N, 1, 2)``
   * - PSBL
     - complex ``(N, 5)``, ``(N, 5)``
     - complex ``(N, 1, 5)``, ``(N, 1, 5)``
   * - PSTL
     - complex ``(N, 10)``, ``(N, 10)``
     - complex ``(N, 1, 10)``, ``(N, 1, 10)``
   * - BSPL
     - did not exist
     - new: complex ``(N, 2, 2)``, ``(N, 2, 2)``
   * - BSBL
     - complex ``(N, 2, 5)``, ``(N, 2, 5)``
     - unchanged
   * - FSPL
     - real ``(N, 2, 2)`` [t, img, EN], ``(N, 2)``
     - complex ``(N, 1, 2)``, ``(N, 1, 2)``
   * - FSBL
     - 4-tuple ``(images (N, 2), None, amps (N,), None)``
     - 2-tuple: complex ``(N, 1, 1)``, ``(N, 1, 1)``
   * - BFSPL
     - real ``(N, 2, 2, 2)``, ``(N, 2, 2)``
     - complex ``(N, 2, 2)``, ``(N, 2, 2)``
   * - BFSBL
     - real ``(N, 2, 2)`` [t, src, EN], ``(N, 2)``
     - complex ``(N, 2, 1)``, ``(N, 2, 1)``

``get_resolved_amplification``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now defined once, on the ``PSPL`` base class, as the ``amps`` from the
family's ``get_all_arrays``. It accepts an optional ``amp_arr=``.

* PSPL_PhotAstrom and FSPL_PhotAstrom: ``(2, N)`` becomes ``(N, 1, 2)``.
* PSBL, BSBL, FSPL_Phot, FSBL and BFSBL previously inherited the
  point-source, point-lens formula and returned the **wrong values**
  (they did not sum to ``get_amplification``). They now return the
  family's real per-image amplifications.
* PSPL_Phot and BSPL already used ``(N, S, I)``; values are unchanged.

``get_resolved_photometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now defined once, on the ``PSPL`` base class, so **every** family has it
(PSPL and BSPL did not before). Signature:
``get_resolved_photometry(t, filt_idx=0, amp_arr=None)``.

.. warning::

   **Behavior change: no blend flux.** The old implementations added the
   *full* blend flux (neighbors plus lens) to *every* image, even though
   their docstrings said they did not. Each image's magnitude is now the
   source flux times that image's amplification alone::

       mag_res[n, s, i] = flux2mag(f_src[s] * amps[n, s, i])

   The resolved fluxes plus the blend flux add up to ``get_photometry``.

Images with non-positive flux are NaN (the minus image of FSPL has a
negative, parity-signed amplification while the lens is inside the
source). The old masked-array returns are gone; you always get a plain
``ndarray`` with NaN.

Old shapes: PSBL ``(N, 5)`` (crashed in ``bagle.model``), PSTL
``(N, 10)``, FSPL ``(2, N)``, FSBL ``(N,)``, BFSPL ``(N, 2, 2)``; BSBL and
BFSBL crashed. PSTL's ``print_warning`` argument is removed.

``get_resolved_astrometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Family
     - Before
     - After
   * - PSPL_PhotAstrom, FSPL
     - ``(2, N, 2)`` [img, t, EN]
     - ``(N, 1, 2, 2)``
   * - PSPL_Phot
     - ``(N, 1, 2, 2)``
     - unchanged
   * - PSBL
     - ``(N, 5, 2)``
     - ``(N, 1, 5, 2)``
   * - PSTL
     - ``(N, 10, 2)``
     - ``(N, 1, 10, 2)``
   * - BSPL, BSBL, BFSPL
     - ``(N, 2, I, 2)``
     - unchanged
   * - FSBL_Phot
     - ``(N, 2)``
     - ``(N, 1, 1, 2)``
   * - FSBL_PhotAstrom
     - ``(N, 1, 2)``
     - ``(N, 1, 1, 2)``
   * - BFSBL
     - crashed
     - ``(N, 2, 1, 2)``

``get_resolved_lens_astrometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lens-first ``(L, N, 2)`` (or a tuple ``(xL1, xL2)`` of ``(N, 2)`` arrays)
becomes a single time-first array ``(N, L, 2)`` in PSBL, PSTL, BSBL, FSBL
and BFSBL.

Fixed along the way
~~~~~~~~~~~~~~~~~~~

These used to raise and now work: PSBL ``get_resolved_photometry``
(``bagle.model``), BSBL and BFSBL ``get_resolved_photometry``, FSPL
``get_amplification(amp_arr=...)``, ``FSPL_Phot.get_resolved_amplification``,
FSBL_Phot ``get_resolved_photometry`` / ``get_astrometry`` /
``get_centroid_shift``, and BFSBL ``get_resolved_astrometry``.

FSPL ``get_amplification`` and ``get_photometry`` also no longer flip the
sign of the total amplification when its most negative value exceeds its
largest in magnitude (a workaround that printed ``True`` in
``bagle.model``). The total is now simply the sum of the signed image
amplifications, which is positive for physical models.


Migration guide
---------------

Work through the recipes below for each place your code calls a resolved
method or ``get_all_arrays``. The grep commands at the end find them.

1. Single-lens plus/minus images (PSPL, FSPL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Old code unpacked the leading image axis::

    # Before
    xS_plus, xS_minus = mod.get_resolved_astrometry(t)   # each (N, 2)
    A_plus, A_minus = mod.get_resolved_amplification(t)  # each (N,)

Index the image axis instead; the source axis is always 0 here::

    # After
    xS = mod.get_resolved_astrometry(t)        # (N, 1, 2, 2)
    xS_plus, xS_minus = xS[:, 0, 0], xS[:, 0, 1]

    A = mod.get_resolved_amplification(t)      # (N, 1, 2)
    A_plus, A_minus = A[:, 0, 0], A[:, 0, 1]

.. warning::

   ``xS_plus, xS_minus = mod.get_resolved_astrometry(t)`` now unpacks along
   **time**. It raises ``ValueError`` unless ``len(t) == 2``, and with
   exactly two times it silently gives you the wrong thing. The same trap
   applies to ``xL1, xL2 = mod.get_resolved_lens_astrometry(t)``. Search
   for every tuple-unpacking of a resolved method.

2. Binary/triple-lens images (PSBL, PSTL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the source index (0) before the image index, and sum over both
axes::

    # Before
    images, amps = psbl.get_all_arrays(t)      # (N, 5)
    img_k = images[:, k]
    A_tot = np.nansum(amps, axis=1)
    xS = psbl.get_resolved_astrometry(t)       # (N, 5, 2)
    plt.plot(xS[:, k, 0], xS[:, k, 1])

    # After
    images, amps = psbl.get_all_arrays(t)      # (N, 1, 5)
    img_k = images[:, 0, k]
    A_tot = np.nansum(amps, axis=(1, 2))
    xS = psbl.get_resolved_astrometry(t)       # (N, 1, 5, 2)
    plt.plot(xS[:, 0, k, 0], xS[:, 0, k, 1])

If a block of old code is long, drop the source axis once at the top
instead: ``images, amps = images[:, 0], amps[:, 0]`` and
``xS = xS[:, 0]``.

.. warning::

   Reductions over ``axis=1`` still run without error but now reduce over
   the (length-1) source axis instead of the images. For example
   ``np.isnan(amps).sum(axis=1) == 5`` is never true any more. Replace
   ``axis=1`` with ``axis=(1, 2)`` or ``axis=-1`` as appropriate.

3. Binary sources (BSPL, BSBL, BFSPL, BFSBL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BSPL, BSBL and BFSPL resolved astrometry was already
``(N, 2, I, 2)``; no change. BSPL now also has ``get_all_arrays`` and
``get_resolved_photometry``. BFSBL's combined-image arrays gain an image
axis of length 1::

    # Before (BFSBL)
    images, amps = bfsbl.get_all_arrays(t)     # real (N, 2, 2), (N, 2)
    x_pri = images[:, 0, :]

    # After
    images, amps = bfsbl.get_all_arrays(t)     # complex (N, 2, 1), (N, 2, 1)
    xS = bfsbl.get_resolved_astrometry(t)      # (N, 2, 1, 2)
    x_pri = xS[:, 0, 0]

The total amplification of a binary source is flux-weighted over sources::

    f = mod._source_fluxes(filt_idx)            # (S,)
    A = np.nansum(amps, axis=2) @ f / f.sum()   # equals get_amplification

4. Real image positions became complex (FSPL, BFSPL, FSBL, BFSBL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``get_all_arrays`` images are now complex in every family::

    # Before (FSPL)
    images, amps = fspl.get_all_arrays(t)      # real (N, 2, 2), (N, 2)
    x_plus_E = images[:, 0, 0]
    x_plus_N = images[:, 0, 1]

    # After
    images, amps = fspl.get_all_arrays(t)      # complex (N, 1, 2), (N, 1, 2)
    x_plus_E = images[:, 0, 0].real
    x_plus_N = images[:, 0, 0].imag

To get a real ``[..., 2]`` array from complex images::

    xy = np.stack((images.real, images.imag), axis=-1)   # (N, S, I, 2)

5. FSBL ``get_all_arrays`` returns a 2-tuple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    # Before
    images, parity, amps, _ = fsbl.get_all_arrays(t)   # (N, 2), None, (N,), None

    # After
    images, amps = fsbl.get_all_arrays(t)              # complex (N, 1, 1), (N, 1, 1)
    A_tot = amps[:, 0, 0]                              # equals get_amplification

There is no per-image parity for the integrated FSBL image. For the
individual image outlines and their parities, use
``FSBL._image_outline`` or ``get_resolved_astrometry_outline``.

6. Resolved photometry no longer includes the blend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you relied on the old behavior (every image carrying the full blend),
rebuild it from the new output::

    from bagle.model import mag2flux, flux2mag

    mag_res = mod.get_resolved_photometry(t, filt_idx=0)       # (N, S, I)
    f_src = mod._source_fluxes(0)                               # (S,)
    b_sff = mod.b_sff[0]
    f_blend = f_src.sum() * (1 - b_sff) / b_sff

    flux_img = np.nan_to_num(mag2flux(mag_res))
    mag_old_style = flux2mag(flux_img + f_blend)

    # Consistency check: images plus blend reproduce get_photometry.
    flux_tot = flux_img.sum(axis=(1, 2)) + f_blend
    np.testing.assert_allclose(flux2mag(flux_tot), mod.get_photometry(t))

Note that ``mag2flux`` maps NaN to 0, which is what you want for missing
images in a sum.

7. Lens positions
~~~~~~~~~~~~~~~~~

::

    # Before
    xL1, xL2 = psbl.get_resolved_lens_astrometry(t)     # each (N, 2)

    # After
    xL = psbl.get_resolved_lens_astrometry(t)           # (N, 2, 2)
    xL1, xL2 = xL[:, 0], xL[:, 1]

PSTL has three lenses: ``xL[:, 0]``, ``xL[:, 1]``, ``xL[:, 2]``.

8. Arrays you build yourself and pass in
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you construct ``amp_arr`` / ``image_arr`` by hand (for example from a
cache), add the source axis::

    amps = amps_old[:, np.newaxis, :]        # (N, I) -> (N, 1, I)
    images = images_old[:, np.newaxis, :]

9. Writing family-agnostic code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The point of the change is that one loop now works for every model::

    images, amps = mod.get_all_arrays(t)               # (N, S, I)
    xS = mod.get_resolved_astrometry(t)                # (N, S, I, 2)
    N, S, I = amps.shape
    for s in range(S):
        for i in range(I):
            ok = np.isfinite(amps[:, s, i])
            plt.plot(xS[ok, s, i, 0], xS[ok, s, i, 1], '.')

Finding code to update
~~~~~~~~~~~~~~~~~~~~~~

Run these from your project root. Every hit needs a look; the
tuple-unpacking ones are the most likely to be wrong silently::

    rg -n "get_all_arrays|get_resolved_(amplification|photometry|astrometry)"
    rg -n "get_resolved_lens_astrometry"
    rg -n "=\s*\w+\.get_resolved_\w+\(" | rg ","     # tuple unpacking
    rg -n "axis\s*=\s*1" | rg -i "amp|img|image"       # reductions to recheck

Then run your code on one model from each family you use and check the
invariants above: ``get_resolved_amplification`` equals the ``amps`` from
``get_all_arrays``, ``get_resolved_astrometry`` equals the complex
``images`` split into ``[East, North]``, and resolved fluxes plus the blend
give ``get_photometry``. ``tests/test_resolved_layouts.py`` checks all
three, for every family, in both ``bagle.model`` and ``bagle.model_jax``.
