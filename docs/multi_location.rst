Multi-location observers
========================

For a microlensing event, the photometry and astrometry can come from 
more than one place: a ground-based telescope and a spacecraft, or two 
catalogs that do not share a sky zero point. BAGLE keeps a single model 
for the event. Each filter has its own observer, and therefore its own parallax vector,
and each filter that has astrometry carries its own origin ``xS0``. 

The numbers in the snippets below are the ones used by
``tests/test_model_multi_location.py`` and
``tests/test_model_fitter_multi_location.py``. They illustrate the
API. They are not a fit to a published event.

Motivation and physics
----------------------

The lens and the source are the same objects in every dataset. What
changes with the observer is the line of sight. ``parallax_in_direction``
returns the East/North parallax vector of that observer 
(or body/spacecraft they reside on) at the observation 
times (barycentric, in AU). Photometry, the lens track, the unlensed
source, and the unresolved centroid all read
``obsLocation[filt_idx]`` and pass it to that function. Annual parallax
and satellite parallax are the same ephemeris call with a different
body name.

``xS0`` is the relative catalog position of the source in the 
Solar System Barycentric frame at ``t0``, in arcseconds. 

Shared across the event: ``t0``, ``u0`` / ``beta``, ``tE``, ``piE``,
``thetaE``, masses, distances, proper motions, binary separation, and
orbit elements. Per filter, at the same index: the source flux
fraction ``b_sff``, magnitudes, ``xS0_E``, ``xS0_N``, ``pi_ref_frame``
on a reference-frame parallax model, and GP or error hyperparameters.
``obsLocation`` is fixed, not sampled.

Joint Photometry and Astrometry Data Sets
-----------------------------------------

Datasets are defined by a dataset name. The model_fitter.Fitter class determins 
how to join the photometry and astrometry data sets together. 
The order of the datasets and their associated filter indices is as follows:

1. Every name in ``phot_data``, in that order.
2. Then every name in ``ast_data`` that is not already in ``phot_data``.
A name in both lists is one joint filter/dataset. A repeated name inside one
list is a ``ValueError``. ``'Keck'`` and ``'keck_Kp'`` are two filters.
The example below shows a single data set.
.. code-block:: python

   data['phot_data'] = 'sim'
   data['ast_data'] = 'sim'

An example of a joint photometry and astrometry data set is shown below. 
This is the layout in
``test_mixed_four_filter_cube`` (that test uses ``mars`` and
``jupiter`` so the ephemeris is drawn from astropy's local ephemeris).
We show ``earth``, ``spitzer`` and ``gaia`` observer locations for different 
data sets. 

.. code-block:: python

   data['phot_data'] = ['ogle', 'spitzer', 'keck']
   data['ast_data'] = ['keck', 'gaia']
   data['obsLocation'] = {
       'ogle': 'earth',
       'spitzer': 'spitzer',
       'keck': 'earth',
       'gaia': 'gaia',
   }

The unified list is ``ogle``, ``spitzer``, ``keck``, ``gaia`` and filter 
indices onto a model object would be as follows:

* 0, OGLE, photometry only. ``xS0`` fixed at 0. Source flux fraction
  and ``mag_src`` sampled.
* 1, Spitzer, photometry only. Same.
* 2, Keck, joint. ``xS0``, source flux fraction, and ``mag_src``
  sampled.
* 3, Gaia, astrometry only. ``xS0`` and the source flux fraction
  sampled. ``mag_src`` fixed at 0.

Array indices for your data objects do not change. ``t_phot1`` is still
``phot_data[0]``. ``t_ast1`` is still ``ast_data[0]``. In the example,
Keck astrometry is ``t_ast1`` and model ``filt_idx=2``. Gaia is
``t_ast2`` and ``filt_idx=3``. ``map_phot_idx_to_ast_idx`` maps back and 
forth between the photometry and astrometry data sets and is
``[2, 3]``: each entry is the unified filter index of that
astrometric series.


Observers
---------

``data['obsLocation']`` accepts three forms, all aligned with the
unified list. 

* Missing, or ``None``. Every filter is ``'earth'``.
* A string. Every filter uses that body.
* A dict keyed by dataset name. A missing name is a ``ValueError``.
  A joint name has one entry. This is the form to use when the
  collection is mixed.
* A list of length ``n_filters``, in unified order (photometric
  names, then astrometry-only names).

The model_fitter ``resolve_obs_locations`` function handles
the mapping between dataset names and observer locations and 
writes ``fitter.obs_locations``.

The ``getdata`` function takes the same idea as ``obs_location``:

.. code-block:: python

   from bagle.data import getdata

   data = getdata(
       'ob120169',
       phot_data=['I_OGLE', 'Ch1_Spitzer'],
       ast_data=['Kp_Keck'],
       obs_location='auto',
   )

``None`` leaves the key unset, so the fitter uses Earth.
``'auto'`` maps ``Ch1_Spitzer`` to ``'spitzer'`` and every other
catalog name (OGLE, Keck, HST, MOA, KMT) to ``'earth'``. A dict is
stored as given.

``Earth`` and ``EARTH`` are aliases of ``'earth'`` in the ephemeris
layer. ``'l2'`` is not a body, and Roman's Horizons id is not in the
alias table. An unknown name is sent to Horizons as typed. Planets
and ``'earth'`` use Astropy's built-in kernel and do not need a
network.

Per-filter parameters
---------------------

Filter-indexed parameters are specified in each model using 
two parallel lists on the parameter class. 

.. code-block:: python

   # PSPL_PhotAstromParam1
   fitter_param_names = [
       'mL', 't0', 'beta', 'dL', 'dL_dS',
       'xS0_E', 'xS0_N',
       'muL_E', 'muL_N', 'muS_E', 'muS_N',
       'b_sff', 'mag_src',
   ]
   filt_param_names = ['xS0_E', 'xS0_N', 'b_sff', 'mag_src']
   filt_param_usage = ['astrom', 'astrom', 'both', 'phot']

Usage is ``'phot'``, ``'astrom'``, or ``'both'``.

* ``'astrom'`` (``xS0``, ``pi_ref_frame``) is fixed at 0 when that
  filter has no astrometry.
* ``'phot'`` (magnitudes) is fixed at 0 when that filter has no
  photometry.
* ``'both'`` (the source flux fraction ``b_sff``, and
  ``dmag_Lp_Ls``, ``fratio_bin``, ``mag_src_sec``) is sampled
  whenever the filter is in the list.

``phot_param_names`` and ``astrom_param_names`` remain as read-only
views. They overlap on ``'both'``. Assigning either name on a class
body raises ``TypeError``. Cube building, priors, packing, and the
JAX likelihood read the parallel lists.

How the fitter cube is expanded
-------------------------------

``expand_fitter_names`` walks ``fitter_param_names``. A shared name
stays once. A contiguous run of filter names expands index-major:
every name in the run for filter 1, then filter 2. Suffixes are
1-based and are not renumbered when a slot is fixed. ``fixed_slots``
then drops a suffix the filter's data do not use.

Four joint filters:

.. code-block:: text

   mL, t0, beta, dL, dL_dS,
   xS0_E1, xS0_N1, xS0_E2, xS0_N2, xS0_E3, xS0_N3, xS0_E4, xS0_N4,
   muL_E, muL_N, muS_E, muS_N,
   b_sff1, mag_src1, b_sff2, mag_src2, b_sff3, mag_src3, b_sff4, mag_src4

One joint filter is the historical column order, with ``xS0_E`` and
``xS0_N`` renamed to ``xS0_E1`` and ``xS0_N1``. The photometric tail
does not move.

The mixed list above does not sample ``xS0_E1`` or ``xS0_E2`` (no
astrometry). It does sample ``xS0_E3``, ``xS0_N3``, ``xS0_E4``, and
``xS0_N4``. ``mag_src4`` is absent and held at 0. ``b_sff4`` is
present because the source flux fraction is ``'both'``.

Preferred constructor: ``xS0`` as arrays
----------------------------------------

Pass ``xS0_E`` and ``xS0_N`` as arrays of length ``n_filters``, in
unified order, the same length as the source flux fraction ``b_sff``
and ``mag_src``. ``xS0`` is always ``(n_filters, 2)``, including one
filter. ``xS0[filt_idx]`` is that filter's East/North pair.
``xS0[0, 0]`` is East and ``xS0[0, 1]`` is North for filter 0.
There is no ``phot_data`` argument on the model.

.. code-block:: python

   event = model.PSPL_PhotAstrom_Par_Param1(
       mL, t0, beta, dL, dL / dS,
       xS0_E=[0.0, 0.0, 0.000, 0.012],
       xS0_N=[0.0, 0.0, 0.000, -0.004],
       muL_E=muL_E, muL_N=muL_N,
       muS_E=muS_E, muS_N=muS_N,
       b_sff=[1.0, 1.0, 0.80, 1.0],
       mag_src=[18.5, 17.9, 19.1, 0.0],
       raL=ra_deg, decL=dec_deg,
       obsLocation=['earth', 'spitzer', 'earth', 'gaia'],
   )
   # Keck is filter 2.
   xS_keck = event.xS0[2]

``PSPL_Param.__init__`` stacks ``xS0`` from those arrays. A mismatched
length raises ``RuntimeError``.

The sampled cube still uses suffixed names (``xS0_E1``, ``xS0_E3``).
Priors are one distribution per sampled name. A length-n vector is
not a legal bound for one suffix: set ``fitter.priors['xS0_E1']``
and ``fitter.priors['xS0_E3']`` separately. A prior whose suffix is
not in the cube warns at the start of ``solve`` and does not replace
the default on the name that is sampled.

``get_model`` takes the same arrays. Element ``k`` is suffix ``k + 1``.

.. code-block:: python

   # Two joint filters. Element k goes to suffix k + 1.
   params['xS0_E'] = np.array([0.1, 0.2])
   params['xS0_N'] = np.array([-0.3, -0.4])
   mod = fitter.get_model(params)
   # mod.xS0[0] == [0.1, -0.3]
   # mod.xS0[1] == [0.2, -0.4]

Photometry-only slots stay at the fixed origin. In a fit whose only
astrometry is filter 3, ``xS0[2]`` is the sampled origin. ``xS0[0]``
and ``xS0[1]`` stay 0.

Getters and ``filt_idx``
------------------------

Every getter takes a 0-based ``filt_idx``. That index selects the
observer, the source flux fraction, and the origin together.

.. code-block:: python

   mag_spitzer = event.get_photometry(t_spitz, filt_idx=1)
   x_keck = event.get_astrometry(t_keck, filt_idx=2)
   x_gaia = event.get_astrometry(t_gaia, filt_idx=3)

``get_photometry`` does not read ``xS0``. Calling a getter on a
filter that has no data of that kind is allowed. It is just not part
of the likelihood. ``get_geoproj_ast_params`` reports filter 0's
origin.

JAX Support
------------

``build_explicit_jax_loglik_fn`` builds one class-order vector per
filter inside the traced likelihood. Shared names are read from the
cube. A sampled suffix is read from its own column. A suffix dropped
by ``fix_fit_param`` is the constant in ``fixed_dataset_params``. A
suffix dropped by ``tie_fit_param`` reads the source column, so the
value and the gradient follow the parameter it is tied to. The
parallax table for that block is precomputed with that filter's
observer, not a default Earth. ``evaluate_loglik_jax``,
``grad_loglik_jax``, the JaxLike solver, PyMC with JAX gradients, and
NumPyro all use this function.

``build_jax_joint_likelihood_context`` is an older helper. It still
packs one base vector and, for a filter-indexed name, uses suffix
``1``. It returns ``None`` when suffix ``1`` was fixed.

``tie_fit_param`` and ``fix_fit_param``
---------------------------------------

``fix_fit_param`` removes a name from the cube and holds it at a
constant. ``tie_fit_param`` removes a suffix and, on every model
build, copies another suffix of the same parameter into that slot.
``n_dims`` shrinks by one. Call either method before ``solve``.

In ``test_model_fitter_multi_location.py``, the 4p2a fit has two 
astrometric tracks. The second is filter 3.
Tying it to filter 1 makes both tracks share one fitted origin, which
is what ``test_lumlens_parallax_fit_4p2a`` does:

.. code-block:: python

   # Hold filter 3 at one number.
   fitter.fix_fit_param('xS0_E3', 0.012)
   fitter.fix_fit_param('xS0_N3', -0.034)

   # Or copy filter 1 on every draw.
   fitter.tie_fit_param('xS0_E3', 'xS0_E1')
   fitter.tie_fit_param('xS0_N3', 'xS0_N1')

After the tie, ``get_model`` writes the same East/North pair into
``xS0[0]`` and ``xS0[2]``. The JAX likelihood does the same copy
inside the traced function.

Backward compatibility
----------------------

A scalar ``xS0_E`` or ``xS0_N``, and an unsuffixed legacy name in an
old results table, are still accepted. That path is kept so older
scripts and chains run. It is not encouraged.

Migration
---------

1. Build the model with ``xS0_E`` and ``xS0_N`` as arrays, one entry
   per filter, the same length as ``b_sff`` and ``mag_src``. Read
   each origin as ``xS0[filt_idx]``.
2. Set priors on the suffixed cube names (``xS0_E1``, ``xS0_N1``).
3. Put real dataset names in ``phot_data`` and ``ast_data`` when more
   than one catalog is present. Set ``obsLocation`` when any filter
   is not Earth. ``getdata(..., obs_location='auto')`` covers the
   usual OGLE / Keck / Spitzer names.
4. Pass ``filt_idx`` into every getter. Do not assume ``t_ast1`` is
   filter 0 once the name lists differ.
5. To rebuild a model, pass ``xS0_E`` and ``xS0_N`` as arrays of
   length ``n_filters``, or the suffixed keys. ``xS0[filt_idx]``
   is that filter's East/North origin.

Known limitations
-----------------

* Astrometric optional parameters (GP hyperparameters on the track)
  keep their ``ast_data`` numbering. Suffix ``1`` is the first
  astrometric series, not necessarily unified filter 1. Photometric
  optional parameters use the unified filter index, which matches
  the photometric index because photometric names are a prefix.
* ``get_geoproj_ast_params`` reports filter 0's origin.
* ``build_jax_joint_likelihood_context`` reads suffix ``1`` only.
