# Multi-observer parallax in one BAGLE model

Planning note for Jessica Lu. No library code has been changed. This is a
design to iterate on before implementation. This revision is v4.

## Changelog versus v3

- Filter-indexed parameters are declared as two parallel lists,
  `filt_param_names` and `filt_param_usage` (`'phot'`, `'astrom'`, or
  `'both'`). `b_sff` is `'both'`, because the blended centroid reads
  it. The v3 line that `b_sff` stays in `phot_param_names` only is
  withdrawn, and so is `phot_zeropoint_names`.
- Fixed versus sampled follows usage alone. `'astrom'` is fixed when
  the filter has no astrometry (`xS0` and `pi_ref_frame` at 0).
  `'phot'` is fixed when the filter has no photometry (magnitude
  zeropoints at 0, the v3 values). `'both'` is sampled whenever the
  filter has either, which is every filter in the list. That covers
  `b_sff`, `dmag_Lp_Ls`, `fratio_bin`, and `mag_src_sec`.
- `phot_param_names` and `astrom_param_names` remain as read-only
  derived views. They overlap on `'both'`. Cube building, priors,
  packing, JAX, and the fixed-slot rule read the parallel lists, not
  those views. Concatenating `fitter_param_names + phot_param_names`
  would sample `b_sff` twice once the photometric names live inside
  `fitter_param_names`.
- Cube order is unchanged from v3. `filt_param_names` is not one block
  glued on the end in the order `b_sff`, `mag_src`, `xS0`. Those names
  stay where they already sit in `fitter_param_names`, and each
  contiguous run expands index-major. `pi_ref_frame` stays after
  `muS_N`, where RefPar puts it today. Moving it next to `xS0_N`,
  which v3 suggested, would slide it in front of `muS` on a one-filter
  chain.
- The audit below is the full name/usage table. Orbit, GP, geoproj,
  and concrete leaf classes inherit it. `model_jax.py` matches
  `model.py` on every shared parameter class. It does not yet define
  the RefPar mixins or `FSPL_PhotParam1`.

The short version: one barycentric event, one list of filters, and a
catalog zero point on each filter that has astrometry. A filter with
only a light curve does not get a free zero point. A filter with only
a track still gets a blend. Annual parallax and satellite parallax
are still the same ephemeris vector.

---


## 1. How the code works today

### Model layout

Instantiable models are mixins, declared in `src/bagle/model.py` and
copied in `src/bagle/model_jax.py`. A typical class is

```text
PSPL_PhotAstrom_Par_Param1(ModelClassABC, PSPL_PhotAstrom,
                           PSPL_Parallax, PSPL_PhotAstromParam1)
```

The four roles are:

- A data class (`PSPL`, `PSPL_Phot`, `PSPL_PhotAstrom`, `PSPL_Astrom`,
  and the PSBL / BSPL / BSBL / FSPL analogues) implements
  `get_photometry`, `get_astrometry`, `get_lens_astrometry`,
  `get_source_astrometry_unlensed`, `get_resolved_astrometry`, `get_u`,
  and the log-likelihood wrappers. Every one of these takes `filt_idx`
  (0-based).
- A parallax class sets `parallaxFlag`. `PSPL_Parallax` also sets
  `fixed_param_names = ['raL', 'decL']` and
  `fixed_phot_param_names = ['obsLocation']`. `PSPL_noParallax` sets
  the flag false. `PSPL_Parallax_RefFrame` adds one catalog parallax
  `pi_ref_frame` on top of the same vector.
- A parameter class stores `t0`, `u0`, `piE`, `b_sff`, `mag_src`, and,
  for astrometric parameterizations, a single `xS0_E`, `xS0_N`. It
  accepts `obsLocation='earth'`.
- `PSPL_Param.__init__` broadcasts a scalar `obsLocation` to one entry
  per photometric parameter. It does not know about astrometric
  datasets.

`model_jax.py` is a parallel copy. Its base `PSPL` methods build a
host-side parallax table in `_parallax_vectors_for_jax` and hand it to
`jax_physics`. Subclasses that override `get_u` or the contour
integrators still call the NumPy ephemeris directly, in both files.

`PSPL_Parallax` is listed before the parameter mixin, so attribute
lookup finds `fixed_phot_param_names = ['obsLocation']` rather than the
empty list on `PSPL_Param`. Keep that base order: data class, then
parallax class, then parameter class.

### What an observer is today

`obsLocation` is a body name, or a sequence of names aligned with the
photometric filters. The default is `'earth'`. After `PSPL_Param`
normalizes a bare string, the PSPL geometry methods index
`self.obsLocation[filt_idx]` and pass it to
`parallax.parallax_in_direction`. That includes amplification,
photometry, the lens track, the unlensed source, both images, the
unresolved centroid, and the centroid shift. PSBL's `get_complex_pos`
and the Keplerian lens branch do the same.

A hand-built PSPL or PSBL model with
`obsLocation=['earth', 'spitzer']` already returns a Spitzer light
curve from `get_photometry(t, filt_idx=1)`. The list length is not
checked. A bare Python string is safe only because `__init__` repeats
it; indexing the string itself would yield `'e'`. The JAX helper
treats a bare string as one location and indexes a list. The NumPy
call sites do not.

### Where the parallax vector comes from

`src/bagle/parallax.py` is the only ephemeris layer.

`parallax_in_direction(RA, Dec, mjd, obsLocation='earth')` is wrapped
in a `joblib.Memory` cache (`PARALLAX_CACHE_DIR`, otherwise
`src/bagle/parallax_cache/`, 1 GB). On a miss it builds a TDB time,
projects `(RA, Dec)` onto a local East/North basis, fetches the
observer's barycentric position, flips the sign (`-obs_pos`, so the
vector points from the observer toward the Solar System barycenter),
converts to AU, and returns shape `(N_times, 2)`. An older
heliocentric branch (Sun minus observer) is commented out. The live
convention is barycentric. `dparallax_dt_in_direction` is used only by
the geocentric-projected conversion.

`get_observer_barycentric` uses Astropy's JPL kernel
(`solar_system_ephemeris.set('jpl')`) when the name is in
`solar_system_ephemeris.bodies`. That covers lowercase `'earth'`, the
Moon, and the planets, with no network. Any other name goes to JPL
Horizons (`location='@0'`) at a step of at least one day, then a linear
interpolation. That is the path for `'spitzer'`, `'jwst'`, `'gaia'`,
`'kepler'`, and Roman. The joblib cache stores the projected East/North
table, and the key includes the whole MJD array. A new plotting grid
is a cache miss.

`src/bagle/ephem/ephem_JWST.txt` is a Horizons dump for JWST. Nothing
reads it.

Nothing is precomputed at model `__init__`. Times show up only when a
getter is called. During a fit the data times are fixed, so each
`(raL, decL, time array, body)` is fetched once and then unpickled.
`raL` and `decL` are fixed parameters, so the projection does not
change from sample to sample.

### How datasets are indexed today

Photometric series are `t_phot1`, `mag1`, `mag_err1`, then `t_phot2`,
and so on. Astrometric series are `t_ast1`, `xpos1`, `ypos1`, and
their errors. `phot_data` and `ast_data` are lists of names.
`MicrolensSolver.setup_params` requires every astrometric name to occur
in `phot_data`. It stores `map_phot_idx_to_ast_idx[i] = phot_index` for
astrometric dataset `i`. The attribute name runs backwards from the
contents: the list is indexed by astrometric dataset and the value is
a photometric index. Astrometry-only input is a hard error.

The NumPy likelihood passes `filt_idx=i` for photometry and
`filt_idx=map_phot_idx_to_ast_idx[i]` for astrometry. One index
therefore selects the observer, the blend, and (today) the single
shared `xS0`. `PSPL_Astrom` documents `b_sff` but the astrometry-only
parameter classes do not sample it, while `PSPL.get_astrometry` still
reads `self.b_sff[filt_idx]`. Mixed fits are not representable.

`model_fitter.plot_astrometry` already separates `data_filt_index`
from the model `filt_index`. `plot_models.py` calls getters with the
default `filt_idx=0`.

### The fitter never applies `data['obsLocation']`

`get_model` passes `fixed_param_names`, which is `['raL', 'decL']`.
`obsLocation` lives only on `fixed_phot_param_names`, and neither
fitter reads that list. The model is built with the default
`'earth'`. `generate_fixed_params_dict` has a branch that would drop
a missing `obsLocation`, but that branch never runs.
`fake_data.fake_data_parallax_multi_location` writes a three-element
list onto the data dictionary, and `test_multi_obsLocation` builds the
comparison models by hand. The likelihood inside the solver does not
see the list. The same hole exists for MultiNest, dynesty, PyMC, and
NumPyro, on both fitter modules.

### Sampled parameter order today

On the parameter class, `fitter_param_names` is the shared geometric
list and does **not** include `b_sff` or `mag_src`.
`phot_param_names` is a separate list. `setup_params` builds the cube
as

```text
model_class.fitter_param_names
    + for each photometric dataset, in order:
          each phot_param_names entry with a 1-based suffix
    + optional astrometric parameters, same idea
```

For `PSPL_PhotAstromParam1` with two photometric datasets the cube is

```text
mL, t0, beta, dL, dL_dS, xS0_E, xS0_N,
muL_E, muL_N, muS_E, muS_N,
b_sff1, mag_src1, b_sff2, mag_src2
```

`xS0_E` and `xS0_N` appear once, unsuffixed, no matter how many
astrometric datasets there are. Photometric parameters are grouped by
dataset (index-major), not by name (`b_sff1, mag_src1` before
`b_sff2`). `make_default_priors` strips a trailing filter number only
when the base name is in the global `multi_filt_params` list. `xS0_E`
is not on that list. The `make_xS0_gen` branch then demands the exact
string `'xS0_E'` or `'xS0_N'` and always reads `xpos1` / `ypos1`.

`generate_params_dict` groups `b_sff1`, `mag_src1`, … back into lists
with a hard-coded `multi_list`. `get_model` splats those lists into
the constructor positionally, in `fitter_param_names` order, then the
photometric lists. `load_mnest_results` renames `col3` onward using
the live `all_param_names`. MultiNest's `.txt` is positional. A `.fits`
written by an older solver already has column names. PyMC and NumPyro
walk `self.fitter_param_names` (the expanded solver list) for priors
and for the parameter vector. The JAX builder takes
`model_class.fitter_param_names` as a geometric cube and
`names.index` of those unsuffixed strings. Per-filter `b_sff` and
`mag_src` are extra arguments, already indexed.

### JAX likelihoods always precompute Earth

`jax_physics.precompute_parallax_vectors(..., obs_location='earth')`
calls `parallax_in_direction` and returns a JAX array. The trajectory
kernels multiply that table by `piE`, `piS`, or `piL`. The observer
is not an argument of the kernel.

These builders omit `obs_location`, so the table is Earth:

- `model_fitter_jax.build_explicit_jax_loglik_fn` (this is what
  `evaluate_loglik_jax`, `grad_loglik_jax`, `MicrolensSolverJaxLike`,
  PyMC with JAX gradients, and NumPyro sample with)
- `jax_physics.build_jax_phot_likelihood_context`
- `jax_physics.build_jax_joint_likelihood_context`

Forward methods on a `model_jax` instance do honor `filt_idx`. A plot
script is fine. A JAX fit is not. PyMC with `use_jax_grad=True` will
mix a NumPy likelihood and a JAX gradient that disagree as soon as
only one of those paths learns the observer. The closure cache
`fitter._explicit_jax_loglik_cache` is not keyed on the data.

`jax_log_likely_astrometry` zips the geometric vector against
`cls.fitter_param_names`, which today includes one `xS0_E` and one
`xS0_N`. That zip is the piece that has to learn per-dataset zero
points. `b_sff` is already passed per call.

### Call sites that drop `filt_idx`

These evaluate observer 0 regardless of the requested filter. The same
patterns are in `model_jax.py`.

- `BSPL.get_u`, when `astrometryFlag` is true, calls
  `get_resolved_source_astrometry_unlensed(t)` and
  `get_lens_astrometry(t)` with no index. `BSPL.get_photometry` goes
  through `get_u`. Photometry-only BSPL (flag false) does pass the
  index. Every BSPL photometry-plus-astrometry model uses the first
  observer for every filter.
- `BSBL.get_u` has the same split.
- `BSPL_Parallax.get_amplification(self, t)` and the no-parallax twin
  omit the index. On the standard concrete classes, `BSPL.get_amplification`
  comes earlier in the MRO and is the one that runs. The override is
  dead, and it has the wrong signature if the MRO changes.
- `FSPL.get_all_arrays_amg` accepts `filt_idx` and then calls
  `get_lens_astrometry(t)` without it. The limb and binary-source
  contour paths do the same.
- `MicrolensSolverWeighted.log_likely_astrometry` in
  `model_fitter_jax.py` omits `filt_idx`. The base solver passes the
  mapped index. Weighted multi-filter astrometry already uses filter
  0's blend for every astrometric dataset.

`tests/test_model.py` and `tests/test_model_jax.py` compare one
satellite (`jwst`, `spitzer`, `gaia`) against a separate Earth model.
They do not put two observers, or two zero points, on one object.

### Fake data

`fake_data_parallax_multi_location` builds one
`PSPL_PhotAstrom_Par_Param1` with three observers, photometry from
each, and astrometry from the third at `filt_idx=2`. It then sets
`phot_data` and `ast_data` to the string `'sim'`. `setup_params`
iterates that string character by character, so the map is not `[2]`.
The bulge wrapper's third observer is also `'earth'`, so pairing the
astrometry with filter 0 is accidentally the same body.
`data.getdata` never sets `obsLocation`.

### Frame conversion, and how `xL0` is made

`frame_convert.convert_helio_geo_phot` and `convert_helio_geo_ast`
match Earth's position and velocity at one epoch `t0par`. They call
the parallax helpers with the default body, Earth. The `*_geoproj`
classes convert once at `__init__` and then store barycentric
parameters. The function names say heliocentric. The vector they
subtract is the barycentric `pvec`. Leave that naming mismatch alone.

Every astrometric parameterization sets

```text
thetaS0 = u0 * thetaE          # mas, source minus lens, shared
xL0     = xS0 - thetaS0 * 1e-3 # arcsec
```

`thetaS0` is the barycentric separation at `t0`. `xS0` is an absolute
sky position at `t0` in whatever frame the astrometry was measured.
Binary-source classes then set `xS0_pri = xS0` and
`xS0_sec = xS0_pri + sep_vec`, or an equivalent center-of-mass split.
The secondary is not an independent zero point. Lens-binary positions
are offsets from `xL0` plus the orbit. Resolved images are offsets
from the lens, so they inherit `xL0`.

---

## 2. User-facing API

### One list of filters

A fit is one list of filters. Each entry is photometry only, astrometry
only, or both. The join key is the dataset name. The fitter builds the
list; the model never sees the names.

Order, which keeps existing photometric indices stable:

1. Every name in `phot_data`, in that order.
2. Then every name in `ast_data` that is not already in `phot_data`,
   in `ast_data` order.

A repeated name inside one list is a `ValueError`. A name in both
lists is one joint filter, not two. `'Keck'` and `'keck_Kp'` are two
filters; the strings have to match.

```python
data['phot_data'] = ['ogle_I', 'spitzer_ch1', 'keck_Kp']
data['ast_data'] = ['keck_Kp', 'gaia']
```

The unified list is `ogle_I`, `spitzer_ch1`, `keck_Kp`, `gaia`.
Indices:

- 0, OGLE, photometry only. `xS0` fixed at 0. `b_sff` and `mag_src`
  sampled.
- 1, Spitzer, photometry only. Same.
- 2, Keck, joint. `xS0`, `b_sff`, and `mag_src` all sampled.
- 3, Gaia, astrometry only. `xS0` and `b_sff` sampled. `mag_src`
  fixed at the placeholder below.

On-disk array numbers do not change. `t_phot1` / `mag1` are still
`phot_data[0]`. `t_ast1` / `xpos1` are still `ast_data[0]`. In the
example, Keck astrometry is `t_ast1` and model `filt_idx=2`. Gaia is
`t_ast2` and `filt_idx=3`. The fitter translates. Users should not
assume `t_ast1` is filter 0 once the lists differ. Renaming the arrays
so the suffix equals `filt_idx` would break every existing reduction,
so the arrays stay numbered inside their own list.

`obsLocation` accepts three forms, all aligned with this one list.

- A string. Every filter uses that body. `'earth'` remains the
  default when the key is absent.
- A dict keyed by dataset name. This is the form to use when the
  collection is mixed. A missing name is an error. A joint name has
  one entry.
- A list of length `n_filters`, in the unified order above (photometric
  names, then astrometry-only names).

`'l2'` is not a body. Roman at L2 means Roman's spacecraft ephemeris.
A small alias table in `parallax.py` folds `'Earth'` to `'earth'` and
maps mission nicknames onto the Horizons id you choose. Unknown names
are sent to Horizons as typed.

### Constructors

The constructor stays close to today's. `b_sff`, `mag_src`, `xS0_E`,
`xS0_N`, and `obsLocation` are sequences of length `n_filters`, in
the same order. A scalar still means "repeat this to every filter",
which is how a one-zero-point call works today and how a hand-built
model can share one `xS0`. There is no `phot_data`, `ast_data`,
`ast_idx`, or `b_sff_astonly` argument. Names are a fitter concern:
the model is a function of parameters and `filt_idx`, and it will
evaluate any index it is given. The fitter decides which of those
values are fixed.

```python
event = model.PSPL_PhotAstrom_Par_Param1(
    mL, t0, beta, dL, dL / dS,
    xS0_E=[0.0, 0.0, 0.000, 0.012],
    xS0_N=[0.0, 0.0, 0.000, -0.004],
    muL_E, muL_N, muS_E, muS_N,
    b_sff=[1.0, 1.0, 0.80, 1.0],
    mag_src=[18.5, 17.9, 19.1, 0.0],
    raL=ra_deg, decL=dec_deg,
    obsLocation=['earth', 'spitzer', 'earth', 'gaia'],
)
```

The last `mag_src` is the astrometry-only placeholder. The first two
`xS0` entries are the photometry-only zeros. A hand-built model may
put other numbers there; the fitter will not, and the light curve
does not read them.

Evaluation uses `filt_idx` only.

```python
mag_spitzer = event.get_photometry(t_spitz, filt_idx=1)
x_keck = event.get_astrometry(t_keck, filt_idx=2)
x_gaia = event.get_astrometry(t_gaia, filt_idx=3)
xL_gaia = event.get_lens_astrometry(t_gaia, filt_idx=3)
images = event.get_resolved_astrometry(t_gaia, filt_idx=3)
```

`get_photometry` does not read `xS0`. `get_astrometry` on a
point-source model does not read `mag_src`. Calling either getter on
a filter that has no data of that kind is allowed; it is just not
part of the likelihood. There is no second index to get wrong, and
no shim that raises.

`get_geoproj_params(t0par)` stays an Earth-at-`t0par` report of the
shared photometric parameters (`t0`, `u0`, `tE`, `piE`). It is not the
integration frame. Per-filter `xS0` is not converted by that helper.

### What is shared, and what is per filter

Shared, one value for the event: `t0`, `u0` / `beta`, `tE`, `piE`,
`thetaE`, masses, distances, `piS`, `piL`, `piRel`, `muS`, `muL`,
`muRel`, binary separation, mass ratio, orbit elements, finite-source
radius.

Per filter, same index: `mag_src`, `mag_base`, `mag_src_pri`,
`mag_src_sec`, `dmag_Lp_Ls`, `fratio_bin`, `b_sff`, `xS0_E`, `xS0_N`,
`pi_ref_frame` on RefPar classes, and the GP / error hyperparameters.
`obsLocation[k]` is fixed, not sampled. `xL0[k]` is computed from
`xS0[k]` and the shared `thetaS0`. It is not sampled.

Which of the per-filter values are actually sampled depends on whether
that filter has photometry, astrometry, or both. That rule is in
section 3. The model object itself always stores the full-length
arrays.

### Sampled names always carry the suffix

Even a one-filter fit writes `xS0_E1` and `xS0_N1`, not `xS0_E` and
`xS0_N`. The same is true of `pi_ref_frame1`. The constructor argument
stays `xS0_E=` (a float or a sequence). Renaming that argument would
break every notebook, and the suffix is a property of the cube, not of
the Python signature. `b_sff1` and `mag_src1` were already suffixed
for one filter; `xS0` joins them.

---

## 3. Internal design

### How the fitter builds the list

`setup_params` in both fitter modules replaces
`map_phot_idx_to_ast_idx`. Nothing in the new design is indexed by a
separate astrometric count.

```python
def build_filt_index(phot_data, ast_data):
    """Join photometric and astrometric names into one filter list.

    Parameters
    ----------
    phot_data : list of str
        Names in ``t_phot`` order. May be empty.
    ast_data : list of str
        Names in ``t_ast`` order. May be empty.

    Returns
    -------
    filt_names : list of str
        Photometric names, then astrometry-only names.
    has_phot : ndarray of bool, shape (n_filters,)
    has_ast : ndarray of bool, shape (n_filters,)
    phot_series : list of int or None
        Index into ``phot_data`` for each filter.
    ast_series : list of int or None
        Index into ``ast_data`` for each filter.
    """
    # Local imports keep this helper usable from either fitter.
    import numpy as np

    if len(phot_data) != len(set(phot_data)):
        raise ValueError('phot_data has a repeated name')
    if len(ast_data) != len(set(ast_data)):
        raise ValueError('ast_data has a repeated name')

    filt_names = list(phot_data)
    for name in ast_data:
        if name not in filt_names:
            filt_names.append(name)

    n_filters = len(filt_names)
    has_phot = np.zeros(n_filters, dtype=bool)
    has_ast = np.zeros(n_filters, dtype=bool)
    phot_series = [None] * n_filters
    ast_series = [None] * n_filters

    for i, name in enumerate(phot_data):
        k = filt_names.index(name)
        has_phot[k] = True
        phot_series[k] = i
    for j, name in enumerate(ast_data):
        k = filt_names.index(name)
        has_ast[k] = True
        ast_series[k] = j

    return filt_names, has_phot, has_ast, phot_series, ast_series
```

`phot_series[k]` is how the likelihood finds `t_phot{i+1}` and
`mag{i+1}`. `ast_series[k]` is how it finds `t_ast{j+1}` and
`xpos{j+1}`. The model is always called with `filt_idx=k`. A `None`
series means that term is skipped, not that the index is illegal.

`check_data` still requires at least one photometric series when
`paramPhotFlag` is set, and at least one astrometric series when
`paramAstromFlag` is set. A phot+ast model may mix the two. A
phot-only model still warns and ignores astrometry. An ast-only model
still warns and ignores photometry. The old error "every astrometric
dataset must have a photometric partner" is removed.

`obsLocation` is resolved onto `self.obsLocation`, a list of length
`n_filters`, and stored on the model. It is not sampled. A joint name
cannot be given two bodies. `fixed_phot_param_names = ['obsLocation']`
is the wrong hook for this: drop `obsLocation` from that list and pass
the resolved list as a keyword from `get_model`. `raL` and `decL`
stay on `fixed_param_names`.

### Parameter declaration

Each parameter class declares filter-indexed quantities as two
parallel lists. `filt_param_names` is every parameter that has one
value per filter. `filt_param_usage` says who that value is for.
Allowed strings are `'phot'`, `'astrom'`, and `'both'`.

`fitter_param_names` is still the full base order of the cube: shared
scalars and the filter-indexed names, in the order they have today.
Photometric names that today live only in `phot_param_names` are
appended at the end of `fitter_param_names`, in the order that list
has now. `xS0_E` and `xS0_N` stay in the middle, where they already
are. `filt_param_names` is not a second ordering. It must list those
same filter-indexed names in the order they appear in
`fitter_param_names`. Expansion walks `fitter_param_names`, not a
rearranged copy of `filt_param_names`.

For `PSPL_PhotAstromParam1` that is

```python
fitter_param_names = [
    'mL', 't0', 'beta', 'dL', 'dL_dS',
    'xS0_E', 'xS0_N',
    'muL_E', 'muL_N',
    'muS_E', 'muS_N',
    'b_sff', 'mag_src',
]
filt_param_names = ['xS0_E', 'xS0_N', 'b_sff', 'mag_src']
filt_param_usage = ['astrom', 'astrom', 'both', 'phot']
```

`b_sff` is `'both'`. The light curve uses it, and
`PSPL.get_astrometry` weights the centroid with `self.b_sff[filt_idx]`.
Putting it in a photometry-only list was the v3 mistake. `mag_src` is
`'phot'` because the point-source track does not read it. `xS0_E` and
`xS0_N` are `'astrom'`.

A sketch that listed `b_sff`, `mag_src`, `xS0_E`, `xS0_N` as the
expansion order would be the right usages in the wrong sequence. As
one block at the end of the cube it would move `xS0` out of its
historical columns. The lists above keep the usages and leave `xS0`
where it sits.

The expanded list, before anything is fixed, walks
`fitter_param_names`. A name that is not in `filt_param_names` is
shared and stays once. A contiguous run of names that are in
`filt_param_names` expands index-major: every name in the run for
filter 1, then every name in the run for filter 2. Suffixes are
1-based. On `PSPL_PhotAstromParam1` the two runs are `xS0_E`, `xS0_N`
and, later, `b_sff`, `mag_src`. They do not merge across `muL` and
`muS`.

With four joint filters the cube is

```text
mL, t0, beta, dL, dL_dS,
xS0_E1, xS0_N1, xS0_E2, xS0_N2, xS0_E3, xS0_N3, xS0_E4, xS0_N4,
muL_E, muL_N, muS_E, muS_N,
b_sff1, mag_src1, b_sff2, mag_src2, b_sff3, mag_src3, b_sff4, mag_src4
```

With one filter it is the historical cube, except unsuffixed `xS0_E`
and `xS0_N` are named `xS0_E1` and `xS0_N1` in the same columns. The
photometric tail does not move. Name-major order (`b_sff1, b_sff2` or
`xS0_E1, xS0_E2`) would reshuffle existing chains. Do not do that.

RefPar does not move `pi_ref_frame`. Today it sits after `muS_N`:

```text
xS0_E, xS0_N, muS_E, muS_N, pi_ref_frame, ... b_sff, mag_base
```

It is its own run of one name, usage `'astrom'`. With two filters the
cube has `xS0_E1, xS0_N1, xS0_E2, xS0_N2`, then the shared proper
motions, then `pi_ref_frame1, pi_ref_frame2`, then the photometric
block. With one filter this is only a rename of `pi_ref_frame` to
`pi_ref_frame1`, still after `muS_N`. Sliding it next to `xS0_N` would
put it in front of `muS` and break those chains.

### Validation

`PSPL_Param.__init_subclass__` checks every parameter class in both
model files. A unit test walks the subclasses so a leaf that shadows
a list still fails in CI.

- `filt_param_names` and `filt_param_usage` have the same length.
- Every usage is `'phot'`, `'astrom'`, or `'both'`.
- Names are unique, and each one appears exactly once in
  `fitter_param_names`.
- The order of `filt_param_names` equals the order of those names in
  `fitter_param_names`.
- The class body does not assign `phot_param_names` or
  `astrom_param_names`. Those are derived. An assignment would shadow
  the descriptor and the two declarations would drift.

`ModelClassABC.__init_subclass__` runs the same check, so a concrete
class that overrides the lists is caught too. Empty lists are legal.
That is `PSPL_Param` itself.

### Derived views

`phot_param_names` and `astrom_param_names` stay, as read-only
descriptors, so existing readers do not break. They are not stored
and they are not edited.

```text
phot_param_names   = names whose usage is 'phot' or 'both'
astrom_param_names = names whose usage is 'astrom' or 'both'
```

On `PSPL_PhotAstromParam1` the derived phot view is `b_sff`,
`mag_src`, which is today's `phot_param_names`. The derived astrom
view is `xS0_E`, `xS0_N`, `b_sff`. They overlap on `b_sff`. That
overlap is the point of `'both'`. A check that the two views are
disjoint would reject `b_sff` and is removed.

Internal code switches to the parallel lists:

- `PSPL_Param.__init__`, which today broadcasts lengths from
  `phot_param_names` only and therefore never repeats `xS0`.
- `setup_params`, which today appends `phot_param_names` after
  `fitter_param_names`. After the move, appending again would put
  `b_sff` in the cube twice. It expands `fitter_param_names` in place.
- `generate_params_dict` and `get_model`, which pack one sequence per
  name in `filt_param_names`.
- `make_default_priors` and `multi_filt_params`. The global multi-filter
  list is the union of `filt_param_names` across classes, including
  `xS0_E`, `xS0_N`, and `pi_ref_frame`.
- The JAX geometric zip. The scalar vector is
  `fitter_param_names` minus `filt_param_names`.
- The fixed-slot rule below.

The derived views are for inspection and for the old-file loader,
whose photometric block is exactly usage `'phot'` or `'both'` and does
not include `xS0`. Do not expand both derived views. `b_sff` would be
sampled twice.

### What is fixed, and what is sampled

Usage is the only rule. After the index-major expansion, the fitter
drops a suffixed name whose filter does not have the data that usage
requires. The suffix is not renumbered. `xS0_E3` stays `xS0_E3` when
`xS0_E1` was removed.

For filter `k` (suffix `k+1`):

- Usage `'astrom'`, and `has_ast[k]` is false. Fix the value at 0 and
  omit it. That is `xS0_E`, `xS0_N`, and `pi_ref_frame`. They do not
  enter `get_photometry` or `get_u`.
- Usage `'phot'`, and `has_phot[k]` is false. Fix the value at 0 and
  omit it. That is `mag_src`, `mag_base`, and `mag_src_pri`. The 0 is
  the v3 placeholder: not a bulge magnitude, finite, and loud if a
  photometry term is included by mistake.
- Usage `'both'`. Sample it whenever the filter has photometry or
  astrometry. Every name in the unified list has one or the other, so
  a `'both'` parameter is sampled on every filter. That is `b_sff`,
  `dmag_Lp_Ls`, `fratio_bin`, and `mag_src_sec`.

`dmag_Lp_Ls` is `'both'` because `PSBL.get_lens_astrometry` turns it
into the flux ratio of the two lenses. `fratio_bin` and `mag_src_sec`
are `'both'` because the binary-source centroid uses the flux ratio.
Fixing `mag_src_pri` and `mag_src_sec` both at 0 would freeze that
ratio at 1. `mag_src_pri` stays `'phot'` so the absolute zeropoint is
the fixed one and `mag_src_sec` carries the ratio.

GP hyperparameters, `add_err`, and `mult_err` are not in
`filt_param_names`. They stay on `phot_optional_param_names` and are
created only for filters with photometry. They are absent on an
astrometry-only filter, not fixed at 0. The `single_gp` switch is
unchanged. The prior on `mag_src_sec` for a filter with no light curve
cannot call `make_mag_base_gen`. Use a uniform prior on `[-5, 5]`.
`fratio_bin` and `dmag_Lp_Ls` keep their existing priors when those
priors do not read a magnitude array; otherwise the same wide uniform.

The OGLE / Spitzer / Keck / Gaia example, after deletion:

```text
mL, t0, beta, dL, dL_dS,
xS0_E3, xS0_N3, xS0_E4, xS0_N4,
muL_E, muL_N, muS_E, muS_N,
b_sff1, mag_src1, b_sff2, mag_src2, b_sff3, mag_src3, b_sff4
```

`xS0_E1`, `xS0_N1`, `xS0_E2`, `xS0_N2`, and `mag_src4` are absent.
`b_sff4` remains, because its usage is `'both'`. The fixed values live
in `fixed_dataset_params`, a dict of suffixed name to float, not in
`data[]` and not in `fixed_param_names`. `raL` and `decL` stay on the
data-dict path they have today. `generate_fixed_params_dict` is not
the right place for a zero that the data file does not contain.

`n_dims` is the length of the sampled list. That list is what
`Prior`, `dyn_prior`, `Prior_copy`, `Prior_from_post`, `log_likely`,
PyMC's stacked priors, and NumPyro's parameter vector iterate. A
positional truth array follows this sampled order. A dict keyed by
name is safer in tests, because a hole cannot slide `muL` into an
`xS0` slot. `get_model` accepts both. Before the constructor is
called, the sampled values and `fixed_dataset_params` are merged into
full-length sequences, one per `filt_param_names` entry. The
constructor always sees `xS0_E` of length `n_filters`, with zeros in
the photometry-only slots.

`all_param_names` is the sampled list plus derived names. Report
`xL0_E{k}` and `xL0_N{k}` only for filters with astrometry. A
photometry-only lens origin would be `-thetaS0` in a frame the data
never used.

### Who has to change, by family

Do this with a mechanical pass over both `model.py` and `model_jax.py`.
Orbit subclasses, GP mixins, geoproj subclasses, and the concrete
leaves inherit the lists. They do not redeclare them. The tables are
the classes that define the lists today, plus the usage to attach.
`xS0_*` and `pi_ref_frame` are written in `fitter_param_names` order,
then the names that today are only in `phot_param_names`, in that
list's current order.

`model_jax.py` has the same `fitter_param_names` and `phot_param_names`
assignments on every class it shares with `model.py`. It does not
define `PSPL_PhotAstromParam3_RefPar`, `PSPL_PhotAstromParam4_RefPar`,
their four GP variants, or `FSPL_PhotParam1`. Those seven get the same
lists when they are ported. `FSPL_Limb_PhotAstromParam1` is marked do
not use and has no `fitter_param_names`. Leave it out.

Photometry only, no `xS0`. Usage `'both'`, `'phot'` for
`b_sff`, `mag_src`.

| Classes | `filt_param_names` | usage |
| --- | --- | --- |
| `PSPL_PhotParam1`, `PSPL_PhotParam1_geoproj`, `PSBL_PhotParam1`, `PSBL_Phot_EllOrbs_Param1`, `PSBL_Phot_CircOrbs_Param1`, `FSPL_PhotParam1` | `b_sff`, `mag_src` | both, phot |
| `PSPL_PhotParam2`, `PSPL_PhotParam3`, `FSPL_PhotParam2` | `b_sff`, `mag_base` | both, phot |
| `BSPL_PhotParam1` | `mag_src_pri`, `mag_src_sec`, `b_sff` | phot, both, both |
| `BSBL_PhotParam1` | `mag_src_pri`, `mag_src_sec`, `b_sff`, `dmag_Lp_Ls` | phot, both, both, both |

Point source, photometry plus astrometry. `xS0_E`, `xS0_N` are
`'astrom'`. `b_sff` is `'both'`. The magnitude is `'phot'`.

| Classes | after the two `xS0` names | usage of that tail |
| --- | --- | --- |
| `PSPL_PhotAstromParam1`, `Param2`, `FSPL_PhotAstromParam1`, `FSPL_PhotAstromParam2`, `BFSPL_PhotAstromParam1` | `b_sff`, `mag_src` | both, phot |
| `PSPL_PhotAstromParam3`, `Param4`, `Param4_geoproj`, `Param5`, `Param6` | `b_sff`, `mag_base` | both, phot |

`BFSPL_PhotAstromParam1` also samples a shared `radiusS`. That name
stays out of `filt_param_names`.

RefPar, `model.py` only today. Same as `PSPL_PhotAstromParam3` or
`Param4`, plus `pi_ref_frame` with usage `'astrom'`, placed where it
already is, after `muS_N`. The GP RefPar classes inherit this.
`filt_param_names` order is `xS0_E`, `xS0_N`, `pi_ref_frame`, `b_sff`,
`mag_base`.

Astrometry only. `PSPL_AstromParam3` and `PSPL_AstromParam4` have
`xS0` and no magnitude. Add `b_sff` at the end of `fitter_param_names`,
usage `'both'`, because the getter already reads it and the class does
not sample it today. There is no `'phot'` name to fix. Old
astrometry-only chains have no `b_sff` column. The loader does not
invent one; those files need a re-run, or an explicit fixed `b_sff`
supplied by the caller. A new fit samples `b_sff1`.

Lens binaries, PSBL, all orbit flavors. `xS0_E`, `xS0_N` are
`'astrom'`. Do not put `xL1` or `xL2` in `filt_param_names`.
`xL0[k] = xS0[k] - thetaS0 * 1e-3` stays derived.

| Classes | tail after `xS0` | usage |
| --- | --- | --- |
| `PSBL_PhotAstromParam1`, `Param2`, `Param6`, `Param7`, `Param8`, and the `LinOrbs` / `AccOrbs` / `EllOrbs` / `CircOrbs` classes that copy those photometric lists, plus `EllOrbs_Param4` and `CircOrbs_Param4` | `b_sff`, `mag_src`, `dmag_Lp_Ls` | both, phot, both |
| `PSBL_PhotAstromParam3`, `Param4`, `Param5`, and their orbit copies | `b_sff`, `mag_base`, `dmag_Lp_Ls` | both, phot, both |

Source binaries, BSPL. The free zero point is the `xS0` the class
already samples. `xS0_pri[k]`, `xS0_sec[k]`, and `xS0_com[k]` are
derived from that plus the shared separation or orbit. Do not sample
a second absolute position per filter.

| Classes | tail after `xS0` | usage |
| --- | --- | --- |
| `BSPL_PhotAstromParam1` and its `LinOrbs`, `AccOrbs`, `EllOrbs`, `CircOrbs` copies (`EllOrbs_Param1`, `EllOrbs_Param4`, `CircOrbs_Param1`) | `mag_src_pri`, `mag_src_sec`, `b_sff` | phot, both, both |
| `BSPL_PhotAstromParam2`, `Param3`, and their orbit copies (`EllOrbs_Param2`, `EllOrbs_Param3`, `CircOrbs_Param2`, `CircOrbs_Param3`) | `fratio_bin`, `mag_base`, `b_sff` | both, phot, both |

Source and lens binaries, BSBL. Same `xS0` rule. `dmag_Lp_Ls` is
`'both'`.

| Classes | tail after `xS0` | usage |
| --- | --- | --- |
| `BSBL_PhotAstromParam1`, `Param2`, `LinOrbs_Param1`, `AccOrbs_Param1`, `EllOrbs_Param1`, `CircOrbs_Param1`, `EllOrbs_Param3`, `CircOrbs_Param3` | `mag_src_pri`, `mag_src_sec`, `b_sff`, `dmag_Lp_Ls` | phot, both, both, both |
| `BSBL_PhotAstrom_EllOrbs_Param2`, `CircOrbs_Param2` | `fratio_bin`, `mag_base`, `b_sff`, `dmag_Lp_Ls` | both, phot, both, both |

Finite source other than the classes already in the point-source
table. No new sampled names. Contour methods must pass the `filt_idx`
they already accepted into `get_lens_astrometry`. The outline radius
is shared.

GP mixins add no rows. Their optional hyperparameters stay
`phot_optional_param_names` and are created only for filters with
photometry.

`PSPL_Param.__init__` normalizes every name in `filt_param_names` to
one length, `n_filters`. A scalar is repeated to that length once any
list has set it. That includes `xS0`. The fitter does not rely on this
broadcast for mixed fits; it passes an explicit list with the fixed
zeros filled in, so a photometry-only slot cannot inherit a sampled
neighbor's zero point. Store

```python
self.xS0 = np.stack([self.xS0_E, self.xS0_N], axis=1)  # (n_filt, 2)
self.xL0 = self.xS0 - (self.thetaS0 * 1e-3)            # (n_filt, 2)
```

`thetaS0` is shape `(2,)` and broadcasts. `self.pi_ref_frame` is shape
`(n_filters,)` on RefPar classes.

### Getters

`get_parallax_vectors(self, t, obs_location)` lives once, on `PSPL`, in
both model files, and calls `parallax.parallax_in_direction`.
`resolve_obs_location` in `parallax.py` turns a string, list, or dict
into the body for one filter and never indexes a character of
`'earth'`. Every getter that needs an observer uses
`self.obsLocation[filt_idx]`.

`get_photometry`, `get_amplification`, and `get_u` for a light curve
use that observer and `b_sff[filt_idx]`. They do not read `xS0`.
`u(t)` is

```text
u0 + ((t - t0) / tE) * thetaE_hat - piE_amp * pvec(t, obsLocation[k])
```

`get_astrometry`, `get_lens_astrometry`,
`get_source_astrometry_unlensed`, `get_resolved_astrometry`, and
`get_centroid_shift` keep `filt_idx` and read `xS0[filt_idx]`.

```text
xS(t, k) = xS0[k] + muS * dt_yr + piS * pvec(t, obsLocation[k])
xL(t, k) = xL0[k] + muL * dt_yr + piL * pvec(t, obsLocation[k])
```

with the usual mas-to-arcsec factor on the parallax term. Because
`xL0[k] = xS0[k] - thetaS0`,

```text
xS - xL = thetaS0 + muRel * dt_yr - piRel * pvec(t, obsLocation[k])
```

The zero point cancels in the separation. Image positions are that
separation, lensed, placed back on the sky by adding `xL(t, k)`, so
they move when `xS0[k]` moves and the image-minus-lens vectors do not.
The unresolved centroid uses `b_sff[filt_idx]`. Reference-frame
parallax adds `pi_ref_frame[k] * pvec` to source and lens together.
It cancels in `u(t)`.

`BSPL.get_u` and `BSBL.get_u` pass `filt_idx` through on the
astrometry branch. Today they drop it and every filter sees observer
0. `FSPL.get_all_arrays_amg` and the limb contour pass `filt_idx` into
`get_lens_astrometry`. Give the dead `BSPL_Parallax.get_amplification`
an index argument and forward it, or delete the override.
`MicrolensSolverWeighted.log_likely_astrometry` must pass `filt_idx`
too. Check the Hobson solver for the same omission.

The joblib cache on `parallax_in_direction` stays the ephemeris cache.
Do not precompute a grid at `__init__`. Horizons remains the spacecraft
backend. The one-day step is enough for Earth, Spitzer, JWST, Gaia,
Kepler, and a Roman halo. Reading `ephem/ephem_JWST.txt` is a later
offline project.

### JAX

Kernels stay observer-agnostic and zero-point-agnostic. The host
builds one `(N_times, 2)` table per filter that has data, from
`obsLocation[k]`.

In `build_explicit_jax_loglik_fn`, and in the two context builders:

- A photometric term for filter `k` uses that table, `b_sff[k]`, and
  the magnitude parameters of filter `k`. Skip the term when
  `has_phot[k]` is false. A fixed zeropoint is a Python float in the
  closure, not a traced parameter.
- An astrometric term for filter `k` uses that table, `xS0_E{k}`,
  `xS0_N{k}`, `pi_ref_frame{k}` when present, and `b_sff[k]`. Skip the
  term when `has_ast[k]` is false. Photometry-only zeros are constants.

The geometric vector that is zipped inside `jax_log_likely_*` is the
scalar subset of `fitter_param_names`: everything not in
`filt_param_names`. Do not subtract the two derived views. They
overlap on `b_sff`, and `xS0` is only in the astrom view, so a zip
built from those views either drops `xS0` from the wrong place or
keeps it in the geometric vector. `xS0_E` and `xS0_N` become
explicit arguments of `jax_log_likely_astrometry`, the way `b_sff`
already is, one pair per filter that has astrometry. `derive_*` is
called with that filter's zero point, so `xL0` is not computed once
from `xS0_E1` and reused for Gaia. Shared quantities (`tE`, `piE`,
`mu`) may be derived once per sample; the zero point may not. Do not
look up `model_class.fitter_param_names` with `names.index` after those
names have been expanded to `xS0_E1` or dropped as fixed.

Do not pass body-name strings into `jax.jit`. `raL` and `decL` stay
fixed; the table is wrong if they become free. `piE`, `piS`, and `piL`
remain traced multiplies. Filters have ragged times, so they stay a
Python loop that JAX unrolls. Do not pad and `vmap` across filters.
Key `_explicit_jax_loglik_cache` on the observer list, the has-phot /
has-ast masks, `raL`, `decL`, and the time-array identities.

NumPy and JAX both call `parallax_in_direction`. Tests compare getters
and likelihoods, not a second ephemeris. PyMC with `use_jax_grad=True`
and NumPyro must see the same fixed zeros and the same tables as the
NumPy `log_likely`. Land the JAX builder in the same merge as
`get_model`.

### Fitter mechanics, both modules

`model_fitter.py` and `model_fitter_jax.py` each carry a full solver.
The following changes land in both, except the JAX builder, which is
only in `model_fitter_jax.py`.

`multi_filt_params` gains `xS0_E`, `xS0_N`, and `pi_ref_frame`.
`make_default_priors` is called only for sampled names, and it keys
off the stripped base name. The `make_xS0_gen` branch today compares
`param_name == 'xS0_E'` and always reads `xpos1`. Change it to match
the base name and to read `xpos` / `ypos` of `ast_series[k]`, not
`xpos{k+1}`, because those suffixes differ when astrometry-only names
were appended. The proper-motion prior stays shared (one `muS`) and
should use the astrometric series with the longest time baseline, not
always `t_ast1`. The `t0` prior uses the first filter that has
photometry, and otherwise the first astrometric series.
`make_invgamma_gen` for a GP hyperparameter uses `t_phot` of that
photometric series. `b_sff` keeps its uniform prior and the existing
upper bound of 1 on any filter that has astrometry.
`_check_b_sff_upper_bound` applies to those filters, including
astrometry-only ones.

`generate_params_dict` packs every name in `filt_param_names` back
into a sequence of length `n_filters`. That is one list for `b_sff`,
one for `mag_src`, one for `xS0_E`, and so on. A suffix that is
missing from the cube is filled from `fixed_dataset_params`.
Do not assume every index from 1 to `n_filters` appears in the sampled
list. `get_model` splats those full sequences into the constructor in
`fitter_param_names` order. `__init__` argument order already matches
if the photometric names are appended at the end and `xS0` stays where
it is. `obsLocation` is a keyword. It is not in the cube.

`log_likely_photometry` loops filters with `has_phot` and calls
`log_likely_photometry(..., filt_idx=k)` on `t_phot` of
`phot_series[k]`. `log_likely_astrometry` loops filters with `has_ast`
and calls `log_likely_astrometry(..., filt_idx=k)` on `t_ast` of
`ast_series[k]`. It does not pass a mapped photometric index.
Weights stay in data-list order: photometric series in `phot_data`
order, then astrometric series in `ast_data` order. That way an
existing weight vector still lines up with the arrays the user
passed. The model index is separate from the weight index, and the
solver translates.

PyMC (`MicrolensSolverPyMC`, including `use_jax_grad=True`) and NumPyro
(`MicrolensSolverNumPyro`) do not have their own parameter lists. They
follow the sampled `self.fitter_param_names` and `self.priors`. Fixed
slots are constants in the likelihood closure, not random variables
and not `pm.Deterministic` nodes. Dynesty and MultiNest use the same
cube. `MicrolensSolverJaxLike` calls `evaluate_loglik_jax`, which must
understand the holes.

### Results files and the loader

New runs write `SCHEMA = 2`. This is the first format that will
actually be written; the v2 two-index idea never reached disk. The
chain columns are the sampled names only, in cube order, holes and
all (`xS0_E3` with no `xS0_E1`). Fixed values are not columns. They go
in a FITS extension `FIXED` with columns `name` and `value`, and the
filter list goes in an extension `FILTERS` (one string column, unified
order). A MultiNest `.txt` gets a JSON sidecar with the same sampled
names, fixed dict, and filter list. Posterior corner plots label the
sampled `all_param_names` and skip the fixed slots, which have no
variance. The summary table includes the fixed parameters with
uncertainty 0 and `fixed=True`. Axis labels should say which filter
the suffix is (`xS0_E3` is `filt_names[2]`).

Rebuilding a model from a saved sample merges the chain row with
`FIXED` before `get_model`, so the constructor again sees full-length
lists.

Old products, by column name:

- Let `N` be the largest 1-based suffix on `b_sff`, `mag_src`,
  `mag_base`, `mag_src_pri`, `mag_src_sec`, `fratio_bin`, or
  `dmag_Lp_Ls`. That is the old filter count. It has to equal the
  destination solver's `n_filters`. Loading an old chain into a run
  that added a new telescope raises; the adapter does not invent a
  slot.
- If `xS0_E1` is already present, keep `xS0_E{k}` as stored. Do not
  overwrite. A missing `k` in a schema-2 file is filled from `FIXED`
  when that name is there, and raises when it is not.
- If the unsuffixed column `xS0_E` is present and `xS0_E1` is not:
  set `xS0_E{k} = xS0_E` for every `k = 1..N`, and the same for
  `xS0_N`. Warn once: the old fit used one zero point, copied into
  every filter. This is the case Jessica specified. An old file with
  several `b_sff` / `mag_src` columns and one `xS0` really did apply
  that single position to every filter's astrometry. Copying it back
  is the faithful reload. Do not then overwrite those copies with 0.
- An unsuffixed `pi_ref_frame` is copied to `pi_ref_frame1` through
  `pi_ref_frameN` the same way, when the destination class is RefPar.
  If the destination is RefPar and the old file has neither the
  unsuffixed name nor `pi_ref_frame1`, raise. If the destination is
  not RefPar, ignore a stray `pi_ref_frame` column.
- If both `xS0_E` and `xS0_E1` exist, raise. The file is ambiguous.
  Same for `pi_ref_frame`.
- Photometric columns are already suffixed. Copy them by exact name.
  Do not replicate them. Index-major order is what makes `b_sff1`
  still mean filter 0.
- `N == 1` and an unsuffixed `xS0_E` is only a rename to `xS0_E1`.

A raw MultiNest `.txt` has no column names. If the sidecar exists and
the column count equals the sampled-name list, load positionally
against that list and fill `FIXED` from the sidecar. If there is no
sidecar, accept the file only when the column count matches the old
layout for this class: the historical unsuffixed geometric list
(one `xS0_E`, one `xS0_N`) plus the photometric block of length
`N` times the derived `phot_param_names` (usage `'phot'` or
`'both'`, which is today's photometric list and does not include
`xS0`). Name those columns the old way, then run
the copy rule above. If the count matches neither the sidecar nor that
old layout, raise. Do not accept a new cube with holes as a bare
positional file: without names, a missing `xS0_E1` is indistinguishable
from a shift. Re-running remains the supported path for an old text
chain whose column count does not match.

### Fake data, the data loader, and plots

`fake_data_parallax_multi_location` writes real name lists, for example
`phot_data=['ogle', 'spitzer', 'keck']` and `ast_data=['keck']`, and an
`obsLocation` dict. The generating model is built with full-length
lists: `xS0` of length `n_filters` (zeros on photometry-only entries
if you want the fitter's convention), and a real `b_sff` on every
entry. Add a noiseless `noise=False` path for the equality tests. Add
one mixed generator: photometry-only Earth, photometry-only Spitzer,
joint ground photometry plus astrometry, and astrometry-only Gaia,
with different `xS0` on the two filters that have astrometry, `xS0=0`
on the photometry-only filters, `mag_src=0` on Gaia, and a sampled
`b_sff` on Gaia.

`data.getdata` grows `obs_location=None`. `None` leaves the key off the
dictionary, and the fitter uses Earth, so existing Spitzer reductions
do not silently change. A dict or `'auto'` opts in. `'auto'` maps
`I_OGLE`, `Kp_Keck`, `HST*`, `MOA`, and `KMT*` to `'earth'` and
`Ch1_Spitzer` to `'spitzer'`. `phot_data` and `ast_data` may overlap
only partly. An astrometry-only request is legal: `t_ast*` is filled
and `t_phot*` is absent. `getdata` does not build the unified index
and does not renumber arrays. The fitter does the join.

`plot_photometry` keeps `filt_index` as this unified index and reads
the data through `phot_series`, not through `mag{filt_index+1}`.
`plot_astrometry` does the same with `ast_series`. There is no
`ast_idx`. `plot_astrometry_multi_filt` loops filters with `has_ast`.
`plot_models` animators keep `filt_idx=0`. A small overlay helper, one
magnitude panel per filter that has photometry and one East/North
panel per filter that has astrometry, labels each panel with
`filt_names[k]` and `obsLocation[k]`. It is a convenience, not part of
the likelihood.

---

## 4. Physics

One set of lens and source parameters describes the event in the Solar
System barycentric frame. Each filter views that event from its own
barycentric ephemeris, and each filter that has astrometry places the
event in its own catalog frame. Annual parallax and satellite parallax
are the same term. Earth's ephemeris is the annual ellipse. A
spacecraft's ephemeris is that motion plus the Earth–spacecraft
vector. There is no second parallax term to add.

### What depends on the observer

Only `pvec(t)`, the East/North projection in AU of the vector from the
observer to the Solar System barycenter, at `(raL, decL)`.

It shifts the light curve through

```text
u(t) = u0 + ((t - t0) / tE) * thetaE_hat - piE_amp * pvec(t)
```

and it shifts the tracks through `piS * pvec` on the source and
`piL * pvec` on the lens. Images, the centroid, and the centroid shift
are built from those. The centroid also depends on `b_sff` of that
same filter, and, for binaries, on the flux ratio of that filter.
Two filters from the same spacecraft differ by blend, zeropoint, and
`xS0`. Two spacecraft differ by those and by `pvec`.

`t0` is the time of closest approach in the barycentric frame. It is
shared. Each observer's magnification peaks at a different time because
`pvec(t)` differs. That peak-time difference is an output. Giving each
filter its own `t0` or `u0` would double-count the parallax.

`calc_piE_ecliptic` projects `piE` onto `pvec(t0)` for one observer.
For Earth that direction is a stand-in for the ecliptic. For a
spacecraft it is not. The method stays diagnostic and observer-specific.
It is not a sampled parameter.

### What `xS0` is, once it is per filter

`xS0[k]` is the catalog position of the source at `t = t0`, in filter
`k`'s frame, before that filter's parallax term is added. It is not a
property of the observer. Two astrometric reductions from the same
spacecraft can have different `xS0` and the same `pvec`. Two spacecraft
can have different `xS0` and different `pvec`.

A photometry-only filter does not have a catalog frame in the fit.
Its `xS0` is fixed at 0 and is not a parameter. `get_photometry` and
`get_u` never read `xS0`, so the light curve is identical to a model
that does not store a zero point for that filter at all. Setting the
fixed value to 0 rather than to some other constant changes nothing
physical. It also does not absorb an Earth–spacecraft offset, because
that filter contributes no astrometric residual. The rest of this
section is about filters whose `xS0` is free.

`xL0` is not free. In filter `k`,

```text
xL0[k] = xS0[k] - thetaS0 * 1e-3
```

`thetaS0` is the shared barycentric source-minus-lens vector at `t0`,
from `u0` and `thetaE`. The lens and the source share a frame. If
`xL0` were sampled on its own, the unresolved centroid's mean position
could trade off against `u0`, and the light curve and the track would
no longer be describing the same separation. Keeping `xL0` slaved is
required.

Resolved image positions on the sky move with `xL0[k]`. Separations
between images, and between an image and the lens, do not. The
unlensed source and the lens both move by the same `xS0[k]`, so a plot
of source and lens in one filter is rigid against that filter's zero
point and flexible against another filter's.

For a source binary, `xS0[k]` is the primary or the center of mass,
whichever that parameterization already uses. The companion is
`xS0[k]` plus the shared separation or the shared orbit evaluated at
`t0`. A Keplerian orbit is integrated in the barycentric frame and then
the whole system is placed at `xS0_com[k]` and given that filter's
`pvec`. The orbital `(x, y)` is not per filter. For a lens binary the
same sentence applies to `xL0[k]` and the lens orbit.

`pi_ref_frame[k]`, sampled on every RefPar filter that has astrometry
and fixed at 0 otherwise, is an extra parallax applied equally to
source and lens in that catalog. It moves the absolute track and
cancels in `u(t)`. It does not replace `xS0`. `xS0` removes a
constant. `pi_ref_frame` removes a parallax-shaped motion of the
reference frame. A photometry-only filter has no track, so a free
`pi_ref_frame` there would be unconstrained; fixing it at 0 changes
no magnification.

### What satellite astrometric parallax still measures

The on-sky difference between two astrometric filters at a single
epoch contains a constant piece,

```text
pi * (pvec_a(t0) - pvec_b(t0))
```

plus the difference of the two zero points. A free `xS0[a]` and
`xS0[b]` absorb every constant vector, including that piece. The mean
positional offset between a ground catalog and a Roman catalog is not
a measurement of the Earth–Roman baseline. Interpreting `xS0`
differences as a parallax, or as a physical source-lens separation, is
wrong.

What is not absorbed:

- The time dependence of `pvec(t)` inside one filter. A free `xS0`
  removes a constant, not the annual ellipse. With a baseline of a
  good fraction of a year, that shape still constrains `piS` and
  `piL`. Proper motion remains partly degenerate with a short arc of
  the ellipse; that is the usual parallax–proper-motion degeneracy,
  and it is not new. It gets worse if an astrometry-only series spans
  only a few days: `xS0` soaks up the mean, and `mu * dt` trades off
  against the parallax arc. The likelihood is not flat, but the
  posterior on `pi` from that series alone will be weak.
- The difference of the time-variable parts. After each track's zero
  point is removed, the residual between Earth and a spacecraft is
  `pi * ((pvec_a - mean_a) - (pvec_b - mean_b))` with a shared `mu`.
  Earth and L2 do not draw the same ellipse. That differential wiggle
  is the satellite-parallax information that survives in the
  astrometry. It is smaller than the full constant baseline, and it
  needs overlapping time coverage and a shared proper motion to be
  useful. State this in the docstring of `xS0` so a fit is not
  over-interpreted.
- The light curves. `u(t)` does not contain `xS0`. Earth plus Spitzer
  photometry still measures `piE` through the different magnifications,
  including when those filters have `xS0` fixed at 0. That channel is
  the robust satellite parallax measurement, and this change does not
  touch it. A joint filter's light curve and its track share `piE`,
  `piRel`, and the observer, so the photometry pins down the parallax
  that the track's shape is also seeing.
- Two astrometric filters from the same observer. They share `pvec`.
  The second filter adds a zero point and, if the blend differs, a
  different centroid weighting. It does not add a new parallax
  baseline. Different blends change the unresolved centroid shift,
  which is a function of `u(t)` and is achromatic in angle but not in
  amplitude. That is lensing astrometry, not satellite parallax.

### Effects this design does not include

Light-travel time between observers is minutes. It is not applied to
the time arrays. Orbital elements stay in the barycentric frame.
Geocentric-projected `t0`, `u0`, `tE`, and `piE` remain an
Earth-at-`t0par` reporting transform of the shared parameters. Each
filter's `xS0` is left in the barycentric frame the model integrates.
A later optional observer argument on `convert_helio_geo_ast` could
project one filter's zero point; the first version does not add it.

---

## 5. Test plan

Tests go in `tests/test_model.py`, `tests/test_model_jax.py`,
`tests/test_model_old_vs_jax.py`, `tests/test_model_fitter.py`, and
`tests/test_model_fitter_jax.py`. Use `PYTHONPATH=src`. Mark live
Horizons tests so they skip when the cache is cold and the network is
down. Earth-only tests must not need the network. Tolerances below are
for noiseless model comparisons.

1. **One filter matches today.** `n_filters == 1`,
   `obsLocation='earth'`, scalar `xS0`. Photometry, astrometry, lens,
   source, images, and centroid match a model built the current way,
   to `1e-12`. The sampled names are the historical names with
   `xS0_E` renamed to `xS0_E1` in the same column, and `b_sff1`,
   `mag_src1` unmoved. Repeat on the JAX model class.

2. **Same observer, two zero points.** One model, two filters, both
   `'earth'`, both with astrometry, different `xS0`, identical `pi`,
   `mu`, and blend. `get_astrometry(t, filt_idx=1) -
   get_astrometry(t, filt_idx=0)` equals `xS0[1] - xS0[0]` at every
   epoch, to `1e-12` arcsec. The centroid shift agrees between the
   two filters. `xL0[1] - xL0[0]` equals `xS0[1] - xS0[0]`.

3. **Each observer matches a single-observer model.** One object with
   Earth photometry, Spitzer photometry, Gaia astrometry, and a joint
   ground series. Filter `k`'s photometry matches a one-filter model
   with that body, that `mag_src`, and that `b_sff`. Filter `k`'s
   astrometry matches a one-filter model with that body and that
   `xS0`, on the lens, the unlensed source, and the centroid.
   Changing `xS0` of a photometry-only filter, including away from the
   fixed 0, does not change `get_photometry` for that filter.
   Include BSPL photometry-only (should already pass) and BSPL
   phot+ast (fails today because `get_u` drops `filt_idx`).

4. **Satellite parallax sign.** At one epoch,
   `u_earth - u_spitzer = piE_amp * (pvec_spitzer - pvec_earth)` to
   `1e-10` in Einstein radii. The light curves differ by more than
   `0.01` mag near peak on the existing Zang or Shvartzvald parameters,
   evaluated as two photometric filters of one model, with `xS0` fixed
   at 0 on both. The existing single-satellite tests stay.

5. **Fitter stores one list.** Data with the mixed names and an
   `obsLocation` dict. `get_model(truth).obsLocation[3]` is the Gaia
   body and `[2]` is the ground body. `xS0` has shape `(4, 2)`.
   `xS0[0]` and `xS0[1]` are exactly 0 and `xS0_E1` is not in
   `fitter_param_names`. `mag_src[3]` is exactly 0 and `mag_src4` is
   not in the cube. `b_sff4` is in the cube. A missing dict entry and
   a list of the wrong length each raise `ValueError` at solver
   construction. Omitting `obsLocation` yields Earth for every filter.
   A repeated name in `phot_data` raises.

6. **Cube order.** For one astrometric filter and two photometric
   filters that are the same two names, the sampled names equal the
   historical names with only the `xS0` rename to `xS0_E1`, `xS0_N1`.
   For the four-filter mixed example, the sampled tuple is the one in
   section 3, with `xS0_E3` still carrying the suffix 3. A unit test
   freezes that tuple so a later edit cannot switch to name-major
   order or renumber the holes. The same test imports every
   `PSPL_Param` subclass in both model modules and checks the parallel
   lists: equal length, allowed usages, order matching
   `fitter_param_names`, and the derived views equal to usage
   `'phot'`/`'both'` and `'astrom'`/`'both'`.

7. **Mixed likelihood adds.** Build photometry-only Earth,
   photometry-only Spitzer, joint ground photometry plus astrometry,
   and astrometry-only Gaia, noiseless. The mixed solver's
   `log_likely(truth)` equals the sum of four single-filter solvers'
   likelihoods, to `1e-6` in log-likelihood, or tighter if the
   Gaussian terms are identical. Each single-filter model gets the
   shared physical parameters, that filter's observer, that filter's
   `xS0` if it has astrometry, and that filter's blend. The
   photometry-only single-filter models do not include an `xS0` term.
   Repeat with `evaluate_loglik_jax`. The NumPy and JAX values agree
   to a fraction of a nat. This fails today on both the missing
   observer and the missing per-filter `xS0`.

8. **Astrometry-only blend.** Gaia as filter 3, `b_sff` of 1 and of
   0.5, produces different centroids and the same unlensed source
   track on a point-source model. The sampled names contain `b_sff4`
   and do not contain `b_sff_ast4` or `mag_src4`. A BSPL
   astrometry-only filter with `mag_src_pri` fixed at 0 changes its
   centroid when `mag_src_sec` changes, and does not change it when
   the fixed primary is edited away from 0 by the same amount added to
   the secondary (the ratio is what matters).

9. **Weighted solver uses `filt_idx`.** Two joint filters, same
   observer, different `b_sff`. The astrometric term of
   `MicrolensSolverWeighted` changes when the second filter's blend
   changes. Today it uses filter 0 for both.

10. **Old results.** A table with unsuffixed `xS0_E`, `xS0_N`,
    `b_sff1`, `mag_src1`, `b_sff2`, `mag_src2` loads as
    `xS0_E1 == xS0_E2 ==` the old value, with a warning, and the
    photometric values land on the right filters. An unsuffixed
    `pi_ref_frame` is copied the same way onto a RefPar destination.
    A schema-2 file round-trips: `xS0_E1 = 0` comes back from `FIXED`,
    not from a chain column, and `xS0_E3` comes back from the chain.
    A positional `.txt` whose column count matches neither the old
    layout nor a sidecar raises. A one-filter unsuffixed column is
    renamed to `xS0_E1`.

11. **FSPL.** `get_all_arrays_amg(t, filt_idx=1)` matches a
    one-filter model of the second observer and the second `xS0`, for
    amplification and image position.

12. **Geoproj is unchanged.** `get_geoproj_params(t0par)` on a
    multi-observer model equals the same call on an Earth-only model
    with the same shared `t0`, `u0`, `tE`, and `piE`. It does not
    depend on `xS0` or on which body sits in filter 0.

`test_multi_obsLocation` can remain a slow sampler smoke test. A 30
percent parameter tolerance is not evidence that the observer or the
zero point was applied. The tests above are the acceptance tests.

---

## 6. Implementation order

Each phase can merge on its own. Later phases assume the earlier ones.

### Phase 1. Declaration schema, every filter joint

Scope: `PSPL_Param` in both model files, `__init_subclass__`
validation, the expand helper used by both fitters,
`multi_filt_params`, `make_default_priors`, `generate_params_dict`,
`get_model`. The family tables in section 3, including `b_sff` on the
astrometry-only classes. Tests 1 and 6 for the all-sampled, no-hole
case, and the class-list check.

Move photometric names onto the end of `fitter_param_names`. Declare
`filt_param_names` and `filt_param_usage` in `fitter_param_names`
order. Leave `phot_param_names` and `astrom_param_names` as derived
views. Expand each contiguous run of `filt_param_names` index-major,
over one `n_filters`, always suffixed. With every name present in both
data lists, nothing is fixed and the cube is the historical cube plus
an `xS0` rename. The old "ast must have phot" check can stay for this
phase. This is the mechanical migration, and it is where a one-filter
chain is protected by a rename rather than a shift.

### Phase 2. Per-filter `xS0` and `xL0` on the model

Scope: `PSPL` getters in both model files, `PSPL_Param.__init__`
normalization, and the binary families so they re-derive `xL0` and
`xS0_pri` / `xS0_sec` from `xS0[k]`. Tests 2 and 3 for PSPL,
Earth-only so Horizons is not required. Getters keep `filt_idx` and
start indexing `xS0[filt_idx]`. No new keyword.

### Phase 3. Mixed filters and fixed slots

Scope: `build_filt_index`, `check_data`, both likelihoods, the
weighted and Hobson solvers, `obsLocation` dict / list / string,
`fixed_dataset_params`, fake-data name lists, tests 5, 7 (NumPy), 8,
and 9. Remove the mandatory phot–ast name match. This is the phase
that makes a Gaia-only series and a Spitzer-only light curve legal in
one fit, with `xS0` fixed at 0 on Spitzer and `mag_src` fixed at 0 on
Gaia.

### Phase 4. JAX tables and per-filter zero points

Scope: `build_explicit_jax_loglik_fn`, the two context builders, the
`jax_log_likely_astrometry` signatures (scalar cube plus per-filter
`xS0` and `b_sff`), cache key, test 7's JAX half and test 4.
Same merge as phase 3 if PyMC-with-gradients is in use, so the
gradient is not the derivative of an Earth model with one `xS0`.

### Phase 5. BSPL, BSBL, and finite source

Scope: pass `filt_idx` through `get_u`, the dead `get_amplification`
overrides, and the FSPL contours. Tests 3 (binary) and 11, plus the
BSPL ratio half of test 8. Re-read PSBL Keplerian branches for a
dropped index; `get_complex_pos` already takes one. GP means follow
the photometric getter and are created only for filters with
photometry. One GP test that the mean, not the kernel, matches per
photometric filter is enough.

### Phase 6. Plots, loader, aliases

Scope: plot helpers reading data through `phot_series` / `ast_series`,
the overlay helper, `data.getdata` `obs_location`, the schema-2 writer
and the old-column adapter (copy unsuffixed `xS0` and `pi_ref_frame`
into every filter), tests 10 and 12, the alias table in `parallax.py`.

Still deferred: reading `ephem/ephem_JWST.txt`, an observer argument
on `convert_helio_geo_*`, and light-travel time.

---

## 7. Open decisions

1. **Shared barycentric `t0`, `u0`, `piE`, and proper motion, with
   `xS0` per filter?**
   Yes. Per-filter `t0` would double-count parallax. Per-filter proper
   motion would absorb the parallax ellipse that `xS0` is deliberately
   not allowed to absorb. A photometry-only filter does not get a free
   `xS0`; it is fixed at 0.

2. **Geocentric-projected conversion?**
   Leave it on Earth, and leave it on the shared photometric
   parameters. Do not project each `xS0` in this feature.

3. **One index. Closed, replaces v2's two index spaces.**
   `filt_idx` everywhere. The fitter's list is `phot_data` order, then
   astrometry-only names. `b_sff{k}`, `mag_src{k}`, `xS0_E{k}`,
   `xS0_N{k}`, `pi_ref_frame{k}`, and `obsLocation[k]` share `k`.
   There is no `ast_idx` and no second observer list. Array suffixes
   in the data dictionary stay in `phot_data` / `ast_data` order; the
   fitter translates. Model constructors do not take the name lists.

4. **Astrometry-only blend. Closed.**
   Ordinary `b_sff{k}`. No `b_sff_ast{k}` and no `b_sff_astonly`. The
   centroid already indexes `b_sff[filt_idx]`.

5. **`pi_ref_frame`. Closed.**
   Per filter on RefPar, usage `'astrom'`, fixed at 0 when that filter
   has no astrometry. It stays after `muS_N` in `fitter_param_names`.
   The loader copies an unsuffixed old column into every filter slot.

6. **Usage, closed. This replaces the v3 zeropoint proposal.**
   `'phot'` (`mag_src`, `mag_base`, `mag_src_pri`) is fixed at 0 when
   the filter has no photometry. `'astrom'` (`xS0_E`, `xS0_N`,
   `pi_ref_frame`) is fixed at 0 when the filter has no astrometry.
   `'both'` (`b_sff`, `dmag_Lp_Ls`, `fratio_bin`, `mag_src_sec`) is
   sampled on every filter. GP and photometric error terms are still
   created only for filters with photometry. `mag_src_sec` with no
   light curve uses a uniform prior on `[-5, 5]`.

7. **Always suffix `xS0`. Closed.**
   Cube, summaries, and new files use `xS0_E1` / `xS0_N1` even for one
   filter. The constructor keyword stays `xS0_E`. Old FITS columns are
   adapted by name: one unsuffixed `xS0` is copied into every filter
   when several photometric columns are present, and the same copy is
   done for `pi_ref_frame`.

8. **Light-travel time?**
   Omit.

9. **Roman's Horizons name?**
   A short alias table, edited by you. Please pick the spacecraft id.
   Do not special-case `'l2'`.

10. **Vendor `ephem/ephem_JWST.txt`?**
    Not in this feature. Horizons plus the joblib cache stays.

11. **Free `raL`, `decL` under JAX?**
    No. The precomputed table depends on them.

12. **Edit both model files?**
    Yes. The body resolver lives in `parallax.py`. The filter-index
    helper and the cube expander are importable by both fitters. The
    two model modules stay copies.

13. **Should `data.getdata` assign `'spitzer'` automatically?**
    No. Default remains "no key, Earth." `'auto'` or a dict is opt-in.

14. **Index-major expansion, then delete fixed names?**
    Yes. Walk `fitter_param_names`. Expand each contiguous run of
    `filt_param_names` index-major over one `n_filters`, then drop
    suffixed names whose usage does not apply to that filter, without
    renumbering. That keeps `b_sff1, mag_src1, b_sff2, mag_src2` and
    lets `xS0_E3` keep the suffix 3 when filters 0 and 1 are
    photometry only. `filt_param_names` is not rearranged into one
    block of `b_sff`, `mag_src`, `xS0`.
