# JAX migration — permanent grad exclusions

Updated: 2026-06-13. **Extension batch B** (+63 applicable): PSBL_Phot /
BSPL_Phot / BSPL_PhotAstrom own-override ``get_source_astrometry_unlensed``
(see §Extension backlog below).

Prior session **batch A** (+59 applicable): BSPL / FSPL_PhotAstrom /
BFSPL own-override ``get_resolved_amplification``.

Prior session recovered the final **12** FSBL
``get_lens_astrometry`` Param4/8 skip rows: static layouts via heliocentric COM
geometry refresh (``t0_com``/``u0_amp_com`` → ``t0``/``u0``/``xL0``); orbit
layouts already had nonzero squared FD at the fixture (status JSON lag).

Prior session recovered **31** ``get_u`` skip rows and **6** FSBL
``get_lens_astrometry`` Param5/6/7 rows.

**Ceiling before extension: 2848 grad pass | 0 grad skip | 0 grad not_run | 2848/2848 closed.**

After batch A: **2907 applicable** (+59 ``get_resolved_amplification`` on BSPL / FSPL_PhotAstrom / BFSPL).

After batch B: **2970 applicable** (+63 ``get_source_astrometry_unlensed`` on PSBL_Phot / BSPL_Phot / BSPL_PhotAstrom).

## Extension backlog (prioritized)

| Priority | Method family | Families / definer | Count | Semantics | JAX forward | Notes |
|----------|---------------|-------------------|------:|-----------|-------------|-------|
| **A (done)** | ``get_resolved_amplification`` | ``BSPL``, ``FSPL_PhotAstrom``, ``BFSPL_PhotAstrom`` | **+59** | Dual-source ± (BSPL) or ``get_all_arrays`` amp (FSPL/BFSPL) | ``bspl_resolved_amplification_from_model``, ``fspl_resolved_amplification_from_model`` | Excludes PSBL/BSBL/FSBL inheriting PSPL ± |
| **B (done)** | ``get_source_astrometry_unlensed`` | ``PSBL_Phot`` (8), ``BSPL_Phot`` (4), ``BSPL_PhotAstrom`` (51) | **+63** | ``get_u`` in θ_E (phot) or arcsec flux-weighted centroid (photastrom) | ``bspl_source_astrometry_unlensed_from_model``, ``bspl_photastrom_source_astrometry_unlensed_from_model``, PSBL ``get_u`` | Excludes PSBL/BSBL PhotAstrom via PSPL ABC (~46) |
| C | ``get_photometry_with_gp`` | GP classes | 0 gap | — | Wired | 54/54 already applicable |
| D | ``get_astrometry_outline_unlensed`` | FSPL PhotAstrom | 0 gap | Outline AMG | Host AMG | 4/4 already applicable |
| — | ``get_resolved_amplification`` (excluded) | PSBL/BSBL/FSBL PhotAstrom via ``PSPL`` definer | ~82 | Wrong: single-lens 2-image ± on binary lens | PSPL formula only | Do not mark applicable |
| — | ``get_resolved_amplification`` (excluded) | ``PSPL_Phot`` on PSBL phot (8) | 8 | Wrong: PSPL ``get_u`` on binary | Inherited | Do not mark applicable |
| — | ``get_source_astrometry_unlensed`` (excluded) | PSBL/BSBL PhotAstrom via ``PSPL`` | ~78 | PSPL astrometry ABC only | — | Use resolved astrometry paths instead |

### Batch A detail (implemented)

- **BSPL (54)**: ``BSPL.get_resolved_amplification`` — per-source PSPL ± from ``get_u`` ``[t, 2 sources, 2]``.
- **FSPL_PhotAstrom (4)**: ``FSPL_PhotAstrom.get_resolved_amplification`` — ``swapaxes(get_all_arrays amp, 0, 1)``.
- **BFSPL_PhotAstrom (1)**: ``BFSPL_PhotAstrom.get_resolved_amplification`` — ``swapaxes(get_all_arrays amp, 1, 2)``.

Harness: ``resolved_amplification_extension_pairs()``, ``test_parity_resolved_amplification_extension``, ``test_grad_resolved_amplification_extension``.

### Batch B detail (implemented)

- **PSBL_Phot (8)**: ``PSBL_Phot.get_source_astrometry_unlensed`` — alias of ``get_u`` (Einstein radii).
- **BSPL_Phot (4)**: ``BSPL_Phot.get_source_astrometry_unlensed`` — flux-weighted dual-source ``get_u`` centroid.
- **BSPL_PhotAstrom (51)**: ``BSPL_PhotAstrom.get_source_astrometry_unlensed`` — flux-weighted arcsec centroid from ``get_resolved_source_astrometry_unlensed``.

Harness: ``source_astrometry_unlensed_extension_pairs()``, ``test_parity_source_astrometry_unlensed_extension``, ``test_grad_source_astrometry_unlensed_extension``.

These skip rows are **not** marked `grad=pass` without a passing finite-difference smoke
test. They are documented here as permanent exclusions until the underlying FD /
autodiff path is fixed or a different grad verification strategy is adopted.

## Summary

| Category | Count (skip) | Primary reason |
|----------|-------------:|----------------|
| Zero / flat `get_u` | 0 | recovered (+31 pass): geometry refresh + squared FD |
| FSBL ``get_lens_astrometry`` | 0 | recovered (+12 pass): Param4/8 COM refresh + squared FD |
| `get_resolved_astrometry` | ~138 | NaN FD (AMG / image-plane Jacobian) — recovered in prior sessions |
| `get_resolved_lens_astrometry` | ~42 | NaN or zero FD (mostly FSBL orbit) — recovered in prior sessions |
| BSBL Param1 phot / ast / likelihood | 0 | recovered: skip ``root_tol`` in host FD |
| PSBL phot orbit Param1 (resolved / u) | 0 | ``get_u`` recovered; resolved ast recovered earlier |
| BSPL phot noPar `get_u` | 0 | recovered: refresh ``u0`` from ``u0_amp`` + squared FD |
| BSPL phot extended (unwired) | 0 | wired in prior session |
| Other phot / likelihood NaN | remainder | Host FD through roots / orbit chain |

Probe artifacts: `docs/grad_probe_nonresolved.json`, `docs/grad_probe_resolved_{bsbl,bspl,psbl,fsbl}.json`,
`docs/grad_probe_skip_batch.json`, `docs/grad_probe_get_u_reprobe.json` (31/31 pass, 2026-06-13).

## 1. Resolved astrometry (`get_resolved_astrometry`, `get_resolved_lens_astrometry`)

**~180 rows skipped in prior sessions; most recovered.** Remaining permanent exclusions
are genuinely flat lens-astrometry paths (see below).

**Root cause (historical):** finite-difference / autodiff through the AMG image solver
and resolved centroid path produced NaN or zero vectors at standard fixture points.
Parity forward passes; grad smoke did not until squared FD and derived-geometry refresh.

## 2. BSBL Param1 phot / ast / likelihood — **recovered (+140 pass)**

**Fix:** skip ``root_tol`` in ``_fd_grad_host`` / ``_fd_grad_jax_eval``
(``GRAD_FD_SKIP_PARAMS``); perturbing the binary-lens root tolerance breaks
root finding and yields NaN photometry / astrometry at the fixture point.

Harness: ``bsbl_param1_phot_ast_likelihood_grad_recovered_pairs``,
``bsbl_param1_phot_grad_recovered_pairs`` (phot-only subset).

## 3. Zero / flat `get_u` — **recovered (+31 pass)**

Previously misclassified as permanently flat: host FD used ``sum(get_u)`` without
refreshing derived ``u0`` after perturbing packed init parameters.

| Family | Count recovered | Fix |
|--------|----------------:|-----|
| PSBL | 13 | Orbit PhotAstrom Param1/7: re-run ``convert_u0_t0_psbl`` after scatter |
| FSBL | 13 | Same refresh on jax-only FSBL PhotAstrom orbit layouts |
| BSPL | 4 | Static phot/photastrom: refresh ``u0`` from ``u0_amp``/``beta`` |
| FSPL | 1 | FSPL PhotAstrom Par Param1 physical refresh |

**Harness:** ``get_u_grad_recovered_pairs()``; squared sum wired via ``_GET_U_FD_METHODS``.

## 4. FSBL ``get_lens_astrometry`` — **recovered (+18 pass total)**

### Recovered (+18 pass)

| Pair group | Fix |
|------------|-----|
| ``FSBL_PhotAstrom_Par_Param5`` | derived-geometry refresh |
| ``FSBL_PhotAstrom_{Par,noPar}_AccOrbs_Param6`` | derived-geometry refresh |
| ``FSBL_PhotAstrom_Par_{Param7,AccOrbs_Param7,LinOrbs_Param7}`` | squared FD objective |
| **Static Param4/8** (4 rows) | ``_refresh_psbl_param4_heliocentric_geometry`` (``t0_com``/``u0_amp_com`` → ``t0``/``u0``/``xL0``) |
| **Orbit Param4/8** (8 rows) | Already nonzero squared FD; ``beta_com`` orbit layouts use existing COM refresh |

**Harness:** ``fsbl_lens_ast_grad_recovered_pairs()``; squared sum wired via
``_RESOLVED_AST_FD_METHODS`` including ``get_lens_astrometry``.

### Param4/8 vs Param7 (why static Param4/8 looked flat)

| | Param7 (recovered earlier) | Param4/8 static (recovered this session) |
|--|---------------------------|------------------------------------------|
| Init frame | ``t0_p`` / ``beta_p`` (primary lens) | ``t0_com`` / ``u0_amp_com`` (COM, heliocentric) |
| Refresh hook | ``_refresh_psbl_prim_u0_geometry`` | **Missing** until ``_refresh_psbl_param4_heliocentric_geometry`` |
| ``get_lens_astrometry`` | Linear ``xL0`` + ``muL`` motion (+ parallax) | Same JAX path; stale ``xL0``/``t0`` after scatter → zero FD |
| Orbit Param4/8 | N/A | Has ``beta_com``; COM refresh already wired; FD sensitive via ``xS0``/``muS`` |

At ``dmag_Lp_Ls = 0`` (canonical fixture), ``sep``/``alpha`` do not affect the
flux-weighted lens centroid (offset cancels); sensitivity enters through ``xL0``,
``muL``, and orbital elements after proper geometry refresh.

## 5. PSBL phot-only orbit Param1 — **recovered (+8 pass, prior session)**

8 rows: `PSBL_Phot_{Par,noPar}_{CircOrbs,EllOrbs}_Param1` × (`get_amplification`,
`get_photometry`).

PSBL PhotAstrom orbit Param1/7 ``get_u`` rows recovered this session (see §3).

## Rows marked this session

1. **FSBL ``get_lens_astrometry`` Param4/8** (+12 pass): ``fsbl_lens_ast_grad_recovered_pairs``;
   ``_refresh_psbl_param4_heliocentric_geometry`` for static COM init.

## Path to 2848/2848 grad pass

**2848 pass + 0 skip = 2848 closed.** All applicable grad rows now pass FD smoke at
the standard fixture (squared objective for resolved astrometry and ``get_u`` where needed).
