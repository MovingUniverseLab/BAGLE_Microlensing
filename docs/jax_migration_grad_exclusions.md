# JAX migration — permanent grad exclusions

Updated: 2026-06-13. This session recovered the final **12** FSBL
``get_lens_astrometry`` Param4/8 skip rows: static layouts via heliocentric COM
geometry refresh (``t0_com``/``u0_amp_com`` → ``t0``/``u0``/``xL0``); orbit
layouts already had nonzero squared FD at the fixture (status JSON lag).

Prior session recovered **31** ``get_u`` skip rows and **6** FSBL
``get_lens_astrometry`` Param5/6/7 rows.

**Ceiling: 2848 grad pass | 0 grad skip | 0 grad not_run | 2848/2848 closed.**

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
