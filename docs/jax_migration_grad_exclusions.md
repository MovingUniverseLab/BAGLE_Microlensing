# JAX migration — permanent grad exclusions

Updated: 2026-06-13. This session recovered **31** ``get_u`` skip rows via
derived-geometry refresh (``t0_com``/``t0_p`` → geometric ``u0``) and squared
FD objective (``nansum(u²)``). Prior session recovered 3 FSBL
``get_lens_astrometry`` Param7 rows the same way.

**Ceiling: 2836 grad pass | 12 grad skip | 0 grad not_run | 2848/2848 closed.**

These skip rows are **not** marked `grad=pass` without a passing finite-difference smoke
test. They are documented here as permanent exclusions until the underlying FD /
autodiff path is fixed or a different grad verification strategy is adopted.

## Summary

| Category | Count (skip) | Primary reason |
|----------|-------------:|----------------|
| Zero / flat `get_u` | 0 | recovered (+31 pass): geometry refresh + squared FD |
| FSBL ``get_lens_astrometry`` | 12 | Zero FD at fixture (Param4/8 + orbit Param4/8; squared FD also zero) |
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

## 4. FSBL ``get_lens_astrometry`` (12 permanent + 6 recovered)

### Recovered (+6 pass)

| Pair | Fix |
|------|-----|
| ``FSBL_PhotAstrom_Par_Param5`` | derived-geometry refresh |
| ``FSBL_PhotAstrom_{Par,noPar}_AccOrbs_Param6`` | derived-geometry refresh |
| ``FSBL_PhotAstrom_Par_{Param7,AccOrbs_Param7,LinOrbs_Param7}`` | squared FD objective |

**Harness:** ``fsbl_lens_ast_grad_recovered_pairs()``; squared sum wired via
``_RESOLVED_AST_FD_METHODS`` including ``get_lens_astrometry``.

### Permanent skip (12 rows)

Param4/Param8 static and CircOrbs/EllOrbs Param4/Param8 (Par + noPar): host FD, jax-eval
FD, and squared FD all return zero norm at the fixture. Lens astrometry is genuinely flat
w.r.t. init parameters for these layouts (binary-lens positions cancel under plain sum and
remain flat under squared objective).

## 5. PSBL phot-only orbit Param1 — **recovered (+8 pass, prior session)**

8 rows: `PSBL_Phot_{Par,noPar}_{CircOrbs,EllOrbs}_Param1` × (`get_amplification`,
`get_photometry`).

PSBL PhotAstrom orbit Param1/7 ``get_u`` rows recovered this session (see §3).

## Rows marked this session

1. **``get_u`` skip batch** (+31 pass): ``get_u_grad_recovered_pairs``;
   ``_refresh_psbl_*`` / BSPL / FSPL geometry helpers; ``_GET_U_FD_METHODS``.

## Path to 2848/2848 grad pass

**2836 pass + 12 skip = 2848 closed.** The **12 skip** rows are documented permanent
exclusions (FSBL ``get_lens_astrometry`` Param4/8). Further grad pass would require:

1. **Different fixture points** where lens astrometry is not flat
2. **Fix resolved astrometry FD** — differentiable AMG or stable host FD through image plane
3. **Accept as skip** — tracked as `grad: skip` in status JSON (current state)

Recommendation: treat **12** as the realistic ceiling; pursue physics/JAX follow-ups
rather than status-json inflation without nonzero FD evidence.
