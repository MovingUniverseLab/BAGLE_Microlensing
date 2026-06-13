# JAX migration — permanent grad exclusions

Updated: 2026-06-13. Commit `514879b` recovered 143 skip rows; this session recovered
**3** FSBL ``get_lens_astrometry`` Param7 rows via squared FD objective
(``nansum(arr²)`` in ``_fd_scalar_from_output``).

**Ceiling: 2805 grad pass | 43 grad skip | 0 grad not_run | 2848/2848 closed.**

These skip rows are **not** marked `grad=pass` without a passing finite-difference smoke
test. They are documented here as permanent exclusions until the underlying FD /
autodiff path is fixed or a different grad verification strategy is adopted.

## Summary

| Category | Count (skip) | Primary reason |
|----------|-------------:|----------------|
| Zero / flat `get_u` | 31 | Zero grad norm at fixture point (host + jax-eval FD) |
| FSBL ``get_lens_astrometry`` | 12 | Zero FD at fixture (Param4/8 + orbit Param4/8; squared FD also zero) |
| `get_resolved_astrometry` | ~138 | NaN FD (AMG / image-plane Jacobian) — recovered in prior sessions |
| `get_resolved_lens_astrometry` | ~42 | NaN or zero FD (mostly FSBL orbit) — recovered in prior sessions |
| BSBL Param1 phot / ast / likelihood | 0 | recovered: skip ``root_tol`` in host FD |
| PSBL phot orbit Param1 (resolved / u) | 16 | NaN/zero resolved ast; zero `get_u` on orbit PhotAstrom |
| BSPL phot noPar `get_u` | 4 | Zero grad norm (Par PhotAstrom variants also flat at fixture) |
| BSPL phot extended (unwired) | 0 | wired in prior session |
| Other phot / likelihood NaN | remainder | Host FD through roots / orbit chain |

Probe artifacts: `docs/grad_probe_nonresolved.json`, `docs/grad_probe_resolved_{bsbl,bspl,psbl,fsbl}.json`,
`docs/grad_probe_skip_batch.json` (46-row skip batch probe, 2026-06-13).

## 1. Resolved astrometry (`get_resolved_astrometry`, `get_resolved_lens_astrometry`)

**~180 rows skipped in prior sessions; most recovered.** Remaining permanent exclusions
are zero/flat `get_u` and genuinely flat lens-astrometry paths (see below).

**Root cause (historical):** finite-difference / autodiff through the AMG image solver
and resolved centroid path produced NaN or zero vectors at standard fixture points.
Parity forward passes; grad smoke did not until squared FD and derived-geometry refresh.

## 2. BSBL Param1 phot / ast / likelihood — **recovered (+140 pass)**

**Fix:** skip ``root_tol`` in ``_fd_grad_host`` / ``_fd_grad_jax_eval``
(``GRAD_FD_SKIP_PARAMS``); perturbing the binary-lens root tolerance breaks
root finding and yields NaN photometry / astrometry at the fixture point.

Harness: ``bsbl_param1_phot_ast_likelihood_grad_recovered_pairs``,
``bsbl_param1_phot_grad_recovered_pairs`` (phot-only subset).

## 3. Zero / flat `get_u` (31 rows, permanent skip)

All 31 skip-batch ``get_u`` rows have identically zero host FD and jax-eval FD at the
standard fixture (verified 2026-06-13). Includes:

| Family | Count | Layouts |
|--------|------:|---------|
| PSBL | 13 | PhotAstrom orbit Param1/7 (CircOrbs, EllOrbs, LinOrbs, AccOrbs) |
| FSBL | 13 | PhotAstrom orbit Param1/7 + AccOrbs Param7 |
| BSPL | 4 | Phot noPar Param1/GP + PhotAstrom Par Param1/GP |
| FSPL | 1 | PhotAstrom Par Param1 |

**Rationale:** at the fixture point, projected source–lens separation ``u`` is flat w.r.t.
all packed init parameters for these orbit/phot-only layouts. Par variants that pass
elsewhere pick up only machine-epsilon noise from the parallax path; these layouts do not.

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

Remaining PSBL phot orbit Param1 rows (resolved ast, `get_u`) stay **`grad: skip`**.

## Rows marked this session

1. **FSBL lens ast Param7** (+3 pass): ``fsbl_lens_ast_grad_recovered_pairs`` extended;
   ``get_lens_astrometry`` added to ``_RESOLVED_AST_FD_METHODS``.

## Path to 2848/2848 grad pass

**2805 pass + 43 skip = 2848 closed.** The **43 skip** rows are documented permanent
exclusions. Further grad pass would require:

1. **Different fixture points** where `get_u` / lens astrometry are not flat
2. **Fix resolved astrometry FD** — differentiable AMG or stable host FD through image plane
3. **Accept as skip** — tracked as `grad: skip` in status JSON (current state)

Recommendation: treat **43** as the realistic ceiling; pursue physics/JAX follow-ups
rather than status-json inflation without nonzero FD evidence.
