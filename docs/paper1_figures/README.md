# Paper 1 figure reproductions (BAGLE binaries paper)

Scripts that regenerate the BAGLE-made figures of the binary-model paper
(Overleaf `6521fc5bf2ddbdd122dabd32`, cloned at
`~/doc/papers/bagle_work/paper1`) directly from the parameters quoted in the
paper. The original figure scripts are not available. Values that the paper
does not state are read off the published PNGs and labelled `ASSUMED` in each
script's docstring.

```bash
cd docs/paper1_figures
PYTHONPATH=../../src:. python compare.py   # all figures + output/compare_*.png
```

| Paper figure | Script | Result |
|---|---|---|
| `com_motion` | `fig_com_motion.py` | Exact match |
| `magmaps_varyq`, `magmaps_varysep` | `fig_caustics.py` | Exact match (CoM at (60, -9.5) mas, face-on circular orbit at tp) |
| `magmap` | `fig_maps.py` | Same caustic and lens geometry |
| `csmap` | `fig_maps.py` | Same structure (cross caustic, two zero-shift lobes) |
| `linorbs` | `fig_bspl_linacc.py` | Matches only with delta_muS_sec = (3, 4); see A1 |
| `lin_lenses` | `fig_psbl_linacc.py` | Matches; needs the flipped alpha of B1 |
| `psbl_keplerian` | `fig_keplerian.py` | Close match (muL along +East, tp = t0_com) |
| `bspl_keplerian` | `fig_keplerian.py` | Close match (muS along +East, tp = t0); see B2 |
| `bsbl_keplerian` | `fig_keplerian.py` | Lens tracks match; source tracks cannot; see A6 |
| `parallax_comparison` | `fig_parallax_comparison.py` | MulensModel - BAGLE = 1.9e-11, VBM - BAGLE = 2.9e-5 |
| `parallax_comparison_bspl` | `fig_parallax_comparison.py` | MulensModel - BAGLE = 7.9e-13, VBM - BAGLE = 1.5e-5; see B3 |

Not reproduced: the hand-drawn schematics (`geometry_schematic_binary_lens`,
`psbl_coord_transform`, `psbl_geometry_projection`), the 12-panel
`geometry_schematic_binary_familes`, and the fit figures (`LinAnal`,
`CircAnal`, `EllAnal` and their Ast versions), which need full fits.

## Conventions confirmed against the code

- Thiele-Innes orbit: i = 0 is face-on; the primary's periastron is at
  PA = Omega + omega for i = 0; the secondary uses omega + 180; both bodies
  cross the sky plane receding (+Z away from the observer) at PA = Omega.
- Binary-lens alpha points from the secondary to the primary, East of North
  (static classes), matching Appendix A ("vector from M2 to M1").
- u0 > 0 puts the source East of the lens (Gould 2004); every figure with a
  stated u0 puts the lens on the side the published figure shows.
- The PSBL Keplerian class reproduces P = 1054.41 d and P_L = 2722.46 d from
  (a, masses, dL), and places the lens CoM at xL0_com at t0_com, as Sec. 4.3
  (line 829) defines.
- Helio -> geo conversions reproduce the paper's geo-projected PSBL and BSPL
  parameters at RA = 267.333, Dec = -21.283 deg (the paper only says
  "Galactic Bulge"; Dec ~ -29 fields give u0_geo ~ 0.93 instead of 0.98).

## Paper text / caption issues (A)

1. `linorbs`: the caption says delta_muS_sec = (9, 7), so the secondary
   should move at muS + delta = (15, 10) mas/yr (Eq. linear_motion2). The
   published secondary track has slope 0.77, and its curvature matches
   (9, 7) mas/yr, i.e. delta_muS_sec = (3, 4).
2. `lin_lenses`: accLsec units should be mas/yr^2, not mas/yr.
3. `psbl_keplerian`: "5 mas between the two lenses at t0_com" is the
   semi-major axis a. The separation at t0_com = tp is a(1 - e) = 2 mas.
4. `csmap`: the text says 5 mas between the lenses at tp; the figure draws
   them 1.8 mas apart (s ~ 0.34 also fits the small caustic).
5. `magmap`: the text says the sources are 13 mas apart at tp; they are
   drawn 5.2 mas apart.
6. `bsbl_keplerian`: the caption says "circular orbits" but quotes e = 0.2
   and 0.4. The published source tracks wiggle East-West and in phase with
   each other. A source orbit with i = 90 and Omega = 10 moves nearly
   North-South, with the two sources in anti-phase, in both today's code and
   the 2025-08 code (`5fe9dfa`). The figure therefore used other source
   inputs.
7. PSBL comparison: the geo-projected CoM t0 is 60459.41 from the paper's
   own geo values (paper: 60460.41, 1.00 d off); u0 = 0.78 matches. The
   event coordinates should be given.
8. BSPL comparison: the quoted heliocentric t0_s = 60476.09 was never used
   by the model (B3). The quoted geo values (60461.01, 1.22) are the
   projection of (t0_s = t0_p = 60478.49, u0_s = 1.25).
9. Line 532: "omega is Eastward of Omega + 180" contradicts the code and
   `com_motion`. omega is measured from Omega (the ascending node, shared by
   both orbits), in the direction of motion.
10. The direction of binary-source alpha is not defined in the paper or the
    BSPL docstrings. The code uses the PA of the secondary measured from the
    primary, the opposite sense to the binary-lens alpha.
11. Previously noted: the Appendix C components use cos(theta) for East
    with theta measured from North; line 1855 `t_prim,0` should be
    `t_com,0`.

## Code issues found (B), present on origin/main

1. Binary-lens Lin/AccOrbs: `PSBL_PhotAstrom.get_resolved_lens_astrometry`
   (model.py ~7656) and the BSBL equivalent (~22978), plus model_jax.py,
   place the primary at -sep/2 along alpha. The static branch places it at
   +sep/2, so switching static <-> Lin/AccOrbs flips the binary by 180 deg.
2. ~~`BSPL_PhotAstrom_EllOrbs_Param1` / `Param2` did not reproduce the input
   xS0 at t0~~ (**fixed**): CoM is now `xS0 - d_pri(t0)` from `oal2xy` at
   `t0`; `sep = aleph + aleph_sec` remains the relative semi-major axis.
3. Phot-only `BSPL.get_u` uses u_sec = u0_sec + tau_pri * muRel_hat. It
   keeps only the separation component along u0_hat, so the secondary
   always peaks at the primary's t0 (only phi = +/-90 deg is right).
4. ~~`plot_models.get_centroid_shift_map` could not run (undefined
   `psbl`, `dist2_mas`, `xCentroid_complex_as`, `zgrid_as`)~~ (**fixed**):
   forward PSBL image solving on a source-plane grid, matching
   `fig_maps.csmap()`.
