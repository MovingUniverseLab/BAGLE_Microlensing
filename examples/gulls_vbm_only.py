"""Predict a prepared Gulls bundle with VBM alone.

Usage: python gulls_vbm_only.py BUNDLE_DIR EPOCHS.csv > predictions.csv
EPOCHS.csv must have columns BJD and observatory_code (0, 1 or 2).
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import VBMicrolensing

bundle = Path(sys.argv[1]).resolve()
config = json.loads((bundle/'vbm.json').read_text())
with open(sys.argv[2], newline='') as f:
    rows = list(csv.DictReader(f))
t = np.array([float(row['BJD']) for row in rows])
codes = np.array([int(row['observatory_code']) for row in rows])
if not len(t) or not np.all(np.isfinite(t)) or np.any((codes < 0) | (codes > 2)):
    raise ValueError('epochs must be finite and observatory codes must be 0, 1 or 2')
if np.min(t) < config['observer_grid_bjd'][0] or np.max(t) > config['observer_grid_bjd'][1]:
    raise ValueError('epochs are outside the supplied observer grid')
vbm = VBMicrolensing.VBMicrolensing()
vbm.LoadSunTable(str(bundle/config['files']['sun']))
vbm.SetObjectCoordinates(str(bundle/config['files']['coordinates']), str(bundle))
if vbm.nsat != 1:
    raise ValueError('expected exactly one satellite ephemeris')
for key, value in config['vbm'].items():
    setattr(vbm, key, value)
writer = csv.writer(sys.stdout)
if config['schema'] == 'gulls_vbm_5.3.5_single_source_1':
    result = np.asarray(vbm.BinaryLightCurveParallax(config['params'], (t-config['time_offset_jd']).tolist()))
    if vbm.parallaxextrapolation or result.shape != (3, len(t)) or not np.all(np.isfinite(result)) or np.any(result[0] <= 0):
        raise ValueError('VBM produced an invalid or extrapolated prediction')
    flux = 1 + np.asarray(config['source_flux_fractions'])[codes]*(result[0]-1)
    writer.writerow(['BJD', 'observatory_code', 'magnification', 'relative_flux', 'source_x', 'source_y'])
    writer.writerows(zip(t, codes, result[0], flux, result[1], result[2]))
elif config['schema'] == 'gulls_vbm_5.3.5_binary_source_1':
    params_by_band = config['params_by_band']
    lum_exponents = config['mass_luminosity_exponents']
    radius_exponents = config['mass_radius_exponents']
    if any(len(items) != 3 for items in (params_by_band, lum_exponents, radius_exponents)):
        raise ValueError('binary VBM bundle must contain three band parameter sets')
    results = []
    for band in range(3):
        vbm.mass_luminosity_exponent = float(lum_exponents[band])
        vbm.mass_radius_exponent = float(radius_exponents[band])
        result = np.asarray(vbm.BinSourceBinLensLightCurve(
            params_by_band[band], (t-config['time_offset_jd']).tolist()
        ))
        if (vbm.parallaxextrapolation or result.shape != (6, len(t))
                or not np.all(np.isfinite(result)) or np.any(result[0] <= 0)):
            raise ValueError(f'VBM produced an invalid or extrapolated band {band} prediction')
        results.append(result)
    source_positions = results[0][1:5]
    if any(not np.allclose(result[1:5], source_positions, rtol=0, atol=1e-8)
           for result in results[1:]):
        raise ValueError('VBM returned band-dependent binary-source trajectories')
    magnification = np.array([results[int(code)][0, i] for i, code in enumerate(codes)])
    source_flux = (
        np.asarray(config['source1_flux_fractions'])
        + np.asarray(config['source2_flux_fractions'])
    )[codes]
    flux = 1 + source_flux*(magnification-1)
    writer.writerow([
        'BJD', 'observatory_code', 'combined_source_magnification',
        'relative_flux', 'source1_x', 'source1_y', 'source2_x', 'source2_y',
        'lens_separation',
    ])
    writer.writerows(zip(t, codes, magnification, flux, *source_positions, results[0][5]))
else:
    raise ValueError(f"unsupported VBM bundle schema: {config['schema']}")
