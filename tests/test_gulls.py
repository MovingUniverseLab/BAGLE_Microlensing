"""Two fixed examples document Gulls truth → BAGLE → high-level VBM.

The reference parameters/fluxes came from the independent native-truth VBM
exporter. They are checked-in values, not recomputed expectations. This is
an alert for changes to this conversion or its dependencies, not a test of
VBM's integration algorithm or the historical simulation's physical validity.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from bagle import frame_convert
from bagle.gulls import JD_ZERO, JD_TO_MJD, load_model


EXAMPLES = json.loads((Path(__file__).parents[1] /
    'examples/data/gulls_beginner_round_trip.json').read_text())['events']


@pytest.mark.parametrize('example', EXAMPLES,
                         ids=[e['truth']['name'] for e in EXAMPLES])
def test_gulls_bagle_vbm_example(example, tmp_path):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    pytest.importorskip('VBMicrolensing')
    truth_path = tmp_path/'truth.parquet'
    pq.write_table(pa.Table.from_pylist([example['truth']]), truth_path)

    model = load_model(example['truth']['name'], truth_path)
    mjd = ((JD_ZERO-JD_TO_MJD) + np.array(example['simulation_time']))
    # The outward converter must use BAGLE's derived variables, not recover
    # the input row and bypass BAGLE's parameter conversion.
    del model.raw
    for band in range(3):
        # Frozen reference and converted VBM both use BinaryMag2 with
        # Tol=1e-4, RelTol=1e-6, minannuli=1, a1=0 (uniform brightness).
        # These integration controls are separate from the assertion's
        # 0.03 ppm comparison threshold below; no parameters are fitted.
        vbm, parameters = frame_convert.get_vbm_model(
            model, times_mjd=mjd, filt_idx=band)
        expected = np.array(example['vbm_parameters_by_filter'][band])
        parameters = np.array(parameters)
        # Angles differing by a full turn encode the same trajectory.
        difference = parameters-expected
        difference[3] = (difference[3]+np.pi) % (2*np.pi)-np.pi
        np.testing.assert_allclose(difference, 0., rtol=0., atol=5e-9)

        routine = (vbm.BinSourceBinLensLightCurve if len(parameters) == 18
                   else vbm.BinaryLightCurveParallax)
        result = np.asarray(routine(parameters.tolist(), (mjd-49999.5).tolist()))
        flux = 1 + model.b_sff[band]*(result[0]-1)
        reference = example['direct_vbm_flux_by_filter'][band]
        # 0.03 ppm allows observed floating-point time/geometry differences
        # (up to 0.013 ppm); a larger change asks for review, not a new fit.
        np.testing.assert_allclose(flux, reference, rtol=3e-8, atol=0.)
