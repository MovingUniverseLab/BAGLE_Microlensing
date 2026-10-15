"""Shared posterior-agreement check for the sampler tests."""
import numpy as np

from bagle.model_fitter import weighted_quantile

# One seed for fake data, MultiNest, PyMC, and NumPyro / JAXNS.
FIT_SEED = 42

# Extra sampler seeds. The default comparison uses FIT_SEED.
FIT_SEEDS_SLOW = (7, 99, 123)

_QUANTILES = np.array([0.16, 0.50, 0.84])


def assert_posteriors_agree(names, samples_a, samples_b,
                            weights_a=None, weights_b=None,
                            n_sigma=3.0):
    """Require 16/50/84% points to agree within the combined width.

    Parameters
    ----------
    names : sequence of str
        Parameter names present in both sample collections.
    samples_a, samples_b : mapping or table
        Posterior samples, one column per parameter.
    weights_a, weights_b : array-like or None
        Sample weights. ``None`` is an unweighted quantile.
        MultiNest tables pass their ``weights`` column.
    n_sigma : float, optional
        Allowed |q_a - q_b|, in units of the combined half-width.
        The half-width is ``0.5 * (q84 - q16)``.

    Raises
    ------
    AssertionError
        When any of the three quantiles is wider apart than
        ``n_sigma * sqrt(sig_a**2 + sig_b**2)``. Equal values
        pass, including two zero-width posteriors.
    """
    failures = []
    for name in list(names):
        qa = _quantiles(samples_a[name], weights_a)
        qb = _quantiles(samples_b[name], weights_b)
        # Half-width of the central 68% interval.
        sig_a = 0.5 * (float(qa[2]) - float(qa[0]))
        sig_b = 0.5 * (float(qb[2]) - float(qb[0]))
        limit = float(n_sigma) * np.sqrt(sig_a ** 2 + sig_b ** 2)
        labels = ('q16', 'q50', 'q84')
        for label, a, b in zip(labels, qa, qb):
            diff = abs(float(a) - float(b))
            bad = not np.isfinite(diff) or not np.isfinite(limit)
            # Strict ``>`` so a difference of zero passes.
            if bad or diff > limit:
                failures.append(
                    f'{name} {label}: |{float(a):.6g} - {float(b):.6g}|'
                    f' = {diff:.6g} > {limit:.6g}'
                    f' (sig {sig_a:.6g}, {sig_b:.6g})'
                )
    if failures:
        msg = 'posteriors disagree:\n' + '\n'.join(failures)
        raise AssertionError(msg)
    return None


def _quantiles(values, weights):
    """16/50/84 percent points, weighted when weights are given."""
    values = np.asarray(values, dtype=float)
    if weights is None:
        return np.quantile(values, _QUANTILES)
    return weighted_quantile(
        values, _QUANTILES, sample_weight=np.asarray(weights, dtype=float))
