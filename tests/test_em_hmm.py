"""Tests for our own Baum-Welch implementation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.em_hmm import fit_em

TRUE_A = np.array([[0.99, 0.01], [0.04, 0.96]])
TRUE_MU = np.array([0.0005, -0.001])
TRUE_SD = np.array([0.008, 0.025])


def _simulate(n=8000, seed=3):
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = rng.random() < TRUE_A[s[t - 1], 1] if s[t - 1] == 0 else rng.random() >= TRUE_A[1, 0]
    return rng.normal(TRUE_MU[s], TRUE_SD[s]), s


def test_em_recovers_parameters():
    """EM, which never sees the states, should get close to the transition
    frequencies of the hidden path that actually occurred, and to the true stds."""
    x, s = _simulate()
    e = fit_em(x)
    realised = np.array([[np.mean(s[1:][s[:-1] == i] == j) for j in (0, 1)] for i in (0, 1)])
    assert np.allclose(e.stds, TRUE_SD, rtol=0.05)
    assert np.allclose(e.transmat, realised, atol=0.01)


def test_log_likelihood_never_decreases():
    x, _ = _simulate(3000, seed=9)
    path = np.array(fit_em(x).log_likelihood_path)
    assert np.all(np.diff(path) >= -1e-6)


def test_matches_hmmlearn_without_prior():
    from hmmlearn.hmm import GaussianHMM
    x, _ = _simulate(4000, seed=5)
    e = fit_em(x, tol=0, max_iter=2000)
    m = GaussianHMM(2, covariance_type="diag", n_iter=2000, tol=1e-10, random_state=0,
                    covars_prior=0.0, covars_weight=0.0, min_covar=1e-12)
    m.fit(x.reshape(-1, 1))
    assert abs(m.score(x.reshape(-1, 1)) - e.log_likelihood) < 1e-3
